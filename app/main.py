from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("structured-parser")

# -------------------- Config --------------------
MODEL_REPO = os.getenv("MODEL_REPO", "microsoft/Phi-3-mini-4k-instruct-gguf")
MODEL_FILE = os.getenv("MODEL_FILE", "Phi-3-mini-4k-instruct-q4.gguf")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
INFERENCE_CONCURRENCY = int(os.getenv("INFERENCE_CONCURRENCY", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "120"))
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "8"))

model: Optional[Llama] = None
inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)
model_lock = asyncio.Lock()

app = FastAPI(
    title="Async Candidate Parser",
    version="1.0.0",
    description="Parses free-form candidate prompts into normalized JSON using a local LLM.",
)


# -------------------- Models --------------------
class ParseRequest(BaseModel):
    prompt: str = Field(
        ...,
        examples=["Write query here."],
    )


class Experience(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None


class CandidateData(BaseModel):
    name: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    optionalSkills: list[str] = Field(default_factory=list)
    instituteName: list[str] = Field(default_factory=list)
    course: list[str] = Field(default_factory=list)
    experience: Experience = Field(default_factory=Experience)
    phoneNumber: Optional[str] = None
    email: Optional[str] = None


class ParseResponse(BaseModel):
    data: CandidateData


# -------------------- Model Load --------------------
def load_model() -> None:
    """Download and load the GGUF model."""
    global model
    if model is not None:
        return

    logger.info("Downloading model %s/%s...", MODEL_REPO, MODEL_FILE)
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        repo_type="model",
    )

    logger.info("Loading model from %s...", model_path)
    model = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=-1,  # Use all available GPU layers (e.g., Metal)
        verbose=False,
    )
    logger.info("Model loaded successfully.")


async def ensure_model_loaded() -> None:
    if model is not None:
        return
    async with model_lock:
        if model is None:
            await asyncio.to_thread(load_model)


# -------------------- Prompting --------------------
def build_json_prompt(user_prompt: str) -> str:
    """Construct an instruction that enforces JSON-only output."""
    # Keep your structure but be very explicit about no extra tokens.
    return (
        "<|system|>\n"
        "You are an expert JSON data extractor for candidate requirements.\n"
        "\n"
        "Rules for SKILL extraction:\n"
        "- Always extract ALL skills mentioned directly or implicitly.\n"
        "- If the user mentions a job role (e.g., 'MERN full stack lead'),\n"
        "  break it into its component skills (e.g., 'MERN', 'full stack').\n"
        "- Do NOT return the whole sentence as a skill.\n"
        "- Skills should be short keywords or phrases only.\n"
        "- If a phrase contains a skill + description, extract ONLY the skill name.\n"
        "- Optional skills: extract only the core skill (e.g., 'Docker' from 'Docker experience would be nice to have').\n"
        "\n"
        "Rules for OPTIONAL SKILLS:\n"
        "- Sentences containing: 'optional', 'nice to have', 'preferably',\n"
        "  'would be nice', 'preferred', 'bonus', 'plus' → classify as optional.\n"
        "- Extract **only the skill keyword**, not the entire sentence.\n"
        "\n"
        "Rules for EXPERIENCE extraction:\n"
        "- Look for keywords: 'fresher', 'fresh graduate', 'entry level', 'junior',\n"
        "  'experienced', 'senior', 'lead', 'principal', 'staff', 'veteran'.\n"
        "- Experience level mapping:\n"
        "  * Fresher/Fresh graduate/Entry level/Junior → {\"min\": 0, \"max\": 2}\n"
        "  * Experienced/Mid-level → {\"min\": 2, \"max\": 5}\n"
        "  * Senior/Lead/Principal → {\"min\": 5, \"max\": 10}\n"
        "  * Staff/Veteran/Expert → {\"min\": 10, \"max\": null}\n"
        "- If specific years mentioned (e.g., '3-5 years', '5+ years'), use those exact values.\n"
        "- If no experience mentioned, leave as {\"min\": null, \"max\": null}.\n"
        "\n"
        "DO NOT invent skills or data that are not explicitly mentioned.\n"
        "Return strictly valid JSON.\n"
        "<|end|>\n"
        "<|user|>\n"
        f"Extract candidate requirements from:\n\n\"{user_prompt.strip()}\"\n\n"
        "Return ONLY valid JSON:\n"
        '{"data":{"name":null,"skills":[],"optionalSkills":[],"instituteName":[],'
        '"course":[],"experience":{"min":null,"max":null},"phoneNumber":null,"email":null}}\n'
        "<|end|>\n"
        "<|assistant|>\n"
    )


# -------------------- Inference --------------------
def generate_completion(prompt: str) -> str:
    """Run blocking generation on the loaded model."""
    assert model is not None, "Model must be loaded."
    response = model(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        echo=False,
        stop=["<|end|>", "</s>"],
    )
    text = response["choices"][0]["text"].strip()
    logger.info("Raw generation: %s", text[:500])
    return text


async def invoke_model(prompt: str) -> str:
    """Async wrapper that throttles concurrent inference with timeout."""
    await ensure_model_loaded()
    async with inference_semaphore:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(generate_completion, prompt),
                timeout=INFERENCE_TIMEOUT,
            )
            logger.info("Model output: %s", result[:300])
            return result
        except asyncio.TimeoutError:
            logger.error("Inference timeout after %d seconds",
                         INFERENCE_TIMEOUT)
            raise HTTPException(
                status_code=504,
                detail=f"Model inference timed out after {INFERENCE_TIMEOUT} seconds",
            )


# -------------------- JSON Extraction & Repair --------------------
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    # Remove ``` and ```json fences if the model ever emits them
    return FENCE_RE.sub("", text)


def extract_first_balanced_json(text: str) -> str:
    """
    Extract the first balanced {...} JSON object from text,
    respecting strings and escape sequences so braces inside strings are ignored.
    """
    s = text
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start: i + 1]
        start = s.find("{", start + 1)
    raise ValueError("No JSON object found in model output.")


def repair_json(json_blob: str) -> str:
    """
    Apply targeted fixes for common LLM JSON glitches without changing valid content.
    """
    j = json_blob

    # Remove trailing commas before } or ]
    j = re.sub(r",\s*([}\]])", r"\1", j)

    # If the model emitted unquoted junk tokens between a comma and the *next* quoted key
    # e.g., ... "phoneNumber": null,\n e \n "email": null ...
    # delete that junk, keeping the comma and whitespace.
    j = re.sub(r",\s*[A-Za-z_][A-Za-z0-9_]*\s*(?=\s*\")", ", ", j)

    # Also handle cases like: null,e"email"  (your original regex, generalized)
    j = re.sub(
        r'(null|true|false|\d+|"[^"\\]*(?:\\.[^"\\]*)*")\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*(?=\s*\")',
        r"\1,",
        j,
    )

    # Very defensive: collapse accidental control chars that sometimes sneak in
    # (keep tabs/newlines/spaces; remove other C0 controls):
    j = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", j)

    return j


def parse_model_output(raw_output: str) -> dict:
    """Convert model completion into a Python dict with robust extraction/repair."""
    raw = strip_code_fences(raw_output)

    # 1) Try to extract first balanced JSON respecting strings/escapes
    json_blob = extract_first_balanced_json(raw)

    # 2) Repair common glitches
    repaired = repair_json(json_blob)

    # 3) Parse; if it fails once, try a second pass with extra hardening
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        logger.error(
            "Failed to parse JSON after repair (first attempt): %s", repaired[:500])

        # Extra-hardening pass: ensure keys are quoted by removing any bare tokens
        # between { or , and the next quote (should already be handled, but belt & suspenders)
        hardened = re.sub(
            r"([{\[,]\s*)[A-Za-z_][A-Za-z0-9_]*\s*(?=\s*\")", r"\1", repaired)

        try:
            parsed = json.loads(hardened)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse JSON after hardening: %s", hardened[:500])
            raise ValueError(
                "Model output could not be parsed as JSON.") from exc

    # 4) Normalize null lists to empty lists for Pydantic validation
    if "data" in parsed and isinstance(parsed["data"], dict):
        data = parsed["data"]
        # Convert null to empty list for list fields
        for field in ["skills", "optionalSkills", "instituteName", "course"]:
            if field in data and data[field] is None:
                data[field] = []

    return parsed


# -------------------- FastAPI Routes --------------------
@app.on_event("startup")
async def startup_event() -> None:
    """Warm the model when the app boots."""
    await ensure_model_loaded()


@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running and model is loaded."""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "starting",
        "model_loaded": model_loaded,
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "max_concurrent_requests": INFERENCE_CONCURRENCY,
    }


@app.post("/parse", response_model=ParseResponse)
async def parse_endpoint(payload: ParseRequest) -> ParseResponse:
    """Parse free-form prompts into the canonical candidate structure."""
    try:
        structured_prompt = build_json_prompt(payload.prompt)
        raw_output = await invoke_model(structured_prompt)

        try:
            parsed = parse_model_output(raw_output)
        except ValueError as exc:
            logger.exception("Failed to parse model response: %s", raw_output)
            raise HTTPException(
                status_code=500, detail="Model output was not valid JSON.") from exc

        candidate_payload = parsed.get("data", {})
        candidate = CandidateData.model_validate(candidate_payload)
        return ParseResponse(data=candidate)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during parsing")
        raise HTTPException(
            status_code=500, detail="Internal server error") from exc
