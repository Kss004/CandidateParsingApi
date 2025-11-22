"""
HR-Only Chat API - ChatGPT-style conversational interface for HR tasks


Supported HR tasks:
- Job description creation
- Resume parsing and analysis
- Candidate ranking and comparison
- Skill extraction
- Interview question generation
- HR policy questions
- Compensation analysis
- Any other HR-related queries

Non-HR tasks are rejected with HR_SCOPE_VIOLATION.
"""

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional
from uuid import uuid4
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import asyncio
import logging
import time
import os
from collections import defaultdict
from datetime import datetime, timedelta
import PyPDF2
import docx
import io
import docx
import io
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --------------------------- Configuration ---------------------------
MODEL_REPO = os.getenv("MODEL_REPO", "bartowski/Qwen2.5-7B-Instruct-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
MODEL_NAME = "hr-assistant-1"

# Generation settings
# 0.1 for structured, 0.7 for conversation
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))  # Max response length
# Context window - reduced for stability with Qwen
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", "8"))  # CPU threads for inference
TOP_P = float(os.getenv("TOP_P", "0.95"))  # Nucleus sampling

# Request settings
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "120"))
INFERENCE_CONCURRENCY = int(os.getenv("INFERENCE_CONCURRENCY", "8"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# HR scope validation keywords (non-HR topics to reject)
NON_HR_KEYWORDS = [
    "docker", "kubernetes", "how to build", "install", "math problem", "physics",
    "legal advice", "contract law", "tax advice", "chemistry", "exploit", "hack",
    "malware", "bomb", "weapon", "cryptocurrency trading", "blockchain development",
    "medical diagnosis", "financial investment", "stock trading"
]

# --------------------------- App Setup ---------------------------
app = FastAPI(
    title="HR Assistant Chat API",
    version="1.0.0",
    description="ChatGPT-style conversational API restricted to HR tasks only"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hr_chat_api")

# --------------------------- Models ---------------------------


class Message(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    userId: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    id: str
    model: str
    created: int
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class StructuredResponse(BaseModel):
    id: str
    model: str
    usage: Dict[str, int]
    response: Dict[str, Any]


class ResumeSessionRequest(BaseModel):
    query: str
    sessionId: Optional[str] = None
    userId: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    userId: Optional[str] = None


class ChatCompletionStructuredResponse(BaseModel):
    id: str
    model: str
    usage: Dict[str, int]
    response: Dict[str, Any]


class ScoreBreakdown(BaseModel):
    skillsMatch: int
    experienceRelevance: int
    educationAlignment: int
    embeddingSimilarity: float


class CandidateScoreDetailed(BaseModel):
    name: str
    filename: str
    overallScore: int
    fitCategory: str  # "Top Fit", "Good Fit", "Potential Fit", "Not Fit"
    breakdown: ScoreBreakdown
    matchedSkills: List[str]
    missingSkills: List[str]
    reasoning: str


class RankingResponseDetailed(BaseModel):
    jobDescription: str
    summaryScore: float
    candidates: List[CandidateScoreDetailed]
    userId: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

# --------------------------- Rate Limiter ---------------------------


class RateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()

    async def check_rate_limit(self, user_id: str) -> bool:
        async with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id]
                if req_time > cutoff
            ]
            if len(self.requests[user_id]) >= self.max_requests:
                return False
            self.requests[user_id].append(now)
            return True


rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)

# --------------------------- Resume Session Storage ---------------------------


class ResumeSessionManager:
    """Manages resume sessions for multi-turn conversations."""

    def __init__(self):
        self.sessions = {}  # sessionId -> {resume_text, messages, created_at}
        self.lock = asyncio.Lock()

    async def create_session(self, resume_text: str) -> str:
        """Create a new session with resume context."""
        session_id = "resume-" + uuid4().hex[:16]
        async with self.lock:
            self.sessions[session_id] = {
                "resume_text": resume_text,
                "messages": [],
                "created_at": datetime.now()
            }
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data."""
        async with self.lock:
            return self.sessions.get(session_id)

    async def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session history."""
        async with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["messages"].append({
                    "role": role,
                    "content": content
                })

    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        async with self.lock:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            to_remove = [
                sid for sid, data in self.sessions.items()
                if data["created_at"] < cutoff
            ]
            for sid in to_remove:
                del self.sessions[sid]


resume_session_manager = ResumeSessionManager()

# --------------------------- Utilities ---------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimation."""
    return len(text) // 4


def estimate_usage(prompt: str, completion: str) -> Dict[str, int]:
    """Estimate token usage from prompt and completion text."""
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }


def gen_id() -> str:
    return "chatcmpl-" + uuid4().hex[:12]


def contains_non_hr_content(text: str) -> bool:
    """Check if text contains non-HR keywords."""
    low = text.lower()
    for k in NON_HR_KEYWORDS:
        if k in low:
            return True
    return False


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        raise HTTPException(status_code=400, detail="Failed to parse PDF file")


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting DOCX: {e}")
        raise HTTPException(
            status_code=400, detail="Failed to parse DOCX file")


def extract_text_from_file(filename: str, file_content: bytes) -> str:
    """Extract text from uploaded file based on extension."""
    filename_lower = filename.lower()

    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(file_content)
    elif filename_lower.endswith('.txt'):
        return file_content.decode('utf-8')
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload PDF, DOCX, or TXT files."
        )


def parse_json_safe(text: str) -> Any:
    """Try to parse JSON from text with repairs."""
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract first {} block
    import re
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        cand = m.group(1)
        cand2 = cand.replace("'", '"')
        cand2 = re.sub(r",\s*\}", "}", cand2)
        cand2 = re.sub(r",\s*\]", "]", cand2)
        try:
            return json.loads(cand2)
        except Exception:
            pass
    return None


def is_hr_related(text: str) -> bool:
    """
    Determine if the query is HR-related.
    Returns True if it's HR-related, False otherwise.
    """
    hr_keywords = [
        "job", "candidate", "resume", "cv", "hire", "hiring", "recruit", "interview",
        "skill", "experience", "qualification", "employee", "hr", "human resource",
        "compensation", "salary", "benefit", "performance", "evaluation", "onboarding",
        "offboarding", "training", "development", "career", "promotion", "jd",
        "job description", "talent", "workforce", "team", "manager", "leadership"
    ]

    low = text.lower()

    # Check for HR keywords
    for keyword in hr_keywords:
        if keyword in low:
            return True

    # If no HR keywords and contains non-HR keywords, reject
    if contains_non_hr_content(text):
        return False

    # Default: allow if ambiguous (can be tuned)
    return True

# --------------------------- System Prompt ---------------------------


HR_SYSTEM_PROMPT = """You are an expert HR Assistant AI. You help with all HR-related tasks including:

- Creating and reviewing job descriptions
- Parsing and analyzing resumes
- Ranking and comparing candidates
- Extracting skills and qualifications
- Generating interview questions
- Providing HR policy guidance
- Analyzing compensation and benefits
- Assisting with performance evaluations
- Helping with onboarding and training
- Any other HR and recruitment tasks

You ONLY handle HR-related queries. If a user asks about non-HR topics (like coding tutorials, math problems, legal advice, medical questions,stocks, investing, etc.), politely decline and remind them you only assist with HR tasks.

Be professional, helpful, and conversational. Provide detailed, actionable responses."""

# --------------------------- Model Client ---------------------------


class HRChatModel:
    """Chat model wrapper for HR assistant."""

    def __init__(self):
        self.model = None
        self.model_lock = asyncio.Lock()
        self.inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)
        self.embedding_model = None
        logger.info("HRChatModel initialized.")

    def _load_embedding_model(self) -> None:
        """Load the sentence transformer model for embeddings."""
        if self.embedding_model is not None:
            return
        
        logger.info("Loading embedding model all-MiniLM-L6-v2...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded.")

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.embedding_model is None:
            self._load_embedding_model()
        return self.embedding_model.encode(text)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Reshape for sklearn
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        return float(cosine_similarity(emb1, emb2)[0][0])

    def _load_model(self) -> None:
        if self.model is not None:
            return

        logger.info("Downloading model %s/%s...", MODEL_REPO, MODEL_FILE)
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            repo_type="model",
        )

        logger.info("Loading model from %s...", model_path)
        self.model = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=-1,
            verbose=False,
            n_batch=256,  # Reduced batch size for stability
            rope_freq_base=0,  # Auto-detect RoPE frequency
            rope_freq_scale=0,  # Auto-detect RoPE scaling
        )
        logger.info("Model loaded successfully with n_ctx=%d", N_CTX)

    async def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        async with self.model_lock:
            if self.model is None:
                await asyncio.to_thread(self._load_model)
            if self.embedding_model is None:
                await asyncio.to_thread(self._load_embedding_model)

    def _build_prompt(self, messages: List[Message]) -> str:
        """Build prompt from message history using Qwen 2.5 format."""
        # Add system prompt if not present
        has_system = any(m.role == "system" for m in messages)

        prompt = "<|im_start|>system\n"
        if has_system:
            # Use provided system message
            for msg in messages:
                if msg.role == "system":
                    prompt += msg.content + "<|im_end|>\n"
                    break
        else:
            # Use default HR system prompt
            prompt += HR_SYSTEM_PROMPT + "<|im_end|>\n"

        # Add conversation messages
        for msg in messages:
            if msg.role == "user":
                prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif msg.role == "assistant":
                prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"

        # Start assistant response
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _call_sync(self, prompt: str, temperature: float, max_tokens: int) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=TOP_P,
            echo=False,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        text = response["choices"][0]["text"].strip()
        return text

    async def generate(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """Generate response from message history."""
        await self.ensure_loaded()

        prompt = self._build_prompt(messages)

        async with self.inference_semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._call_sync, prompt,
                                      temperature, max_tokens),
                    timeout=INFERENCE_TIMEOUT
                )
                logger.info("Generated response: %d chars", len(result))
                return result
            except asyncio.TimeoutError:
                logger.error("Inference timeout after %d seconds",
                             INFERENCE_TIMEOUT)
                raise HTTPException(
                    status_code=504,
                    detail=f"Request timed out after {INFERENCE_TIMEOUT} seconds"
                )


chat_model = HRChatModel()

# --------------------------- Main Processing ---------------------------


async def process_structured_chat_completion(req: ChatCompletionRequest) -> ChatCompletionStructuredResponse:
    """
    Process structured chat completion with JSON validation and regeneration.
    Follows the HR-SLM specification exactly.
    """
    if not req.messages or len(req.messages) == 0:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Get user ID
    user_id = req.userId or "anonymous"

    # Rate limiting
    if not await rate_limiter.check_rate_limit(user_id):
        return ChatCompletionStructuredResponse(
            id=gen_id(),
            model="slm-hr-1",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            response={
                "data": "RATE_LIMIT_EXCEEDED",
                "type": "error"
            }
        )

    # Extract last user message for HR scope check
    last_user_msg = None
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_msg = m.content
            break

    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    # HR scope validation
    if not is_hr_related(last_user_msg):
        logger.info("HR scope violation for user: %s", user_id)
        return ChatCompletionStructuredResponse(
            id=gen_id(),
            model="slm-hr-1",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            response={
                "data": "HR_SCOPE_VIOLATION",
                "type": "error"
            }
        )

    # Build structured prompt with specific schemas
    system_prompt = """You are an HR-only SLM. You can perform resume parsing, job descriptions, skill extraction, candidate search criteria creation, HR documentation, and evaluation tasks only.

Anything outside HR must return HR_SCOPE_VIOLATION.

Always return a response in the required { data, type } format with correct data types.
No additional text or explanation.

REQUIRED OUTPUT SCHEMAS (Choose the one that matches the user's request):

1. RESUME PARSING:
{
  "data": {
    "name": "string",
    "email": "string",
    "phone": "string",
    "skills": ["string"],
    "education": ["string"],
    "experienceInYears": number,
    "workHistory": [{"company": "string", "role": "string"}]
  },
  "type": "object"
}

2. JOB DESCRIPTION GENERATION:
{
  "data": {
    "title": "string",
    "summary": "string",
    "responsibilities": ["string"],
    "requirements": ["string"],
    "preferredQualifications": ["string"]
  },
  "type": "object"
}

3. SKILL EXTRACTION & CODING REQUIREMENT:
{
  "data": {
    "isCodingJob": boolean,
    "coreSkills": ["string"],
    "softSkills": ["string"],
    "codingQuestionsRequired": boolean,
    "codingSkills": ["string"]
  },
  "type": "object"
}

4. CANDIDATE SEARCH CRITERIA:
{
  "data": {
    "name": "string",
    "skills": ["string"],
    "optionalSkills": ["string"],
    "instituteName": ["string"],
    "course": ["string"],
    "experience": {"min": number, "max": number},
    "phoneNumber": "string",
    "email": "string"
  },
  "type": "object"
}

Return ONLY valid JSON in the exact format above. Do not include markdown formatting like ```json."""

    # Add system prompt if not present
    messages_with_system = []
    has_system = any(m.role == "system" for m in req.messages)
    if not has_system:
        messages_with_system.append(
            Message(role="system", content=system_prompt))
    else:
        # If system prompt exists, we prepend our instructions to it or just add ours as a second system message
        # But for this specific requirement, we should probably force our context.
        # Let's prepend our context to the existing system message or add it as the first message.
        # The requirement says "Internal System Context (Injected Automatically)".
        # So we should probably force it.
        pass

    # Actually, let's just reconstruct the messages list to ensure our system prompt is first.
    # If the user provided a system prompt, we can keep it as a secondary instruction or merge it.
    # The requirement says: "Internal System Context (Injected Automatically)"
    
    final_messages = [Message(role="system", content=system_prompt)]
    for m in req.messages:
        if m.role != "system":
            final_messages.append(m)
        else:
            # If user provided system prompt, append it as additional context
            final_messages.append(Message(role="system", content=m.content))

    # Regeneration loop (up to 3 attempts)
    final_parsed = None
    final_error = None
    all_completions = []

    for attempt in range(1, 4):  # 3 attempts max
        logger.info("Structured completion attempt %d for user %s",
                    attempt, user_id)

        try:
            start_time = time.time()
            # Low temp for structured
            response_text = await chat_model.generate(final_messages, 0.1, MAX_TOKENS)
            all_completions.append(response_text)
            duration = time.time() - start_time

            logger.info("Generated response in %.2fs (attempt %d)",
                        duration, attempt)

            # Try to parse JSON
            parsed = parse_json_safe(response_text)
            
            # Handle the case where the model returns a raw string "HR_SCOPE_VIOLATION"
            # which parse_json_safe might return as a string if it was quoted, or None if not.
            if not parsed:
                if "HR_SCOPE_VIOLATION" in response_text:
                    parsed = {"data": "HR_SCOPE_VIOLATION", "type": "error"}
                else:
                    final_error = "Unable to parse JSON from model output"
                    logger.warning("Parse failure on attempt %d. Output: %s", attempt, response_text[:100])
                    continue

            # If parsed is just the string "HR_SCOPE_VIOLATION", wrap it
            if isinstance(parsed, str) and "HR_SCOPE_VIOLATION" in parsed:
                parsed = {"data": "HR_SCOPE_VIOLATION", "type": "error"}

            # Validate structure
            if not isinstance(parsed, dict) or "data" not in parsed or "type" not in parsed:
                final_error = "Missing required keys 'data' or 'type' in model output"
                logger.warning("Schema keys missing on attempt %d", attempt)
                continue
            
            # Check for HR_SCOPE_VIOLATION in data if it's a string
            if isinstance(parsed["data"], str) and parsed["data"] == "HR_SCOPE_VIOLATION":
                 # This is a valid response type for violations
                 pass

            # Success!
            final_parsed = parsed
            logger.info(
                "Successfully parsed and validated on attempt %d", attempt)
            break

        except Exception as e:
            logger.exception(
                "Error during generation attempt %d: %s", attempt, e)
            final_error = str(e)
            continue

    # Build prompt for usage calculation
    prompt_text = "\n".join([m.content for m in final_messages])
    completion_text = "\n".join(all_completions)
    usage = estimate_usage(prompt_text, completion_text)

    if final_parsed is None:
        # Exhausted all attempts
        logger.error("Failed after 3 attempts. Last error: %s", final_error)
        return ChatCompletionStructuredResponse(
            id=gen_id(),
            model="slm-hr-1",
            usage=usage,
            response={
                "data": "INVALID_OUTPUT",
                "type": "error",
                "error": final_error or "Failed to generate valid output after 3 attempts"
            }
        )

    # Success
    return ChatCompletionStructuredResponse(
        id=gen_id(),
        model="slm-hr-1",
        usage=usage,
        response=final_parsed
    )


async def process_chat(req: ChatRequest) -> ChatResponse:
    """Process chat request and generate response."""

    if not req.messages or len(req.messages) == 0:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Get user ID for rate limiting
    user_id = req.userId or "anonymous"

    # Rate limiting
    if not await rate_limiter.check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} requests per minute."
        )

    # Get last user message for HR scope check
    last_user_msg = None
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_msg = m.content
            break

    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    # HR scope validation
    if not is_hr_related(last_user_msg):
        logger.info("Non-HR query rejected for user: %s", user_id)
        response_text = (
            "I apologize, but I can only assist with HR-related tasks such as job descriptions, "
            "resume analysis, candidate evaluation, interview questions, and other human resources topics. "
            "Your query appears to be outside my area of expertise. "
            "Please ask me about HR-related matters, and I'll be happy to help!"
        )

        return ChatResponse(
            id=gen_id(),
            model=MODEL_NAME,
            created=int(time.time()),
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": estimate_tokens(last_user_msg),
                "completion_tokens": estimate_tokens(response_text),
                "total_tokens": estimate_tokens(last_user_msg + response_text)
            }
        )

    # Generate response
    temperature = req.temperature if req.temperature is not None else TEMPERATURE
    max_tokens = req.max_tokens if req.max_tokens is not None else MAX_TOKENS

    start_time = time.time()
    response_text = await chat_model.generate(req.messages, temperature, max_tokens)
    duration = time.time() - start_time

    logger.info("Generated response in %.2fs for user %s", duration, user_id)

    # Build response
    prompt_text = "\n".join([m.content for m in req.messages])

    return ChatResponse(
        id=gen_id(),
        model=MODEL_NAME,
        created=int(time.time()),
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": estimate_tokens(prompt_text),
            "completion_tokens": estimate_tokens(response_text),
            "total_tokens": estimate_tokens(prompt_text + response_text)
        }
    )

# --------------------------- Routes ---------------------------


@app.on_event("startup")
async def startup_event():
    """Preload model on startup."""
    logger.info("Starting up - preloading model...")
    await chat_model.ensure_loaded()
    logger.info("Model ready.")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple web interface for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HR Assistant Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            #chat-container { border: 1px solid #ddd; padding: 20px; height: 400px; overflow-y: auto; margin-bottom: 20px; background: #f9f9f9; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #e3f2fd; text-align: right; }
            .assistant { background: #f1f8e9; }
            .error { background: #ffebee; color: #c62828; }
            #input-container { display: flex; gap: 10px; margin-bottom: 10px; }
            #user-input { flex: 1; padding: 10px; font-size: 14px; }
            button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; font-size: 14px; }
            button:hover { background: #45a049; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            #file-upload-container { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
            #file-input { flex: 1; }
            .upload-btn { background: #2196F3; }
            .upload-btn:hover { background: #0b7dda; }
        </style>
    </head>
    <body>
        <h1>🤖 HR Assistant Chat</h1>
        <p>Ask me anything about HR tasks: job descriptions, resume analysis, candidate ranking, interview questions, etc.</p>
        <p><strong>📄 Upload Resume:</strong> You can upload PDF, DOCX, or TXT files for analysis.</p>
        <div id="chat-container"></div>
        <div id="file-upload-container">
            <input type="file" id="file-input" accept=".pdf,.docx,.txt" />
            <button class="upload-btn" onclick="uploadResume()">Upload & Analyze Resume</button>
        </div>
        <div id="input-container">
            <input type="text" id="user-input" placeholder="Type your HR question here..." />
            <button onclick="sendMessage()">Send</button>
        </div>
        
        <script>
            let messages = [];
            
            function addMessage(role, content) {
                const container = document.getElementById('chat-container');
                const div = document.createElement('div');
                div.className = 'message ' + role;
                div.innerHTML = '<strong>' + (role === 'user' ? 'You' : 'HR Assistant') + ':</strong><br>' + content.replace(/\n/g, '<br>');
                container.appendChild(div);
                container.scrollTop = container.scrollHeight;
            }
            
            async function uploadResume() {
                const fileInput = document.getElementById('file-input');
                const file = fileInput.files[0];
                if (!file) {
                    alert('Please select a file first');
                    return;
                }
                
                addMessage('user', '📄 Uploaded: ' + file.name);
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/v1/upload-resume', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        const assistantMessage = data.choices[0].message.content;
                        addMessage('assistant', assistantMessage);
                        messages.push({role: 'user', content: 'Analyze this resume: ' + file.name});
                        messages.push({role: 'assistant', content: assistantMessage});
                    } else {
                        addMessage('error', 'Error: ' + (data.detail || 'Unknown error'));
                    }
                } catch (error) {
                    addMessage('error', 'Error: ' + error.message);
                }
                
                fileInput.value = '';
            }
            
            async function sendMessage() {
                const input = document.getElementById('user-input');
                const message = input.value.trim();
                if (!message) return;
                
                addMessage('user', message);
                input.value = '';
                input.disabled = true;
                
                messages.push({role: 'user', content: message});
                
                try {
                    const response = await fetch('/v1/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({messages: messages})
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        const assistantMessage = data.choices[0].message.content;
                        addMessage('assistant', assistantMessage);
                        messages.push({role: 'assistant', content: assistantMessage});
                    } else {
                        addMessage('error', 'Error: ' + (data.detail || 'Unknown error'));
                    }
                } catch (error) {
                    addMessage('error', 'Error: ' + error.message);
                }
                
                input.disabled = false;
                input.focus();
            }
            
            document.getElementById('user-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
            
            // Welcome message
            addMessage('assistant', 'Hello! I\'m your HR Assistant. I can help you with job descriptions, resume analysis, candidate evaluation, and other HR tasks. You can also upload resumes (PDF, DOCX, TXT) for analysis. What would you like help with today?');
        </script>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": chat_model.model is not None,
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "model_name": MODEL_NAME,
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint - OpenAI-compatible format.

    Example request:
    {
        "messages": [
            {"role": "user", "content": "Create a job description for a Senior Python Developer"}
        ],
        "userId": "user123"
    }
    """
    return await process_chat(request)


@app.post("/v1/chat-completion", response_model=ChatCompletionStructuredResponse)
async def chat_completion_endpoint(request: ChatCompletionRequest):
    """
    Structured chat completion endpoint - Returns validated JSON with {data, type} format.

    Follows the HR-SLM specification with:
    - HR-only scope validation
    - Structured JSON output
    - Regeneration up to 3 times on validation failure
    - Strict JSON format enforcement

    Example request:
    {
        "model": "slm-hr-1",
        "messages": [
            {"role": "user", "content": "Parse this resume: ..."}
        ],
        "userId": "USR-001"
    }

    Response format:
    {
        "id": "chatcmpl-xxx",
        "model": "slm-hr-1",
        "usage": {"prompt_tokens": 123, "completion_tokens": 456, "total_tokens": 579},
        "response": {"data": {...}, "type": "object"}
    }
    """
    return await process_structured_chat_completion(request)


@app.post("/v1/create-jd", response_model=ChatResponse)
async def create_jd_endpoint(request: Request):
    """
    Create a job description from structured JSON input.

    Accepts either:
    1. Structured format with specific fields
    2. Natural language in 'content' field

    Example structured request:
    {
        "jobTitle": "Senior Python Developer",
        "minExperience": 5,
        "maxExperience": 8,
        "employmentType": "Full-time",
        "salaryRange": "$120k-$150k",
        "qualification": "Bachelor's in Computer Science",
        "jobLevel": "Senior",
        "workplaceType": "Remote",
        "technicalSkills": ["Python", "Django", "REST APIs"],
        "nonTechnicalSkills": ["Leadership", "Communication"],
        "tone": "Professional",
        "jdType": "Detailed",
        "additionalInfo": "Experience with microservices",
        "userId": "user123"
    }

    Example natural language request:
    {
        "content": "Create a JD for Senior Python Developer with 5-8 years experience",
        "userId": "user123"
    }
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    user_id = body.get("userId", "anonymous")

    # Check if it's natural language format (has 'content' field)
    if "content" in body:
        chat_request = ChatRequest(
            messages=[Message(role="user", content=body["content"])],
            userId=user_id
        )
        return await process_chat(chat_request)

    # Otherwise, it's structured format - build prompt from fields
    job_title = body.get("jobTitle", "")
    min_exp = body.get("minExperience", 0)
    max_exp = body.get("maxExperience", 0)
    employment_type = body.get("employmentType", "")
    salary_range = body.get("salaryRange", "")
    qualification = body.get("qualification", "")
    job_level = body.get("jobLevel", "")
    workplace_type = body.get("workplaceType", "")
    technical_skills = body.get("technicalSkills", [])
    non_technical_skills = body.get("nonTechnicalSkills", [])
    tone = body.get("tone", "Professional")
    jd_type = body.get("jdType", "Detailed")
    additional_info = body.get("additionalInfo", "")

    # Build structured prompt
    prompt = f"""Create a {jd_type.lower()} job description with a {tone.lower()} tone for the following position:

Job Title: {job_title}
Experience Required: {min_exp}-{max_exp} years
Employment Type: {employment_type}
Salary Range: {salary_range}
Qualification: {qualification}
Job Level: {job_level}
Workplace Type: {workplace_type}

Technical Skills Required:
{', '.join(technical_skills) if technical_skills else 'Not specified'}

Non-Technical Skills Required:
{', '.join(non_technical_skills) if non_technical_skills else 'Not specified'}

Additional Information:
{additional_info if additional_info else 'None'}

Please create a comprehensive job description including:
1. Job Summary
2. Key Responsibilities
3. Required Qualifications
4. Preferred Qualifications
5. Benefits (if applicable)"""

    chat_request = ChatRequest(
        messages=[Message(role="user", content=prompt)],
        userId=user_id
    )

    logger.info(f"Creating JD for: {job_title} (structured input)")
    return await process_chat(chat_request)


@app.post("/v1/upload-resume")
async def upload_resume_endpoint(
    file: UploadFile = File(...),
    userId: Optional[str] = Form(None)
):
    """
    Upload a resume and create a session for multi-turn conversation.

    Returns:
    - sessionId: Use this for subsequent queries about the resume
    - summary: Initial analysis of the resume

    After uploading, use /v1/resume-query with the sessionId to ask questions.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = ['.pdf', '.docx', '.txt']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file")

    # Extract text from file
    try:
        resume_text = extract_text_from_file(file.filename, file_content)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to extract text from file")

    if not resume_text or len(resume_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Resume appears to be empty or too short. Please upload a valid resume."
        )

    # Create session
    session_id = await resume_session_manager.create_session(resume_text)

    # Generate initial summary
    prompt = f"""Analyze this resume and provide a brief summary:

Resume Content:
{resume_text}

Provide:
1. Candidate name
2. Current role/title
3. Years of experience
4. Top 5 skills
5. Education
6. One-line assessment"""

    chat_request = ChatRequest(
        messages=[Message(role="user", content=prompt)],
        userId=userId
    )

    summary_response = await process_chat(chat_request)
    summary_text = summary_response.choices[0]["message"]["content"]

    # Store initial interaction
    await resume_session_manager.add_message(session_id, "system", f"Resume uploaded: {file.filename}")
    await resume_session_manager.add_message(session_id, "assistant", summary_text)

    logger.info(
        f"Created resume session: {session_id} for file: {file.filename}")

    return {
        "sessionId": session_id,
        "filename": file.filename,
        "summary": summary_text,
        "message": "Resume uploaded successfully. Use this sessionId to query about the resume."
    }


@app.post("/v1/resume-query", response_model=StructuredResponse)
async def resume_query_endpoint(request: ResumeSessionRequest):
    """
    Query about an uploaded resume with structured JSON responses.

    Example queries:
    - "Extract all technical skills from this resume"
    - "Is this candidate suitable for a Backend Engineer role requiring Java and Spring Boot?"
    - "What coding questions should I ask this candidate?"
    - "Determine if this is a coding-focused resume and list core skills"

    Returns structured JSON response based on the query type.
    """
    if not request.sessionId:
        raise HTTPException(status_code=400, detail="sessionId is required")

    # Get session
    session = await resume_session_manager.get_session(request.sessionId)
    if not session:
        raise HTTPException(
            status_code=404, detail="Session not found or expired")

    resume_text = session["resume_text"]
    user_id = request.userId or "anonymous"

    # Rate limiting
    if not await rate_limiter.check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} requests per minute."
        )

    # Build prompt with resume context
    full_prompt = f"""You are analyzing a resume. Here is the resume content:

{resume_text}

User Query: {request.query}

Based on the resume above, answer the query and return a JSON response with the following structure:
{{
  "data": {{
    // Your analysis here as a structured object
  }},
  "type": "object"
}}

For skill extraction queries, include: isCodingJob, coreSkills, softSkills, codingQuestionsRequired, codingSkills
For suitability queries, include: suitable (boolean), matchingSkills, missingSkills, recommendation
For general queries, structure the response appropriately.

Return ONLY valid JSON, no additional text."""

    # Generate response
    messages = [Message(role="user", content=full_prompt)]

    start_time = time.time()
    response_text = await chat_model.generate(messages, TEMPERATURE, MAX_TOKENS)
    duration = time.time() - start_time

    # Parse JSON response
    try:
        parsed = parse_json_safe(response_text)
        if not parsed or "data" not in parsed:
            # Fallback: wrap response
            parsed = {"data": {"response": response_text}, "type": "object"}
    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        parsed = {"data": {"response": response_text,
                           "error": "Failed to parse as JSON"}, "type": "object"}

    # Store in session
    await resume_session_manager.add_message(request.sessionId, "user", request.query)
    await resume_session_manager.add_message(request.sessionId, "assistant", json.dumps(parsed))

    logger.info(
        f"Resume query processed in {duration:.2f}s for session {request.sessionId}")

    # Calculate usage
    usage = estimate_usage(full_prompt, response_text)

    return StructuredResponse(
        id=gen_id(),
        model=MODEL_NAME,
        usage=usage,
        response=parsed
    )


@app.post("/v1/rank-candidates", response_model=RankingResponseDetailed)
async def rank_candidates_endpoint(
    files: List[UploadFile] = File(...),
    jobDescription: str = Form(...),
    criteria: Optional[str] = Form(None)
):
    """
    Rank multiple candidates against a Job Description using Hybrid approach (Embeddings + LLM).
    
    Upload multiple resume files (PDF/DOCX/TXT) and provide a JD.
    The system will evaluate each resume and return a ranked list with detailed breakdown.
    """
    candidates = []
    
    # Process each file
    for file in files:
        try:
            # Read and extract text
            content = await file.read()
            resume_text = extract_text_from_file(file.filename, content)
            
            if not resume_text or len(resume_text.strip()) < 50:
                logger.warning(f"Skipping empty file: {file.filename}")
                continue
            
            # 1. Calculate Semantic Similarity (Embedding)
            similarity_score = await asyncio.to_thread(
                chat_model.calculate_similarity, jobDescription, resume_text[:2000]
            )
            similarity_percent = int(similarity_score * 100)
            
            # 2. Build prompt with semantic context
            prompt = f"""Evaluate this resume against the Job Description.
            
Job Description:
{jobDescription}

{f'Additional Criteria: {criteria}' if criteria else ''}

Resume Content:
{resume_text[:3500]}

Semantic Match Score (Calculated by Embedding Model): {similarity_score:.2f} ({similarity_percent}%)

Task:
1. Analyze the resume against the JD.
2. Provide a detailed score breakdown (0-100) for Skills, Experience, and Education.
3. Determine the overall fit category (Top Fit, Good Fit, Potential Fit, Not Fit).
4. List matched and missing skills.
5. Provide a reasoning.

Return ONLY valid JSON in this format:
{{
  "data": {{
    "name": "Candidate Name",
    "overallScore": 85,
    "fitCategory": "Top Fit",
    "breakdown": {{
      "skillsMatch": 90,
      "experienceRelevance": 80,
      "educationAlignment": 70,
      "embeddingSimilarity": {similarity_score:.2f}
    }},
    "matchedSkills": ["Skill1", "Skill2"],
    "missingSkills": ["Skill3"],
    "reasoning": "Detailed explanation..."
  }},
  "type": "object"
}}"""

            # Generate evaluation
            messages = [Message(role="user", content=prompt)]
            
            # We process sequentially
            response_text = await chat_model.generate(messages, 0.1, 1024)
            
            # Parse response
            parsed = parse_json_safe(response_text)
            
            if parsed and "data" in parsed and isinstance(parsed["data"], dict):
                data = parsed["data"]
                candidates.append(CandidateScoreDetailed(
                    name=data.get("name", "Unknown"),
                    filename=file.filename,
                    overallScore=data.get("overallScore", 0),
                    fitCategory=data.get("fitCategory", "Potential Fit"),
                    breakdown=ScoreBreakdown(**data.get("breakdown", {
                        "skillsMatch": 0, "experienceRelevance": 0, 
                        "educationAlignment": 0, "embeddingSimilarity": similarity_score
                    })),
                    matchedSkills=data.get("matchedSkills", []),
                    missingSkills=data.get("missingSkills", []),
                    reasoning=data.get("reasoning", "No reasoning provided")
                ))
            else:
                logger.warning(f"Failed to parse score for {file.filename}")
                # Fallback for parse error
                candidates.append(CandidateScoreDetailed(
                    name="Parse Error",
                    filename=file.filename,
                    overallScore=0,
                    fitCategory="Not Fit",
                    breakdown=ScoreBreakdown(
                        skillsMatch=0, experienceRelevance=0, 
                        educationAlignment=0, embeddingSimilarity=similarity_score
                    ),
                    matchedSkills=[],
                    missingSkills=[],
                    reasoning="Failed to parse model response"
                ))
                
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            # Fallback for exception
            candidates.append(CandidateScoreDetailed(
                name="Error",
                filename=file.filename,
                overallScore=0,
                fitCategory="Not Fit",
                breakdown=ScoreBreakdown(
                    skillsMatch=0, experienceRelevance=0, 
                    educationAlignment=0, embeddingSimilarity=0.0
                ),
                matchedSkills=[],
                missingSkills=[],
                reasoning=f"Processing error: {str(e)}"
            ))

    # Sort by overallScore descending
    candidates.sort(key=lambda x: x.overallScore, reverse=True)
    
    # Calculate summary score (average of top 3 or all)
    if candidates:
        avg_score = sum(c.overallScore for c in candidates) / len(candidates)
    else:
        avg_score = 0.0
    
    return RankingResponseDetailed(
        jobDescription=jobDescription[:100] + "...",
        summaryScore=round(avg_score, 1),
        candidates=candidates
    )


# CLI Runner

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main3:app", host="0.0.0.0", port=8002, reload=True)

