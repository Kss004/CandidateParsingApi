import json
import logging
import re

logger = logging.getLogger("structured-parser")

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
