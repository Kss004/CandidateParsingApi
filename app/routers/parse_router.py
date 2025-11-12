import logging

from fastapi import APIRouter, HTTPException

from app.models import ParseRequest, ParseResponse, CandidateData
from app.services.llm_service import invoke_model
from app.services.prompt_builder import build_json_prompt
from app.services.json_parser import parse_model_output

logger = logging.getLogger("structured-parser")

router = APIRouter()


@router.post("/parse", response_model=ParseResponse)
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
