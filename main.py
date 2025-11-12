from __future__ import annotations

import logging

from fastapi import FastAPI

from app.routers import parse_router
from app.services.llm_service import ensure_model_loaded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("structured-parser")

app = FastAPI(
    title="Async Candidate Parser",
    version="1.0.0",
    description="Parses free-form candidate prompts into normalized JSON using a local LLM.",
)

# Include routers
app.include_router(parse_router.router)


@app.on_event("startup")
async def startup_event() -> None:
    """Warm the model when the app boots."""
    await ensure_model_loaded()


@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running and model is loaded."""
    from app.config import MODEL_REPO, MODEL_FILE, INFERENCE_CONCURRENCY
    from app.services.llm_service import model

    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "starting",
        "model_loaded": model_loaded,
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "max_concurrent_requests": INFERENCE_CONCURRENCY,
    }
