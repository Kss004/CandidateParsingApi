from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import HTTPException
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from app.config import (
    MODEL_REPO,
    MODEL_FILE,
    MAX_NEW_TOKENS,
    INFERENCE_CONCURRENCY,
    TEMPERATURE,
    TOP_P,
    INFERENCE_TIMEOUT,
    N_CTX,
    N_THREADS,
)

logger = logging.getLogger("structured-parser")

model: Optional[Llama] = None
inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)
model_lock = asyncio.Lock()


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
