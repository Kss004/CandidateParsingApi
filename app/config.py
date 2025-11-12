import os

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
