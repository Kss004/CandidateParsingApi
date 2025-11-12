# Candidate Parser API

FastAPI-based service that uses Phi-3.5 to parse free-form candidate search prompts into structured JSON.

## Project Structure

```
.
├── main.py                         # Main application entry point
├── app/
│   ├── __init__.py
│   ├── config.py                   # Configuration and environment variables
│   ├── models.py                   # Pydantic models
│   ├── routers/                    # API endpoints
│   │   ├── __init__.py
│   │   └── parse_router.py         # /parse endpoint
│   └── services/                   # Business logic
│       ├── __init__.py
│       ├── llm_service.py          # LLM model loading and inference
│       ├── prompt_builder.py       # Prompt construction
│       └── json_parser.py          # JSON extraction and repair
└── requirements.txt
```

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /parse` - Parse candidate search prompts

