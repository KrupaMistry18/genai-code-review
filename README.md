# GenAI Code Review Assistant (MVP)

FastAPI backend that reviews code diffs and returns a summary + findings.

## Modes
- Dummy (default): `USE_OPENAI=false`, `USE_OLLAMA=false`
- OpenAI: `USE_OPENAI=true` + `OPENAI_API_KEY` + `OPENAI_MODEL`


## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.app:app --reload
