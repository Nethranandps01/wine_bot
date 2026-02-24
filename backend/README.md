# Wine Bot Backend (Python + Pipecat + Gemini)

## Run

```bash
cd /Users/nethranand/Downloads/wine_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8001 --reload
```

The backend reads `GEMINI_API_KEY` from `/Users/nethranand/Downloads/wine_bot/.env`.

## Endpoints

- `GET /health`
- `POST /stt` (multipart file upload for mic transcription)
- `WS /ws/chat?session_id=<id>`

## Dummy Mode

For local verification without live API calls:

```bash
export USE_DUMMY_GEMINI=true
pytest -q backend/tests
```
