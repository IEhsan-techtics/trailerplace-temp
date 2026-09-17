# Luna — TrailerPlace chatbot

## Running it with the test frontend

Two processes, both from this folder, both with this project's venv.

```powershell
# 1. The backend (FastAPI, port 8000)
.\.venv\Scripts\python.exe main.py

# 2. The frontend (Streamlit, port 8501) - in a second terminal
.\.venv\Scripts\streamlit.exe run app.py
```

Open http://localhost:8501 and sign in with `TRAILERPLACE_APP_USERNAME` /
`TRAILERPLACE_APP_PASSWORD` from `.env`. The frontend waits on `/health` before it lets you
type, so it shows "initializing" until the backend is up and the database answers.

If another project's virtualenv is active, deactivate it first
(`Remove-Item Env:\VIRTUAL_ENV`), or the wrong packages get imported.

### Settings the frontend reads

| Variable | Default | |
|---|---|---|
| `CHATBOT_API_URL` | `http://127.0.0.1:8000` | Where the backend is |
| `CHAT_STREAM_ENABLED` | `1` | Replies arrive as bubbles, one per trailer. `0` = one blocking reply |
| `SHOW_LISTING_CARDS` | `0` | Also draw Streamlit's own listing cards under the reply |

### Model settings

| Variable | Default | |
|---|---|---|
| `CHAT_REASONING_EFFORT` | `none` | gpt-5.6-luna. It rejects `minimal`. |
| `FEATURE_RERANK_REASONING_EFFORT` | `minimal` | gpt-5-nano. It rejects `none`. |
| `SEARCH_MAX_RECOMMENDATIONS` | `5` | Trailers per search |

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
