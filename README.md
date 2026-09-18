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

## Question rules

Which questions each category asks, and when that changes, are data, not code. They live in
one versioned document (`src/rules/`):

- `categories`: each category's required and optional questions, with their wording.
- `cargo_traits`: labels the model may give the cargo (`lightweight`, `large_or_heavy`), each
  with a definition and examples. The model only labels the cargo.
- `rules`: one action each, `skip_question`, `ask_question` or `set_default`, with a `when`
  of `categories_in`, `categories_not_in`, `traits_any` and `traits_none`.

The bot serves the active row of `chatbot_question_rules`. When that table is empty or
persistence is off, it serves `src/rules/seed.json`. Precedence:

1. A value the customer gave beats everything.
2. Skip beats ask.
3. A default counts as answered.

| Variable | Default | |
|---|---|---|
| `ADMIN_API_TOKEN` | *(empty)* | Enables `/admin/rules`, which is unmounted when this is empty. Send it as `X-Admin-Token` |
| `RULES_REFRESH_SECONDS` | `30` | How soon other processes pick up a newly activated version |

| Endpoint | |
|---|---|
| `GET /admin/rules` | Active document and version (`0` = seed) |
| `GET /admin/rules/catalog` | Categories, slots and their kinds, traits, actions and conditions for dropdowns |
| `POST /admin/rules/validate` | `{"document": ...}` → `{"valid", "errors"}`; saves nothing |
| `PUT /admin/rules` | `{"document", "note", "author"}`. Saves and activates a new version, or returns 422 with errors |
| `GET /admin/rules/versions` | History |
| `GET /admin/rules/versions/{v}` | One stored document |
| `POST /admin/rules/versions/{v}/activate` | Roll back or forward |

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
