"""FastAPI surface for the chatbot, shaped for the Streamlit frontend in app.py.

The turn is ``run_turn`` and nothing here reimplements any part of it. What this file does
is speak app.py's contract, which was written against New Prompt's API:

    GET  /health                           {"status": "ok"} once the backend can serve a turn
    POST /chat                             one turn, blocking
    POST /chat/stream                      the same turn as Server-Sent Events
    GET  /session/{id}                     200 with exists=false for a session never seen
    GET  /session/{id}/turn-status         {"search_status_message": ...} while a turn runs
    POST /session/reset                    a fresh session id

Run with the project venv:
    .venv\\Scripts\\python.exe main.py
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from src import conversation_store, warmup, db, turn_saver, turn_status  # noqa: E402
from src.config import settings  # noqa: E402
from src.domain.reply_chunks import split_reply_into_chunks  # noqa: E402
from src.graph.build import run_turn  # noqa: E402
from src.graph.state import from_snapshot  # noqa: E402
from src.log_setup import configure_trailerplace_logging  # noqa: E402

configure_trailerplace_logging()
logger = logging.getLogger(__name__)

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Get the schema out of the way before the first customer, not during their turn.

    ``ensure_schema`` memoises, so this is the only time it costs anything in this process -
    it used to run on every turn, and the reflection round trips to Azure Postgres were
    about 2.7 s of an 11 s answer.

    With DB_AUTO_CREATE=1 the full migration runs instead: ``upgrade head`` and then the
    create-if-missing. Every revision is guarded, so it is safe on the live database whose
    tables predate Alembic.
    """
    if conversation_store.persistence_enabled():
        try:
            if settings.db_auto_create:
                db.run_migrations()
            else:
                db.ensure_schema()
        except Exception:  # the bot still serves; the first turn will try again
            logger.exception("Schema check at startup failed")

    # The connection pool, the question rules and the make inventory: about ten seconds of
    # first-turn latency that belongs to startup, not to whoever happens to send the first
    # message. On a serverless deployment that is every scale-from-zero.
    warmup.warm_everything()
    yield

    # On the way down. A finished turn whose commit is still queued was answered but not
    # saved, so the container waits for the queue before it goes - and says so, loudly, if
    # the clock runs out first.
    turn_saver.drain()


app = FastAPI(title="TrailerPlace chatbot", version="1.0", lifespan=lifespan)

# The question-rules admin API (src/api/admin_rules.py), for the control panel. Mounted only
# when a token is configured, so a deploy without one has no admin surface at all.
if settings.admin_api_token:
    from src.api.admin_rules import router as admin_rules_router  # noqa: E402

    app.include_router(admin_rules_router)

# The 5-minute rule's minute tick (src/api/idle.py). Mounted only with a token, like the admin
# API: it messages customers, so a deploy without one exposes no such route at all.
if settings.idle_sweep_token:
    from src.api.idle import router as idle_router  # noqa: E402

    app.include_router(idle_router)

# The Facebook Messenger webhook (src/api/messenger.py). Always mounted; both routes 404
# unless MESSENGER_ENABLED is on and the app secret and page token are configured, so a
# deployment that is not the Messenger one has no webhook surface at all.
from src.api.messenger import router as messenger_router  # noqa: E402

app.include_router(messenger_router)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


# Streaming is typing, not token streaming: the reply is produced whole by run_turn, then sent
# the way a person would send it - the intro, one message per trailer, then the question.
STREAM_ENABLED = (os.getenv("CHAT_STREAM_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
STREAM_WORDS_PER_DELTA = int(_env_float("CHAT_STREAM_WORDS_PER_DELTA", 3))
STREAM_DELTA_SECONDS = _env_float("CHAT_STREAM_DELTA_SECONDS", 0.02)
STREAM_CHUNK_PAUSE_SECONDS = _env_float("CHAT_STREAM_CHUNK_PAUSE_SECONDS", 0.35)
STREAM_STATUS_POLL_SECONDS = 0.3

# Turns run here so the stream can keep forwarding the search line while one is in progress.
_STREAM_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chat_stream")


class ChatRequest(BaseModel):
    # app.py also sends sales_phase, onboarding_api_messages, customer_* and
    # already_shown_listing_urls. They are New Prompt's; Luna keeps all of that in the session
    # state, so they are accepted and ignored rather than rejected.
    model_config = ConfigDict(extra="ignore")

    message: str = Field(min_length=1)
    session_id: str | None = None
    # Send the SAME id when retrying a message and the stored reply comes back instead of a
    # second turn: no second model call, no second email, and the conversation does not move
    # on twice on one message. A fresh id per attempt means every attempt is a new turn.
    turn_id: str | None = None


def _require_uuid(value: str | None, name: str) -> str:
    """Session ids are UUID columns. A bad one is a 422 now, not a 500 halfway through."""
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"{name} must be a UUID")


def _contact_fields(contact: dict[str, Any] | None) -> dict[str, Any]:
    contact = contact or {}
    return {
        "customer_full_name": contact.get("name"),
        "customer_email": contact.get("email"),
        "customer_phone": contact.get("phone"),
    }


def _chat(request: ChatRequest) -> dict[str, Any]:
    if len(request.message) > settings.chat_max_message_chars:
        raise HTTPException(status_code=413, detail="Message too long.")
    session_id = _require_uuid(request.session_id or str(uuid.uuid4()), "session_id")

    # Cleared at both ends: at the start so a poll can never show the PREVIOUS turn's search
    # line, and at the end so the status dict does not keep one entry per session forever.
    turn_status.clear(session_id)
    try:
        result = run_turn(session_id, request.message, turn_id=request.turn_id)
    except Exception:
        # The turn degrades internally on a model failure, so reaching here means something
        # structural. The customer gets a message rather than a stack trace.
        logger.exception("Turn failed: session=%s", session_id)
        raise HTTPException(status_code=500, detail="Something went wrong on our side.")
    finally:
        turn_status.clear(session_id)

    return {
        **result,
        **_contact_fields(result.get("contact")),
        # app.py has an onboarding phase of its own. Luna's contact gate lives in the graph,
        # so the UI is always in the main phase and simply renders what the bot says.
        "sales_phase": "main",
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    return _chat(request)


# ----------------------------------------------------------------------------- streaming
# Events: status | chunk_start | delta | chunk_end | done | error - the set app.py reads.
# `done` carries exactly what POST /chat returns, plus `chunks`.
_WORD_RE = re.compile(r"\S+\s*")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _typing_deltas(chunk: str) -> list[str]:
    """The chunk in keystroke-sized pieces that join back into it exactly."""
    words = _WORD_RE.findall(chunk)
    if not words:
        return [chunk] if chunk else []
    step = max(1, STREAM_WORDS_PER_DELTA)
    return ["".join(words[i:i + step]) for i in range(0, len(words), step)]


def _chat_event_stream(request: ChatRequest, session_id: str):
    # ThreadPoolExecutor does not carry contextvars across, and the turn's usage scope lives
    # in one - the copy keeps it request-scoped.
    context = contextvars.copy_context()
    future = _STREAM_EXECUTOR.submit(context.run, _chat, request)
    last_note: str | None = None
    while not future.done():
        time.sleep(STREAM_STATUS_POLL_SECONDS)
        note = turn_status.peek(session_id)
        if note and note != last_note:
            last_note = note
            yield _sse("status", {"search_status_message": note})

    try:
        body = future.result()
    except HTTPException as exc:
        # The response has already started, so the status code is spent; the client reads the
        # failure off the event instead.
        yield _sse("error", {"status_code": exc.status_code, "detail": str(exc.detail)})
        return
    except Exception as exc:  # a stream must end with an event, never a traceback
        logger.exception("Streaming turn failed: session=%s", session_id)
        yield _sse("error", {"status_code": 500, "detail": str(exc)})
        return

    # _chat clears the live line on its way out, so a search that finished between two polls
    # was never forwarded. The response still carries it.
    note = (body.get("search_status_message") or "").strip()
    if note and note != last_note:
        yield _sse("status", {"search_status_message": note})

    chunks = split_reply_into_chunks(body.get("assistant_text") or "")
    body["chunks"] = chunks
    for index, chunk in enumerate(chunks):
        yield _sse("chunk_start", {"index": index, "total": len(chunks)})
        for delta in _typing_deltas(chunk):
            yield _sse("delta", {"index": index, "text": delta})
            if STREAM_DELTA_SECONDS:
                time.sleep(STREAM_DELTA_SECONDS)
        yield _sse("chunk_end", {"index": index, "text": chunk})
        if STREAM_CHUNK_PAUSE_SECONDS and index < len(chunks) - 1:
            time.sleep(STREAM_CHUNK_PAUSE_SECONDS)
    yield _sse("done", body)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    if not STREAM_ENABLED:
        # app.py treats 404 as "no streaming here" and runs the turn through POST /chat.
        raise HTTPException(status_code=404, detail="Streaming is disabled.")
    # Validated before the first byte: after that the status code can no longer change.
    session_id = _require_uuid(request.session_id, "session_id")
    request = request.model_copy(update={"session_id": session_id})
    return StreamingResponse(
        _chat_event_stream(request, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this a proxy in front buffers the whole stream and nothing appears until
            # the last event.
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------------------ sessions
@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """The conversation so far, for app.py to redraw after a reload.

    A session with no history is a 200 with exists=false, not a 404: app.py opens every new
    chat by asking for its session, and treats any error as "could not restore".
    """
    session_id = _require_uuid(session_id, "session_id")
    snapshot, conversation, lead_id = conversation_store.load_session(session_id)
    exists = snapshot is not None or bool(conversation)
    state = from_snapshot(session_id, snapshot)
    messages = [
        {**entry, "listings": None}
        for entry in (conversation or [])
        if isinstance(entry, dict) and entry.get("role") in {"user", "assistant"}
    ]
    slots = state.get("slots") or {}
    declined = state.get("declined_slots") or []
    return {
        "session_id": session_id,
        "exists": exists,
        "closed": False,
        "messages": messages,
        "sales_phase": "main",
        **_contact_fields(state.get("contact")),
        # Not read by app.py; kept for inspecting a session by hand.
        "lead_id": lead_id,
        "category": state.get("category"),
        "slots": slots,
        "declined_slots": declined,
        "required_remaining": [
            slot for slot in (state.get("required_slots") or [])
            if slot not in slots and slot not in declined
        ],
        "qualification_complete": state.get("qualification_complete"),
        "contact": state.get("contact"),
    }


class FeedbackRequest(BaseModel):
    """A tester's verdict on one of Luna's replies."""

    model_config = ConfigDict(extra="ignore")

    # Counts Luna's REPLIES from zero, which is the position a frontend knows a message by.
    # The store resolves it against the assistant entries in the transcript.
    turn_idx: int = Field(ge=0)
    rating: Literal["up", "down"] | None = None
    text: str | None = None
    timestamp: str | None = None


@app.post("/session/{session_id}/feedback")
def post_feedback(session_id: str, request: FeedbackRequest) -> dict[str, Any]:
    """Record a thumb and/or a note against one reply.

    Written straight onto the message inside the stored transcript, so the reply and the
    verdict on it are read back together and cannot drift apart.

    Queued rather than awaited: nobody waits on a thumbs-up, and a note is never worth
    holding the UI for. The answer is 'accepted', not 'stored' - a turn_idx pointing at a
    reply that does not exist is logged and dropped.
    """
    session_id = _require_uuid(session_id, "session_id")
    if request.rating is None and request.text is None:
        raise HTTPException(status_code=422, detail="Send a rating, a note, or both.")
    conversation_store.enqueue_save_user_feedback(
        session_id, request.turn_idx, request.text, request.timestamp, rating=request.rating
    )
    return {"session_id": session_id, "turn_idx": request.turn_idx, "status": "accepted"}


@app.post("/session/reset")
def reset_session() -> dict[str, str]:
    """A fresh session id. Existing rows are left alone - a conversation is a record, and
    starting a new one is not a reason to delete the old one."""
    return {"session_id": str(uuid.uuid4())}


@app.get("/session/{session_id}/turn-status")
def get_turn_status(session_id: str) -> dict[str, Any]:
    """What the running turn is doing, for a UI that is mid-wait. Takes no lock: it reads the
    line the search node publishes, while /chat is still busy with that very turn."""
    note = turn_status.peek(session_id)
    return {"session_id": session_id, "search_status_message": note, "status": note}


# -------------------------------------------------------------------------------- health
@app.get("/health")
def health(response: Response) -> dict[str, Any]:
    """"ok" only when a turn could actually be served.

    app.py waits on this before letting anyone type, and retries on any non-ok answer, so a
    503 while the database is unreachable is safe - and far better than an input box that
    fails on the first message.
    """
    from sqlalchemy import text

    from src import db

    # A serverless platform pings this to wake the container, so it is the right place to
    # load the caches: by the time health says "ok", a turn can be served without paying for
    # any of it. Free after the first call - warm_everything is idempotent.
    warmup.warm_everything()

    body: dict[str, Any] = {
        "status": "ok",
        "warm": warmup.is_warm(),
        "model": settings.chat_model,
        "reasoning_effort": settings.chat_reasoning_effort,
        "database": db.database_enabled(),
        "persistence": conversation_store.persistence_enabled(),
    }
    if not settings.openai_api_key:
        body["status"] = "error"
        body["error"] = "OPENAI_API_KEY is not set."
    elif conversation_store.persistence_enabled():
        try:
            with db.get_engine().connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:  # reported, not raised: this endpoint must always answer
            body["status"] = "error"
            body["error"] = f"Database unreachable: {type(exc).__name__}"
    if body["status"] != "ok":
        response.status_code = 503
    return body


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.chatbot_api_port)
