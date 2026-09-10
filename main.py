"""FastAPI surface for the chatbot.

Deliberately small: the turn is ``run_turn`` and nothing here reimplements any part of it.
No streaming and no thinking-agent endpoints - those belong to the Streamlit stack in
app.py, which is out of scope for this build.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import conversation_store, turn_status
from src.config import settings
from src.graph.build import run_turn
from src.graph.state import from_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TrailerPlace chatbot", version="1.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """One turn. Exactly one model call happens inside."""
    if len(request.message) > settings.chat_max_message_chars:
        raise HTTPException(status_code=413, detail="Message too long.")

    session_id = request.session_id or str(uuid.uuid4())
    try:
        return run_turn(session_id, request.message)
    except Exception:
        # The turn already degrades internally on a model failure, so reaching here means
        # something structural. The customer gets a reply rather than a stack trace.
        logger.exception("Turn failed: session=%s", session_id)
        raise HTTPException(status_code=500, detail="Something went wrong on our side.")


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
    snapshot, conversation, lead_id = conversation_store.load_session(session_id)
    if snapshot is None and not conversation:
        raise HTTPException(status_code=404, detail="No such session.")
    state = from_snapshot(session_id, snapshot)
    return {
        "session_id": session_id,
        "lead_id": lead_id,
        "category": state.get("category"),
        "slots": state.get("slots"),
        "declined_slots": state.get("declined_slots"),
        "required_remaining": [
            slot for slot in (state.get("required_slots") or [])
            if slot not in (state.get("slots") or {})
            and slot not in (state.get("declined_slots") or [])
        ],
        "qualification_complete": state.get("qualification_complete"),
        "contact": state.get("contact"),
        "messages": conversation,
    }


@app.post("/session/reset")
def reset_session() -> dict:
    """Hand back a fresh session id. Existing rows are left alone - a conversation is a
    record, and starting a new one is not a reason to delete the old one."""
    return {"session_id": str(uuid.uuid4())}


@app.get("/session/{session_id}/turn-status")
def get_turn_status(session_id: str) -> dict:
    """What the turn is doing right now, for a UI that is mid-wait."""
    return {"session_id": session_id, "status": turn_status.peek(session_id)}


@app.get("/health")
def health() -> dict:
    from src import db

    return {
        "ok": True,
        "model": settings.chat_model,
        "database": db.database_enabled(),
        "persistence": conversation_store.persistence_enabled(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.chatbot_api_port)
