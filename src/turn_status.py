"""Live, mid-turn status for the UI, published while the graph is still running.

A /chat turn is one blocking request: the graph runs analyze -> search -> respond and the
client sees nothing until all of it finishes. The session state is no help either — routes
only writes `_sessions[session_id] = result` once the graph has RETURNED, so a second request
reading it mid-turn sees the previous turn's state.

So the search node publishes here the moment it fires, and the UI polls
GET /session/{id}/turn-status while it waits. That is what turns "Let me see what we have on
the lot for you." from a line printed alongside the results into one the customer reads while
we are actually looking.

Deliberately in src/ rather than src/api/: the graph writes it and the API reads it, so it
belongs to neither. Process-local by design, exactly like routes._sessions — a multi-worker
deployment would need a shared store, and would already need one for _sessions first.
"""
from __future__ import annotations

import threading

# session_id -> the line to show for the turn currently in flight.
_STATUS: dict[str, str] = {}
_LOCK = threading.Lock()


def publish(session_id: str | None, message: str) -> None:
    """Make ``message`` visible to pollers for this session's in-flight turn."""
    key = str(session_id or "").strip()
    if not key or not str(message or "").strip():
        return
    with _LOCK:
        _STATUS[key] = str(message).strip()


def peek(session_id: str | None) -> str | None:
    """The current line, left in place — the UI polls repeatedly and must see it each time."""
    key = str(session_id or "").strip()
    if not key:
        return None
    with _LOCK:
        return _STATUS.get(key)


def clear(session_id: str | None) -> None:
    """Drop this session's line. Called at both ends of a turn.

    At the start so a poll can never surface the PREVIOUS turn's line, and at the end so the
    dict does not grow one permanent entry per session for the life of the process.
    """
    key = str(session_id or "").strip()
    if not key:
        return
    with _LOCK:
        _STATUS.pop(key, None)
