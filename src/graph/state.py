"""The conversation state carried between turns.

Key names are not free. ``category``, ``slots``, ``brand_preference``,
``non_metadata_features``, ``shown_urls``, ``qualification_complete`` and ``turn_outcome``
are the contract ``src/graph/nodes/search.py`` already reads - renaming any of them breaks
the finished retrieval path, so they stay exactly as they are.

Everything except ``turn_outcome`` is persisted to
``chatbot_conversations.state_snapshot``; ``turn_outcome`` is per-turn scratch and is
rebuilt from empty each turn.
"""
from __future__ import annotations

from typing import Any, TypedDict

# Bumped when the persisted shape changes in a way an old snapshot cannot satisfy.
STATE_SCHEMA_VERSION = 2

# A required question is asked at most twice, then dropped (brief S23).
MAX_ASKS_PER_SLOT = 2


class SessionState(TypedDict, total=False):
    session_id: str
    lead_id: str
    messages: list[dict[str, Any]]

    # ---- the search_node contract: do not rename ----
    category: str | None
    slots: dict[str, Any]
    brand_preference: str | None
    non_metadata_features: list[str]
    shown_urls: list[str]
    qualification_complete: bool
    turn_outcome: dict[str, Any]

    # ---- qualification bookkeeping ----
    required_slots: list[str]
    optional_slots: list[str]
    slot_questions: dict[str, str]
    asked_counts: dict[str, int]
    declined_slots: list[str]
    volunteered_slots: list[str]
    pending_slot: str | None
    invalid_retry_slot: str | None
    invalid_retry_reason: str | None

    # ---- pending confirmations ----
    pending_keep_filters: dict[str, Any] | None
    pending_category_switch: dict[str, Any] | None
    rejected_switches: list[str]
    # "gooseneck" is both a hitch type and a make we stock, so a bare mention is worth one
    # short question rather than a guess. Holds the text that was ambiguous.
    pending_gooseneck_clarification: str | None

    # ---- contact / lead ----
    contact: dict[str, Any]
    # Notifications waiting on a name and a way to reach them. Persisted, so a request made
    # three turns ago still goes out when the customer finally hands over their number.
    pending_email_actions: list[dict[str, Any]]
    contact_followup_pending: str | None

    turn_index: int
    # Whether listings have been put in front of them at least once. It changes what a
    # later requirement change means: the first search is a search, the second is a redo.
    results_shown: bool


# Scratch that must never be written to state_snapshot: it is rebuilt every turn and
# holds listing dicts that would bloat the row for no benefit.
_TRANSIENT_KEYS = ("turn_outcome",)


def new_state(session_id: str) -> SessionState:
    """A fresh session. Every collection is materialised so no node has to None-guard."""
    return SessionState(
        session_id=session_id,
        lead_id="",
        messages=[],
        category=None,
        slots={},
        brand_preference=None,
        non_metadata_features=[],
        shown_urls=[],
        qualification_complete=False,
        turn_outcome={},
        required_slots=[],
        optional_slots=[],
        slot_questions={},
        asked_counts={},
        declined_slots=[],
        volunteered_slots=[],
        pending_slot=None,
        invalid_retry_slot=None,
        invalid_retry_reason=None,
        pending_keep_filters=None,
        pending_category_switch=None,
        rejected_switches=[],
        pending_gooseneck_clarification=None,
        contact={
            "name": None, "email": None, "phone": None,
            "asked": False, "declined": False,
            # Asks that produced nothing new. Progress resets it; the gate stops at 2.
            "asks_without_progress": 0,
            # Whether the "great to have your details" line has already been said.
            "greeted": False,
        },
        pending_email_actions=[],
        contact_followup_pending=None,
        turn_index=0,
        results_shown=False,
    )


def to_snapshot(state: SessionState) -> dict[str, Any]:
    """The JSONB payload for chatbot_conversations.state_snapshot."""
    return {key: value for key, value in state.items() if key not in _TRANSIENT_KEYS}


def from_snapshot(session_id: str, snapshot: dict[str, Any] | None) -> SessionState:
    """Rebuild a session from a persisted snapshot.

    Defaults are laid down first and the snapshot merged over them, so a snapshot written
    by an earlier version simply lacks the newer keys rather than crashing a node that
    reads one. Unknown keys are dropped for the same reason in reverse.
    """
    state = new_state(session_id)
    if not snapshot:
        return state
    known = set(SessionState.__annotations__)
    for key, value in snapshot.items():
        if key in known and key not in _TRANSIENT_KEYS:
            state[key] = value  # type: ignore[literal-required]
    state["session_id"] = session_id
    state["turn_outcome"] = {}
    return state
