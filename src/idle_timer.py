"""The 5-minute rule. LLM_WRITES_REPLY only.

A customer who has chosen a category and been asked a question, then goes quiet, is shown
that category's trailers after IDLE_RESULTS_MINUTES - whatever we know by then, even nothing -
and asked who they are afterwards, so the team hears which trailers they saw and who saw
them. A quiet customer is otherwise a lost one: the conversation stops on our question.

A customer who gave specs but no trailer type ("7x14, 4 ft walls, 14 ply tires") and was
asked which type they want is treated the same way, under ALL_TYPES: after the same wait they
are shown trailers of every type, ranked on those specs, and asked which type suits them.

Once per category. When trailers have been shown for a category, by this rule or because they
asked, the timer does not run for it again; a change of category starts it afresh, because on
Messenger a customer's whole history is one conversation.

This module only decides. The row is written with the turn (conversation_store.save_turn), so
a timer can never disagree with the conversation it belongs to, and the sweep in
src/idle_sweep.py is what fires it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src import config

# Python's own questions. Waiting on one of these is waiting on an answer too.
_OPEN_CONFIRMATIONS = (
    "pending_gooseneck_clarification", "pending_category_switch", "pending_keep_filters",
    "pending_axle_basis", "pending_axle_count",
)


# The timer row's category when no type has been chosen. The column is NOT NULL, and the
# idle turn reads this as "search every type".
ALL_TYPES = "all types"


def enabled() -> bool:
    settings = config.settings
    return bool(settings.llm_writes_reply) and float(settings.idle_results_minutes or 0) > 0


def question_waiting(state: dict) -> bool:
    return (
        bool(state.get("pending_slot"))
        or bool(state.get("pending_type_question"))
        or any(state.get(key) for key in _OPEN_CONFIRMATIONS)
    )


def shown_categories(state: dict) -> list[str]:
    return list(state.get("results_categories") or [])


def note_results_shown(state: dict) -> None:
    """Trailers went out for the current category: its timer is spent."""
    category = state.get("category") or ALL_TYPES
    if category and category not in shown_categories(state):
        state["results_categories"] = shown_categories(state) + [category]


def target_category(state: dict) -> str | None:
    """The category they are after. Mid-switch it is the new one: while we ask whether to
    keep their earlier answers the old category is still set, but they have already chosen -
    live, "actually I need a utility trailer instead" after livestock listings left no timer,
    because Livestock had been shown."""
    keep = state.get("pending_keep_filters") or {}
    chosen = keep.get("new_category") or state.get("category")
    if chosen:
        return chosen
    # No type yet, and we asked which one they want: the search runs across every type.
    return ALL_TYPES if state.get("pending_type_question") else None


def plan(state: dict, channel_id: str | None, now: datetime | None = None) -> dict[str, Any] | None:
    """The timer this turn leaves behind: a row to arm, or None to cancel any that is armed."""
    if not enabled():
        return None
    category = target_category(state)
    if not category or category in shown_categories(state) or not question_waiting(state):
        return None
    now = now or datetime.now(timezone.utc)
    return {
        "channel": "messenger" if channel_id else "web",
        "channel_id": str(channel_id) if channel_id else None,
        "category": str(category),
        "due_at": now + timedelta(minutes=float(config.settings.idle_results_minutes)),
    }
