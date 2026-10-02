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
Messenger a customer's whole history is one conversation. So does a new detail after the
trailers (a size, a weight, a feature): if they then leave our question unanswered, they are
shown a fresh set on everything they have said, without the ones they have already seen.

Several types wanted at once ("Equipment + Car Hauler") count as one category here.

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
        # Several types shown and none chosen: "which suits you better?" is waiting.
        or (not state.get("category") and bool(state.get("candidate_categories")))
        or any(state.get(key) for key in _OPEN_CONFIRMATIONS)
    )


def shown_categories(state: dict) -> list[str]:
    return list(state.get("results_categories") or [])


def _several_key(state: dict) -> str | None:
    """The timer's name for several types at once: "Equipment + Car Hauler"."""
    wanted = list(state.get("candidate_categories") or [])
    return " + ".join(wanted) if wanted and not state.get("category") else None


def note_results_shown(state: dict) -> None:
    """Trailers went out for the current category: its timer is spent."""
    category = state.get("category") or _several_key(state) or ALL_TYPES
    if category and category not in shown_categories(state):
        state["results_categories"] = shown_categories(state) + [category]


def allow_another_showing(state: dict) -> None:
    """They told us something new after seeing trailers: one more showing if they go quiet.

    Live, a customer who had been shown trailers of every type then gave his cargo and a 14k
    GVWR, and nothing more ever came - the timer had been spent on the first showing. A new
    detail un-spends it; "ok thanks" or a question about our hours is not one.
    """
    if not state.get("category") and not state.get("candidate_categories"):
        # The trailers they saw asked which type they want; that is still the open question.
        state["pending_type_question"] = True
    key = target_category(state)
    if key in shown_categories(state):
        state["results_categories"] = [c for c in shown_categories(state) if c != key]


def target_category(state: dict) -> str | None:
    """The category they are after. Mid-switch it is the new one: while we ask whether to
    keep their earlier answers the old category is still set, but they have already chosen -
    live, "actually I need a utility trailer instead" after livestock listings left no timer,
    because Livestock had been shown."""
    keep = state.get("pending_keep_filters") or {}
    chosen = keep.get("new_category") or state.get("category")
    if chosen:
        return chosen
    several = _several_key(state)
    if several:
        return several
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
