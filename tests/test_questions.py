"""Attempt bookkeeping: ask at most twice, never re-ask a resolved question."""
from __future__ import annotations

from src.graph.state import MAX_ASKS_PER_SLOT, new_state
from src.tools.questions import (
    all_required_resolved,
    at_attempt_cap,
    decline_slot,
    is_resolved,
    mark_asked,
    next_unanswered_slot,
    record_no_preference,
    required_remaining,
    resolve_pending_slot,
)


def qualified_state(**kwargs):
    state = new_state("s1")
    state["category"] = "Dump"
    state["required_slots"] = ["haul_item", "payload_capacity", "length", "hitch_type"]
    state["slot_questions"] = {s: f"Q:{s}" for s in state["required_slots"]}
    state.update(kwargs)
    return state


# ------------------------------------------------------------------------ never re-ask (S29)
def test_an_answered_slot_is_never_returned_again():
    state = qualified_state()
    state["slots"]["haul_item"] = "gravel"
    assert next_unanswered_slot(state) == "payload_capacity"


def test_questions_come_in_the_categorys_own_order():
    state = qualified_state()
    assert next_unanswered_slot(state) == "haul_item"
    state["slots"]["haul_item"] = "gravel"
    assert next_unanswered_slot(state) == "payload_capacity"
    state["slots"]["payload_capacity"] = 6000.0
    assert next_unanswered_slot(state) == "length"


def test_a_none_value_does_not_count_as_answered():
    state = qualified_state()
    state["slots"]["haul_item"] = None
    assert next_unanswered_slot(state) == "haul_item"


# ------------------------------------------------------------------- the two-attempt cap (S23)
def test_a_slot_is_asked_at_most_twice():
    state = qualified_state()
    mark_asked(state, "haul_item")
    assert state["asked_counts"]["haul_item"] == 1
    assert not at_attempt_cap(state, "haul_item")

    mark_asked(state, "haul_item")
    assert state["asked_counts"]["haul_item"] == MAX_ASKS_PER_SLOT
    assert at_attempt_cap(state, "haul_item")

    # At the cap it stops being offered, so a third ask can never be composed.
    assert next_unanswered_slot(state) == "payload_capacity"


def test_the_count_increments_on_the_ask_not_on_the_failure():
    """One ask, one increment - otherwise the first ask is counted twice and the customer
    gets a single attempt instead of two."""
    state = qualified_state()
    mark_asked(state, "length")
    resolve_pending_slot(state, answered=False)
    assert state["asked_counts"]["length"] == 1
    assert not is_resolved(state, "length")


def test_an_unanswered_slot_at_the_cap_is_declined_on_the_next_turn():
    state = qualified_state()
    mark_asked(state, "length")
    resolve_pending_slot(state, answered=False)
    mark_asked(state, "length")
    resolve_pending_slot(state, answered=False)

    assert "length" in state["declined_slots"]
    assert state["pending_slot"] is None


def test_answering_clears_the_pending_slot_without_declining_it():
    state = qualified_state()
    mark_asked(state, "length")
    state["slots"]["length"] = 20.0
    resolve_pending_slot(state, answered=True)
    assert state["pending_slot"] is None
    assert "length" not in state["declined_slots"]


# ------------------------------------------- vague and refusal resolve without a second ask
def test_a_vague_answer_declines_the_slot_immediately():
    """Brief S18: no preference IS an answer. It is never asked a second time."""
    state = qualified_state()
    mark_asked(state, "length")
    record_no_preference(state, ["length"])
    assert "length" in state["declined_slots"]
    assert next_unanswered_slot(state) != "length"


def test_an_explicit_refusal_declines_the_slot_immediately():
    state = qualified_state()
    mark_asked(state, "length")
    decline_slot(state, "length", reason="skip_current")
    assert is_resolved(state, "length")


def test_declining_removes_a_lingering_none_so_it_cannot_become_a_filter():
    state = qualified_state()
    state["slots"]["length"] = None
    decline_slot(state, "length")
    assert "length" not in state["slots"]


# ------------------------------------------------------------------ the invalid-value retry
def test_an_invalid_value_is_re_asked_before_moving_on():
    state = qualified_state()
    state["slots"]["haul_item"] = "gravel"
    state["invalid_retry_slot"] = "payload_capacity"
    assert next_unanswered_slot(state) == "payload_capacity"


def test_the_invalid_retry_is_bounded_by_the_same_cap():
    """-500 lbs three times must not loop."""
    state = qualified_state(invalid_retry_slot="payload_capacity")
    state["asked_counts"] = {"payload_capacity": MAX_ASKS_PER_SLOT}
    assert next_unanswered_slot(state) != "payload_capacity"


# ------------------------------------------------------------------- the completion gate
def test_qualification_is_complete_only_when_every_required_slot_is_resolved():
    state = qualified_state()
    assert not all_required_resolved(state)

    state["slots"].update({"haul_item": "gravel", "payload_capacity": 6000.0, "length": 20.0})
    assert not all_required_resolved(state)

    state["declined_slots"] = ["hitch_type"]
    assert all_required_resolved(state)
    assert required_remaining(state) == []


def test_qualification_is_never_complete_without_a_category():
    """An empty required list must not read as "everything answered"."""
    state = new_state("s1")
    assert not all_required_resolved(state)


def test_no_question_left_returns_none():
    state = qualified_state()
    state["declined_slots"] = list(state["required_slots"])
    assert next_unanswered_slot(state) is None
