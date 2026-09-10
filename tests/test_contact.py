"""Contact capture and the lead row. Asked once, then never again."""
from __future__ import annotations

import pytest

from src.conversation_store import load_lead, load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


@pytest.fixture
def capture(fake_llm):
    """Run one turn carrying contact details; return the resulting lead row."""

    def _capture(**contact):
        fake_llm.push(turn_output(intent="contact_info_provided", **contact))
        run_turn("s1", "here are my details")
        return load_lead("s1")

    return _capture


# ------------------------------------------------------------------------ storing the lead
def test_name_and_email_make_a_hard_lead(capture):
    """The dealership's minimum for a real lead: a name plus one way to reach them."""
    lead = capture(name="Dave", email="dave@example.com")
    assert lead["name"] == "Dave"
    assert lead["email"] == "dave@example.com"
    assert lead["lead_type"] == "hard"
    assert lead["contact_status"] == "complete"


def test_name_and_phone_also_make_a_hard_lead(capture):
    lead = capture(name="Dave", phone="979-555-0100")
    assert lead["phone_number"] == "979-555-0100"
    assert lead["lead_type"] == "hard"


def test_a_name_alone_is_not_a_lead_yet(capture):
    lead = capture(name="Dave")
    assert lead["lead_type"] == "soft"
    assert lead["contact_status"] == "partial"


def test_contact_details_without_a_name_are_not_a_lead(capture):
    lead = capture(email="dave@example.com")
    assert lead["lead_type"] == "soft"


def test_a_session_that_gives_nothing_stays_a_soft_lead(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hi")
    lead = load_lead("s1")
    assert lead["lead_type"] == "soft"
    assert lead["contact_status"] == "missing_contact"


# ------------------------------------------------------------------- asked once, then never
def test_the_opener_is_marked_asked_after_the_first_turn(fake_llm):
    """Whatever they replied, they have now been asked - that is the whole policy."""
    assert state_after()["contact"]["asked"] is False

    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hi")

    assert state_after()["contact"]["asked"] is True


def test_ignoring_the_opener_does_not_get_a_second_ask(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hi")
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "I need a dump trailer")

    state = state_after()
    assert state["contact"]["asked"] is True
    assert state["contact"]["name"] is None
    # The prompt is what carries the instruction; assert it says so for this state.
    from src.llm.prompt import state_block

    assert "Never ask again" in state_block(state)


def test_declining_is_remembered_and_never_pursued(fake_llm):
    fake_llm.push(turn_output(intent="contact_declined", declined=True))
    run_turn("s1", "I'd rather not share that")

    assert state_after()["contact"]["declined"] is True

    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    assert state_after()["contact"]["declined"] is True


def test_a_decline_on_a_message_that_does_something_else_still_counts(fake_llm):
    """ContactInfo.declined is set even when the dominant intent is something bigger."""
    fake_llm.push(
        turn_output(category_mentioned="dump", intent="category_selection", declined=True)
    )
    run_turn("s1", "dump trailer, and no I won't give my number")

    state = state_after()
    assert state["contact"]["declined"] is True
    assert state["category"] == "Dump", "the rest of the message still applied"


def test_details_are_never_overwritten_by_a_later_turn(fake_llm):
    fake_llm.push(turn_output(intent="contact_info_provided", name="Dave", email="dave@x.com"))
    run_turn("s1", "I'm Dave, dave@x.com")

    fake_llm.push(turn_output(intent="contact_info_provided", name="Sam", email="sam@x.com"))
    run_turn("s1", "actually I'm Sam")

    state = state_after()
    assert state["contact"]["name"] == "Dave", "the lead keeps the details it qualified on"
    assert state["contact"]["email"] == "dave@x.com"


def test_partial_details_across_two_turns_add_up_to_a_hard_lead(fake_llm):
    fake_llm.push(turn_output(intent="contact_info_provided", name="Dave"))
    run_turn("s1", "I'm Dave")
    assert load_lead("s1")["lead_type"] == "soft"

    fake_llm.push(turn_output(intent="contact_info_provided", phone="979-555-0100"))
    run_turn("s1", "my number is 979-555-0100")
    assert load_lead("s1")["lead_type"] == "hard"


# ------------------------------------------------------------------------ item of interest
def test_the_lead_records_what_they_are_shopping_for(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}, name="Dave", phone="979-555-0100"))
    run_turn("s1", "gravel, I'm Dave on 979-555-0100")

    lead = load_lead("s1")
    assert "Dump" in lead["item_of_interest"]
    assert "gravel" in lead["item_of_interest"]


def test_a_session_with_nothing_yet_still_has_a_valid_item_of_interest(fake_llm):
    """item_of_interest is NOT NULL, so it always resolves to something."""
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hi")
    assert load_lead("s1")["item_of_interest"] == "Unspecified"
