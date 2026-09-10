"""Changing category, keeping or dropping filters, and the haul-item suggestion."""
from __future__ import annotations

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def start_with_filters(fake_llm):
    """A Dump session carrying two real answers."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel", "payload_capacity": "6000 lbs"}))
    run_turn("s1", "gravel, 6000 lbs")


# ------------------------------------------------------------- the keep question (S12)
def test_changing_category_asks_whether_to_keep_collected_filters(fake_llm):
    start_with_filters(fake_llm)

    fake_llm.push(turn_output(category_mentioned="equipment trailer", intent="category_change"))
    result = run_turn("s1", "actually make it an equipment trailer")

    state = state_after()
    assert state["pending_keep_filters"] is not None
    assert state["category"] == "Dump", "not switched until they answer"
    assert "keep those" in result["assistant_text"]


def test_only_filters_with_real_values_are_mentioned(fake_llm):
    """Brief S12: offering to keep a blank is noise."""
    start_with_filters(fake_llm)

    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    result = run_turn("s1", "equipment trailer instead")

    text = result["assistant_text"]
    assert "6000" in text
    assert "width" not in text and "None" not in text


def test_the_keep_question_never_offers_to_keep_the_haul_item(fake_llm):
    """What they are hauling is what DEFINES the category. Offering to keep it invites the
    customer to preserve the very thing that just changed, and it is a required question
    for the new category anyway."""
    start_with_filters(fake_llm)

    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    result = run_turn("s1", "equipment trailer instead")

    assert "gravel" not in result["assistant_text"]
    assert "haul item" not in result["assistant_text"].lower()


def test_the_old_haul_item_is_dropped_and_asked_again(fake_llm):
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="all"))
    run_turn("s1", "keep them")

    state = state_after()
    assert "haul_item" not in state["slots"], "gravel is not an answer for an Equipment trailer"
    assert "haul_item" in state["required_slots"]


def test_a_haul_item_restated_in_the_same_message_survives(fake_llm):
    """The usual way a category change arrives: "actually I need to move a tractor"."""
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="all", slots={"haul_item": "a tractor"}))
    run_turn("s1", "keep them, I'm moving a tractor")

    assert state_after()["slots"]["haul_item"] == "a tractor"


def test_keeping_everything_carries_the_filters_over(fake_llm):
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="all"))
    run_turn("s1", "yes keep them")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["slots"]["payload_capacity"] == 6000.0


def test_starting_fresh_drops_them(fake_llm):
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="none"))
    run_turn("s1", "no, start over")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["slots"] == {}


def test_keeping_a_subset_keeps_only_what_they_named(fake_llm):
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="some", kept_fields=["payload_capacity"]))
    run_turn("s1", "just the weight")

    state = state_after()
    assert state["slots"] == {"payload_capacity": 6000.0}


def test_changing_with_nothing_collected_switches_without_asking(fake_llm):
    """There is nothing to keep, so the question would be noise."""
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "actually equipment")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["pending_keep_filters"] is None


# ------------------------------------------------- the haul-item suggestion (your addition)
def test_a_haul_item_suiting_another_category_offers_a_switch(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="utility", intent="category_selection"))
    run_turn("s1", "utility trailer")

    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    result = run_turn("s1", "a skid steer")

    state = state_after()
    assert state["pending_category_switch"]["suggested"] == "Equipment"
    assert "Equipment" in result["assistant_text"]
    assert "switch" in result["assistant_text"].lower()


def test_saying_yes_switches_without_asking_about_filters(fake_llm):
    """Your instruction: on a suggested switch, do NOT ask the keep-filters question."""
    fake_llm.push(turn_output(category_mentioned="utility", intent="category_selection"))
    run_turn("s1", "utility trailer")
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    run_turn("s1", "a skid steer")

    fake_llm.push(turn_output(category_confirm_answer="yes"))
    result = run_turn("s1", "yes please")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["pending_keep_filters"] is None, "no keep question on a suggested switch"
    assert state["slots"]["haul_item"] == "a skid steer", "their answer carried over"
    assert "keep those" not in result["assistant_text"]


def test_saying_no_stays_put_and_never_suggests_it_again(fake_llm):
    fake_llm.push(turn_output(category_mentioned="utility", intent="category_selection"))
    run_turn("s1", "utility trailer")
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    run_turn("s1", "a skid steer")

    fake_llm.push(turn_output(category_confirm_answer="no"))
    run_turn("s1", "no thanks")

    state = state_after()
    assert state["category"] == "Utility"
    assert state["pending_category_switch"] is None
    assert "Utility->Equipment" in state["rejected_switches"]

    # Saying the same thing again must not re-open it.
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    run_turn("s1", "still a skid steer")
    assert state_after()["pending_category_switch"] is None
