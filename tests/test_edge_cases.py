"""The brief's enumerated edge cases, plus the gooseneck flow end to end.

Each test names the behaviour it protects rather than the case number, so a failure says
what broke rather than which paragraph it came from.
"""
from __future__ import annotations

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def pick_dump(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "dump trailer")


# ------------------------------------------------------------------- unresolvable category
def test_an_unrecognised_category_leaves_it_unset(fake_llm):
    fake_llm.push(turn_output(category_mentioned="spaceship trailer", intent="category_selection"))
    run_turn("s1", "I want a spaceship trailer")
    assert state_after()["category"] is None


def test_not_sure_leaves_the_category_unset(fake_llm):
    fake_llm.push(turn_output(category_mentioned="not sure", intent="category_exploration"))
    run_turn("s1", "not sure what I need")
    assert state_after()["category"] is None


def test_asking_about_a_category_does_not_select_it(fake_llm):
    """"What's a tilt trailer for?" is a question, not a choice."""
    fake_llm.push(
        turn_output(
            category_mentioned="tilt", is_category_info_only=True, intent="category_exploration"
        )
    )
    run_turn("s1", "what's a tilt trailer for?")
    assert state_after()["category"] is None


# ------------------------------------------------------------------------- numeric handling
def test_a_range_is_stored_at_its_smallest(fake_llm):
    pick_dump(fake_llm)
    fake_llm.push(
        turn_output(slots={"payload_capacity": "5000-10000 lbs"},
                    extracted={"payload_capacity": 7500.0})
    )
    run_turn("s1", "5000-10000 lbs")
    assert state_after()["slots"]["payload_capacity"] == 5000.0


def test_a_stated_zero_is_not_stored_as_a_filter(fake_llm):
    pick_dump(fake_llm)
    fake_llm.push(turn_output(slots={"payload_capacity": "0"}))
    run_turn("s1", "0")
    state = state_after()
    assert "payload_capacity" not in state["slots"]
    assert "payload_capacity" in state["declined_slots"]


def test_tons_are_converted_to_pounds(fake_llm):
    pick_dump(fake_llm)
    fake_llm.push(turn_output(slots={"payload_capacity": "3 tons"}))
    run_turn("s1", "3 tons")
    assert state_after()["slots"]["payload_capacity"] == 6000.0


def test_spelled_out_numbers_are_stored_numerically(fake_llm):
    pick_dump(fake_llm)
    fake_llm.push(turn_output(slots={"payload_capacity": "two thousand pounds"}))
    run_turn("s1", "two thousand pounds")
    stored = state_after()["slots"]["payload_capacity"]
    assert stored == 2000.0 and isinstance(stored, (int, float))


def test_a_model_number_that_disagrees_with_the_raw_text_loses(fake_llm):
    """The live probe's failure mode: the model averaged "18-20 ft" to 19."""
    pick_dump(fake_llm)
    fake_llm.push(
        turn_output(slots={"length": "around 18-20 ft"}, extracted={"length": 19.0})
    )
    run_turn("s1", "around 18-20 ft")
    assert state_after()["slots"]["length"] == 18.0


# ------------------------------------------------------------------------------ hitch type
def test_either_hitch_stores_nothing_rather_than_both(fake_llm):
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_selection"))
    run_turn("s1", "equipment trailer")
    fake_llm.push(turn_output(slots={"hitch_type": "either is fine"}))
    run_turn("s1", "either is fine")

    state = state_after()
    assert state["slots"].get("hitch_type") is None
    assert "hitch_type" in state["declined_slots"]


# --------------------------------------------------------------------------- the gooseneck
def test_a_bare_gooseneck_mention_asks_which_one_is_meant(fake_llm):
    complete_welcome(fake_llm)
    pick_dump(fake_llm)
    fake_llm.push(turn_output(intent="general_question"))
    result = run_turn("s1", "I want the gooseneck trailer")

    state = state_after()
    assert state["pending_gooseneck_clarification"] is not None
    assert "gooseneck hitch" in result["assistant_text"]
    assert "brand" in result["assistant_text"].lower()


def test_answering_hitch_stores_the_hitch(fake_llm):
    pick_dump(fake_llm)
    fake_llm.push(turn_output(intent="general_question"))
    run_turn("s1", "I want the gooseneck trailer")

    fake_llm.push(turn_output(intent="qualification_answer"))
    run_turn("s1", "the hitch")

    state = state_after()
    assert state["slots"]["hitch_type"] == ["Gooseneck"]
    assert state["brand_preference"] is None
    assert state["pending_gooseneck_clarification"] is None


def test_answering_brand_stores_the_brand(fake_llm):
    pick_dump(fake_llm)
    fake_llm.push(turn_output(intent="general_question"))
    run_turn("s1", "I want the gooseneck trailer")

    fake_llm.push(turn_output(intent="qualification_answer"))
    run_turn("s1", "the brand")

    state = state_after()
    assert state["brand_preference"] == "Gooseneck"
    assert state["slots"].get("hitch_type") is None


def test_gooseneck_hitch_is_never_ambiguous(fake_llm):
    pick_dump(fake_llm)
    fake_llm.push(turn_output(intent="qualification_answer"))
    run_turn("s1", "I want a gooseneck hitch")
    assert state_after()["pending_gooseneck_clarification"] is None


def test_gooseneck_with_another_brand_named_is_the_hitch(fake_llm):
    """The model reads the make - it has the list in its prompt - and that settles it."""
    pick_dump(fake_llm)
    fake_llm.push(
        turn_output(intent="qualification_answer", extracted={"brand_preference": "Diamond C"})
    )
    run_turn("s1", "a gooseneck Diamond C")

    state = state_after()
    assert state["brand_preference"] == "Diamond C"
    assert state["slots"]["hitch_type"] == ["Gooseneck"]
    assert state["pending_gooseneck_clarification"] is None


def test_gooseneck_answering_the_hitch_question_is_never_ambiguous(fake_llm):
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_selection"))
    run_turn("s1", "equipment trailer")
    # Walk to the hitch question.
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer", "payload_capacity": "8000 lbs",
                                     "length": "20 ft"}))
    run_turn("s1", "skid steer, 8000 lbs, 20 ft")

    if state_after()["pending_slot"] == "hitch_type":
        fake_llm.push(turn_output(slots={"hitch_type": "gooseneck"}))
        run_turn("s1", "gooseneck")
        state = state_after()
        assert state["pending_gooseneck_clarification"] is None
        assert state["slots"]["hitch_type"] == ["Gooseneck"]


# ------------------------------------------------------------------------ degraded turns
def test_a_failed_model_call_still_produces_a_reply(fake_llm):
    """The queue is empty, so the fixture returns client.empty_output - the degraded path."""
    pick_dump(fake_llm)
    result = run_turn("s1", "something the model could not read")

    assert result["assistant_text"], "the customer still gets words"
    assert state_after()["category"] == "Dump", "nothing was corrupted"


def test_an_empty_message_does_not_crash(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other"))
    result = run_turn("s1", "")
    assert result["assistant_text"]


# ---------------------------------------------------------------- state the model invents
def test_a_slot_the_model_invents_never_reaches_the_state(fake_llm):
    pick_dump(fake_llm)
    fake_llm.push(turn_output(slots={"colour": "red", "haul_item": "gravel"}))
    run_turn("s1", "red one, for gravel")

    state = state_after()
    assert "colour" not in state["slots"]
    assert state["slots"]["haul_item"] == "gravel"


def test_a_category_the_model_invents_never_reaches_the_state(fake_llm):
    fake_llm.push(turn_output(category_mentioned="Hovercraft", intent="category_selection"))
    run_turn("s1", "a hovercraft trailer")
    assert state_after()["category"] is None
