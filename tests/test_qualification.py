"""Whole turns through run_turn(): qualification, questions, attempts, single-call."""
from __future__ import annotations

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output


def state_after(session_id):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


# ------------------------------------------------------------------- one call per turn (S31)
def test_exactly_one_model_call_per_turn(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    result = run_turn("s1", "I need a dump trailer")

    assert fake_llm.calls == 1
    assert result["usage"]["chat_completions"] == 1


def test_still_one_call_on_a_turn_that_searches(fake_llm, no_search):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    result = run_turn("s1", "just show me what you have")

    assert result["usage"]["chat_completions"] == 1
    assert len(no_search) == 1


# ------------------------------------------------------------ capture before category (S11)
def test_values_given_before_a_category_are_kept(fake_llm):
    fake_llm.push(turn_output(slots={"length": "24 ft"}, intent="feature_request_no_category"))
    run_turn("s1", "I need something around 24 ft")

    state = state_after("s1")
    assert state["category"] is None
    assert state["slots"]["length"] == 24.0

    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "a dump trailer")

    state = state_after("s1")
    assert state["category"] == "Dump"
    assert state["slots"]["length"] == 24.0, "the length survived the category selection"


# --------------------------------------------------------------- multi-field answers (S14)
def test_several_fields_in_one_message_are_all_stored_and_never_re_asked(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(slots={"haul_item": "gravel", "payload_capacity": "3 tons"}))
    result = run_turn("s1", "gravel, about 3 tons")

    state = state_after("s1")
    assert state["slots"]["haul_item"] == "gravel"
    assert state["slots"]["payload_capacity"] == 6000.0
    # Dump requires exactly these two, so both being answered ends qualification.
    assert result["qualification_complete"] is True


# -------------------------------------------------------------------- the two-ask cap (S23)
def test_a_question_is_asked_twice_then_dropped(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    assert state_after("s1")["asked_counts"].get("haul_item") == 1

    fake_llm.push(turn_output(intent="smalltalk_other", answered_current_question=False))
    run_turn("s1", "hmm")
    assert state_after("s1")["asked_counts"].get("haul_item") == 2

    fake_llm.push(turn_output(intent="smalltalk_other", answered_current_question=False))
    run_turn("s1", "hmm again")

    state = state_after("s1")
    assert "haul_item" in state["declined_slots"]
    assert state["asked_counts"]["haul_item"] == 2, "never asked a third time"


def test_a_counter_question_consumes_an_attempt(fake_llm):
    """Your call: being asked is what costs an attempt, whatever they send back."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(
        turn_output(
            intent="general_question",
            answered_current_question=False,
            user_question_to_answer="what sizes do you have?",
            answer_to_customer_question="We carry 10 to 24 ft dump trailers.",
        )
    )
    result = run_turn("s1", "what sizes do you have?")

    assert "We carry 10 to 24 ft" in result["assistant_text"]
    assert state_after("s1")["asked_counts"]["haul_item"] == 2, "the re-ask cost the attempt"


def test_an_explicit_skip_declines_immediately_without_a_second_ask(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(intent="skip_current", answered_current_question=False))
    run_turn("s1", "skip that")

    state = state_after("s1")
    assert "haul_item" in state["declined_slots"]
    assert state["asked_counts"]["haul_item"] == 1, "no second ask was spent"


def test_a_vague_answer_declines_immediately(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")

    fake_llm.push(turn_output(slots={"payload_capacity": "whatever works"}))
    run_turn("s1", "whatever works")

    state = state_after("s1")
    assert "payload_capacity" in state["declined_slots"]
    assert "payload_capacity" not in state["slots"]


# --------------------------------------------------------------- never re-ask (S29)
def test_an_answered_question_is_never_asked_again(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    first = run_turn("s1", "dump trailer")
    assert "haul" in first["assistant_text"].lower() or "material" in first["assistant_text"].lower()

    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    second = run_turn("s1", "gravel")

    assert "material will you be hauling" not in second["assistant_text"]


def test_the_models_question_is_overridden_when_it_names_a_resolved_slot(fake_llm):
    """The model proposing an answered slot must not produce a repeat question."""
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(
        turn_output(
            slots={"haul_item": "gravel"},
            next_question_slot="haul_item",
            next_question_text="What will you be hauling?",
        )
    )
    result = run_turn("s1", "gravel")

    assert "What will you be hauling?" not in result["assistant_text"]


def test_the_models_phrasing_is_used_when_it_names_the_right_slot(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(
        turn_output(
            category_mentioned="dump",
            intent="category_selection",
            next_question_slot="haul_item",
            next_question_text="What are you planning to haul in it?",
        )
    )
    result = run_turn("s1", "dump trailer")
    assert "What are you planning to haul in it?" in result["assistant_text"]


# ------------------------------------------------------------- invalid values (S22)
def test_a_negative_weight_is_re_asked_not_stored(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")

    fake_llm.push(
        turn_output(slots={"payload_capacity": "-500 lbs"}, extracted={"payload_capacity": 500.0})
    )
    result = run_turn("s1", "-500 lbs")

    state = state_after("s1")
    assert "payload_capacity" not in state["slots"], "the sign-flipped value was not stored"
    assert "negative number" in result["assistant_text"]


def test_the_negative_retry_cannot_loop(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")

    for _ in range(3):
        fake_llm.push(turn_output(slots={"payload_capacity": "-500 lbs"}))
        run_turn("s1", "-500 lbs")

    state = state_after("s1")
    assert state["asked_counts"]["payload_capacity"] <= 2
    assert "payload_capacity" in state["declined_slots"]


# ------------------------------------------------------------------- counter-questions (S24)
def test_a_question_mid_flow_is_answered_without_losing_state(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")

    fake_llm.push(
        turn_output(
            intent="faq",
            answered_current_question=False,
            user_question_to_answer="where are you located?",
            answer_to_customer_question="We're in Wharton, TX.",
        )
    )
    result = run_turn("s1", "where are you located?")

    state = state_after("s1")
    assert "Wharton" in result["assistant_text"]
    assert state["category"] == "Dump", "the category survived the interruption"
    assert state["slots"]["haul_item"] == "gravel", "the answers survived"
