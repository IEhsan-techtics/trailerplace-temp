"""LLM_WRITES_REPLY: the model's whole reply goes out as written, or not at all.

Live, compose's name check read "You're welcome! What material will you be hauling?" as a
greeting to someone called What, cut the word out, and sent "You're welcome material will
you be hauling". With the flag on, the model's reply is checked and never edited - so a
reply either reaches the customer exactly as written, or compose builds one as before.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.graph.nodes import compose
from src.llm.client import empty_output
from src.llm.schemas import ChatbotTurnReplyOutput


@pytest.fixture(autouse=True)
def writes_reply(monkeypatch):
    monkeypatch.setattr(compose, "settings", replace(compose.settings, llm_writes_reply=True))


def _output(reply: str, asked: list[str] | None = None, **pieces):
    fields = empty_output().model_dump()
    fields.update(pieces)
    return ChatbotTurnReplyOutput.model_validate({**fields, "reply": reply, "asked_slots": asked or []})


def _state(**overrides):
    """Mid-qualification on a Dump trailer, contact settled, so no contact ask is due."""
    state = {
        "session_id": "s1",
        "turn_index": 3,
        "category": "Dump",
        "required_slots": ["haul_item", "payload_capacity"],
        "slots": {"length": 10.0, "width": 6.0},
        "asked_counts": {},
        "declined_slots": [],
        "contact": {"name": "Dave", "phone": "979-555-0100", "greeted": True, "asked": True},
        "turn_outcome": {},
    }
    state.update(overrides)
    return state


def _send(state, output):
    compose.compose_node(state, output)
    return state["turn_outcome"]


def test_the_live_what_reply_goes_out_word_for_word():
    reply = "You're welcome! What material will you be hauling (dirt, gravel, debris, etc.)?"
    outcome = _send(_state(), _output(reply, ["haul_item"]))

    assert outcome["assistant_text"] == reply
    assert outcome["asked_slot"] == "haul_item"


def test_the_question_it_asks_is_counted():
    state = _state()
    _send(state, _output("Got it. What will you be hauling?", ["haul_item"]))

    assert state["asked_counts"] == {"haul_item": 1}
    assert state["pending_slot"] == "haul_item"


def test_it_may_pick_any_open_question_not_only_the_first():
    state = _state()
    reply = "Thanks! What's the approximate weight of the load?"
    outcome = _send(state, _output(reply, ["payload_capacity"]))

    assert outcome["assistant_text"] == reply
    assert state["pending_slot"] == "payload_capacity"


def _falls_back(state, output):
    outcome = _send(state, output)
    return outcome["assistant_text"] != output.reply


def test_two_questions_fall_back():
    reply = "What will you be hauling? And how heavy is it?"
    assert _falls_back(_state(), _output(reply, ["haul_item"]))


def test_two_slots_claimed_fall_back():
    reply = "What will you be hauling, and how heavy is it?"
    assert _falls_back(_state(), _output(reply, ["haul_item", "payload_capacity"]))


def test_a_question_already_answered_falls_back():
    state = _state(slots={"length": 10.0, "width": 6.0, "haul_item": "gravel"})
    assert _falls_back(state, _output("What will you be hauling?", ["haul_item"]))


def test_a_question_asked_twice_already_falls_back():
    state = _state(asked_counts={"haul_item": 2})
    assert _falls_back(state, _output("What will you be hauling?", ["haul_item"]))


def test_a_question_that_is_not_about_its_slot_falls_back():
    """The model named haul_item but asked the length - the count would go to the wrong
    question, and length is already known."""
    assert _falls_back(_state(), _output("How long should the trailer be?", ["haul_item"]))


def test_a_slot_question_it_did_not_name_falls_back():
    assert _falls_back(_state(), _output("What will you be hauling?", []))


def test_asking_nothing_with_questions_left_falls_back():
    assert _falls_back(_state(), _output("Great, thanks for that.", []))


def test_a_confirmation_python_asks_owns_the_turn():
    state = _state(pending_category_switch={"suggested": "Equipment", "from_haul_item": "a skid steer"})
    assert _falls_back(state, _output("What will you be hauling?", ["haul_item"]))


def _retry(**overrides):
    return _state(**{"invalid_retry_slot": "payload_capacity", "invalid_retry_reason": "negative", **overrides})


def test_a_rejected_value_moving_on_to_another_question_falls_back():
    assert _falls_back(_retry(), _output("Thanks! What will you be hauling?", ["haul_item"]))


def test_a_rejected_value_re_asked_with_the_reason_goes_out():
    state = _retry()
    reply = "That came through as a negative number. What's the rough haul weight per load?"
    outcome = _send(state, _output(reply, ["payload_capacity"]))

    assert outcome["assistant_text"] == reply
    assert state["asked_counts"] == {"payload_capacity": 1}


def test_thanks_for_a_rejected_value_falls_back():
    """Live: "Thanks, I've noted that. That came through as a negative number..." """
    reply = "Thanks, I've noted that. That came through as a negative number. What's the rough haul weight per load?"
    assert _falls_back(_retry(), _output(reply, ["payload_capacity"]))


def test_a_re_ask_that_does_not_say_what_was_wrong_falls_back():
    reply = "What's the rough haul weight per load?"
    assert _falls_back(_retry(), _output(reply, ["payload_capacity"]))


def test_an_implausible_value_re_asked_in_its_own_words_goes_out():
    state = _retry(invalid_retry_reason="implausible")
    reply = "That seems unusual for a trailer load - could you double-check the weight and unit?"
    assert _send(state, _output(reply, ["payload_capacity"]))["assistant_text"] == reply


def test_a_re_ask_that_does_not_name_what_it_asks_falls_back():
    """"the amount" of what? A question the check cannot place is not counted as that slot."""
    state = _retry(invalid_retry_reason="implausible")
    reply = "That seems unusual for a trailer - could you double-check the amount and unit?"
    assert _falls_back(state, _output(reply, ["payload_capacity"]))


def test_the_axle_count_is_still_re_asked_by_python():
    state = _state(invalid_retry_slot="axle_count", invalid_retry_reason="axle_range",
                   required_slots=["haul_item", "axle_count"])
    reply = "We carry one to four axles, so that number seems off. How many axles would you like?"
    assert _falls_back(state, _output(reply, ["axle_count"]))


def test_a_rejected_value_already_asked_twice_is_not_asked_again():
    state = _retry(asked_counts={"payload_capacity": 2})
    reply = "That came through as a negative number. What's the rough haul weight per load?"
    assert _falls_back(state, _output(reply, ["payload_capacity"]))


def test_asking_for_contact_when_it_is_not_due_falls_back():
    reply = "What will you be hauling? Could you share your name and the best email or phone to reach you on?"
    assert _falls_back(_state(), _output(reply, ["haul_item"]))


def test_a_due_contact_ask_rides_along_as_the_second_question():
    state = _state(contact={})
    reply = "What will you be hauling? Could I get your name and the best email or phone to reach you on?"
    outcome = _send(state, _output(reply, ["haul_item"]))

    assert outcome["assistant_text"] == reply
    assert state["contact"]["asks_without_progress"] == 1


def test_a_due_contact_ask_it_left_out_is_added_on_the_end():
    from src.graph.nodes import greeting

    state = _state(contact={})
    outcome = _send(state, _output("Got it. What will you be hauling?", ["haul_item"]))

    assert outcome["assistant_text"].startswith("Got it. What will you be hauling?")
    assert outcome["assistant_text"].endswith(greeting.gate_ask(state))


def test_no_category_the_type_question_needs_no_slot():
    state = _state(category=None, required_slots=[], slots={})
    reply = "Happy to help! Which type fits what you need - Utility, Dump, Enclosed or something else?"
    outcome = _send(state, _output(reply, []))

    assert outcome["assistant_text"] == reply
    assert outcome["asked_slot"] is None


def test_the_fallback_is_compose_as_before():
    """A turned-down reply costs nothing: the pieces are assembled exactly as with the flag off."""
    output = _output(
        "What will you be hauling? And how heavy?", ["haul_item"],
        acknowledgement="Thanks for that.",
        next_question_slot="haul_item",
        next_question_text="What will you be hauling?",
    )
    outcome = _send(_state(), output)

    assert outcome["assistant_text"] == "Thanks for that. What will you be hauling?"


def test_with_the_flag_off_the_reply_is_ignored(monkeypatch):
    monkeypatch.setattr(compose, "settings", replace(compose.settings, llm_writes_reply=False))
    output = _output(
        "You're welcome! What material will you be hauling?", ["haul_item"],
        acknowledgement="Thanks for that.",
        next_question_slot="haul_item",
        next_question_text="What will you be hauling?",
    )
    outcome = _send(_state(), output)

    assert outcome["assistant_text"] == "Thanks for that. What will you be hauling?"


def test_the_prompt_only_asks_for_a_reply_with_the_flag_on():
    from src.llm.prompt import _system_prompt_for

    assert "THE REPLY -> reply" in _system_prompt_for(0, True)
    assert "THE REPLY -> reply" not in _system_prompt_for(0, False)


# ---- off topic: the whole turn, written by the model ----


def _off_topic(state=None, **kwargs):
    state = state or _state()
    state["turn_outcome"]["off_topic"] = True
    return state


def test_an_off_topic_decline_goes_out_as_written():
    reply = "Sorry, I can only help with trailers here. What will you be hauling?"
    outcome = _send(_off_topic(), _output(reply, ["haul_item"]))

    assert outcome["assistant_text"] == reply
    assert outcome["asked_slot"] == "haul_item"


def test_an_off_topic_reply_that_does_what_they_asked_falls_back():
    """Live, the model apologised and then gave the recipe anyway."""
    reply = (
        "I mainly help with trailers, but here you go: toast two slices of bread, add ham, "
        "cheese and lettuce, then close it up and cut it in half. What will you be hauling?"
    )
    assert _falls_back(_off_topic(), _output(reply, ["haul_item"]))


def test_an_off_topic_reply_with_code_falls_back():
    reply = "Only trailers here. ```print('hi')``` What will you be hauling?"
    assert _falls_back(_off_topic(), _output(reply, ["haul_item"]))


def test_an_off_topic_reply_must_decline():
    assert _falls_back(_off_topic(), _output("What will you be hauling?", ["haul_item"]))


def test_off_topic_with_nothing_left_to_ask_may_just_stop():
    state = _off_topic(_state(slots={"haul_item": "gravel", "payload_capacity": 5000}))
    reply = "Sorry, I can only help with trailers and TrailerPlace here."
    outcome = _send(state, _output(reply, []))

    assert outcome["assistant_text"] == reply


def test_an_off_topic_first_message_still_gets_the_welcome():
    from src.graph.nodes import greeting

    state = _off_topic(_state(turn_index=1, category=None, required_slots=[], slots={}))
    bare = "Sorry, I can only help with trailers here. Which type fits what you need - Utility, Dump or Enclosed?"
    assert _falls_back(state, _output(bare, []))

    state = _off_topic(_state(turn_index=1, category=None, required_slots=[], slots={}))
    welcomed = f"{greeting.OPENING} {bare}"
    assert _send(state, _output(welcomed, []))["assistant_text"] == welcomed


def test_accepting_a_refusal_is_not_asking_for_contact():
    """Live: "Understood - no contact details needed." read as asking again."""
    reply = "Understood - no contact details needed. What will you be hauling?"
    outcome = _send(_state(contact={"declined": True}), _output(reply, ["haul_item"]))

    assert outcome["assistant_text"] == reply


def test_a_request_without_a_question_mark_still_counts():
    reply = "What will you be hauling? Please share your name and the best email or phone to reach you on."
    assert _falls_back(_state(), _output(reply, ["haul_item"]))


def test_our_own_menu_shape_is_one_question():
    """Two question marks, one question - word for word what orientation_question sends."""
    state = _state(category=None, required_slots=[], slots={})
    reply = (
        "No problem at all. What type of trailer are you looking for? We have Utility, "
        "Enclosed, Equipment, Dump and Car Hauler, and many more - which one fits what you need?"
    )
    assert _send(state, _output(reply, []))["assistant_text"] == reply


def test_two_real_questions_with_no_category_still_fall_back():
    state = _state(category=None, required_slots=[], slots={})
    reply = "What type of trailer are you looking for? And what's your budget?"
    assert _falls_back(state, _output(reply, []))


def test_a_due_contact_ask_may_be_the_one_question():
    state = _state(contact={})
    reply = "You're welcome! Could you share your name and either an email or phone number so our team can follow up?"
    outcome = _send(state, _output(reply, []))

    assert outcome["assistant_text"] == reply
    assert outcome["asked_slot"] is None
    assert state["contact"]["asks_without_progress"] == 1
