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


def test_a_rejected_value_is_re_asked_by_python():
    state = _state(invalid_retry_slot="payload_capacity", invalid_retry_reason="negative")
    assert _falls_back(state, _output("Thanks! What will you be hauling?", ["haul_item"]))


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
