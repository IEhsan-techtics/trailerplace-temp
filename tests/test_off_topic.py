"""We sell trailers, and only trailers.

Two halves, and they are not symmetric. Refusing a real customer loses a lead, so the bar
for calling a message off topic is high and Python holds a veto over the model. Answering a
stray question costs one turn, so the decline is short and the conversation carries on from
exactly where it was.
"""
from __future__ import annotations

import pytest

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot
from src.tools import scope

from tests.factories import complete_welcome, turn_output


def off_topic_output(**kwargs):
    kwargs.setdefault("intent", "general_question")
    kwargs.setdefault("off_topic", True)
    return turn_output(**kwargs)


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


# ------------------------------------------------------------------ what counts as off topic
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"intent": "smalltalk_other"},
        {"acknowledgement": "I only help with trailers here."},
    ],
)
def test_the_model_saying_so_is_enough_when_nothing_contradicts_it(kwargs):
    assert scope.is_off_topic(off_topic_output(**kwargs)) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"category_mentioned": "dump"},
        {"slots": {"length": "20 ft"}},
        {"name": "Ibrahim"},
        {"phone": "03304388550"},
        {"declined": True},
        {"faq_key": "financing"},
        {"listing_reference": 2},
        {"unavailable_type_requested": "boat trailer"},
        {"extracted": {"haul_item": "gravel"}},
        {"extracted": {"payload_capacity": 6000.0}},
        {"extracted": {"non_metadata_features": ["ramps"]}},
        {"dropped_fields": ["length"]},
    ],
)
def test_a_turn_carrying_trailer_content_is_on_topic_whatever_the_model_said(kwargs):
    """The veto. Every one of these could only have come from a real customer, so the label
    is wrong even though the model wrote it - and acting on it would refuse them."""
    assert scope.is_off_topic(off_topic_output(**kwargs)) is False


def test_an_intent_that_is_not_a_question_or_chat_overrules_the_label():
    """It cannot be both a category selection and nothing to do with us. The intent is what
    the rest of the turn is routed on, so the intent wins."""
    assert scope.is_off_topic(off_topic_output(intent="category_selection")) is False


def test_the_label_is_needed_at_all():
    assert scope.is_off_topic(turn_output(intent="general_question")) is False


# ------------------------------------------------------------------------------ the reply
def test_an_off_topic_question_is_declined_and_not_answered(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(off_topic_output(
        acknowledgement="I'm sorry, I only help with trailers.",
        answer_to_customer_question="Start with two slices of bread and add filling.",
    ))
    result = run_turn("s1", "how do I make a sandwich?")

    assert "bread" not in result["assistant_text"].lower(), "the answer must not go out"
    assert "trailers" in result["assistant_text"]


def test_the_decline_still_moves_the_conversation_on(fake_llm):
    """A refusal that ends on our full stop has closed the customer down. It ends on the
    question the flow was already owed, so declining costs them nothing."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(off_topic_output())
    result = run_turn("s1", "write me a python script")

    assert result["assistant_text"].rstrip().endswith("?")
    assert "Dump" in result["assistant_text"]


def test_no_search_runs_on_an_off_topic_turn(fake_llm, no_search):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel", "payload_capacity": "6000 lbs"}))
    run_turn("s1", "gravel, 6000 lbs")
    before = len(no_search)

    fake_llm.push(off_topic_output())
    run_turn("s1", "who won the match last night?")

    assert len(no_search) == before, "a qualified session does not re-search to say no"


def test_the_reply_pass_never_runs_on_an_off_topic_turn(fake_llm, no_reply_pass):
    """The one place a decline could leak: a second model call, asked to write freely about
    a question the first one was told to refuse."""
    complete_welcome(fake_llm)
    before = no_reply_pass.calls

    fake_llm.push(off_topic_output())
    run_turn("s1", "what is the capital of France?")

    assert no_reply_pass.calls == before


def test_an_off_topic_first_message_still_gets_the_opening_and_the_contact_ask(fake_llm):
    fake_llm.push(off_topic_output())
    result = run_turn("s1", "how do I make a sandwich?")
    text = result["assistant_text"]

    assert text.startswith("Thank you for contacting TrailerPlace")
    assert "trailers" in text
    assert "name" in text.lower()


def test_a_long_proposed_decline_is_replaced_with_ours(fake_llm):
    """A decline is one line. Anything longer is the model answering anyway, whatever it
    opened with."""
    complete_welcome(fake_llm)
    fake_llm.push(off_topic_output(
        acknowledgement=(
            "I mainly help with trailers, but here you go: take two slices of bread, add "
            "cheese, lettuce and tomato, season it, then close the sandwich and cut it in "
            "half diagonally for the best result."
        )
    ))
    result = run_turn("s1", "how do I make a sandwich?")

    assert "bread" not in result["assistant_text"].lower()
    assert scope.DECLINE in result["assistant_text"]


def test_a_decline_that_asks_its_own_question_is_replaced(fake_llm):
    """The steer below it is the turn's question, and a reply that asks two of its own gets
    neither answered - so a proposed decline with a question mark in it is not one."""
    complete_welcome(fake_llm)
    fake_llm.push(off_topic_output(acknowledgement="I can't help with that - shall we talk trailers?"))
    result = run_turn("s1", "write me a python script")

    assert "shall we talk trailers" not in result["assistant_text"]
    assert scope.DECLINE in result["assistant_text"]


# ---------------------------------------------------------------- light chat still works
def test_a_greeting_is_not_refused(fake_llm):
    """The model should never label this off topic, and if it does the veto does not apply -
    there is no trailer content in "how are you". So the prompt is the only guard, and this
    test pins the behaviour we want from it rather than the code path."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(
        intent="smalltalk_other", acknowledgement="Doing well, thanks for asking!"
    ))
    result = run_turn("s1", "how are you?")

    assert scope.DECLINE not in result["assistant_text"]
