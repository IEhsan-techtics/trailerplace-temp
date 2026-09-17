"""Saying the same thing twice in one reply.

Every case here was produced by the live bot, not invented. The acknowledgement field is
meant to react to what the customer said; once the prompt insisted a reply must respond
before it asks for anything, the model started writing one even when the answer beside it
already did the job, and both went out.
"""
from __future__ import annotations

import pytest

from src.graph.nodes.compose import _restates


@pytest.mark.parametrize(
    "candidate, established",
    [
        # Live, turn 3: "do you offer financing?"
        (
            "Yes, we do offer financing.",
            "We offer financing. Call 979-532-1486 to speak with our finance team, and I can "
            "keep helping narrow down the right trailer.",
        ),
        # Live, turn 5: "what are your opening hours?"
        (
            "We're open from 8:00 AM to 6:00 PM.",
            "We're located in Wharton, TX and open 8:00 AM to 6:00 PM. Call 979-532-1486 or "
            "visit https://trailerplace.com.",
        ),
        # Live, turn 4: the welcome, where "info" and "information" are the same word.
        (
            "Thanks, Ibrahim - I've got your contact information.",
            "Great to have your contact info, Ibrahim!",
        ),
    ],
)
def test_a_restatement_is_recognised(candidate, established):
    assert _restates(candidate, established)


@pytest.mark.parametrize(
    "candidate, established",
    [
        # Shares the topic and nothing else. Dropping this would lose the apology.
        (
            "I'm sorry your trailer arrived damaged.",
            "Our team will reach out about the trailer as soon as I know how to contact you.",
        ),
        # A real second point: the answer says nothing about delivery.
        (
            "We can deliver to most of Texas.",
            "We offer financing. Call 979-532-1486 to speak with our finance team.",
        ),
        # Their name is not a restatement of anything.
        ("Thanks, Ibrahim!", "We're open 8:00 AM to 6:00 PM."),
    ],
)
def test_a_second_point_is_kept(candidate, established):
    assert not _restates(candidate, established)


def test_an_empty_line_restates_nothing():
    assert not _restates("", "We offer financing.")
    assert not _restates("We offer financing.", "")
    assert not _restates("Thanks!", "Sure, okay.")


# ------------------------------------------------------------ the handoff confirmation
# The pattern that spots "the agent already said it" was written with a backspace character
# where its word boundary belonged, so it matched nothing and the confirmation was appended to
# every reply that already carried one.
@pytest.mark.parametrize(
    "reply",
    [
        "Thanks, Dave - I've passed it on to our team.",
        "I've passed your request to the team.",
        "Our team will reach out shortly.",
        "I've notified our team about the damaged trailer.",
        "I've logged your interest in that trailer.",
    ],
)
def test_a_reply_that_already_confirms_the_handoff_is_left_alone(reply):
    from src.graph.nodes.compose import _with_handoff

    assert _with_handoff({"emails_flushed": 1}, reply) == reply


def test_a_reply_that_does_not_mention_it_gets_the_confirmation():
    from src.graph.nodes.compose import _with_handoff

    text = _with_handoff({"emails_flushed": 2}, "What type of trailer are you after?")
    assert "I've passed your requests on to our team" in text


def test_nothing_is_confirmed_when_nothing_went_out():
    from src.graph.nodes.compose import _with_handoff

    assert _with_handoff({}, "Hello.") == "Hello."
