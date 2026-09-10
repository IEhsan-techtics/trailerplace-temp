"""Escalation: the escalate tool, and the turns it owns.

The bug this exists to prevent, seen end to end before the fix:

    "I'm sorry your order arrived damaged. What material will you be hauling?"

Someone who has just told us something went wrong is not being sold to on that turn.
"""
from __future__ import annotations

import pytest
from factories import complete_welcome, turn_output

from src.domain import canned_responses
from src.llm.tools import ToolRunner


@pytest.fixture(autouse=True)
def no_mail(monkeypatch):
    """Nothing leaves the machine; record what would have."""
    from src.tools import email_sender

    sent = []
    monkeypatch.setattr(email_sender, "send_email", lambda subject, body: sent.append((subject, body)) or True)
    return sent


def _runner(**contact):
    state = {
        "session_id": "s1",
        "category": "Dump",
        "contact": {"name": "Dave", "email": "d@x.ai", "phone": None, "declined": False, **contact},
        "turn_outcome": {},
    }
    return ToolRunner(state, turn_output())


def _escalate(runner, reason="complaint", summary="order arrived damaged"):
    import json

    return runner.call("escalate", json.dumps({"reason": reason, "summary": summary}))


# ------------------------------------------------------------------ the email it builds
def test_escalating_queues_an_email_in_the_spec_format():
    runner = _runner()
    _escalate(runner)
    events = runner.state["turn_outcome"]["outbox_events"]

    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["subject"] == "TrailerPlace Lead — Escalation — Dave"
    assert payload["body"] == (
        "Full Name: Dave\n"
        "Email: d@x.ai\n"
        "Phone Number: Not provided\n"
        "\n"
        "[Escalation] order arrived damaged"
    )


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("complaint", "Escalation"),
        ("callback", "Team Request"),
        ("meeting", "Team Request"),
        ("quote", "Team Request"),
        ("pricing", "Team Request"),
        ("delivery", "Team Request"),
        ("paperwork", "Team Request"),
        ("viewing", "Team Request"),
        ("stock_question", "Team Request"),
        ("unstocked_type", "Team Request"),
        ("listing_interest", "Listing Interest"),
        ("other", "Team Request"),
        ("something the model invented", "Team Request"),
    ],
)
def test_every_reason_maps_to_the_fixed_reason_vocabulary(reason, expected):
    """company_and_email_scenarios.md fixes these strings; the team filters on them."""
    runner = _runner()
    _escalate(runner, reason=reason)
    assert runner.state["turn_outcome"]["outbox_events"][0]["reason"] == expected


def test_a_missing_summary_still_produces_a_usable_email():
    runner = _runner()
    _escalate(runner, summary="")
    assert "No detail given." in runner.state["turn_outcome"]["outbox_events"][0]["payload"]["body"]


# ------------------------------------------------------- what it tells the agent to say
def test_a_complaint_owns_the_turn():
    runner = _runner()
    _escalate(runner, reason="complaint")
    assert runner.state["turn_outcome"]["escalation_owns_turn"] is True


def test_a_team_request_does_not_own_the_turn():
    runner = _runner()
    _escalate(runner, reason="callback")
    assert "escalation_owns_turn" not in runner.state["turn_outcome"]


def test_the_reply_instruction_carries_the_phone_number():
    result = _escalate(_runner())
    assert "979-532-1486" in result
    assert "Ask no qualification question" in result


def test_it_asks_for_contact_details_when_we_have_none():
    runner = _runner(name=None, email=None, phone=None)
    result = _escalate(runner)
    assert canned_responses.CONTACT_FOLLOWUP in result


def test_it_does_not_chase_someone_who_declined():
    runner = _runner(name=None, email=None, phone=None, declined=True)
    assert canned_responses.CONTACT_FOLLOWUP not in _escalate(runner)


def test_it_does_not_ask_again_when_we_already_have_them():
    assert canned_responses.CONTACT_FOLLOWUP not in _escalate(_runner())


# ------------------------------------------------------------------------- routing
def test_a_complaint_reaches_the_reply_pass(fake_llm, no_reply_pass):
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="Dump"))
    run_turn("s1", "I need a dump trailer")

    fake_llm.push(turn_output(intent="team_request_escalation"))
    run_turn("s1", "my last order arrived damaged")

    assert no_reply_pass.calls == 1, "escalation needs the agent and its escalate tool"


def test_a_complaint_is_never_answered_with_a_qualification_question(fake_llm, no_reply_pass):
    """Even when the reply pass fails, the fallback must not sell to them."""
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="Dump"))
    run_turn("s1", "I need a dump trailer")

    fake_llm.push(turn_output(intent="team_request_escalation"))
    # no_reply_pass.queue is empty -> returns None, exactly as a failed call does.
    text = run_turn("s1", "my last order arrived damaged")["assistant_text"]

    assert "979-532-1486" in text
    assert "hauling" not in text.lower()
    assert "?" not in text.split("979-532-1486")[0], "no question before the number"


def test_an_faq_does_not_reach_the_reply_pass(fake_llm, no_reply_pass):
    """The five FAQs are ours to answer, and the qualification flow survives them."""
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="Dump"))
    run_turn("s1", "I need a dump trailer")

    fake_llm.push(turn_output(
        intent="faq",
        answer_to_customer_question="We offer financing. Call 979-532-1486.",
    ))
    result = run_turn("s1", "do you offer financing?")

    assert no_reply_pass.calls == 0, "an FAQ costs no second model call"
    assert "financing" in result["assistant_text"].lower()


def test_the_contact_gate_still_owns_the_first_turn(fake_llm, no_reply_pass):
    """A first message we cannot act on still gets the welcome, not an escalation reply."""
    from src.graph.build import run_turn

    fake_llm.push(turn_output(intent="team_request_escalation"))
    text = run_turn("s1", "have someone call me")["assistant_text"]

    assert no_reply_pass.calls == 0
    assert "Thank you for contacting TrailerPlace" in text
