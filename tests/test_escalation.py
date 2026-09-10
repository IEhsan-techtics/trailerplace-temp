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


# ------------------------------------------------------------------------- the outbox
# Queued inside save_turn's transaction, drained after the commit. Tests run with
# persistence off, so the drain is exercised against a stand-in session rather than Postgres.
class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.committed = False

    def query(self, *a, **k):
        return _FakeQuery(self._rows)

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Row:
    def __init__(self, subject="s", body="b"):
        self.event_id = "e1"
        self.event_type = "escalation"
        self.session_id = "s1"
        self.payload = {"subject": subject, "body": body}
        self.status = "pending"
        self.attempt_count = 0
        self.claimed_at = None
        self.last_error = None


@pytest.fixture
def draining(monkeypatch):
    """Persistence on, with a stand-in database session."""
    from src import conversation_store, db

    def _install(rows):
        monkeypatch.setattr(conversation_store, "persistence_enabled", lambda: True)
        monkeypatch.setattr(db, "get_session_factory", lambda: (lambda: _FakeSession(rows)))
        return rows

    return _install


def test_a_queued_row_is_sent_and_marked(draining, no_mail):
    from src import conversation_store

    rows = draining([_Row("TrailerPlace Lead — Escalation — Dave", "body here")])
    conversation_store.deliver_pending_outbox()

    assert len(no_mail) == 1
    assert no_mail[0][0] == "TrailerPlace Lead — Escalation — Dave"
    assert rows[0].status == "sent"
    assert rows[0].attempt_count == 1


def test_a_failed_send_stays_retryable(draining, monkeypatch):
    from src import conversation_store
    from src.tools import email_sender

    monkeypatch.setattr(email_sender, "send_email", lambda s, b: False)
    rows = draining([_Row()])
    conversation_store.deliver_pending_outbox()

    assert rows[0].status == "pending", "still pending, so the next drain retries it"
    assert rows[0].attempt_count == 1
    assert rows[0].last_error


def test_a_row_gives_up_after_the_attempt_cap(draining, monkeypatch):
    from src import conversation_store
    from src.tools import email_sender

    monkeypatch.setattr(email_sender, "send_email", lambda s, b: False)
    row = _Row()
    row.attempt_count = conversation_store.MAX_OUTBOX_ATTEMPTS - 1
    draining([row])
    conversation_store.deliver_pending_outbox()

    assert row.status == "failed", "visible in the table rather than retried forever"


def test_the_drain_never_raises(draining, monkeypatch):
    """A broken mailbox must not take the bot down."""
    from src import conversation_store
    from src.tools import email_sender

    def _boom(subject, body):
        raise RuntimeError("smtp exploded")

    monkeypatch.setattr(email_sender, "send_email", _boom)
    draining([_Row()])
    conversation_store.deliver_pending_outbox()  # must not raise


def test_the_drain_is_a_no_op_without_persistence(no_mail):
    from src import conversation_store

    conversation_store.deliver_pending_outbox()
    assert no_mail == []


def test_save_turn_accepts_outbox_events_on_the_memory_path():
    """The in-memory fallback has no outbox; passing events must not blow up."""
    from src import conversation_store

    conversation_store.ensure_session("s-mem")
    conversation_store.save_turn(
        "s-mem",
        conversation=[],
        state_snapshot={},
        request_message="hi",
        response={},
        outbox_events=[{"event_key": "k", "event_type": "escalation", "payload": {}}],
    )
