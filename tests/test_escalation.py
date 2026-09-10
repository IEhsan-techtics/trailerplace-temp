"""Escalation: the escalate tool, and the turns it owns.

The bug this exists to prevent, seen end to end before the fix:

    "I'm sorry your order arrived damaged. What material will you be hauling?"

Someone who has just told us something went wrong is not being sold to on that turn.
"""
from __future__ import annotations

import pytest
from factories import complete_welcome, turn_output

from src.domain import canned_responses, company
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


def test_it_asks_for_both_when_we_have_neither():
    runner = _runner(name=None, email=None, phone=None)
    result = _escalate(runner)
    assert "your name and an email or phone number" in result
    assert "follow up" in result


def test_it_asks_only_for_the_name_when_that_is_all_that_is_missing():
    """Asking for a number they already gave reads as not having listened."""
    result = _escalate(_runner(name=None, email="d@x.ai"))
    assert "Could I take your name" in result
    assert "email or phone number so our team" not in result


def test_it_asks_only_for_a_number_when_that_is_all_that_is_missing():
    result = _escalate(_runner(name="Dave", email=None, phone=None))
    assert "Could I take an email or phone number" in result


def test_it_does_not_chase_someone_who_declined():
    result = _escalate(_runner(name=None, email=None, phone=None, declined=True))
    assert "Could I take" not in result
    assert "declined" in result


def test_it_does_not_ask_again_when_we_already_have_them():
    assert "Could I take" not in _escalate(_runner())


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


def test_an_escalation_on_the_very_first_turn_is_not_swallowed_by_the_contact_gate(
    fake_llm, no_reply_pass
):
    """The gate used to own turn one outright, and the request was silently dropped.

    A live run: "I have a complaint about my last order, the trailer arrived damaged" was
    answered with "Before we go on - could you please provide your name..." and nothing else.
    No apology, no phone number, and no email to the team. Nothing carries an unhandled
    intent forward, so when the conversation moved on the complaint was gone for good.

    The escalation now owns the turn. The contact details are still asked for - the canned
    answer carries the request - so the gate gives up nothing by losing.
    """
    from src.graph.build import run_turn

    fake_llm.push(turn_output(intent="team_request_escalation"))
    text = run_turn("s1", "have someone call me")["assistant_text"]

    assert no_reply_pass.calls == 1, "the escalation must reach the agent, gate or no gate"
    assert company.PHONE in text, "someone we cannot help ourselves gets the number"
    assert "name" in text.lower(), "and is still asked for the details, in the same breath"


def test_an_escalation_before_we_have_contact_details_is_stashed_not_dropped(fake_llm):
    """The request survives the turn it could not be sent on."""
    from src.graph.build import run_turn
    from src.graph.state import from_snapshot
    from src import conversation_store

    fake_llm.push(turn_output(intent="team_request_escalation"))
    run_turn("s1", "I have a complaint about my last order")

    snapshot, _conversation, _lead = conversation_store.load_session("s1")
    state = from_snapshot("s1", snapshot)
    assert state["pending_email_actions"], "the complaint must be waiting, not gone"
    assert state["contact_followup_pending"] == "name, contact"


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


# ------------------------------------------------------- the contact gate on sending
# Nothing goes out until we hold a name AND an email or phone. Short of that the request is
# stashed, never dropped, and every stashed request flushes on the turn they become reachable.
def _state(**contact):
    return {
        "session_id": "s1",
        "contact": {"name": None, "email": None, "phone": None, "declined": False, **contact},
        "turn_outcome": {},
    }


def test_nothing_is_sent_without_a_name():
    from src.tools import team_notify

    state = _state(email="d@x.ai")
    assert team_notify.record(state, reason="Escalation", description="x") == "stashed"
    assert state["turn_outcome"].get("outbox_events") is None
    assert len(state["pending_email_actions"]) == 1


def test_nothing_is_sent_without_a_way_to_reach_them():
    from src.tools import team_notify

    state = _state(name="Dave")
    assert team_notify.record(state, reason="Escalation", description="x") == "stashed"
    assert len(state["pending_email_actions"]) == 1


def test_it_sends_once_both_pieces_are_present():
    from src.tools import team_notify

    state = _state(name="Dave", phone="979-555-0100")
    assert team_notify.record(state, reason="Escalation", description="x") == "sent"
    assert len(state["turn_outcome"]["outbox_events"]) == 1
    assert not state.get("pending_email_actions")


def test_several_stashed_requests_all_go_out_together():
    """Five requests made across five turns become five emails on the turn we can reach them."""
    from src.tools import team_notify

    state = _state()
    for reason, description in [
        ("FAQ - financing", "asked about financing"),
        ("Team Request", "wants a callback"),
        ("Escalation", "order arrived damaged"),
        ("Listing Interest", "likes the Iron Bull DTB"),
        ("FAQ - trade_in", "asked about trade-ins"),
    ]:
        team_notify.record(state, reason=reason, description=description)
    assert len(state["pending_email_actions"]) == 5

    state["contact"].update({"name": "Dave", "email": "d@x.ai"})
    assert team_notify.flush(state) == 5
    assert len(state["turn_outcome"]["outbox_events"]) == 5
    assert state["pending_email_actions"] == []
    assert state["contact_followup_pending"] is None


def test_a_flushed_email_carries_the_contact_details_we_finally_got():
    """Rendered at flush time - the details we were waiting for are the ones in the body."""
    from src.tools import team_notify

    state = _state()
    team_notify.record(state, reason="Escalation", description="order arrived damaged")
    state["contact"].update({"name": "Dave", "email": "d@x.ai"})
    team_notify.flush(state)

    body = state["turn_outcome"]["outbox_events"][0]["payload"]["body"]
    assert "Full Name: Dave" in body
    assert "Email: d@x.ai" in body
    assert "Not provided" in body  # the phone, which they never gave


def test_half_the_details_is_still_not_enough_to_flush():
    from src.tools import team_notify

    state = _state()
    team_notify.record(state, reason="Escalation", description="x")
    state["contact"]["name"] = "Dave"
    assert team_notify.flush(state) == 0
    assert len(state["pending_email_actions"]) == 1
    assert state["contact_followup_pending"] == "contact"


def test_a_refusal_drops_the_stash_and_stops_the_asking():
    from src.tools import team_notify

    state = _state()
    team_notify.record(state, reason="Escalation", description="x")
    state["contact"]["declined"] = True
    assert team_notify.flush(state) == 0
    assert state["pending_email_actions"] == []
    assert team_notify.record(state, reason="Team Request", description="y") == "dropped"


def test_the_same_request_re_raised_is_not_emailed_twice():
    """The agent re-words a request every time it raises it, so keying on wording double-sent."""
    from src.tools import team_notify

    state = _state()
    team_notify.record(state, reason="Team Request", description="wants a callback about a dump trailer")
    team_notify.record(state, reason="Team Request", description="Wants a callback about a dump trailer!")
    assert len(state["pending_email_actions"]) == 1


def test_two_genuinely_different_requests_both_survive():
    from src.tools import team_notify

    state = _state()
    team_notify.record(state, reason="Team Request", description="wants a callback")
    team_notify.record(state, reason="Team Request", description="asked us to beat $8,000 on a dump trailer")
    assert len(state["pending_email_actions"]) == 2


# ------------------------------------------------------------------- FAQs notify too
def test_every_faq_notifies_the_team(fake_llm, no_reply_pass):
    from src.graph.build import run_turn
    from src.graph.state import from_snapshot
    from src import conversation_store

    complete_welcome(fake_llm)  # gives us name + phone
    fake_llm.push(turn_output(
        intent="faq",
        faq_key="financing",
        user_question_to_answer="do you offer financing?",
        answer_to_customer_question="We offer financing. Call 979-532-1486.",
    ))
    run_turn("s1", "do you offer financing?")

    snapshot, _c, _l = conversation_store.load_session("s1")
    state = from_snapshot("s1", snapshot)
    # Contact was complete, so it went straight out rather than into the stash.
    assert state.get("pending_email_actions") in (None, [])


def test_an_faq_before_we_have_contact_details_is_stashed(fake_llm, no_reply_pass):
    from src.graph.nodes.apply import apply_node
    from src.graph.state import new_state

    state = new_state("s1")
    apply_node(state, turn_output(intent="faq", faq_key="financing"), "do you offer financing?")

    assert len(state["pending_email_actions"]) == 1
    assert state["pending_email_actions"][0]["reason"] == "FAQ - financing"


def test_an_faq_still_costs_no_second_model_call(fake_llm, no_reply_pass):
    """Notifying the team must not cost the customer their place in the flow."""
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="Dump"))
    run_turn("s1", "I need a dump trailer")
    fake_llm.push(turn_output(intent="faq", faq_key="trade_in",
                              answer_to_customer_question="Our sales team handles trade-in appraisals."))
    result = run_turn("s1", "do you take trade-ins?")

    assert no_reply_pass.calls == 0
    assert "trade-in" in result["assistant_text"].lower()
