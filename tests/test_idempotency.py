"""A message that reaches us twice is answered once.

The turn row is the receipt. It is written in the same transaction as the state snapshot,
so a row means that turn finished: the state was saved and its emails were queued. A
caller that can name the turn - a platform message id, or a hash of one - therefore gets
the stored reply back instead of a second turn, which is what stops a retry costing a
second model call, a second email to the team, and a conversation that has moved on twice
on one thing the customer said.
"""
from __future__ import annotations

import pytest

from src import conversation_store, db
from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output

TURN = "3f1b0c8e-0000-4000-8000-000000000001"
OTHER = "3f1b0c8e-0000-4000-8000-000000000002"


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


# ------------------------------------------------------------------------- replaying
def test_the_same_turn_id_twice_runs_the_turn_once(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    first = run_turn("s1", "dump trailer", turn_id=TURN)
    second = run_turn("s1", "dump trailer", turn_id=TURN)

    assert fake_llm.calls == 1, "the second call paid for no model call at all"
    assert second["assistant_text"] == first["assistant_text"]
    assert second["replayed"] is True
    assert "replayed" not in first


def test_a_replay_does_not_advance_the_conversation(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer", turn_id=TURN)
    before = state_after()

    run_turn("s1", "dump trailer", turn_id=TURN)
    after = state_after()

    assert after["turn_index"] == before["turn_index"], "one message, one turn"
    assert after["asked_counts"] == before["asked_counts"], "the question was not re-asked"
    assert len(after["messages"]) == len(before["messages"])


def test_a_replay_sends_the_team_no_second_email(fake_llm, monkeypatch):
    sent = []
    from src.tools import email_sender

    monkeypatch.setattr(email_sender, "send_email", lambda subject, body: sent.append(subject) or True)

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(intent="team_request_escalation", name="Dana", phone="979-555-0100"))
    run_turn("s1", "I need to speak to someone", turn_id=TURN)
    after_first = len(sent)

    fake_llm.push(turn_output(intent="team_request_escalation", name="Dana", phone="979-555-0100"))
    run_turn("s1", "I need to speak to someone", turn_id=TURN)

    assert after_first >= 1, "the first one did go out"
    assert len(sent) == after_first, "the retry sent nothing"


def test_a_different_turn_id_is_a_different_turn(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer", turn_id=TURN)
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel", turn_id=OTHER)

    assert fake_llm.calls == 3
    assert state_after()["slots"]["haul_item"] == "gravel"


def test_without_a_turn_id_every_call_is_a_new_turn(fake_llm):
    """Which is what a browser holding its own request open actually wants."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    assert fake_llm.calls == 3, "nothing to recognise it by, so it ran again"


def test_one_customers_turn_id_cannot_replay_anothers(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer", turn_id=TURN)

    fake_llm.push(turn_output(category_mentioned="tilt", intent="category_selection"))
    run_turn("s2", "tilt trailer", turn_id=TURN)

    assert state_after("s2")["category"] == "Tilt", "a receipt belongs to one session"


def test_every_turn_reports_the_id_it_was_stored_under(fake_llm):
    fake_llm.push(turn_output())
    assert run_turn("s1", "hi")["turn_id"], "so a turn can always be found in the logs"


def test_a_receipt_lookup_that_fails_does_not_lose_the_message(monkeypatch, fake_llm):
    """A dedupe check that throws would cost a customer their message - far worse than the
    duplicate it was guarding against."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(conversation_store, "persistence_enabled", boom)

    assert conversation_store.stored_turn_response("s1", TURN) is None


# ------------------------------------------------------------------- the schema check
def test_the_schema_is_checked_once_per_process_not_once_per_turn(monkeypatch, fake_llm):
    """It used to run every turn: about 2.7 s of an 11 s answer, reflecting Azure Postgres
    for a result that cannot change while the process is alive."""
    calls = []
    monkeypatch.setattr(db.Base.metadata, "create_all", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(db, "get_engine", lambda: None)
    db.reset_schema_cache()

    db.ensure_schema()
    db.ensure_schema()
    db.ensure_schema()

    assert len(calls) == 1
    db.ensure_schema(force=True)
    assert len(calls) == 2, "force re-checks, for a test pointing the engine somewhere new"

    db.reset_schema_cache()


@pytest.fixture(autouse=True)
def _forget_schema_cache():
    """The flag is process-wide; a test that sets it must not leak into the next one."""
    yield
    db.reset_schema_cache()
