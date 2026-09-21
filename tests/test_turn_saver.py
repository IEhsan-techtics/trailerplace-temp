"""Committing a finished turn off the reply path, without losing one.

The 2.4 s commit happens after the reply is written, so the customer waits on a result they
never see. Moving it is worth the whole 2.4 s and is the most dangerous thing in the app to
move, because that transaction is the turn's receipt, its outbox and its state at once.

What has to hold: strict order, never dropped, still idempotent while in flight, drained on
the way out, and - when it fails anyway - a log that says exactly what the customer was
promised and did not get.
"""
from __future__ import annotations

import threading
import time

import pytest

from src import conversation_store, turn_saver


@pytest.fixture(autouse=True)
def fresh_saver():
    turn_saver.reset()
    yield
    turn_saver.reset()


# ------------------------------------------------------------------------------- ordering
def test_turns_commit_in_the_order_they_were_produced():
    """Two messages in quick succession must not land out of order and leave turn 1's state
    on top of turn 2's."""
    done: list[int] = []
    for index in range(12):
        turn_saver.submit(
            lambda i=index: done.append(i), session_id="s1", turn_id=f"t{index}"
        )
    assert turn_saver.drain(timeout=10)
    assert done == list(range(12))


# ----------------------------------------------------------------------------- never lost
def test_a_full_queue_saves_inline_rather_than_dropping():
    """A backlog is a database problem. The right answer is to wait for it, not to lose a
    lead - so a full queue makes the bot slow again, never lossy."""
    release = threading.Event()
    done: list[str] = []

    turn_saver.submit(lambda: release.wait(5), session_id="s1", turn_id="blocker")
    for index in range(turn_saver.MAX_PENDING):
        turn_saver.submit(lambda: None, session_id="s1", turn_id=f"queued{index}")

    queued = turn_saver.submit(
        lambda: done.append("inline"), session_id="s1", turn_id="overflow"
    )
    release.set()

    assert queued is False, "the queue was full"
    assert done == ["inline"], "it ran here instead of being dropped"
    turn_saver.drain(timeout=10)


def test_a_failure_is_logged_with_what_the_customer_was_promised(caplog):
    """At that point they have been told their request reached the team, and it has not."""
    def _boom():
        raise RuntimeError("Azure is having a day")

    with caplog.at_level("ERROR"):
        turn_saver.submit(
            _boom,
            session_id="s1",
            turn_id="t1",
            outbox_events=[{"reason": "Listing Interest"}, {"reason": "Team Request"}],
        )
        assert turn_saver.drain(timeout=10)

    logged = caplog.text
    assert "BACKGROUND SAVE FAILED" in logged
    assert "session=s1" in logged and "turn=t1" in logged
    assert "2 notification(s) LOST" in logged
    assert "Listing Interest" in logged and "Team Request" in logged


def test_one_failure_does_not_stop_the_next_turn_saving():
    done: list[str] = []
    turn_saver.submit(lambda: (_ for _ in ()).throw(RuntimeError("no")),
                      session_id="s1", turn_id="t1")
    turn_saver.submit(lambda: done.append("t2"), session_id="s1", turn_id="t2")

    assert turn_saver.drain(timeout=10)
    assert done == ["t2"]


# --------------------------------------------------------------------------- idempotency
def test_a_redelivery_inside_the_save_window_is_answered_not_re_run():
    """The turn row is the receipt, and while the commit is in flight there is no row.
    Without this, a Messenger retry would run the whole turn a second time - two model
    calls, the conversation advanced twice, the team emailed twice."""
    conversation_store.remember_inflight("s1", "t1", {"assistant_text": "here you go"})

    assert conversation_store.stored_turn_response("s1", "t1") == {"assistant_text": "here you go"}
    assert conversation_store.stored_turn_response("s1", "t2") is None, "a different turn"

    conversation_store.clear_inflight("s1", "t1")
    assert conversation_store.stored_turn_response("s1", "t1") is None, "the row answers now"


def test_the_commit_clears_it():
    conversation_store.remember_inflight("s2", "t9", {"assistant_text": "hi"})
    turn_saver.submit(lambda: None, session_id="s2", turn_id="t9")
    assert turn_saver.drain(timeout=10)

    assert conversation_store.stored_turn_response("s2", "t9") is None


def test_a_failed_commit_keeps_it(caplog):
    """The row never arrived, so the reply in memory is still the only receipt there is."""
    conversation_store.remember_inflight("s3", "t9", {"assistant_text": "hi"})
    with caplog.at_level("ERROR"):
        turn_saver.submit(lambda: (_ for _ in ()).throw(RuntimeError("no")),
                          session_id="s3", turn_id="t9")
        assert turn_saver.drain(timeout=10)

    assert conversation_store.stored_turn_response("s3", "t9") == {"assistant_text": "hi"}
    conversation_store.clear_inflight("s3", "t9")


# ------------------------------------------------------------------------------- shutdown
def test_the_drain_waits_for_a_slow_commit():
    done: list[str] = []
    turn_saver.submit(lambda: (time.sleep(0.4), done.append("saved")),
                      session_id="s1", turn_id="t1")

    assert turn_saver.drain(timeout=10)
    assert done == ["saved"]


def test_anything_left_behind_is_logged_as_lost(caplog):
    release = threading.Event()
    turn_saver.submit(lambda: release.wait(5), session_id="s1", turn_id="slow")
    turn_saver.submit(lambda: None, session_id="s1", turn_id="stuck-behind-it")

    with caplog.at_level("ERROR"):
        drained = turn_saver.drain(timeout=0.2)
    release.set()

    assert drained is False
    assert "BACKGROUND SAVE ABANDONED" in caplog.text
    turn_saver.drain(timeout=10)


# ------------------------------------------------------------------- the in-memory store
def test_the_in_memory_store_still_saves_inline(fake_llm, no_search):
    """It is instant, and the tests read it straight back - there is nothing to gain and a
    race to lose."""
    from src.graph.build import run_turn

    from tests.factories import turn_output

    fake_llm.push(turn_output(intent="general_question"))
    run_turn("s-inline", "hi")

    snapshot, conversation, _lead = conversation_store.load_session("s-inline")
    assert snapshot is not None, "already on disk by the time run_turn returned"
    assert len(conversation) == 2
    assert turn_saver.pending() == 0
