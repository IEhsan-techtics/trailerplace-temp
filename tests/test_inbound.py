"""A push channel's messages: answered once each, in order, and never twice.

None of this is about the web chat, where the browser holds its own request open. It is
about what happens when the platform in front of Luna retries a delivery, splits a
customer across two instances, or hands us three messages before we have answered the
first - and what happens when the Azure container handling one of them simply goes away.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from src import conversation_store, inbound
from src.graph.build import run_turn

from tests.factories import turn_output

PSID = "8675309"
T0 = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def record(external_id: str, body: str, *, seconds: int = 0, session_id: str = PSID,
           channel: str = "messenger") -> bool:
    return inbound.record_inbound_message(
        session_id=session_id, external_id=external_id, body=body, channel=channel,
        sent_at=T0 + timedelta(seconds=seconds),
    )


def bodies(session_id: str = PSID) -> list[str]:
    return [row["body"] for row in inbound.pending_inbound_batch(session_id)]


# ------------------------------------------------------------------- the queue itself
def test_the_same_delivery_twice_is_recorded_once():
    """Meta reuses the message id on a retry, so the second one is not a new message."""
    assert record("mid-1", "got any dump trailers?") is True
    assert record("mid-1", "got any dump trailers?") is False
    assert bodies() == ["got any dump trailers?"]


def test_messages_come_back_in_the_order_they_were_SENT():
    """Not the order they reached us - two instances can hand them over either way round."""
    record("mid-late", "about 20 ft", seconds=30)
    record("mid-early", "dump trailer", seconds=0)

    assert bodies() == ["dump trailer", "about 20 ft"]


def test_one_customers_queue_is_their_own():
    record("mid-1", "mine")
    record("mid-2", "theirs", session_id="other-psid")

    assert bodies() == ["mine"]
    assert bodies("other-psid") == ["theirs"]


def test_answered_messages_leave_the_queue():
    record("mid-1", "dump trailer")
    batch = inbound.pending_inbound_batch(PSID)

    inbound.mark_inbound_answered([row["message_id"] for row in batch], turn_id=None)

    assert bodies() == []
    assert inbound.has_pending_inbound(PSID) is False


def test_a_message_that_could_not_be_answered_still_leaves_the_queue():
    """Left pending, it would sit at the head and block everything they said after it."""
    record("mid-1", "???")
    batch = inbound.pending_inbound_batch(PSID)

    inbound.mark_inbound_answered([row["message_id"] for row in batch], error="boom")

    assert bodies() == []


def test_the_drain_lock_is_held_by_one_holder_at_a_time():
    with inbound.inbound_drain_lock(PSID) as first:
        assert first is True
        with inbound.inbound_drain_lock(PSID) as second:
            assert second is False, "the other instance is draining; we do nothing"
    with inbound.inbound_drain_lock(PSID) as third:
        assert third is True, "released on the way out"


def test_the_lock_is_per_customer():
    with inbound.inbound_drain_lock(PSID):
        with inbound.inbound_drain_lock("other-psid") as other:
            assert other is True


def test_a_newer_message_is_visible_to_the_turn_that_is_already_running():
    record("mid-1", "dump trailer")
    known = [row["message_id"] for row in inbound.pending_inbound_batch(PSID)]
    assert inbound.has_inbound_beyond(PSID, known) is False

    record("mid-2", "actually, make it a tilt", seconds=5)
    assert inbound.has_inbound_beyond(PSID, known) is True


# ------------------------------------------------------------------------- the drain
def test_three_messages_in_a_row_become_ONE_turn():
    """Three replies that each ignore the other two is not what a salesperson would send."""
    seen = []

    def answer(session_id, message, *, turn_id=None, abandon_if=None):
        seen.append(message)
        return {"assistant_text": "Sure - what will you haul?", "listings": []}

    record("mid-1", "hey", seconds=0)
    record("mid-2", "looking for a dump trailer", seconds=1)
    record("mid-3", "around 20 ft", seconds=2)

    results = inbound.drain_inbound(PSID, answer=answer)

    assert seen == ["hey\nlooking for a dump trailer\naround 20 ft"]
    assert len(results) == 1 and bodies() == []


def test_the_turn_runs_against_the_session_that_psid_maps_to():
    sessions = []

    def answer(session_id, message, *, turn_id=None, abandon_if=None):
        sessions.append(session_id)
        return {"assistant_text": "hi", "listings": []}

    record("mid-1", "hello")
    inbound.drain_inbound(PSID, answer=answer)

    assert sessions == [conversation_store.session_uuid_for(PSID)]
    assert sessions[0] != PSID, "the chatbot tables are keyed by a UUID, not by a PSID"


def test_a_turn_they_interrupt_is_thrown_away_and_rerun_with_both_messages():
    """A reply to a question they have already followed up on is worse than no reply."""
    calls = []

    def answer(session_id, message, *, turn_id=None, abandon_if=None):
        calls.append(message)
        if len(calls) == 1:
            record("mid-2", "and it needs a ramp", seconds=5)
        # What run_turn does with the check: consulted once, just before the save.
        if abandon_if is not None and abandon_if():
            return {"abandoned": True}
        return {"assistant_text": f"answering: {message}", "listings": []}

    record("mid-1", "dump trailer")
    results = inbound.drain_inbound(PSID, answer=answer)

    both = "\n".join(["dump trailer", "and it needs a ramp"])
    assert calls == ["dump trailer", both]
    assert len(results) == 1, "the discarded turn produced no reply at all"
    assert results[0]["assistant_text"] == f"answering: {both}"
    assert bodies() == []


def test_someone_who_never_stops_typing_is_still_answered():
    """After MAX_REGENERATIONS the check is switched off, or they would wait forever."""
    calls = []

    def answer(session_id, message, *, turn_id=None, abandon_if=None):
        calls.append(message)
        if abandon_if is not None:
            # Still typing. On the last attempt the check is None, so they stop and the
            # drain can finish - otherwise this test would feed it forever.
            record(f"mid-{len(calls) + 1}", "and another thing", seconds=len(calls))
            if abandon_if():
                return {"abandoned": True}
        return {"assistant_text": "here you go", "listings": []}

    record("mid-1", "dump trailer")
    results = inbound.drain_inbound(PSID, answer=answer)

    assert len(calls) == inbound.MAX_REGENERATIONS + 1, "it gave up discarding and answered"
    assert results[0]["assistant_text"] == "here you go"


def test_a_turn_that_blows_up_does_not_wedge_the_queue():
    calls = []

    def answer(session_id, message, *, turn_id=None, abandon_if=None):
        calls.append(message)
        if message == "bad":
            raise RuntimeError("model down")
        return {"assistant_text": "fine", "listings": []}

    record("mid-1", "bad", seconds=0)
    results = inbound.drain_inbound(PSID, answer=answer)
    record("mid-2", "good", seconds=10)
    results += inbound.drain_inbound(PSID, answer=answer)

    assert calls == ["bad", "good"]
    assert len(results) == 1, "the failed one produced nothing to send"
    assert bodies() == []


def test_nothing_is_answered_while_another_instance_holds_the_lock():
    record("mid-1", "hello")

    with inbound.inbound_drain_lock(PSID):
        results = inbound.drain_inbound(PSID, answer=lambda *a, **k: pytest.fail("ran anyway"))

    assert results == [] and bodies() == ["hello"], "left for the holder to pick up"


def test_an_empty_delivery_is_closed_off_rather_than_blocking_the_queue():
    record("mid-1", "   ", seconds=0)
    record("mid-2", "dump trailer", seconds=1)

    answered = []
    inbound.drain_inbound(PSID, answer=lambda s, m, **k: answered.append(m) or {"assistant_text": "x"})

    assert answered == ["dump trailer"], "the blank one adds nothing and blocks nothing"


def test_two_threads_draining_the_same_customer_answer_it_once():
    started = threading.Barrier(2)
    calls: list[str] = []

    def answer(session_id, message, *, turn_id=None, abandon_if=None):
        calls.append(message)
        return {"assistant_text": "ok", "listings": []}

    def drain():
        started.wait()
        inbound.drain_inbound(PSID, answer=answer)

    record("mid-1", "dump trailer")
    threads = [threading.Thread(target=drain) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == ["dump trailer"], "the loser of the lock did nothing"


# ---------------------------------------------------------------- the turn id it uses
def test_the_same_messages_always_produce_the_same_turn_id():
    """This is what makes a redelivered batch replayable rather than answered twice."""
    batch = [{"external_id": "mid-1"}, {"external_id": "mid-2"}]
    assert inbound.turn_id_for(batch) == inbound.turn_id_for(list(batch))
    assert inbound.turn_id_for(batch) != inbound.turn_id_for([{"external_id": "mid-1"}])


def test_a_redelivered_batch_is_answered_from_the_stored_reply(fake_llm):
    """The container died after the turn committed but before Meta heard back.

    Meta redelivers, the duplicate guard is gone with the process - and the customer must
    still not be charged a second model call, advanced two turns on one message, or have
    the team emailed twice.
    """
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    record("mid-1", "dump trailer")
    first = inbound.drain_inbound(PSID, answer=run_turn)
    assert fake_llm.calls == 1

    # The queue is emptied by a drain, so a redelivery is the same batch arriving again.
    inbound.reset_memory()
    record("mid-1", "dump trailer")
    second = inbound.drain_inbound(PSID, answer=run_turn)

    assert fake_llm.calls == 1, "no second model call"
    assert second[0]["assistant_text"] == first[0]["assistant_text"]
    assert second[0]["replayed"] is True


def test_the_reply_comes_back_ready_to_send_as_messenger_cards():
    """A webhook should have nothing left to decide - see src/domain/cards.py."""
    def answer(session_id, message, *, turn_id=None, abandon_if=None):
        return {
            "assistant_text": "\n\n".join([
                "Here are two that fit:",
                "1. [2026 Diamond C Dump](https://x/2)\n   - Price: $12,500",
                "Do either of these work?",
            ]),
            "listings": [{"title": "2026 Diamond C Dump", "url": "https://x/2",
                          "price_display": "$12,500", "image_url": "https://img/2.jpg"}],
        }

    record("mid-1", "show me some")
    results = inbound.drain_inbound(PSID, answer=answer)

    kinds = [kind for kind, _ in results[0]["sends"]]
    assert "card" in kinds, "the trailer goes as a generic template, not as markdown"
    assert kinds[0] == "text", "the intro is still a plain bubble"


def test_a_channel_with_nothing_to_render_gets_no_sends():
    record("mid-1", "hello", session_id="web-1", channel="web")
    results = inbound.drain_inbound(
        "web-1", channel="web", answer=lambda s, m, **k: {"assistant_text": "hi", "listings": []}
    )
    assert results[0]["sends"] is None


# ----------------------------------------------------- the whole exchange, end to end
class _Transport:
    def __init__(self):
        self.calls = []

    def send_text(self, session_id, text):
        self.calls.append(("text", text))

    def send_card(self, session_id, element):
        self.calls.append(("card", element))

    def send_action(self, session_id, action):
        self.calls.append(("action", action))


def test_a_transport_gets_typing_the_search_line_and_then_the_paced_answer():
    """Everything the customer sees, in the order they see it - with nothing left for a
    webhook to decide but how to call the Send API."""
    from src import turn_status

    transport = _Transport()

    def answer(session_id, message, *, turn_id=None, abandon_if=None):
        # What the search node does the moment it starts looking.
        turn_status.publish(session_id, "One moment while I check what we have in stock.")
        # Longer than the keep-alive's poll interval, so this stands in for a real turn -
        # the point being that the line goes out BEFORE the turn returns.
        time.sleep(0.8)
        return {
            "assistant_text": "\n\n".join([
                "Here is one that fits:",
                "1. [2026 Diamond C Dump](https://x/2)\n   - Price: $12,500",
                "Does that work?",
            ]),
            "listings": [{"title": "2026 Diamond C Dump", "url": "https://x/2",
                          "price_display": "$12,500"}],
        }

    record("mid-1", "show me a dump trailer")
    try:
        inbound.drain_inbound(PSID, answer=answer, transport=transport)
    finally:
        turn_status.clear(conversation_store.session_uuid_for(PSID))

    kinds = [kind for kind, _ in transport.calls]
    texts = [payload for kind, payload in transport.calls if kind == "text"]

    assert kinds[:2] == ["action", "action"], "seen, then typing, before anything else"
    assert "One moment while I check what we have in stock." in texts, (
        "the search line reached them WHILE we were looking, not with the results"
    )
    assert texts.index("One moment while I check what we have in stock.") < texts.index(
        "Here is one that fits:"
    ), "and it came first"
    assert "card" in kinds, "the trailer went as a card"
    # Typing stops the moment the answer is ready, and the bubbles follow it - so the last
    # thing they see is the closing question, not an indicator still running.
    assert transport.calls[-1][1] == "Does that work?"
    assert "typing_off" in [p for k, p in transport.calls if k == "action"]


def test_a_discarded_reply_never_reaches_them_and_leaves_no_trace():
    """It is thrown away before the save, so there is nothing to send and nothing stored."""
    transport = _Transport()

    def answer(session_id, message, *, turn_id=None, abandon_if=None):
        if message == "dump trailer":
            record("mid-2", "actually a tilt", seconds=5)
        if abandon_if is not None and abandon_if():
            return {"abandoned": True}
        return {"assistant_text": f"answering: {message}", "listings": []}

    record("mid-1", "dump trailer")
    results = inbound.drain_inbound(PSID, answer=answer, transport=transport)

    texts = [payload for kind, payload in transport.calls if kind == "text"]
    both = "\n".join(["dump trailer", "actually a tilt"])
    assert "answering: dump trailer" not in texts, "they never saw the superseded reply"
    assert f"answering: {both}" in texts
    assert len(results) == 1 and results[0]["delivered"]
