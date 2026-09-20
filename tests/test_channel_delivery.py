"""The ten seconds between their message and the answer, and the pace the answer arrives at.

A turn is long. On a push channel that silence looks like a broken bot unless something
fills it, and an answer that lands as six notifications at once reads like a machine. None
of it may ever delay or fail the turn itself.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src import channel_delivery, config, turn_status
from src.channel_delivery import TurnKeepAlive, deliver

SESSION = "s-keepalive"


class FakeTransport:
    """Records what a channel was asked to put in front of the customer, in order."""

    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple[str, object]] = []
        self.fail_on = fail_on

    def send_text(self, session_id, text):
        if self.fail_on == "text":
            raise RuntimeError("send failed")
        self.calls.append(("text", text))

    def send_card(self, session_id, element):
        if self.fail_on == "card":
            raise RuntimeError("send failed")
        self.calls.append(("card", element))

    def send_action(self, session_id, action):
        if self.fail_on == "action":
            raise RuntimeError("send failed")
        self.calls.append(("action", action))

    def kinds(self):
        return [kind for kind, _ in self.calls]

    def texts(self):
        return [payload for kind, payload in self.calls if kind == "text"]

    def actions(self):
        return [payload for kind, payload in self.calls if kind == "action"]


@pytest.fixture(autouse=True)
def clean_status():
    turn_status.clear(SESSION)
    yield
    turn_status.clear(SESSION)


@pytest.fixture
def fast(monkeypatch):
    """Refresh typing every tick, so a test does not wait ten seconds to see one."""
    monkeypatch.setattr(
        config, "settings",
        replace(config.settings, messenger_typing_refresh_seconds=0.05),
    )
    monkeypatch.setattr(channel_delivery, "settings", config.settings)


# --------------------------------------------------------------- what fills the silence
def test_they_are_shown_typing_before_the_turn_even_starts():
    transport = FakeTransport()

    with TurnKeepAlive(SESSION, transport):
        pass

    assert transport.actions()[:2] == ["mark_seen", "typing_on"]
    assert transport.actions()[-1] == "typing_off", "and it is cleared afterwards"


def test_typing_is_re_sent_so_it_never_lapses(fast):
    """Messenger drops the indicator after about 20 s and a turn takes longer than that."""
    transport = FakeTransport()

    alive = TurnKeepAlive(SESSION, transport, poll_seconds=0.02)
    alive.start()
    _wait_until(lambda: transport.actions().count("typing_on") >= 3)
    alive.stop()

    assert transport.actions().count("typing_on") >= 3


def test_the_search_line_reaches_them_while_we_are_still_looking(fast):
    """Sent WITH the results it would be worthless - they can already see the trailers."""
    transport = FakeTransport()

    alive = TurnKeepAlive(SESSION, transport, poll_seconds=0.02)
    alive.start()
    turn_status.publish(SESSION, "Give me a moment - I'll check what matches.")
    _wait_until(lambda: alive.status_sent is not None)
    alive.stop()

    assert transport.texts() == ["Give me a moment - I'll check what matches."]


def test_the_search_line_is_sent_once_not_on_every_tick(fast):
    transport = FakeTransport()

    alive = TurnKeepAlive(SESSION, transport, poll_seconds=0.02)
    alive.start()
    turn_status.publish(SESSION, "One moment while I check what we have in stock.")
    _wait_until(lambda: alive.status_sent is not None)
    _wait_until(lambda: transport.actions().count("typing_on") >= 3)
    alive.stop()

    assert len(transport.texts()) == 1


def test_a_turn_that_never_searches_says_nothing(fast):
    """Only a search publishes a line. A turn that just asks a question has nothing to say
    beyond the answer itself."""
    transport = FakeTransport()

    alive = TurnKeepAlive(SESSION, transport, poll_seconds=0.02)
    alive.start()
    _wait_until(lambda: transport.actions().count("typing_on") >= 2)
    alive.stop()

    assert transport.texts() == []


def test_a_channel_that_cannot_be_reached_costs_the_turn_nothing(fast):
    """All of this is cosmetic. It may never raise into the turn."""
    transport = FakeTransport(fail_on="action")

    with TurnKeepAlive(SESSION, transport, poll_seconds=0.02) as alive:
        turn_status.publish(SESSION, "Checking the lot.")
        _wait_until(lambda: alive.status_sent is not None, timeout=1.0)

    assert True, "no exception escaped"


def test_with_both_switched_off_no_thread_is_started(monkeypatch):
    monkeypatch.setattr(
        config, "settings",
        replace(config.settings, messenger_typing_refresh_seconds=0,
                messenger_send_search_status=False),
    )
    monkeypatch.setattr(channel_delivery, "settings", config.settings)
    transport = FakeTransport()

    with TurnKeepAlive(SESSION, transport):
        pass

    assert transport.calls == [], "nothing sent, nothing started"


# ------------------------------------------------------------------------ the pacing
SENDS = [
    ("text", "Here are two that fit:"),
    ("card", {"title": "2026 Diamond C Dump"}),
    ("text", "- Price: $12,500"),
    ("text", "Do either of these work?"),
]


def test_the_answer_arrives_a_bubble_at_a_time_in_order():
    transport = FakeTransport()
    slept: list[float] = []

    sent = deliver(SESSION, SENDS, transport, pause_seconds=0.8, sleep=slept.append)

    assert sent == 4
    assert transport.kinds() == ["text", "card", "text", "text"]
    assert slept == [0.8, 0.8, 0.8], "a hold before each piece but the first"


def test_the_first_piece_is_not_held_back():
    """The answer starts the moment it is ready; only what follows is spaced out."""
    slept: list[float] = []
    deliver(SESSION, SENDS[:1], FakeTransport(), pause_seconds=0.8, sleep=slept.append)
    assert slept == []


def test_one_bubble_failing_does_not_swallow_the_rest():
    """Losing a trailer is bad; losing the closing question along with it is worse."""
    class OneBadCard(FakeTransport):
        def send_card(self, session_id, element):
            raise RuntimeError("Meta rejected it")

    transport = OneBadCard()
    sent = deliver(SESSION, SENDS, transport, pause_seconds=0, sleep=lambda _s: None)

    assert sent == 3
    assert transport.texts()[-1] == "Do either of these work?"


def test_the_pause_comes_from_settings_by_default():
    slept: list[float] = []
    deliver(SESSION, SENDS[:2], FakeTransport(), sleep=slept.append)
    assert slept == [config.settings.messenger_chunk_pause_seconds]


def _wait_until(predicate, timeout: float = 2.0) -> None:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition never became true")
