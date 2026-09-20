"""What the customer sees while a turn runs, and the pace the answer arrives at.

A turn takes ten to fifteen seconds, and on a push channel that is a long silence. Worse,
it is a silence that looks broken: Messenger dismisses a typing indicator after about
twenty seconds, so even the "..." stops.

Two things fill it, both of them cosmetic, both best effort, and neither able to delay or
fail the turn:

* the **typing indicator**, re-sent on an interval so it never lapses;
* the **search line** - "Give me a moment, I'll check what matches your requirements." -
  which the search node publishes to ``src/turn_status`` the instant it starts looking.
  Sent straight away rather than with the results, because a line saying we are looking is
  worth nothing once the trailers are already on screen. It is the same line, from the same
  place, that the web chat shows while it waits, so the two channels say the same thing.

Then the answer itself arrives **paced**: the intro, a short hold, the first trailer, a
hold, the next - the way a person sends them, rather than six notifications in one burst.

Nothing here knows how to talk to Meta. A caller passes a ``Transport``; this module
decides what to send and when.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterable, Protocol

from src import turn_status
from src.config import settings

logger = logging.getLogger(__name__)


class Transport(Protocol):
    """How a channel actually puts something in front of the customer.

    Three calls, because a generic template, a text bubble and a typing indicator are three
    different requests to the Send API. Every one of them may fail; none of them may raise
    in a way that costs the customer their answer.
    """

    def send_text(self, session_id: str, text: str) -> None: ...

    def send_card(self, session_id: str, element: dict[str, Any]) -> None: ...

    def send_action(self, session_id: str, action: str) -> None: ...


class TurnKeepAlive:
    """Everything shown BETWEEN their message and the answer. Runs on its own thread.

    Started before the turn and stopped after it. Its failures are logged and swallowed:
    a typing indicator that did not send is a cosmetic loss, and a customer who gets their
    trailers without one has lost nothing that matters.
    """

    def __init__(self, session_id: str, transport: Transport, *, poll_seconds: float = 0.5):
        self._session_id = session_id
        self._transport = transport
        self._poll = max(0.05, poll_seconds)
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        # Exposed for the tests and the logs: the line we actually sent, if any.
        self.status_sent: str | None = None

    def __enter__(self) -> "TurnKeepAlive":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        refresh = max(0.0, settings.messenger_typing_refresh_seconds)
        if refresh <= 0 and not settings.messenger_send_search_status:
            return  # both switched off: nothing is sent and no thread is started
        self._started = True
        self._safely(self._transport.send_action, self._session_id, "mark_seen")
        self._safely(self._transport.send_action, self._session_id, "typing_on")
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"keepalive:{self._session_id[:12]}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        if self._thread is not None:
            # Bounded: the loop only ever waits on the event, so it returns promptly.
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._started:
            # Only when we turned it on. Clearing an indicator we never set would be one
            # more request to the channel for no reason at all.
            self._started = False
            self._safely(self._transport.send_action, self._session_id, "typing_off")

    def _run(self) -> None:
        refresh = max(0.0, settings.messenger_typing_refresh_seconds)
        # The status line is polled far more often than typing is refreshed, because its
        # whole value is being early - one delivered ten seconds late has missed the point.
        # Capped by the refresh interval too, or a shorter interval could never come due.
        tick = min(self._poll, refresh) if refresh > 0 else self._poll
        since_typing = 0.0
        while not self._done.wait(tick):
            if self.status_sent is None and settings.messenger_send_search_status:
                self._send_status_if_ready()
            since_typing += tick
            if refresh > 0 and since_typing >= refresh:
                since_typing = 0.0
                self._safely(self._transport.send_action, self._session_id, "typing_on")

    def _send_status_if_ready(self) -> None:
        line = (turn_status.peek(self._session_id) or "").strip()
        if not line:
            return  # this turn is not searching, or has not started looking yet
        self.status_sent = line
        self._safely(self._transport.send_text, self._session_id, line)
        logger.info("KEEPALIVE sent the search line early: session=%s", self._session_id)

    def _safely(self, call: Callable[..., Any], *args: Any) -> None:
        try:
            call(*args)
        except Exception:  # noqa: BLE001 - cosmetic; it must never touch the turn
            logger.exception("KEEPALIVE send failed: session=%s", self._session_id)


def deliver(
    session_id: str,
    sends: Iterable[tuple[str, Any]],
    transport: Transport,
    *,
    pause_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Send one rendered reply, paced. Returns how many pieces went out.

    The hold goes BEFORE each piece but the first, so the answer starts the moment it is
    ready and only the pieces after it are spaced out. A piece that fails to send is logged
    and the rest still go: losing one trailer is bad, losing the closing question with it
    is worse.
    """
    pause = settings.messenger_chunk_pause_seconds if pause_seconds is None else pause_seconds
    pause = max(0.0, pause)
    sent = 0
    for index, (kind, payload) in enumerate(sends):
        if index and pause:
            sleep(pause)
        try:
            if kind == "card":
                transport.send_card(session_id, payload)
            else:
                transport.send_text(session_id, payload)
            sent += 1
        except Exception:  # noqa: BLE001 - one bad bubble must not swallow the rest
            logger.exception("DELIVER failed: session=%s kind=%s", session_id, kind)
    return sent
