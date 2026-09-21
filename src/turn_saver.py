"""Committing a finished turn off the reply path, without losing one.

``save_turn`` is about 2.4 s against Azure Postgres from outside the region, and it happens
AFTER the reply is written - so the customer is made to wait for a commit whose result they
never see. Moving it off the turn is worth the whole 2.4 s.

It is also the most dangerous thing in the app to move, because that one transaction is what
makes three separate promises true:

* the turn row is the RECEIPT. ``stored_turn_response`` reads it to answer a redelivery from
  the stored reply instead of running the turn again - and running it again charges for two
  model calls, advances the conversation twice on one message and emails the team twice;
* the outbox rows commit with the state that produced them, so a lead is never recorded
  without the turn that earned it, nor the other way round;
* until it commits, the conversation has not happened. A container that exits here has told
  the customer their request reached the team, and it has not.

So this is not fire-and-forget. What it is:

ONE WORKER, STRICT FIFO
    Turns commit in the order they were produced. Two messages from the same customer in
    quick succession cannot land out of order and leave turn 1's state on top of turn 2's.

NEVER DROPPED
    The queue is bounded. When it is full the save runs inline, on the caller's thread,
    exactly as it did before. A slow database makes the bot slow again; it never makes it
    lossy.

ANSWERED FROM MEMORY WHILE IN FLIGHT
    The reply is registered before the job is queued, so a redelivery arriving inside the
    save window is still answered from it rather than re-run. See
    ``conversation_store.stored_turn_response``.

DRAINED ON THE WAY OUT
    The app's lifespan waits for the queue before the process exits, and ``atexit`` catches
    the paths that do not run a lifespan at all.

READ-YOUR-OWN-WRITES, PER CUSTOMER
    A turn that begins inside the previous turn's save window would load the session as it
    was BEFORE that turn - their phone number missing, their last answer gone - and answer
    from it. ``wait_for`` holds the new turn until THIS customer's save has landed. Waiting
    on the global drain would make one slow commit everybody's problem, so the count is kept
    per session and nobody waits on a conversation that is not theirs.

AND WHEN IT STILL FAILS, IT SAYS SO
    A failed save is logged at ERROR with the session, the turn and every notification that
    died with it - because at that point the customer has been told something that is no
    longer true, and that has to be findable in the logs.
"""
from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# How many finished turns may be waiting to commit before we stop queueing and save inline.
# Small on purpose: a backlog is a database problem, and a long queue only widens the window
# in which a crash loses something.
MAX_PENDING = 32

# How long the shutdown drain waits. Long enough for a queue of slow saves, short enough that
# a container being scaled down is not held past its grace period.
DRAIN_TIMEOUT_SECONDS = 25.0

# How long a new turn waits for its own session's save. A save is about 2.4 s, so this is
# several times the expected wait and still short enough that a stuck database costs one
# customer a stale read rather than their reply: past it we go on and log it.
WAIT_FOR_SAVE_SECONDS = 5.0

_queue: queue.Queue | None = None
_worker: threading.Thread | None = None

# A Condition, not a Lock, because ``wait_for`` has to sleep until a save it did not start
# reports in. It is still used as a plain lock everywhere else.
_lock = threading.Condition()

# Turns accepted and not yet committed. NOT the queue length: a job is off the queue while it
# is being written, and a drain that trusted qsize() would report "all clear" with a commit
# still in flight - then the process exits mid-save, which is the exact thing this file is
# here to prevent.
_outstanding = 0
_idle = threading.Event()
_idle.set()

# The same count, broken down by session, so a turn can wait for ITS customer's save without
# waiting on anyone else's. A session is removed the moment it reaches zero, which keeps the
# common case a miss on an empty dict.
_by_session: dict[str, int] = {}


def _describe(job: dict[str, Any]) -> str:
    return f"session={job.get('session_id')} turn={job.get('turn_id')}"


def _lost(job: dict[str, Any]) -> str:
    """What the customer was promised that this failure has just broken."""
    events = job.get("outbox_events") or []
    if not events:
        return "no notifications were pending"
    reasons = ", ".join(sorted({str(event.get("reason") or "?") for event in events}))
    return f"{len(events)} notification(s) LOST: {reasons}"


def _run(job: dict[str, Any]) -> None:
    """Commit one turn. Never raises - a failure here must not kill the worker."""
    from src import conversation_store

    save = job["save"]
    try:
        save()
    except Exception:  # noqa: BLE001 - reported, never re-raised into the worker loop
        logger.error(
            "BACKGROUND SAVE FAILED: the turn was answered but NOT saved. %s - %s. "
            "The customer has their reply; the conversation, the turn row and the lead "
            "update are all gone, and a redelivery of this message will run it again.",
            _describe(job), _lost(job),
            exc_info=True,
        )
        return

    conversation_store.clear_inflight(job["session_id"], job["turn_id"])
    if job.get("outbox_events"):
        # Only after the commit: the rows it delivers are the ones that transaction wrote.
        conversation_store.deliver_pending_outbox_async()


def _loop() -> None:
    assert _queue is not None
    while True:
        job = _queue.get()
        try:
            if job is None:  # shutdown sentinel
                return
            _run(job)
        finally:
            _queue.task_done()
            if job is not None:
                _finished(job["session_id"])


def _finished(session_id: str) -> None:
    global _outstanding
    with _lock:
        _outstanding = max(_outstanding - 1, 0)
        left = _by_session.get(session_id, 0) - 1
        if left > 0:
            _by_session[session_id] = left
        else:
            _by_session.pop(session_id, None)
        # Whether it committed or blew up. A failed save must release the next turn, not
        # leave it sitting here until the timeout - it has nothing left to wait for.
        _lock.notify_all()
        if _outstanding == 0:
            _idle.set()


def _accepted(session_id: str) -> None:
    global _outstanding
    with _lock:
        _outstanding += 1
        _by_session[session_id] = _by_session.get(session_id, 0) + 1
        _idle.clear()


def _ensure_worker() -> queue.Queue:
    global _queue, _worker
    with _lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=MAX_PENDING)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="turn-saver", daemon=True)
            _worker.start()
            atexit.register(drain)
    return _queue


def submit(
    save: Callable[[], None],
    *,
    session_id: str,
    turn_id: Any,
    outbox_events: list[dict[str, Any]] | None = None,
) -> bool:
    """Queue a finished turn's commit. Returns True when it was queued, False when it ran here.

    Never raises and never drops: a full queue means the database is slower than the
    conversation, and the right answer to that is to wait for it, not to lose a lead.
    """
    job = {
        "save": save,
        "session_id": session_id,
        "turn_id": turn_id,
        "outbox_events": list(outbox_events or []),
    }
    worker_queue = _ensure_worker()
    _accepted(session_id)
    try:
        worker_queue.put_nowait(job)
        return True
    except queue.Full:
        _finished(session_id)
        logger.warning(
            "TURN saver queue is full (%d pending) - saving inline instead. %s",
            MAX_PENDING, _describe(job),
        )
        _run(job)
        return False


def pending() -> int:
    """Turns accepted and not yet committed, including the one being written right now."""
    return _outstanding


def wait_for(session_id: str, timeout: float = WAIT_FOR_SAVE_SECONDS) -> bool:
    """Block until this session has no save in flight. Returns False if the clock ran out.

    Called at the top of a turn, before the session is read. Almost always a miss on an
    empty dict and no wait at all: it only bites when the same customer sends again inside
    the previous turn's save window - about 2.4 s - and that is exactly the case that would
    otherwise answer from the session as it was BEFORE their last message, with the contact
    they just gave us missing and the email it should have sent stashed instead.

    It waits on ONE session. Another customer's slow commit is not this customer's problem.

    Past the timeout it goes ahead anyway and says so. A stale read is bad; refusing to
    answer at all because the database is wedged is worse.
    """
    if not session_id:
        return True
    # Lock-free fast path. A dict lookup under the GIL, on the turn's hottest line: the
    # answer for a customer whose last save landed long ago is "nothing to wait for".
    if session_id not in _by_session:
        return True

    deadline = time.monotonic() + max(timeout, 0.0)
    started = time.monotonic()
    with _lock:
        while _by_session.get(session_id):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "TURN starting on a session whose previous save has not landed after "
                    "%.1fs: session=%s. This turn may not see the last one's state.",
                    timeout, session_id,
                )
                return False
            _lock.wait(remaining)

    waited = (time.monotonic() - started) * 1000
    logger.info(
        "TURN waited %.0fms for the previous turn's save: session=%s", waited, session_id
    )
    return True


def drain(timeout: float = DRAIN_TIMEOUT_SECONDS) -> bool:
    """Wait for every queued turn to commit. Returns False if any were still waiting.

    Called from the app's lifespan on the way down, and from ``atexit`` for every process
    that never had one. Anything still queued when the clock runs out is logged at ERROR:
    those turns were answered and are about to be lost.
    """
    if _outstanding == 0:
        return True

    logger.info("TURN saver draining %d pending turn(s)", _outstanding)
    # Waits for the commit to FINISH, not merely to leave the queue.
    if _idle.wait(max(timeout, 0.0)):
        logger.info("TURN saver drained")
        return True

    left = _outstanding
    if left:
        logger.error(
            "BACKGROUND SAVE ABANDONED: %d finished turn(s) never reached the database "
            "before shutdown. Those conversations were answered and are now lost.", left,
        )
    return left == 0


def reset() -> None:
    """For the tests, so one does not inherit another's worker or backlog."""
    global _queue, _worker, _outstanding
    drain(timeout=5.0)
    with _lock:
        _queue = None
        _worker = None
        _outstanding = 0
        _by_session.clear()
        _lock.notify_all()
    _idle.set()
