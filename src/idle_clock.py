"""The 5-minute rule's clock: a background thread that runs the sweep every minute.

Inside the API rather than an Azure job calling it, because the arithmetic allows it: a timer
is due five minutes after the customer's last message, and Azure keeps trailerplace-api up
for ten minutes (cooldownPeriod 600) after the last request. So whenever a timer comes due,
this process is still running to fire it - with no job waking the API every minute, which
would keep it up for good and cost what an always-on replica costs.

The sweep's own work is not a request, so the clock never keeps the API awake by itself; it
shuts down on the usual ten minutes after the last customer. Timers live in the database, so
a restart loses nothing: the first tick, a few seconds after startup, fires anything that
came due while the process was down. The sweep claims rows with SKIP LOCKED, so a second
replica, or the manual POST /internal/idle-sweep, can never fire the same timer twice.
"""
from __future__ import annotations

import logging
import threading

from src import config, idle_timer

logger = logging.getLogger(__name__)

# The first tick comes soon after startup, for timers that fell due while we were down.
FIRST_TICK_SECONDS = 5.0

_thread: threading.Thread | None = None
_stop = threading.Event()


def start() -> bool:
    """Start the clock, if the 5-minute rule is on. True when it is running."""
    global _thread
    if not idle_timer.enabled():
        return False
    if _thread is not None and _thread.is_alive():
        return True
    _stop.clear()
    _thread = threading.Thread(target=_run, name="idle-clock", daemon=True)
    _thread.start()
    logger.info("IDLE clock started: every %.0fs", _interval())
    return True


def stop(timeout: float = 5.0) -> None:
    """Stop the clock. A sweep already running finishes its current timer first."""
    global _thread
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=timeout)
        _thread = None


def _interval() -> float:
    return max(10.0, float(config.settings.idle_sweep_interval_seconds or 60))


def _run() -> None:
    from src import idle_sweep

    wait = FIRST_TICK_SECONDS
    while not _stop.wait(wait):
        try:
            idle_sweep.sweep()
        except Exception:  # noqa: BLE001 - the clock outlives any one bad tick
            logger.exception("IDLE clock tick failed")
        wait = _interval()
