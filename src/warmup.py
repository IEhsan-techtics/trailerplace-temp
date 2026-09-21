"""Everything a first turn would otherwise pay for, done before the first customer arrives.

Measured against Azure Postgres from outside the region, a cold process spent about ten
seconds on work that is then free for the rest of its life:

    connection pool       4.3 s   the first connect, TLS and all
    question rules        5.0 s   one read, cached for the process
    make inventory        1.3 s   one read, lru_cached

All of it landed on whoever happened to send the first message. On a serverless deployment
that is every scale-from-zero, so it is not a rare event - it is the experience of the first
customer after every quiet spell.

Each step is independent and none may raise: a warm-up that fails has cost nothing, because
the thing it was warming is still loaded lazily on the turn path exactly as before.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_done = False


def _warm_pool() -> None:
    from sqlalchemy import text

    from src import db

    if not db.database_enabled():
        return
    with db.get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


def _warm_rules() -> None:
    from src.rules.store import current_rules

    current_rules()


def _warm_inventory() -> None:
    from src.domain.brands import load_make_inventory

    load_make_inventory()


_STEPS = (("pool", _warm_pool), ("rules", _warm_rules), ("inventory", _warm_inventory))


def warm_everything(force: bool = False) -> dict[str, float]:
    """Load the caches. Idempotent, never raises; returns what each step cost in ms.

    Called from the app's lifespan and again from /health, which is what a serverless
    platform pings to wake the container - so by the time the health check says "ok", a turn
    can be served without paying for any of this.
    """
    global _done
    if _done and not force:
        return {}

    timings: dict[str, float] = {}
    for name, step in _STEPS:
        started = time.perf_counter()
        try:
            step()
            timings[name] = round((time.perf_counter() - started) * 1000, 1)
        except Exception:  # noqa: BLE001 - a cold cache is not a broken app
            logger.exception("Warm-up step %r failed; it will load lazily instead", name)
            timings[name] = -1.0
    _done = True
    logger.info("WARMUP done: %s", ", ".join(f"{k}={v}ms" for k, v in timings.items()))
    return timings


def is_warm() -> bool:
    return _done


def reset() -> None:
    """For the tests, which must not inherit a warm flag from another one."""
    global _done
    _done = False
