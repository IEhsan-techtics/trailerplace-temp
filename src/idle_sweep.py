"""Fire the 5-minute timers whose time has come. Called every minute.

POST /internal/idle-sweep runs this; in Azure a Container Apps job calls it on a one-minute
schedule, which also wakes the API when it has scaled to zero. src/idle_timer.py decides when
a timer is armed; this is only what happens when one is due:

1. claim it, so no overlapping sweep can take it too;
2. check the customer has not come back meanwhile - a Messenger message still queued, or any
   turn since the one that armed it - and let their own turn answer them if so;
3. run the idle turn (build.run_idle_turn), under a turn id derived from the timer, so a
   retried sweep can never send the same trailers twice;
4. deliver it: Messenger gets the paced cards; web chat fetches it on its own poll;
5. close the timer.

Never raises: one conversation failing must not stop the rest of the sweep.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src import conversation_store, inbound

logger = logging.getLogger(__name__)

# Where the idle turn's id comes from. Fixed, so the same timer always yields the same id.
_TURN_NAMESPACE = uuid.UUID("5b1c2a8e-4f7d-4f0e-9a51-6c1f1e0d2a77")


def idle_turn_id(session_id: str, due_at: Any) -> uuid.UUID:
    return uuid.uuid5(_TURN_NAMESPACE, f"{session_id}|{due_at}")


def sweep(now: datetime | None = None, limit: int = 10, transport: Any = None) -> list[dict[str, Any]]:
    """Fire every due timer, up to ``limit``. Returns one line per timer, for the log."""
    now = now or datetime.now(timezone.utc)
    results = []
    for timer in conversation_store.claim_due_timers(now, limit=limit):
        results.append(_fire(timer, transport))
    if results:
        logger.info("IDLE sweep: %d fired of %d due", sum(r["status"] == "fired" for r in results), len(results))
    return results


def _fire(timer: dict[str, Any], transport: Any) -> dict[str, Any]:
    session_id = timer["session_id"]
    psid = timer.get("channel_id")
    try:
        if timer.get("channel") == "messenger" and psid and inbound.has_pending_inbound(psid):
            # They wrote in; their own turn is on its way and answers them.
            conversation_store.finish_timer(session_id, "cancelled", "customer wrote in")
            return {"session_id": session_id, "status": "cancelled", "why": "customer wrote in"}

        from src.graph.build import run_idle_turn

        result = run_idle_turn(
            session_id, timer["category"],
            turn_id=idle_turn_id(session_id, timer["due_at"]),
            channel_id=psid,
        )
        if result is None:
            conversation_store.finish_timer(session_id, "cancelled", "no longer needed")
            return {"session_id": session_id, "status": "cancelled", "why": "no longer needed"}

        delivered = _deliver(timer, result, transport)
        conversation_store.finish_timer(session_id, "fired")
        logger.info(
            "IDLE fired: session=%s channel=%s category=%s listings=%d delivered=%s",
            session_id, timer.get("channel"), timer["category"], len(result.get("listings") or []), delivered,
        )
        return {"session_id": session_id, "status": "fired", "delivered": delivered}
    except Exception as exc:  # noqa: BLE001 - one conversation must not stop the sweep
        logger.exception("IDLE fire failed: session=%s", session_id)
        conversation_store.finish_timer(session_id, "failed", f"{type(exc).__name__}: {exc}")
        return {"session_id": session_id, "status": "failed", "why": type(exc).__name__}


def _deliver(timer: dict[str, Any], result: dict[str, Any], transport: Any) -> int | None:
    """Messenger is pushed to. Web chat is not: the page polls /session/{id} and draws it."""
    if timer.get("channel") != "messenger" or not timer.get("channel_id"):
        return None
    from src import channel_delivery
    from src.config import settings
    from src.domain.cards import messenger_sends

    if transport is None:
        from src.api.messenger import MessengerTransport, messenger_enabled

        if not messenger_enabled():
            logger.warning("IDLE fired for Messenger, but Messenger is not configured: session=%s", timer["session_id"])
            return 0
        transport = MessengerTransport()
    sends = messenger_sends(
        result.get("assistant_text") or "", result.get("listings") or [],
        cards_enabled=settings.messenger_listing_cards,
    )
    return channel_delivery.deliver(timer["channel_id"], sends, transport)
