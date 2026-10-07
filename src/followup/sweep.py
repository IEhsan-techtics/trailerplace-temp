"""One run of the follow-up agent: find quiet customers, decide, send, record.

For each customer due a follow-up:

1. claim the attempt (a row in chatbot_followups, unique per silence and attempt), so two runs
   can never message the same customer twice;
2. ask the model whether to send, and what;
3. send it on Messenger under the same per-customer lock a Messenger turn holds, so it can
   never cross a reply the bot is writing;
4. on success, put it in the conversation history (store.record_sent).

A customer whose message arrived while we held their lock is answered here, by the same drain
the webhook would have run - otherwise their message would wait for their next one.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src import db, inbound
from src.config import settings
from src.followup import decide as deciding
from src.followup import store

logger = logging.getLogger(__name__)


def run(now: datetime | None = None, *, send: bool = True) -> dict[str, Any]:
    """One sweep. Returns counts for the log. Never raises for one customer."""
    if not settings.followup_enabled:
        logger.info("FOLLOWUP disabled (FOLLOWUP_ENABLED is off) - nothing to do")
        return {"enabled": False}
    if not db.database_enabled():
        logger.error("FOLLOWUP no database configured")
        return {"enabled": True, "error": "no database"}

    db.ensure_schema()
    now = now or store.now_utc()
    candidates = store.find_candidates(now)
    counts = {"due": len(candidates), "sent": 0, "skipped": 0, "failed": 0, "busy": 0}
    logger.info("FOLLOWUP run: %d due", len(candidates))

    for candidate in candidates[: max(0, settings.followup_max_per_run)]:
        try:
            outcome = _one(candidate, now, send=send)
        except Exception:  # noqa: BLE001 - one customer must not stop the run
            logger.exception("FOLLOWUP failed: session=%s", candidate.session_id)
            outcome = "failed"
        counts[outcome] = counts.get(outcome, 0) + 1

    logger.info("FOLLOWUP done: %s", counts)
    return counts


def _one(candidate: store.Candidate, now: datetime, *, send: bool) -> str:
    row_id = store.claim(candidate)
    if row_id is None:
        return "busy"  # another run holds this attempt

    decision = deciding.decide(candidate, now)
    logger.info(
        "FOLLOWUP decision: session=%s attempt=%d send=%s scenario=%s reason=%s",
        candidate.session_id, candidate.attempt, decision.send, decision.scenario, decision.reason,
    )
    if not decision.send:
        store.update(row_id, status="skipped", scenario=decision.scenario, reason=decision.reason)
        return "skipped"
    if not send:
        store.update(row_id, status="skipped", scenario=decision.scenario,
                     reason=f"dry run: {decision.reason}", message=decision.message)
        return "skipped"

    from src.api.messenger import CHANNEL, MessengerTransport

    with inbound.inbound_drain_lock(candidate.psid, CHANNEL) as acquired:
        if not acquired:
            # The bot is answering them right now. Give the attempt back; the next run looks
            # again, and the new turn will have moved the anchor anyway.
            store.release(row_id)
            return "busy"
        if inbound.has_pending_inbound(candidate.psid, CHANNEL):
            store.update(row_id, status="skipped", scenario=decision.scenario,
                         reason="they wrote in before we sent", message=decision.message)
            outcome = "skipped"
        else:
            store.update(row_id, status="sending", scenario=decision.scenario,
                         reason=decision.reason, message=decision.message)
            if MessengerTransport().send_text(candidate.psid, decision.message):
                store.record_sent(row_id, candidate, decision.message, store.now_utc())
                logger.info("FOLLOWUP sent: session=%s attempt=%d", candidate.session_id, candidate.attempt)
                outcome = "sent"
            else:
                store.update(row_id, status="failed", error="Messenger rejected the send (see log)")
                outcome = "failed"

    # A message that landed while we held their lock found it taken and was left queued.
    # Answer it now, the way the webhook would have.
    if inbound.has_pending_inbound(candidate.psid, CHANNEL):
        from src.api.messenger import _drain

        logger.info("FOLLOWUP answering a message that arrived meanwhile: session=%s", candidate.session_id)
        _drain(candidate.psid)
    return outcome
