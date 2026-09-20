"""One line of JSON per turn - the machine feed, and the cost report's data source.

Written to the ``trailerplace.turn`` logger, so an operator sees every record in the normal
log stream without configuring anything. Set TURN_LOG_PATH as well and the same lines also
land in a dedicated append-only JSONL file, which ``scripts/cost_report.py`` reads.

Deliberately flat: the cost audit asserts on ``llm_calls`` and ops greps on ``session_id``
and ``latency_ms``. Anything a human wants to READ belongs in src/conversation_log.py.
"""
from __future__ import annotations

import json
import logging
from typing import Any

# Read through the module, not from it, so a test can swap the frozen Settings wholesale.
from src import config

TURN_LOGGER_NAME = "trailerplace.turn"

logger = logging.getLogger(TURN_LOGGER_NAME)

_file_handler_installed = False


class _JsonLineFormatter(logging.Formatter):
    """Emits the record's ``turn`` payload as bare JSON - no level or timestamp prefix.

    The cost report json.loads every line of this file, so anything that is not the payload
    would only have to be stripped back out.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "turn", None)
        if payload is None:
            return record.getMessage()
        return json.dumps(payload, default=str)


def _ensure_file_handler() -> None:
    global _file_handler_installed
    if _file_handler_installed or not config.settings.turn_log_path:
        return
    handler = logging.FileHandler(config.settings.turn_log_path, encoding="utf-8")
    handler.setFormatter(_JsonLineFormatter())
    logger.addHandler(handler)
    # The dedicated file is a machine feed; the record still propagates to root for humans.
    # Setting the level here keeps it independent of root's.
    logger.setLevel(logging.INFO)
    _file_handler_installed = True


def reset_for_tests() -> None:
    """Drop the file handler, so a test can point TURN_LOG_PATH somewhere new."""
    global _file_handler_installed
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    _file_handler_installed = False


def tools_fired(turn_outcome: dict[str, Any]) -> list[str]:
    """Which tools ran this turn - the ops log and the cost assertion read the same list.

    ``search`` implies one catalogue query; ``inventory_lookup`` implies no model call at
    all, since the matcher is plain Python over the prepared inventory.
    """
    outcome = turn_outcome or {}
    tools: list[str] = []
    if outcome.get("search_ran"):
        tools.append("search")
    if outcome.get("inventory_lookup_ran"):
        tools.append("inventory_lookup")
    if outcome.get("outbox_events") or outcome.get("emails_flushed"):
        tools.append("email")
    if outcome.get("escalated") or outcome.get("link_interest"):
        tools.append("escalate")
    return tools


def log_turn(
    *,
    session_id: str,
    turn_id: str | None,
    turn_index: int | None = None,
    intent: str | None,
    category: str | None,
    latency_ms: float,
    turn_outcome: dict[str, Any],
    usage: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Emit - and return - one turn record. Never raises."""
    try:
        return _log_turn(
            session_id=session_id, turn_id=turn_id, turn_index=turn_index, intent=intent,
            category=category, latency_ms=latency_ms, turn_outcome=turn_outcome,
            usage=usage, error=error,
        )
    except Exception:  # noqa: BLE001 - a log line is never worth the customer's reply
        logger.exception("Turn logging failed: session=%s", session_id)
        return {}


def _log_turn(
    *,
    session_id: str,
    turn_id: str | None,
    turn_index: int | None,
    intent: str | None,
    category: str | None,
    latency_ms: float,
    turn_outcome: dict[str, Any],
    usage: Any,
    error: str | None,
) -> dict[str, Any]:
    _ensure_file_handler()
    outcome = turn_outcome or {}
    fired = tools_fired(outcome)
    if usage is not None and int(getattr(usage, "feature_reranks", 0) or 0):
        fired.append("feature_rerank")
    record: dict[str, Any] = {
        "event": "chat_turn",
        "session_id": session_id,
        "turn_id": turn_id,
        "turn_index": turn_index,
        "intent": intent,
        "category": category,
        "latency_ms": round(float(latency_ms), 2),
        "tools_fired": fired,
        "result_count": outcome.get("result_count", 0),
        "email_status": outcome.get("email_status"),
        "llm_calls": usage.as_dict() if usage is not None else None,
    }
    if error:
        record["error"] = error
    logger.info("chat_turn", extra={"turn": record})
    return record
