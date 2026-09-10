"""Reading and writing the three chatbot tables.

``chatbot_conversations.lead_id`` is NOT NULL with a RESTRICT foreign key, so a lead row
has to exist before the conversation row that points at it. Every session therefore opens
with a placeholder lead (soft / missing_contact / "Unspecified") which is upgraded in place
as the conversation learns who they are - rather than a lead being created at the moment
contact details arrive, which would leave the first turns with nowhere to persist.

When the database is off or persistence is disabled, everything falls back to a
process-local dict with the same interface. Tests and offline runs use that path; nothing
in the graph knows the difference.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src import db
from src.config import settings
from src.db_models import ChatbotConversation, ChatbotLead, ChatbotTurn

logger = logging.getLogger(__name__)

# The in-memory fallback: session_id -> {"lead_id", "conversation", "state_snapshot",
# "state_version"}. Process-local and deliberately not thread-partitioned; it exists for
# tests and for a developer running without Postgres.
_MEMORY: dict[str, dict[str, Any]] = {}

_PLACEHOLDER_ITEM = "Unspecified"


def persistence_enabled() -> bool:
    return bool(settings.trailerplace_persist_chats) and db.database_enabled()


def reset_memory() -> None:
    """Drop the in-memory store. Tests use this between cases."""
    _MEMORY.clear()


def _as_uuid(value: str) -> uuid.UUID:
    """Session ids are UUIDs in the schema but arrive as strings from the API.

    A caller-supplied id that is not a UUID is hashed into one deterministically, so an
    id like "demo" still round trips instead of raising at the database boundary.
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return uuid.uuid5(uuid.NAMESPACE_URL, f"trailerplace-session:{value}")


# --------------------------------------------------------------------------- session setup
def ensure_session(session_id: str) -> str:
    """Make sure a lead and a conversation row exist. Returns the lead id.

    Idempotent: calling it on an existing session returns the lead already attached.
    """
    if not persistence_enabled():
        record = _MEMORY.setdefault(
            session_id,
            {"lead_id": str(uuid.uuid4()), "conversation": [], "state_snapshot": None,
             "state_version": 0},
        )
        return record["lead_id"]

    session_uuid = _as_uuid(session_id)
    db.ensure_schema()
    with db.get_session_factory()() as sql:
        existing = sql.get(ChatbotConversation, session_uuid)
        if existing:
            return str(existing.lead_id)

        # The lead first, and flushed before the conversation: the FK is RESTRICT, so the
        # order is not merely tidy, it is required.
        lead = ChatbotLead(
            lead_id=uuid.uuid4(),
            psid=None,
            name=None,
            phone_number=None,
            email=None,
            lead_type="soft",
            contact_status="missing_contact",
            item_of_interest=_PLACEHOLDER_ITEM,
        )
        sql.add(lead)
        sql.flush()

        sql.add(
            ChatbotConversation(
                session_id=session_uuid,
                lead_id=lead.lead_id,
                conversation=[],
                state_snapshot=None,
                state_version=0,
            )
        )
        sql.commit()
        logger.info("SESSION opened: session=%s lead=%s", session_id, lead.lead_id)
        return str(lead.lead_id)


def load_session(session_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    """Return ``(state_snapshot, conversation, lead_id)`` for a session."""
    if not persistence_enabled():
        record = _MEMORY.get(session_id)
        if not record:
            return None, [], ""
        return record["state_snapshot"], list(record["conversation"]), record["lead_id"]

    with db.get_session_factory()() as sql:
        row = sql.get(ChatbotConversation, _as_uuid(session_id))
        if not row:
            return None, [], ""
        return row.state_snapshot, list(row.conversation or []), str(row.lead_id)


# ------------------------------------------------------------------------------ writing
def save_turn(
    session_id: str,
    *,
    conversation: list[dict[str, Any]],
    state_snapshot: dict[str, Any],
    request_message: str,
    response: dict[str, Any],
    contact: dict[str, Any] | None = None,
    item_of_interest: str | None = None,
) -> None:
    """Persist one completed turn.

    One transaction: the conversation row, the turn row and the lead update commit together
    or not at all, so a crash can never leave a transcript that disagrees with the state
    snapshot it was produced from.
    """
    if not persistence_enabled():
        record = _MEMORY.setdefault(
            session_id,
            {"lead_id": str(uuid.uuid4()), "conversation": [], "state_snapshot": None,
             "state_version": 0},
        )
        record["conversation"] = list(conversation)
        record["state_snapshot"] = dict(state_snapshot)
        record["state_version"] += 1
        record.setdefault("turns", []).append(
            {"request_message": request_message, "response": response}
        )
        if contact:
            record["contact"] = dict(contact)
        if item_of_interest:
            record["item_of_interest"] = item_of_interest
        return

    session_uuid = _as_uuid(session_id)
    with db.get_session_factory()() as sql:
        row = sql.get(ChatbotConversation, session_uuid)
        if row is None:
            logger.warning("SAVE skipped: no conversation row for session=%s", session_id)
            return

        row.conversation = list(conversation)
        row.state_snapshot = dict(state_snapshot)
        row.state_version = (row.state_version or 0) + 1
        row.updated_at = datetime.now(timezone.utc)

        # Appended through the relationship, not by raw FK, so the unit of work orders the
        # inserts correctly on a session's very first flush.
        row.turns.append(
            ChatbotTurn(
                session_id=session_uuid,
                turn_id=uuid.uuid4(),
                request_message=request_message[: settings.chat_max_message_chars],
                response=response,
            )
        )

        if contact or item_of_interest:
            _update_lead(sql, row.lead_id, contact or {}, item_of_interest)

        sql.commit()


def _update_lead(sql, lead_id, contact: dict[str, Any], item_of_interest: str | None) -> None:
    """Fill in what we have learned about who this is.

    A lead becomes 'hard' the moment we hold a name AND either an email or a phone number -
    the minimum the dealership treats as a real lead. Below that it stays 'soft', and a
    declined contact simply stays soft forever; we do not keep asking and we do not record
    a half-lead as though it were a whole one.
    """
    lead = sql.get(ChatbotLead, lead_id)
    if lead is None:
        return

    # Only ever fill a blank. A later turn saying "actually I'm Dave's colleague" must not
    # silently overwrite the contact details the lead was qualified on.
    if contact.get("name") and not lead.name:
        lead.name = str(contact["name"])[:255]
    if contact.get("email") and not lead.email:
        lead.email = str(contact["email"])[:255]
    if contact.get("phone") and not lead.phone_number:
        lead.phone_number = str(contact["phone"])[:64]

    if item_of_interest:
        lead.item_of_interest = item_of_interest[:2000]

    if lead.name and (lead.email or lead.phone_number):
        lead.lead_type = "hard"
        lead.contact_status = "complete"
    elif lead.name or lead.email or lead.phone_number:
        lead.contact_status = "partial"


def describe_interest(state: dict) -> str:
    """A one-line summary of what they are shopping for, for ``item_of_interest``.

    That column is NOT NULL, so it always has to resolve to something.
    """
    category = state.get("category")
    slots = state.get("slots") or {}
    if not category and not slots:
        return _PLACEHOLDER_ITEM

    parts: list[str] = [str(category)] if category else []
    for slot in ("haul_item", "length", "payload_capacity", "hitch_type"):
        value = slots.get(slot)
        if value is None or value == [] or value == "":
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        parts.append(f"{slot}={value}")
    return " | ".join(parts) or _PLACEHOLDER_ITEM


def load_lead(session_id: str) -> dict[str, Any] | None:
    """The lead row behind a session, as a plain dict. For tests and the debug endpoint."""
    if not persistence_enabled():
        record = _MEMORY.get(session_id)
        if not record:
            return None
        contact = record.get("contact") or {}
        name, email, phone = contact.get("name"), contact.get("email"), contact.get("phone")
        complete = bool(name and (email or phone))
        return {
            "lead_id": record["lead_id"],
            "name": name,
            "email": email,
            "phone_number": phone,
            "lead_type": "hard" if complete else "soft",
            "contact_status": "complete" if complete
            else ("partial" if (name or email or phone) else "missing_contact"),
            "item_of_interest": record.get("item_of_interest") or _PLACEHOLDER_ITEM,
        }

    with db.get_session_factory()() as sql:
        row = sql.get(ChatbotConversation, _as_uuid(session_id))
        if row is None:
            return None
        lead = sql.get(ChatbotLead, row.lead_id)
        if lead is None:
            return None
        return {
            "lead_id": str(lead.lead_id),
            "name": lead.name,
            "email": lead.email,
            "phone_number": lead.phone_number,
            "lead_type": lead.lead_type,
            "contact_status": lead.contact_status,
            "item_of_interest": lead.item_of_interest,
        }
