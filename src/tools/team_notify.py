"""Telling the team something, under the contact gate.

Every email the bot sends is an internal notification about a customer, and it is worth
nothing without a way to reach that customer. So the rule is absolute:

    NOTHING IS SENT UNTIL WE HOLD A NAME **AND** AN EMAIL OR A PHONE NUMBER.

A request that arrives before then is STASHED, never dropped - ``state["pending_email_actions"]``
carries it, and it survives to the next turn in the state snapshot like everything else. The
moment the missing piece arrives, every stashed request is flushed together: five requests
made across five turns become five emails on the turn they finally become reachable.

The customer is never made to wait for their answer. They get the canned line immediately,
plus a request for whichever piece we are missing, worded so they understand WHY - it is so
someone can get back to them, not paperwork for its own sake.

A refusal ends it. We stop asking, the stash is dropped, and the conversation carries on.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.tools import email_sender

logger = logging.getLogger(__name__)


def contact_pieces(state: dict) -> dict[str, Any]:
    return state.get("contact") or {}


def has_name(state: dict) -> bool:
    return bool(contact_pieces(state).get("name"))


def has_reachable(state: dict) -> bool:
    contact = contact_pieces(state)
    return bool(contact.get("email") or contact.get("phone"))


def contact_complete(state: dict) -> bool:
    """The minimum the dealership treats as a real lead."""
    return has_name(state) and has_reachable(state)


def declined(state: dict) -> bool:
    return bool(contact_pieces(state).get("declined"))


def missing_pieces(state: dict) -> list[str]:
    """Everything still missing, so we can ask for it all at once rather than one per turn."""
    missing: list[str] = []
    if not has_name(state):
        missing.append("name")
    if not has_reachable(state):
        missing.append("contact")
    return missing


def ask_for_missing(state: dict) -> str:
    """The request for what we are missing, saying why we want it.

    Phrased around the follow-up rather than around our records: a customer asked for their
    number with no reason given is being processed, one asked so the team can get back to
    them is being helped.
    """
    missing = missing_pieces(state)
    if missing == ["name"]:
        return (
            "Could I take your name so our team can follow up with you? It makes it much "
            "easier for them to get back to you."
        )
    if missing == ["contact"]:
        return (
            "Could I take an email or phone number so our team can follow up with you? It "
            "makes it much easier for them to get back to you."
        )
    if missing:
        return (
            "Could I take your name and an email or phone number so our team can follow up "
            "with you? It makes it much easier for them to get back to you."
        )
    return ""


def _normalized(text: Any) -> str:
    """A description reduced to what it is ABOUT, for de-duplication.

    The agent re-words the same request every time it re-raises it ("wants a callback about a
    dump trailer" / "asked us to call him back"), so keying on the exact string sent the team
    the same lead twice. Keyed on this instead, and truncated, so a genuinely different
    request still gets through.
    """
    return re.sub(r"[^a-z0-9 ]+", "", str(text or "").lower())[:60]


def _key(event: dict) -> tuple:
    return (event.get("reason"), _normalized(event.get("description")))


def _dedupe(events: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []
    for event in events:
        key = _key(event)
        if key in seen:
            logger.info("team_notify | dropping duplicate %s", event.get("reason"))
            continue
        seen.add(key)
        unique.append(event)
    return unique


def build_event(state: dict, *, reason: str, description: str) -> dict[str, Any]:
    """One notification, rendered against whatever contact details we hold RIGHT NOW.

    Rendered at flush time rather than here would be better in principle, but the details we
    are waiting for are exactly the ones that go in the body - so a stashed event is
    re-rendered when it is flushed. See ``flush``.
    """
    return {
        "reason": reason,
        "description": str(description or "").strip() or "No detail given.",
        "event_type": reason.lower().replace(" ", "_").replace("–", "-")[:64],
    }


def _rendered(state: dict, event: dict) -> dict[str, Any]:
    """Turn a stored event into the outbox row payload, with the contact we now hold."""
    contact = contact_pieces(state)
    return {
        "event_key": f"{event['event_type']}:{_normalized(event['description'])[:32]}",
        "event_type": event["event_type"],
        "reason": event["reason"],
        "payload": {
            "subject": email_sender.render_subject(
                reason=event["reason"],
                name=contact.get("name"),
                session_id=str(state.get("session_id") or ""),
            ),
            "body": email_sender.render_email_body(
                name=contact.get("name"),
                email=contact.get("email"),
                phone=contact.get("phone"),
                reason=event["reason"],
                description=event["description"],
            ),
        },
    }


def record(state: dict, *, reason: str, description: str) -> str:
    """Note something for the team. Sends it, or stashes it until we can.

    Returns "sent", "stashed" or "dropped" - the caller turns that into what we tell the
    customer.
    """
    outcome = state.setdefault("turn_outcome", {})
    event = build_event(state, reason=reason, description=description)

    if declined(state):
        # They refused to be contacted. Recording a lead we can never follow up on is not a
        # lead, and asking again is the one thing we promised not to do.
        logger.info("team_notify | %s dropped: contact declined", reason)
        outcome["email_status"] = "dropped (contact declined)"
        return "dropped"

    if contact_complete(state):
        queued = outcome.setdefault("outbox_events", [])
        queued.extend(_rendered(state, item) for item in _dedupe([event]))
        outcome["email_status"] = f"sent: {reason}"
        logger.info("team_notify | %s queued for delivery", reason)
        return "sent"

    stash = state.setdefault("pending_email_actions", [])
    stash.append(event)
    state["pending_email_actions"] = _dedupe(stash)
    state["contact_followup_pending"] = ", ".join(missing_pieces(state))
    outcome["email_status"] = f"stashed: {reason} (waiting on {state['contact_followup_pending']})"
    logger.info(
        "team_notify | %s stashed, waiting on %s (%d pending)",
        reason, state["contact_followup_pending"], len(state["pending_email_actions"]),
    )
    return "stashed"


def flush(state: dict) -> int:
    """Send everything that was waiting, now that we can reach them.

    Called once per turn from apply, after the contact details have been merged. Returns how
    many went out, so the reply can tell the customer their request is on its way.
    """
    stash = state.get("pending_email_actions") or []
    if not stash:
        return 0

    if declined(state):
        state["pending_email_actions"] = []
        state["contact_followup_pending"] = None
        logger.info("team_notify | dropping %d stashed notification(s): declined", len(stash))
        return 0

    if not contact_complete(state):
        # Still short. Keep waiting, and keep the record of what we are waiting for current -
        # they may have given us the name and still owe us a number.
        state["contact_followup_pending"] = ", ".join(missing_pieces(state))
        return 0

    outcome = state.setdefault("turn_outcome", {})
    queued = outcome.setdefault("outbox_events", [])
    events = _dedupe(stash)
    queued.extend(_rendered(state, event) for event in events)

    state["pending_email_actions"] = []
    state["contact_followup_pending"] = None
    outcome["emails_flushed"] = len(events)
    outcome["email_status"] = f"sent {len(events)} stashed notification(s)"
    logger.info("team_notify | flushed %d stashed notification(s)", len(events))
    return len(events)
