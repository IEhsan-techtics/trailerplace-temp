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

    Long enough to tell two listing URLs apart: those differ only in the slug on the end,
    and truncated at 60 characters two trailers from the same make became "the same
    request" and the team heard about one of them.
    """
    return re.sub(r"[^a-z0-9 ]+", "", str(text or "").lower())[:120]


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


# How long the one-line description may run. It is a LABEL, not a report: the reason line
# already says what kind of thing this is, and the chat link carries the whole conversation,
# so the words in between only have to say which customer and which thing. Everything Python
# authors is written to fit; this is the backstop for the one description it does not write,
# the agent's own summary on an escalation.
MAX_DESCRIPTION_WORDS = 7


def shorten(description: Any) -> str:
    """The description, cut to ``MAX_DESCRIPTION_WORDS``.

    Cut on whole words with an ellipsis, so a line that was too long still reads as a
    sentence that trails off rather than a string that stops mid-word.
    """
    words = str(description or "").strip().split()
    if len(words) <= MAX_DESCRIPTION_WORDS:
        return " ".join(words)
    kept = " ".join(words[:MAX_DESCRIPTION_WORDS]).rstrip(".,;:-")
    logger.info("team_notify | description shortened from %d words", len(words))
    return f"{kept}…"


def build_event(state: dict, *, reason: str, description: str) -> dict[str, Any]:
    """One notification, rendered against whatever contact details we hold RIGHT NOW.

    Rendered at flush time rather than here would be better in principle, but the details we
    are waiting for are exactly the ones that go in the body - so a stashed event is
    re-rendered when it is flushed. See ``flush``.
    """
    return {
        "reason": reason,
        "description": shorten(description) or "No detail given.",
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
                shared_platforms=list(state.get("shared_platforms") or []),
                chat_url=email_sender.chat_session_url(str(state.get("session_id") or "")),
            ),
        },
    }


def _deliver(state: dict, rendered: list[dict]) -> None:
    """Queue for the outbox, or send now when there is no outbox to queue into.

    The outbox is the right home for a notification: it commits in the same transaction as
    the turn that produced it, and a send that fails stays pending for the next drain. But it
    only exists when persistence does. With it off - a developer without Postgres, or, far
    worse, a database outage in production - ``save_turn`` discards ``outbox_events`` and the
    drain returns early, so every lead was silently lost while this function logged "queued
    for delivery" and the customer was told their request had been passed on.

    So when there is no transaction to be atomic with, we give up atomicity rather than
    delivery and send inline. A failed send is logged and lost here, which is worse than a
    retryable row and much better than never attempting one.
    """
    if not rendered:
        return

    from src import conversation_store

    # Recorded on the turn either way: it is what save_turn persists, and what the tests and
    # the logs read to see what this turn generated.
    state.setdefault("turn_outcome", {}).setdefault("outbox_events", []).extend(rendered)

    if conversation_store.persistence_enabled():
        logger.info("team_notify | %d notification(s) queued for delivery", len(rendered))
        return

    # No outbox to queue into, so send here. run_turn's post-commit drain returns early in
    # this mode, so there is no second attempt and nothing is sent twice.
    from src.tools import email_sender

    for item in rendered:
        payload = item.get("payload") or {}
        sent = email_sender.send_email(
            str(payload.get("subject") or "TrailerPlace Lead"),
            str(payload.get("body") or ""),
        )
        logger.log(
            logging.INFO if sent else logging.ERROR,
            "team_notify | %s sent inline (no outbox): %s",
            item.get("event_type"), "ok" if sent else "FAILED",
        )


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
        _deliver(state, [_rendered(state, item) for item in _dedupe([event])])
        outcome["email_status"] = f"sent: {reason}"
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
    events = _dedupe(stash)
    _deliver(state, [_rendered(state, event) for event in events])

    state["pending_email_actions"] = []
    state["contact_followup_pending"] = None
    outcome["emails_flushed"] = len(events)
    outcome["email_status"] = f"sent {len(events)} stashed notification(s)"
    logger.info("team_notify | flushed %d stashed notification(s)", len(events))
    return len(events)


# The team hears about every set of trailers a customer was actually shown, not only about
# the ones they said yes to. A browsing session that goes quiet is still a lead worth
# knowing about - somebody looked at six livestock trailers this afternoon - and it is the
# one thing the transcript cannot tell them at a glance.
RESULTS_SHOWN_REASON = "Results Shown to User"

def record_results_shown(state: dict, listings: list[Any], category: str = "") -> str:
    """Tell the team which trailers were just put in front of this customer.

    Called with what the reply actually SHOWED, never with what the search found: a trailer
    the reply dropped was never presented, and telling the team otherwise is telling them
    something that did not happen.

    It goes through the same gate as everything else, so with no way to reach the customer
    it waits rather than going out as an anonymous "someone saw six trailers" - and it is
    what keeps the contact request alive on the following turn (greeting.contact_ask_is_due).

    The stock numbers used to be listed here. They are in the conversation the chat link
    opens, and six of them made this the longest line the team ever read.
    """
    if not listings:
        return "dropped"
    count = len(listings)
    kind = f"{category} " if category else ""
    description = f"Showed {count} {kind}trailer{'' if count == 1 else 's'}"
    return record(state, reason=RESULTS_SHOWN_REASON, description=description)
