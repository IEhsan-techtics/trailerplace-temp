"""The follow-up agent's database work: who is due, and what was done about them.

Every read here is one query per table for the whole run, not one per customer: the job wakes
once an hour and looks at a day's worth of Messenger conversations at most.

A silence is anchored on the customer's LAST TURN. Our replies after it - the reply to that
turn, a 5-minute idle turn, follow-up 1 - are all part of the same silence, and a new message
from them is a new anchor, which is what restarts the cycle.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from src import db
from src.config import settings
from src.conversation_store import COMMENT_HANDOFF_MARKER
from src.db_models import (
    ChatbotConversation,
    ChatbotFollowup,
    ChatbotIdleTimer,
    ChatbotInboundMessage,
    ChatbotLead,
    ChatbotTurn,
)

logger = logging.getLogger(__name__)

# How a follow-up turn is recorded in chatbot_turns.request_message. The prefix is how a later
# run tells our follow-ups apart from the customer's own turns.
FOLLOWUP_MARKER = "(follow-up"

# The idle turn's request_message (src/graph/build.py IDLE_MESSAGE): ours, not the customer's.
_IDLE_MESSAGE = "(The customer has not replied for a few minutes.)"

_TURN_NAMESPACE = uuid.UUID("0c6f7f43-9a51-4f1e-8d0a-5b2a1f0e7c11")

# Slack on the gap between follow-up 1 and 2, for the minute a run takes to get round to sending.
_RUN_MARGIN = timedelta(minutes=10)


@dataclass
class Candidate:
    session_id: str
    psid: str
    anchor_turn_id: uuid.UUID
    attempt: int
    last_bot_at: datetime
    last_customer_at: datetime
    conversation: list[dict[str, Any]]
    state: dict[str, Any]
    contact: dict[str, Any] = field(default_factory=dict)

    def hours_silent(self, now: datetime) -> float:
        return (now - self.last_bot_at).total_seconds() / 3600


def marker(attempt: int, hours: float) -> str:
    return f"{FOLLOWUP_MARKER} {attempt}: no reply for {hours:.0f} hours)"


def followup_turn_id(session_id: str, anchor: uuid.UUID, attempt: int) -> uuid.UUID:
    return uuid.uuid5(_TURN_NAMESPACE, f"{session_id}|{anchor}|{attempt}")


def _is_followup(request_message: str) -> bool:
    return str(request_message or "").startswith(FOLLOWUP_MARKER)


def _is_comment_handoff(request_message: str) -> bool:
    return str(request_message or "").startswith(COMMENT_HANDOFF_MARKER)


def _is_customer_turn(request_message: str) -> bool:
    text = str(request_message or "")
    return not (_is_followup(text) or text == _IDLE_MESSAGE or _is_comment_handoff(text))


def find_candidates(now: datetime, *, ignore_timing: bool = False, lookback_hours: float | None = None) -> list[Candidate]:
    """Messenger conversations owed a follow-up right now, oldest silence first.

    ``ignore_timing``: the preview script's mode - every quiet conversation in the lookback is
    returned as if attempt 1 were due, so the decisions can be read before anything is live.
    """
    lookback = timedelta(hours=lookback_hours or settings.followup_window_hours + 1)
    with db.get_session_factory()() as sql:
        rows = sql.execute(
            select(ChatbotConversation, ChatbotLead)
            .join(ChatbotLead, ChatbotLead.lead_id == ChatbotConversation.lead_id)
            .where(ChatbotLead.psid.is_not(None), ChatbotConversation.updated_at > now - lookback)
        ).all()
        if not rows:
            return []
        session_ids = [conv.session_id for conv, _ in rows]
        psids = [lead.psid for _, lead in rows]

        turns: dict[uuid.UUID, list[ChatbotTurn]] = {}
        for turn in sql.execute(
            select(ChatbotTurn).where(ChatbotTurn.session_id.in_(session_ids)).order_by(ChatbotTurn.created_at)
        ).scalars():
            turns.setdefault(turn.session_id, []).append(turn)

        inbound = {
            psid: (last_sent, pending)
            for psid, last_sent, pending in sql.execute(
                select(
                    ChatbotInboundMessage.session_id,
                    func.max(ChatbotInboundMessage.sent_at),
                    func.count().filter(ChatbotInboundMessage.status == "pending"),
                )
                .where(ChatbotInboundMessage.session_id.in_(psids))
                .group_by(ChatbotInboundMessage.session_id)
            ).all()
        }
        timers = {
            session_id: status
            for session_id, status in sql.execute(
                select(ChatbotIdleTimer.session_id, ChatbotIdleTimer.status)
                .where(ChatbotIdleTimer.session_id.in_(session_ids))
            ).all()
        }
        done: dict[tuple[uuid.UUID, uuid.UUID], dict[int, str]] = {}
        sent_at: dict[tuple[uuid.UUID, uuid.UUID], datetime] = {}
        for row in sql.execute(
            select(ChatbotFollowup).where(ChatbotFollowup.session_id.in_(session_ids))
        ).scalars():
            done.setdefault((row.session_id, row.anchor_turn_id), {})[row.attempt] = row.status
            if row.attempt == 1 and row.sent_at:
                sent_at[(row.session_id, row.anchor_turn_id)] = row.sent_at

    window = timedelta(hours=settings.followup_window_hours)
    first = timedelta(hours=settings.followup_first_after_hours)
    second = timedelta(hours=settings.followup_second_after_hours)
    found: list[Candidate] = []
    for conv, lead in rows:
        if settings.followup_only_psid and lead.psid != settings.followup_only_psid:
            continue  # a test run, limited to one account
        history = turns.get(conv.session_id) or []
        customer_turns = [t for t in history if _is_customer_turn(t.request_message)]
        if not customer_turns:
            continue
        anchor = customer_turns[-1]
        ours = [t for t in history if not _is_followup(t.request_message)]
        if ours and _is_comment_handoff(ours[-1].request_message):
            # Our last word is the private reply to their comment. Meta allows nothing more
            # until they answer it, so there is nobody here to follow up with yet.
            continue
        last_bot_at = ours[-1].created_at
        last_sent, pending = inbound.get(lead.psid, (None, 0))
        last_customer_at = last_sent or anchor.created_at

        if pending:
            continue  # they wrote in; the bot answers them
        if timers.get(conv.session_id) in ("armed", "firing"):
            continue  # the 5-minute rule goes first
        if now - last_customer_at >= window and not ignore_timing:
            continue  # outside Facebook's 24-hour window

        attempts = done.get((conv.session_id, anchor.turn_id), {})
        silent = now - last_bot_at
        if ignore_timing:
            attempt = 1
        elif 1 not in attempts:
            if silent < first:
                continue
            attempt = 1
        elif attempts.get(1) == "sent" and 2 not in attempts:
            if silent < second:
                continue
            # The same gap after follow-up 1 as before it. Measured from our last message alone,
            # a customer already 4 h quiet when follow-up 1 went out (the 24-hour backfill on
            # the first live run) would get follow-up 2 an hour later, on the next run.
            # The margin: a run sends a minute or so after the hour, and without it two hours
            # later would read as 1 h 59 m and slip to the run after.
            first_sent = sent_at.get((conv.session_id, anchor.turn_id))
            if first_sent and now - first_sent < second - first - _RUN_MARGIN:
                continue
            attempt = 2
        else:
            continue

        found.append(Candidate(
            session_id=str(conv.session_id),
            psid=lead.psid,
            anchor_turn_id=anchor.turn_id,
            attempt=attempt,
            last_bot_at=last_bot_at,
            last_customer_at=last_customer_at,
            conversation=[m for m in (conv.conversation or []) if m.get("role") in ("user", "assistant")],
            state=dict(conv.state_snapshot or {}),
            contact={"name": lead.name, "email": lead.email, "phone": lead.phone_number},
        ))
    found.sort(key=lambda c: c.last_bot_at)
    return found


def claim(candidate: Candidate) -> uuid.UUID | None:
    """Write the decision row first. None when another run already holds this attempt."""
    row_id = uuid.uuid4()
    with db.get_session_factory()() as sql:
        sql.add(ChatbotFollowup(
            id=row_id,
            session_id=uuid.UUID(candidate.session_id),
            psid=candidate.psid,
            anchor_turn_id=candidate.anchor_turn_id,
            attempt=candidate.attempt,
            status="deciding",
        ))
        try:
            sql.commit()
        except IntegrityError:
            sql.rollback()
            return None
    return row_id


def update(row_id: uuid.UUID, **fields: Any) -> None:
    with db.get_session_factory()() as sql:
        row = sql.get(ChatbotFollowup, row_id)
        if row is None:
            return
        for name, value in fields.items():
            setattr(row, name, value)
        sql.commit()


def release(row_id: uuid.UUID) -> None:
    """Give the attempt back, so the next run can try again."""
    with db.get_session_factory()() as sql:
        row = sql.get(ChatbotFollowup, row_id)
        if row is not None:
            sql.delete(row)
            sql.commit()


def record_sent(row_id: uuid.UUID, candidate: Candidate, message: str, now: datetime) -> None:
    """The follow-up went out: mark it, and put it in the conversation like any reply.

    One transaction, so the log and the transcript can never disagree. The bot's next turn
    reads it from the conversation, and the chat link shows it. The state is not touched -
    a follow-up answers nothing and asks nothing the state needs to know about.
    """
    session_uuid = uuid.UUID(candidate.session_id)
    hours = candidate.hours_silent(now)
    with db.get_session_factory()() as sql:
        row = sql.get(ChatbotFollowup, row_id)
        row.status = "sent"
        row.sent_at = now
        row.message = message

        conv = sql.get(ChatbotConversation, session_uuid, with_for_update=True)
        conv.conversation = list(conv.conversation or []) + [
            {"role": "assistant", "content": message, "followup": candidate.attempt}
        ]
        conv.state_version = (conv.state_version or 0) + 1
        conv.updated_at = now
        sql.add(ChatbotTurn(
            session_id=session_uuid,
            turn_id=followup_turn_id(candidate.session_id, candidate.anchor_turn_id, candidate.attempt),
            request_message=marker(candidate.attempt, hours),
            response={"assistant_text": message, "listings": [], "followup": candidate.attempt},
        ))
        sql.commit()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
