"""The inbound queue: a push channel's messages, answered once each and in order.

Web chat needs none of this. The browser holds its request open, so there is exactly one
delivery, one turn and one reply, and the reply goes back on the connection that asked
for it.

A push channel - Messenger, or anything sitting in front of Luna that retries - breaks
all three of those assumptions:

* it **redelivers** a message it did not hear back about quickly enough, and on Azure the
  container that was mid-turn may well be gone by then;
* it can land two messages the customer sent seconds apart on **two instances**, with
  nothing to say which came first;
* a customer who sends three short messages in a row wants **one reply that read all
  three**, not three replies that each ignore the others.

So a delivery is recorded here and acknowledged at once. Whichever instance wins the
per-customer lock then drains the queue: it takes everything pending, oldest first, joins
it into one message, runs one turn, and marks the lot answered. Duplicates never get that
far - the unique key on (channel, external_id) rejects them on the way in.

Everything works with or without Postgres. Without it the same semantics run against a
process-local list, which is what the tests exercise and what a developer gets locally.
"""
from __future__ import annotations

import logging
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from sqlalchemy import func, select

from src import conversation_store, db
from src.db_models import ChatbotInboundMessage

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL = "messenger"

# How many times a reply may be thrown away because the customer carried on typing while
# it was being written. After this the next answer goes out regardless - otherwise someone
# who never stops typing would never be answered at all.
MAX_REGENERATIONS = 3

# A deliberately different key from anything a turn locks on. The drain lock is held across
# several transactions - one per turn - so it must not be a transaction-scoped lock, or the
# drain would block on itself the moment a turn began.
_DRAIN_LOCK_PREFIX = "inbound-drain:"

# The in-memory backend: the same semantics without Postgres. A list of row dicts, plus the
# set of customers currently being drained. Guarded by one lock, because the point of the
# drain lock is that two threads cannot both hold it.
_MEMORY: list[dict[str, Any]] = []
_MEMORY_DRAINING: set[str] = set()
_MEMORY_GUARD = threading.Lock()


def _enabled() -> bool:
    return conversation_store.persistence_enabled()


def reset_memory() -> None:
    """Drop the in-memory queue. Tests use this between cases."""
    with _MEMORY_GUARD:
        _MEMORY.clear()
        _MEMORY_DRAINING.clear()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- recording
def record_inbound_message(
    *,
    session_id: str,
    external_id: str,
    body: str,
    sent_at: datetime | None = None,
    channel: str = DEFAULT_CHANNEL,
) -> bool:
    """Record one delivery. False means we have already seen this exact message.

    The duplicate answer comes from the unique constraint rather than a read-then-write, so
    two instances handed the same retry at the same moment cannot both decide it is new.
    """
    sent_at = sent_at or _now()
    if not _enabled():
        with _MEMORY_GUARD:
            if any(row["channel"] == channel and row["external_id"] == external_id for row in _MEMORY):
                return False
            _MEMORY.append({
                "message_id": uuid.uuid4(), "channel": channel, "session_id": session_id,
                "external_id": external_id, "sent_at": sent_at, "created_at": _now(),
                "body": body, "status": "pending", "turn_id": None, "last_error": None,
                "answered_at": None,
            })
        return True

    from sqlalchemy.exc import IntegrityError

    db.ensure_schema()
    with db.get_session_factory()() as sql:
        sql.add(ChatbotInboundMessage(
            channel=channel, session_id=session_id, external_id=external_id,
            body=body, sent_at=sent_at,
        ))
        try:
            sql.commit()
        except IntegrityError:
            sql.rollback()
            logger.info("INBOUND duplicate ignored: channel=%s external_id=%s", channel, external_id)
            return False
    return True


# ------------------------------------------------------------------------------ the lock
@contextmanager
def inbound_drain_lock(session_id: str, channel: str = DEFAULT_CHANNEL):
    """The exclusive right to answer this customer, held across every instance.

    Yields False rather than waiting when someone else holds it. That instance drains until
    the queue is empty, so it will pick up whatever we have just recorded; blocking here
    would only pile up threads waiting to do nothing.

    A session-level advisory lock, not a transaction one, because it has to span several
    transactions. Released in the finally - and by Postgres itself if the process dies,
    which on a serverless container is not a hypothetical.
    """
    key_text = f"{_DRAIN_LOCK_PREFIX}{channel}:{session_id}"
    if not _enabled():
        with _MEMORY_GUARD:
            acquired = key_text not in _MEMORY_DRAINING
            if acquired:
                _MEMORY_DRAINING.add(key_text)
        try:
            yield acquired
        finally:
            if acquired:
                with _MEMORY_GUARD:
                    _MEMORY_DRAINING.discard(key_text)
        return

    key = func.hashtext(key_text)
    with db.get_session_factory()() as sql:
        acquired = bool(sql.execute(select(func.pg_try_advisory_lock(key))).scalar())
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    sql.execute(select(func.pg_advisory_unlock(key)))
                    sql.commit()
                except Exception:  # noqa: BLE001 - the lock dies with the connection anyway
                    logger.exception("INBOUND lock release failed: session=%s", session_id)


# ------------------------------------------------------------------------------- reading
def _as_row(row: Any) -> dict[str, Any]:
    return {
        "message_id": row.message_id, "session_id": row.session_id,
        "external_id": row.external_id, "body": row.body, "sent_at": row.sent_at,
    }


def pending_inbound_batch(session_id: str, channel: str = DEFAULT_CHANNEL) -> list[dict[str, Any]]:
    """Everything this customer is still waiting on, oldest first.

    A batch, not one message: anything that arrived while we were busy is answered together
    as a single turn, so they get one reply that read all of it.

    Ordered by sent_at - when they pressed send - not by created_at, which only records when
    the delivery happened to reach us.
    """
    if not _enabled():
        with _MEMORY_GUARD:
            rows = [
                dict(row) for row in _MEMORY
                if row["session_id"] == session_id and row["channel"] == channel
                and row["status"] == "pending"
            ]
        rows.sort(key=lambda row: (row["sent_at"], row["created_at"]))
        return [
            {key: row[key] for key in ("message_id", "session_id", "external_id", "body", "sent_at")}
            for row in rows
        ]

    with db.get_session_factory()() as sql:
        rows = sql.execute(
            select(ChatbotInboundMessage)
            .where(
                ChatbotInboundMessage.session_id == session_id,
                ChatbotInboundMessage.channel == channel,
                ChatbotInboundMessage.status == "pending",
            )
            .order_by(ChatbotInboundMessage.sent_at, ChatbotInboundMessage.created_at)
        ).scalars().all()
        return [_as_row(row) for row in rows]


def has_inbound_beyond(
    session_id: str, known_ids: Iterable[Any], channel: str = DEFAULT_CHANNEL
) -> bool:
    """Has the customer sent something we were NOT already answering?

    The question a finished turn asks about its own reply. True means they carried on while
    we were writing, so the reply is about to be out of date and is better not sent - the
    next pass round the drain answers all of it together instead.
    """
    known = list(known_ids or [])
    if not _enabled():
        with _MEMORY_GUARD:
            return any(
                row["session_id"] == session_id and row["channel"] == channel
                and row["status"] == "pending" and row["message_id"] not in known
                for row in _MEMORY
            )

    with db.get_session_factory()() as sql:
        query = select(ChatbotInboundMessage.message_id).where(
            ChatbotInboundMessage.session_id == session_id,
            ChatbotInboundMessage.channel == channel,
            ChatbotInboundMessage.status == "pending",
        )
        if known:
            query = query.where(ChatbotInboundMessage.message_id.notin_(known))
        return sql.execute(query.limit(1)).first() is not None


def has_pending_inbound(session_id: str, channel: str = DEFAULT_CHANNEL) -> bool:
    """Used after the drain lock is released, to catch a message that arrived in the gap."""
    return has_inbound_beyond(session_id, [], channel)


# --------------------------------------------------------------------------- closing off
def mark_inbound_answered(
    message_ids: Iterable[Any], *, turn_id: Any = None, error: str | None = None
) -> None:
    """Close off every message the turn covered, in one write.

    A failure is marked answered WITH the error rather than left pending. A message that
    cannot be answered would otherwise sit at the head of the queue and block everything
    the customer says after it, for good.
    """
    ids = list(message_ids or [])
    if not ids:
        return
    answered_at = _now()
    if not _enabled():
        with _MEMORY_GUARD:
            for row in _MEMORY:
                if row["message_id"] in ids:
                    row["status"] = "done"
                    row["answered_at"] = answered_at
                    if turn_id is not None:
                        row["turn_id"] = turn_id
                    if error:
                        row["last_error"] = error[:2000]
        return

    with db.get_session_factory()() as sql:
        for message_id in ids:
            row = sql.get(ChatbotInboundMessage, message_id)
            if row is None:
                continue
            row.status = "done"
            row.answered_at = answered_at
            if turn_id is not None:
                row.turn_id = conversation_store.as_session_uuid(turn_id)
            if error:
                row.last_error = error[:2000]
        sql.commit()


# ------------------------------------------------------------------------------ the drain
def combine_messages(bodies: Iterable[str]) -> str:
    """Several messages sent in a row, as the one message the turn reads.

    Newline-joined and nothing more: they are the customer's own consecutive lines, and a
    prefix like "they also said" would be words we put in their mouth for the model to read
    back as theirs.
    """
    return "\n".join(str(body).strip() for body in bodies if str(body or "").strip())


def turn_id_for(batch: list[dict[str, Any]]) -> uuid.UUID:
    """A turn id derived from the messages it answers, so a replay lands on the same turn.

    This is what makes the turn idempotent. If this exact batch is drained twice - a
    container that died between the turn and the acknowledgement, say - the second pass
    recognises the turn as already handled and returns the stored reply, instead of paying
    for another one and sending the team a second email about it.
    """
    key = "|".join(str(row["external_id"]) for row in batch)
    return uuid.uuid5(uuid.NAMESPACE_URL, f"trailerplace-inbound-turn:{key}")


def drain_inbound(
    session_id: str,
    *,
    channel: str = DEFAULT_CHANNEL,
    answer: Callable[..., dict[str, Any]] | None = None,
    transport: Any = None,
) -> list[dict[str, Any]]:
    """Answer everything this customer is waiting on, oldest first. Never raises.

    Returns one result per turn it ran, each the payload ``run_turn`` returns plus:

    ``message_ids``  the inbound rows that turn covered;
    ``turn_id``      the id it was recorded under;
    ``sends``        the reply already rendered for this channel;
    ``delivered``    how many of those pieces actually went out, when a transport was given.

    A turn the customer interrupts produces NO result: it is discarded whole and rerun with
    their newer message included, so only the reply that read everything they said is ever
    returned or sent.

    Pass a ``transport`` (see src/channel_delivery.py) and the whole exchange is handled:
    the customer is shown typing while the turn runs, the search line reaches them the
    moment we start looking rather than with the results, and the answer arrives paced
    bubble by bubble. Without a transport nothing is sent and the caller does the delivering.

    An empty list means either that there was nothing to do, or that another instance holds
    the lock and is doing it. Either way there is nothing for the caller to send.
    """
    if answer is None:
        from src.graph.build import run_turn as answer  # noqa: N813 - the real turn

    results: list[dict[str, Any]] = []
    with inbound_drain_lock(session_id, channel) as acquired:
        if not acquired:
            logger.info("INBOUND drain skipped, another instance holds it: session=%s", session_id)
            return results

        attempt = 0
        while True:
            batch = pending_inbound_batch(session_id, channel)
            if not batch:
                return results

            ids = [row["message_id"] for row in batch]
            turn_id = turn_id_for(batch)
            message = combine_messages(row["body"] for row in batch)
            if not message:
                # Nothing to answer - an attachment-only delivery, say. Closed off rather
                # than left to block everything queued behind it.
                mark_inbound_answered(ids, error="empty message")
                continue

            # The turn's own session id: the graph publishes its search line under this,
            # not under the channel's raw identity.
            turn_session = conversation_store.session_uuid_for(session_id)

            # A reply to a question they have already followed up on is worse than no reply
            # at all, so a turn they interrupt is thrown away and rerun with what they went
            # on to say folded in. On the last attempt the check is switched off and the
            # answer goes out, so someone typing continuously is still answered.
            last_attempt = attempt >= MAX_REGENERATIONS
            if last_attempt:
                logger.info(
                    "INBOUND answering after %d interruptions: session=%s", attempt, session_id
                )

            def said_more(_ids=tuple(ids)) -> bool:
                return has_inbound_beyond(session_id, _ids, channel)

            try:
                with _keep_alive(turn_session, transport):
                    result = answer(
                        turn_session, message, turn_id=turn_id,
                        abandon_if=None if last_attempt else said_more,
                    )
            except Exception as exc:  # noqa: BLE001 - one bad message must not wedge the queue
                logger.exception("INBOUND turn failed: session=%s messages=%d", session_id, len(ids))
                mark_inbound_answered(ids, turn_id=turn_id, error=f"{type(exc).__name__}: {exc}")
                attempt = 0
                continue

            if result.get("abandoned"):
                # Nothing was written and nothing was sent. The messages stay pending, so
                # the next pass picks them up together with whatever arrived.
                attempt += 1
                continue
            attempt = 0

            mark_inbound_answered(ids, turn_id=turn_id)
            # Rendered for the channel it is going to, so a webhook only has to loop and
            # send. Messenger draws no markdown, so a trailer has to become an actual card
            # - see src/domain/cards.py.
            sends = _sends_for(channel, result)
            delivered = None
            if transport is not None and sends:
                from src import channel_delivery

                delivered = channel_delivery.deliver(session_id, sends, transport)
            results.append({
                **result,
                "message_ids": ids,
                "turn_id": str(turn_id),
                "sends": sends,
                "delivered": delivered,
            })


@contextmanager
def _keep_alive(turn_session: str, transport: Any):
    """Show them we are on it while the turn runs. Nothing at all without a transport."""
    if transport is None:
        yield None
        return
    from src.channel_delivery import TurnKeepAlive

    with TurnKeepAlive(turn_session, transport) as alive:
        yield alive


def _sends_for(channel: str, result: dict[str, Any]) -> list[tuple[str, Any]] | None:
    """The turn's reply, rendered for the channel that asked for it. None if it needs none."""
    if channel != DEFAULT_CHANNEL:
        return None
    from src.config import settings
    from src.domain.cards import messenger_sends

    return messenger_sends(
        result.get("assistant_text") or "",
        result.get("listings") or [],
        cards_enabled=settings.messenger_listing_cards,
    )
