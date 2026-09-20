"""Facebook Messenger webhook.

Meta POSTs a customer's message here; we run the ordinary Luna turn and push the reply
back through the Send API. Three things shape every decision in this file.

1. **META RETRIES, SO NOTHING MAY RUN TWICE.** Meta expects a 200 within about twenty
   seconds and redelivers when it does not get one. A turn takes far longer than that, so
   the webhook acknowledges IMMEDIATELY and does the work on a background thread. That
   makes retries routine rather than rare - and every retry carries the same message id,
   which the unique key on chatbot_inbound_messages rejects before any work is done.

2. **A CUSTOMER SENDS ONE THOUGHT AS SEVERAL MESSAGES.** "I need a trailer", then "20ft",
   then "for hay" - often while we are still answering the first. Answered separately,
   each reply asks what the next message already said. So everything pending is answered
   as ONE turn, and a turn they interrupt is discarded and rerun with the new message
   folded in. src/inbound.py owns all of that, including the ordering across instances:
   behind a load balancer two deliveries can land on two machines, so the queue is in the
   database and only the instance holding that customer's advisory lock answers it.

3. **THE PSID IS THE CUSTOMER.** It goes in unchanged and
   ``conversation_store.session_uuid_for`` maps it onto the UUID the chatbot_* tables are
   keyed by, so the same person always resumes the same conversation with no lookup table
   to keep in step.

What the customer sees while the turn runs - typing that never lapses, the search line the
moment we start looking, then the answer paced bubble by bubble - is src/channel_delivery.py,
and how a trailer becomes a card is src/domain/cards.py. This file is the transport: it
verifies Meta's signature, turns a webhook body into messages, and knows how to call the
Send API. Nothing here decides what to say.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Request, Response

from src import inbound
from src.config import settings
from src.domain import links

logger = logging.getLogger(__name__)

router = APIRouter()

_GRAPH_URL = "https://graph.facebook.com/{version}/me/messages"

CHANNEL = "messenger"


def messenger_enabled() -> bool:
    """Both routes 404 unless this is the Messenger deployment AND it is configured.

    A page token with no app secret would serve a bot that anyone who guessed the URL
    could speak through, so the secret is part of "configured", not an optional extra.
    """
    return bool(
        settings.messenger_enabled
        and settings.messenger_app_secret
        and settings.messenger_page_access_token
    )


# ------------------------------------------------------------------ the Send API
class MessengerTransport:
    """How we actually put something in front of a Messenger customer.

    Three calls, because a card, a text bubble and a typing indicator are three different
    requests. None of them raises: a send that fails is logged, and the pieces after it
    still go out. Losing one trailer is bad; losing the closing question with it is worse.
    """

    def send_text(self, psid: str, text: str) -> None:
        if not str(text or "").strip():
            return
        self._post({
            "recipient": {"id": psid},
            "messaging_type": "RESPONSE",
            "message": {"text": text},
        })

    def send_card(self, psid: str, element: dict[str, Any]) -> None:
        """One trailer, as a generic template.

        One element at a time rather than a carousel: the reply interleaves each card with
        that trailer's own bullets, and a carousel would have to hoist every card above all
        of the text to group them.
        """
        self._post({
            "recipient": {"id": psid},
            "messaging_type": "RESPONSE",
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {"template_type": "generic", "elements": [element]},
                }
            },
        })

    def send_action(self, psid: str, action: str) -> None:
        """mark_seen / typing_on / typing_off. Cosmetic, so a failure is never fatal."""
        self._post({"recipient": {"id": psid}, "sender_action": action})

    def _post(self, payload: dict) -> None:
        url = _GRAPH_URL.format(version=settings.messenger_graph_api_version)
        try:
            response = requests.post(
                url,
                params={"access_token": settings.messenger_page_access_token},
                json=payload,
                timeout=settings.messenger_send_timeout_seconds,
            )
        except requests.RequestException:
            logger.exception("MESSENGER send failed (network)")
            return
        if response.status_code >= 400:
            # The token and the recipient are the usual culprits. Log the body, never the
            # token - it is in the query string and would end up in the log file.
            logger.error(
                "MESSENGER send rejected: status=%s body=%s",
                response.status_code, response.text[:500],
            )


# ------------------------------------------------------- duplicate suppression
# Meta's `mid` is unique per message and constant across retries, so it is the natural
# idempotency key. This cache only rejects a retry that reaches THIS instance; the check
# that matters is the unique key on chatbot_inbound_messages, which holds across instances
# and across a container being recycled.
_seen_mids: OrderedDict[str, float] = OrderedDict()
_seen_lock = threading.Lock()


def _claim_mid(mid: str) -> bool:
    """Record `mid` as being handled. False if another delivery got there first.

    Claimed BEFORE the work starts, not after, so a retry arriving while the first copy is
    still running is rejected too - which is the common case, since the retry is triggered
    by exactly the slowness that means we are still busy.
    """
    with _seen_lock:
        if mid in _seen_mids:
            return False
        _seen_mids[mid] = time.time()
        while len(_seen_mids) > max(1, settings.messenger_seen_mid_cache_size):
            _seen_mids.popitem(last=False)
        return True


def reset_seen_mids() -> None:
    """Forget every claimed message id. Tests use this between cases."""
    with _seen_lock:
        _seen_mids.clear()


# ------------------------------------------------------------------- dispatch
def _enqueue(psid: str, text: str, mid: str, sent_at: datetime) -> None:
    """Record the message, then try to become this customer's drain worker."""
    if not _claim_mid(mid):
        logger.info("MESSENGER duplicate suppressed in process: psid=%s mid=%s", psid, mid)
        return
    if not inbound.record_inbound_message(
        session_id=psid, external_id=mid, body=text, sent_at=sent_at, channel=CHANNEL
    ):
        # The unique constraint rejected it: a retry of something already queued or already
        # answered. Durable, so it holds across instances and across restarts.
        logger.info("MESSENGER duplicate suppressed at the queue: psid=%s mid=%s", psid, mid)
        return
    threading.Thread(
        target=_drain, args=(psid,), daemon=True, name=f"messenger:{psid[:12]}"
    ).start()


def _drain(psid: str) -> None:
    """Answer this customer's queue, unless another instance already is.

    The re-check after the lock is released closes the one gap in the hand-off: another
    instance can record a message and fail to take the lock in the moment between our last
    look at the queue and us letting go.
    """
    try:
        while True:
            results = inbound.drain_inbound(psid, channel=CHANNEL, transport=MessengerTransport())
            if not results or not inbound.has_pending_inbound(psid, CHANNEL):
                return
    except Exception:  # noqa: BLE001 - a webhook thread has nobody to raise to
        logger.exception("MESSENGER drain failed: psid=%s", psid)


# --------------------------------------------------------------------- routes
def _signature_is_valid(body: bytes, header: str | None) -> bool:
    """Verify X-Hub-Signature-256 over the RAW body.

    The raw bytes, not a re-serialised dict: any difference in key order or spacing changes
    the digest. compare_digest rather than ==, so the check does not leak through its timing
    how much of a forged signature was correct.
    """
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(
        settings.messenger_app_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1].strip())


@router.get("/webhooks/messenger")
def verify_webhook(request: Request) -> Response:
    """Meta's one-time subscription handshake: echo hub.challenge if the token matches."""
    if not messenger_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    params = request.query_params
    if (params.get("hub.mode") == "subscribe"
            and params.get("hub.verify_token") == settings.messenger_verify_token):
        logger.info("MESSENGER webhook verified")
        return Response(content=params.get("hub.challenge", ""), media_type="text/plain")
    logger.warning("MESSENGER webhook verification rejected")
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhooks/messenger")
async def receive_webhook(request: Request) -> Response:
    """Acknowledge in milliseconds, answer in the background.

    Returning 200 only once the turn had finished would blow Meta's twenty-second budget on
    every single message and put us in a permanent retry storm, so the ack is unconditional
    the moment the signature checks out.
    """
    if not messenger_enabled():
        raise HTTPException(status_code=404, detail="Not found")

    body = await request.body()
    if not _signature_is_valid(body, request.headers.get("X-Hub-Signature-256")):
        logger.warning("MESSENGER webhook rejected: bad signature")
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        logger.warning("MESSENGER webhook rejected: the body was not JSON")
        return Response(status_code=200)

    for psid, text, mid, sent_at in extract_messages(payload):
        _enqueue(psid, text, mid, sent_at)
    return Response(status_code=200)


# ------------------------------------------------------- reading a webhook body
def _sent_at(event: dict) -> datetime:
    """When the CUSTOMER pressed send, per Meta's millisecond timestamp.

    This is the ordering key and it has to come from Facebook: our own clock only knows
    when a delivery reached us, which for a retry is long after the message it repeats.
    Falls back to now for an event carrying no timestamp, which puts it at the back of the
    queue rather than dropping it.
    """
    raw = event.get("timestamp")
    try:
        return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return datetime.now(timezone.utc)


# Attachments that are the customer's OWN media. Their payload url is a CDN copy of the
# photo or clip, not a link they are pointing us at, so it is not a shared link.
_MEDIA_ATTACHMENT_TYPES = {"image", "video", "audio", "file", "sticker", "animated_image"}


def _attachment_url(attachment: dict) -> str:
    """The link a shared post, reel or link attachment points at, or ''."""
    payload = attachment.get("payload") or {}
    candidates = [payload.get("url"), attachment.get("url")]
    # A share can also arrive as a generic template, with the link on its elements.
    for element in payload.get("elements") or []:
        candidates.append(element.get("item_url") or element.get("url"))
        candidates.append((element.get("default_action") or {}).get("url"))
        candidates.extend(button.get("url") for button in element.get("buttons") or [])
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip().startswith(("http://", "https://")):
            return links.unwrap_redirect(candidate.strip())
    return ""


def _shared_link_text(message: dict, text: str = "") -> str:
    """A post shared with Messenger's Share button, written out as text Luna can read.

    A shared post arrives with NO ``text`` at all - only an attachment - so without this it
    is dropped and the customer gets no reply whatsoever. Written out WITH the platform
    named, so the model knows what it is looking at and the lead email can say where the
    trailer was seen.
    """
    lines: list[str] = []
    for attachment in message.get("attachments") or []:
        if (attachment.get("type") or "").lower() in _MEDIA_ATTACHMENT_TYPES:
            continue
        url = _attachment_url(attachment)
        # A pasted link arrives as text AND as a preview attachment. Say it once.
        if not url or url in text:
            continue
        platform = links.platform_of(url)
        article = "an" if (platform or "")[:1] in {"A", "E", "I", "O", "U"} else "a"
        lines.append(
            f"Shared {article} {platform} post: {url}" if platform else f"Shared a link: {url}"
        )
    return "\n".join(lines)


def extract_messages(payload: dict) -> list[tuple[str, str, str, datetime]]:
    """(psid, text, mid, sent_at) for everything in a webhook body we can answer.

    Deliberately ignored: ``is_echo`` - our OWN outgoing message handed back to us, which
    answering would turn into a conversation with ourselves - delivery and read receipts,
    and attachment-only messages with nothing readable in them (photos, stickers). A post
    shared with the Share button has no text either, but its link becomes text.
    ``postback`` IS included, because the Get Started button arrives as one.
    """
    found: list[tuple[str, str, str, datetime]] = []
    if payload.get("object") != "page":
        return found
    for entry in payload.get("entry") or []:
        for event in entry.get("messaging") or []:
            psid = ((event.get("sender") or {}).get("id") or "").strip()
            if not psid:
                continue
            message = event.get("message") or {}
            if message.get("is_echo"):
                continue
            text = (message.get("text") or "").strip()
            mid = (message.get("mid") or "").strip()
            shared = _shared_link_text(message, text)
            if shared:
                text = f"{text}\n{shared}" if text else shared
            if not text:
                postback = event.get("postback") or {}
                text = (postback.get("payload") or postback.get("title") or "").strip()
                # A postback carries no mid, so key the dedupe on what does identify it.
                mid = mid or f"postback:{psid}:{event.get('timestamp')}"
            if not text or not mid:
                if message:
                    logger.info(
                        "MESSENGER ignored, nothing to answer: psid=%s attachments=%s",
                        psid, [a.get("type") for a in message.get("attachments") or []],
                    )
                continue
            found.append((psid, text, mid, _sent_at(event)))
    return found


def turn_id_for(mid: str) -> str:
    """Meta's message id -> a turn id, for a caller that wants one outside the drain."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"trailerplace-messenger-turn:{mid}"))
