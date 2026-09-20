"""The Facebook Messenger webhook: what Meta sends us, and what we send back.

The turn itself is tested elsewhere. This is the edge - the signature check, reading a
webhook body, and the Send API calls - plus the two things that only go wrong in
production: a forged request, and Meta retrying a message we are already answering.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from src import config, inbound
from src.api import messenger

SECRET = "app-secret"
PSID = "8675309"
GALYEAN = "https://www.trailerplace.com/inventory/2026-galyean-32-cattle-trailer-015087/"


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr(
        config, "settings",
        replace(
            config.settings,
            messenger_enabled=True,
            messenger_app_secret=SECRET,
            messenger_page_access_token="page-token",
            messenger_verify_token="verify-me",
        ),
    )
    monkeypatch.setattr(messenger, "settings", config.settings)
    messenger.reset_seen_mids()
    yield
    messenger.reset_seen_mids()


@pytest.fixture
def client():
    import main

    return TestClient(main.app)


@pytest.fixture(autouse=True)
def no_threads(monkeypatch):
    """Record what would have been drained instead of starting a background turn."""
    drained = []
    monkeypatch.setattr(messenger.threading, "Thread",
                        lambda **kwargs: type("T", (), {"start": lambda _self: drained.append(kwargs["args"][0])})())
    return drained


def signed(payload: dict) -> tuple[bytes, dict]:
    body = json.dumps(payload).encode()
    digest = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return body, {"X-Hub-Signature-256": f"sha256={digest}", "Content-Type": "application/json"}


def message_event(text: str = "got any dump trailers?", mid: str = "m_1", **message) -> dict:
    return {
        "object": "page",
        "entry": [{"messaging": [{
            "sender": {"id": PSID},
            "timestamp": 1758297600000,
            "message": {"mid": mid, "text": text, **message},
        }]}],
    }


# ------------------------------------------------------------------ the handshake
def test_the_subscription_handshake_echoes_the_challenge(client):
    response = client.get("/webhooks/messenger", params={
        "hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "42",
    })
    assert response.status_code == 200 and response.text == "42"


def test_a_wrong_verify_token_is_refused(client):
    response = client.get("/webhooks/messenger", params={
        "hub.mode": "subscribe", "hub.verify_token": "guess", "hub.challenge": "42",
    })
    assert response.status_code == 403


def test_both_routes_are_invisible_when_messenger_is_off(client, monkeypatch):
    """A deployment that is not the Messenger one has no webhook surface at all."""
    monkeypatch.setattr(config, "settings", replace(config.settings, messenger_enabled=False))
    monkeypatch.setattr(messenger, "settings", config.settings)

    assert client.get("/webhooks/messenger").status_code == 404
    body, headers = signed(message_event())
    assert client.post("/webhooks/messenger", content=body, headers=headers).status_code == 404


def test_a_page_token_without_an_app_secret_is_not_configured(monkeypatch):
    """It would serve a bot that anyone who guessed the URL could speak through."""
    monkeypatch.setattr(config, "settings", replace(config.settings, messenger_app_secret=""))
    monkeypatch.setattr(messenger, "settings", config.settings)
    assert messenger.messenger_enabled() is False


# -------------------------------------------------------------------- the signature
def test_an_unsigned_request_is_refused(client):
    response = client.post("/webhooks/messenger", json=message_event())
    assert response.status_code == 403
    assert inbound.pending_inbound_batch(PSID) == [], "and nothing was queued"


def test_a_forged_signature_is_refused(client):
    body, _headers = signed(message_event())
    response = client.post(
        "/webhooks/messenger", content=body,
        headers={"X-Hub-Signature-256": "sha256=" + "0" * 64},
    )
    assert response.status_code == 403


def test_the_signature_is_checked_against_the_raw_body(client):
    """Re-serialising the dict would change the digest - key order and spacing count."""
    body, headers = signed(message_event())
    assert client.post("/webhooks/messenger", content=body, headers=headers).status_code == 200


# ---------------------------------------------------------------------- receiving
def test_a_message_is_queued_and_acknowledged_at_once(client, no_threads):
    """A turn takes far longer than Meta's ~20 s budget, so the ack cannot wait for it."""
    body, headers = signed(message_event())

    assert client.post("/webhooks/messenger", content=body, headers=headers).status_code == 200
    assert [row["body"] for row in inbound.pending_inbound_batch(PSID)] == [
        "got any dump trailers?"
    ]
    assert no_threads == [PSID], "and a drain was started for that customer"


def test_the_send_order_comes_from_facebooks_timestamp(client):
    body, headers = signed(message_event())
    client.post("/webhooks/messenger", content=body, headers=headers)

    sent_at = inbound.pending_inbound_batch(PSID)[0]["sent_at"]
    assert sent_at.year == 2025 or sent_at.timestamp() == 1758297600.0


def test_a_retry_of_the_same_message_is_not_queued_twice(client):
    body, headers = signed(message_event())
    client.post("/webhooks/messenger", content=body, headers=headers)
    client.post("/webhooks/messenger", content=body, headers=headers)

    assert len(inbound.pending_inbound_batch(PSID)) == 1


def test_our_own_reply_echoed_back_is_ignored(client):
    """Answering it would be a conversation with ourselves, forever."""
    event = message_event(text="Here are two that fit:", mid="m_echo", is_echo=True)
    body, headers = signed(event)
    client.post("/webhooks/messenger", content=body, headers=headers)

    assert inbound.pending_inbound_batch(PSID) == []


def test_a_photo_with_no_words_is_ignored(client):
    event = message_event(text="", mid="m_img", attachments=[
        {"type": "image", "payload": {"url": "https://cdn.fb/photo.jpg"}}
    ])
    body, headers = signed(event)
    client.post("/webhooks/messenger", content=body, headers=headers)

    assert inbound.pending_inbound_batch(PSID) == []


def test_the_get_started_button_is_answered(client):
    event = {"object": "page", "entry": [{"messaging": [{
        "sender": {"id": PSID}, "timestamp": 1758297600000,
        "postback": {"payload": "GET_STARTED", "title": "Get Started"},
    }]}]}
    body, headers = signed(event)
    client.post("/webhooks/messenger", content=body, headers=headers)

    assert [row["body"] for row in inbound.pending_inbound_batch(PSID)] == ["GET_STARTED"]


def test_a_body_that_is_not_json_is_acknowledged_and_dropped(client):
    digest = hmac.new(SECRET.encode(), b"not json", hashlib.sha256).hexdigest()
    response = client.post("/webhooks/messenger", content=b"not json",
                           headers={"X-Hub-Signature-256": f"sha256={digest}"})
    assert response.status_code == 200, "a retry storm over a bad body helps nobody"


# ----------------------------------------------------------------- a shared post
def test_a_shared_post_with_no_words_becomes_text_luna_can_read():
    """It arrives with NO text at all, so without this the customer gets no reply."""
    message = {"mid": "m_share", "attachments": [
        {"type": "fallback", "payload": {"url": "https://www.facebook.com/share/p/1AbCdEf/"}}
    ]}
    found = messenger.extract_messages({"object": "page", "entry": [{"messaging": [
        {"sender": {"id": PSID}, "timestamp": 1758297600000, "message": message}
    ]}]})

    assert len(found) == 1
    assert found[0][1] == "Shared a Facebook post: https://www.facebook.com/share/p/1AbCdEf/"


def test_one_of_our_listings_shared_from_facebook_arrives_as_the_listing():
    """Facebook wraps an outbound link in l.facebook.com. Unwrapped, it is the listing
    itself - which is what the lookup needs to recognise the trailer."""
    message = {"mid": "m_wrap", "attachments": [
        {"type": "fallback", "payload": {"url": f"https://l.facebook.com/l.php?u={GALYEAN}&h=AT0"}}
    ]}
    found = messenger.extract_messages({"object": "page", "entry": [{"messaging": [
        {"sender": {"id": PSID}, "timestamp": 1758297600000, "message": message}
    ]}]})

    assert found[0][1] == f"Shared a link: {GALYEAN}", "the wrapper is gone"


def test_an_instagram_share_is_named_as_one():
    message = {"mid": "m_ig", "attachments": [
        {"type": "fallback", "payload": {"url": "https://www.instagram.com/p/C9xYz/"}}
    ]}
    found = messenger.extract_messages({"object": "page", "entry": [{"messaging": [
        {"sender": {"id": PSID}, "timestamp": 1758297600000, "message": message}
    ]}]})

    assert found[0][1].startswith("Shared an Instagram post: ")


def test_a_link_pasted_as_text_is_not_repeated_by_its_preview():
    """A pasted link arrives as text AND as a preview attachment. Say it once."""
    message = {"mid": "m_paste", "text": f"is this one still available? {GALYEAN}",
               "attachments": [{"type": "fallback", "payload": {"url": GALYEAN}}]}
    found = messenger.extract_messages({"object": "page", "entry": [{"messaging": [
        {"sender": {"id": PSID}, "timestamp": 1758297600000, "message": message}
    ]}]})

    assert found[0][1].count(GALYEAN) == 1


def test_the_customers_own_photo_is_not_treated_as_a_shared_link():
    """Its payload url is a CDN copy of their photo, not a link they are pointing us at."""
    message = {"mid": "m_own", "text": "here is my old trailer",
               "attachments": [{"type": "image", "payload": {"url": "https://cdn.fb/mine.jpg"}}]}
    found = messenger.extract_messages({"object": "page", "entry": [{"messaging": [
        {"sender": {"id": PSID}, "timestamp": 1758297600000, "message": message}
    ]}]})

    assert found[0][1] == "here is my old trailer"


# --------------------------------------------------------------------- the Send API
@pytest.fixture
def posted(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        text = ""

    monkeypatch.setattr(messenger.requests, "post",
                        lambda url, **kwargs: calls.append((url, kwargs)) or Response())
    return calls


def test_a_trailer_goes_out_as_a_generic_template(posted):
    messenger.MessengerTransport().send_card(PSID, {"title": "2026 Diamond C Dump"})

    _url, kwargs = posted[0]
    payload = kwargs["json"]["message"]["attachment"]["payload"]
    assert payload["template_type"] == "generic"
    assert payload["elements"] == [{"title": "2026 Diamond C Dump"}], "one element, not a carousel"


def test_a_text_bubble_goes_out_as_a_message(posted):
    messenger.MessengerTransport().send_text(PSID, "Here are two that fit:")

    assert posted[0][1]["json"] == {
        "recipient": {"id": PSID}, "messaging_type": "RESPONSE",
        "message": {"text": "Here are two that fit:"},
    }


def test_an_empty_bubble_is_not_sent_at_all(posted):
    messenger.MessengerTransport().send_text(PSID, "   ")
    assert posted == []


def test_typing_goes_out_as_a_sender_action(posted):
    messenger.MessengerTransport().send_action(PSID, "typing_on")
    assert posted[0][1]["json"] == {"recipient": {"id": PSID}, "sender_action": "typing_on"}


def test_the_page_token_is_sent_as_a_parameter_never_in_the_body(posted):
    messenger.MessengerTransport().send_text(PSID, "hello")
    _url, kwargs = posted[0]
    assert kwargs["params"] == {"access_token": "page-token"}
    assert "page-token" not in json.dumps(kwargs["json"])


def test_a_send_that_fails_does_not_raise(monkeypatch):
    """The pieces after it still have to go out."""
    def boom(*_args, **_kwargs):
        raise messenger.requests.RequestException("network down")

    monkeypatch.setattr(messenger.requests, "post", boom)
    messenger.MessengerTransport().send_text(PSID, "hello")  # no exception


def test_a_rejection_from_meta_does_not_raise(monkeypatch):
    class Rejected:
        status_code = 400
        text = '{"error": {"message": "Invalid recipient"}}'

    monkeypatch.setattr(messenger.requests, "post", lambda *a, **k: Rejected())
    messenger.MessengerTransport().send_text(PSID, "hello")  # no exception
