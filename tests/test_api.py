"""The FastAPI surface. The turn itself is tested elsewhere; this is the wiring."""
from __future__ import annotations

import json
import uuid
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from tests.factories import turn_output


@pytest.fixture
def client():
    import main

    return TestClient(main.app)


def _with_api_key(monkeypatch, key="test-key"):
    import main

    monkeypatch.setattr(main, "settings", replace(main.settings, openai_api_key=key))


def test_health_is_ok_in_the_shape_app_py_waits_for(client, monkeypatch):
    """app.py polls until body["status"] == "ok". The old {"ok": true} kept the frontend on
    "initializing" forever."""
    _with_api_key(monkeypatch)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "gpt-5.6-luna"


def test_health_is_not_ok_without_an_api_key(client, monkeypatch):
    """Otherwise the input box opens and the first message fails."""
    _with_api_key(monkeypatch, key="")
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "error"


def test_chat_returns_a_reply_and_a_session_id(client, fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    body = client.post("/chat", json={"message": "I need a dump trailer"}).json()

    assert body["assistant_text"]
    assert body["session_id"]
    assert body["category"] == "Dump"
    assert body["usage"]["chat_completions"] == 1


def test_a_session_id_carries_the_conversation_forward(client, fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    first = client.post("/chat", json={"message": "dump trailer"}).json()

    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    second = client.post(
        "/chat", json={"message": "gravel", "session_id": first["session_id"]}
    ).json()

    assert second["category"] == "Dump"
    assert second["slots"]["haul_item"] == "gravel"


def test_an_empty_message_is_rejected_by_validation(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_an_over_long_message_is_rejected(client):
    response = client.post("/chat", json={"message": "x" * 5000})
    assert response.status_code == 413


def test_reading_a_session_shows_what_is_known(client, fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    session_id = client.post("/chat", json={"message": "dump trailer"}).json()["session_id"]

    body = client.get(f"/session/{session_id}").json()
    assert body["category"] == "Dump"
    assert "haul_item" in body["required_remaining"]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]


def test_an_unknown_session_is_a_200_that_says_it_does_not_exist(client):
    """app.py asks for its session before the first message and shows "could not be
    restored" on any error - so a brand-new chat must not be a 404."""
    response = client.get(f"/session/{uuid.uuid4()}")
    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is False
    assert body["messages"] == []
    assert body["sales_phase"] == "main"


def test_a_malformed_session_id_is_rejected_up_front(client):
    assert client.get("/session/does-not-exist").status_code == 422


def test_reset_hands_back_a_new_session_id(client):
    body = client.post("/session/reset").json()
    assert body["session_id"]


def test_turn_status_is_readable_under_the_key_app_py_reads(client):
    from src import turn_status

    turn_status.publish("abc", "Searching dump trailers...")
    try:
        body = client.get("/session/abc/turn-status").json()
    finally:
        turn_status.clear("abc")
    assert body["session_id"] == "abc"
    assert body["search_status_message"] == "Searching dump trailers..."


# --------------------------------------------------------------- app.py's request shape
def _app_payload(message, session_id):
    """What app.py actually posts - New Prompt's fields included."""
    return {
        "session_id": session_id,
        "turn_id": str(uuid.uuid4()),
        "sales_phase": "main",
        "message": message,
        "onboarding_api_messages": [],
        "customer_full_name": None,
        "customer_email": None,
        "customer_phone": None,
        "already_shown_listing_urls": [],
    }


def test_the_fields_app_py_sends_are_accepted(client, fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    response = client.post("/chat", json=_app_payload("dump trailer", str(uuid.uuid4())))
    assert response.status_code == 200
    body = response.json()
    assert body["sales_phase"] == "main"
    for key in ("customer_full_name", "customer_email", "customer_phone"):
        assert key in body


def test_contact_details_come_back_under_app_py_names(client, fake_llm):
    fake_llm.push(turn_output(intent="contact_info_provided", name="Dave", email="d@x.ai"))
    body = client.post("/chat", json=_app_payload("I'm Dave, d@x.ai", str(uuid.uuid4()))).json()
    assert body["customer_full_name"] == "Dave"
    assert body["customer_email"] == "d@x.ai"


def test_a_search_line_from_one_turn_is_gone_by_the_next(client, fake_llm):
    """Nothing used to clear it, so the UI polling during turn two saw turn one's line."""
    from src import turn_status

    session_id = str(uuid.uuid4())
    turn_status.publish(session_id, "Searching dump trailers...")
    fake_llm.push(turn_output())
    client.post("/chat", json=_app_payload("hello", session_id))
    assert turn_status.peek(session_id) is None


# ---------------------------------------------------------------------------- streaming
def _events(raw: str) -> list[tuple[str, dict]]:
    events = []
    for frame in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_the_stream_types_the_reply_out_and_ends_with_the_chat_body(client, fake_llm, monkeypatch):
    import main

    monkeypatch.setattr(main, "STREAM_DELTA_SECONDS", 0)
    monkeypatch.setattr(main, "STREAM_CHUNK_PAUSE_SECONDS", 0)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    response = client.post("/chat/stream", json=_app_payload("dump trailer", str(uuid.uuid4())))
    assert response.status_code == 200
    events = _events(response.text)
    kinds = [kind for kind, _ in events]

    assert kinds[-1] == "done"
    assert "chunk_start" in kinds and "delta" in kinds and "chunk_end" in kinds
    done = events[-1][1]
    assert done["category"] == "Dump"
    typed = "\n\n".join(body["text"] for kind, body in events if kind == "chunk_end")
    assert typed == "\n\n".join(done["chunks"])


def test_the_stream_needs_a_real_session_id_before_it_starts(client):
    response = client.post("/chat/stream", json={"message": "hi", "session_id": "nope"})
    assert response.status_code == 422


def test_streaming_can_be_switched_off_and_app_py_falls_back(client, monkeypatch):
    """app.py reads a 404 as "no streaming" and runs the turn through POST /chat."""
    import main

    monkeypatch.setattr(main, "STREAM_ENABLED", False)
    response = client.post("/chat/stream", json=_app_payload("hi", str(uuid.uuid4())))
    assert response.status_code == 404
