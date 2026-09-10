"""The FastAPI surface. The turn itself is tested elsewhere; this is the wiring."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.factories import turn_output


@pytest.fixture
def client():
    import main

    return TestClient(main.app)


def test_health_reports_the_configured_model(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["model"] == "gpt-5.6-luna"


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


def test_reading_an_unknown_session_is_a_404(client):
    assert client.get("/session/does-not-exist").status_code == 404


def test_reset_hands_back_a_new_session_id(client):
    body = client.post("/session/reset").json()
    assert body["session_id"]


def test_turn_status_is_readable(client):
    body = client.get("/session/abc/turn-status").json()
    assert body["session_id"] == "abc"
