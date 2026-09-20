"""A thumb or a note against one of Luna's replies.

Stored on the message itself inside the transcript, not in a table of its own, so the
reply and the verdict on it are read back together and can never drift apart.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import conversation_store
from src.conversation_store import load_session, save_user_feedback
from src.graph.build import run_turn

from tests.factories import turn_output

SESSION = "b1a7c0de-0000-4000-8000-000000000001"


@pytest.fixture
def client():
    import main

    return TestClient(main.app)


def two_replies(fake_llm, session_id=SESSION):
    fake_llm.push(turn_output())
    run_turn(session_id, "hi")
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn(session_id, "dump trailer")


def transcript(session_id=SESSION):
    _snapshot, conversation, _lead = load_session(session_id)
    return conversation


# ------------------------------------------------------------------------- the store
def test_the_verdict_lands_on_the_reply_it_was_given_about(fake_llm):
    """turn_idx counts REPLIES. The transcript alternates user and assistant, so a raw
    index would land on the wrong message - half the time on the customer's own."""
    two_replies(fake_llm)

    assert save_user_feedback(SESSION, 1, "spot on", rating="up") is True

    messages = transcript()
    assert messages[3]["role"] == "assistant"
    assert messages[3]["feedback"] == "spot on"
    assert messages[3]["feedback_rating"] == "up"
    assert "feedback" not in messages[1], "the first reply is untouched"


def test_a_thumb_alone_does_not_erase_the_note(fake_llm):
    two_replies(fake_llm)
    save_user_feedback(SESSION, 0, "asked the wrong thing", rating="down")

    save_user_feedback(SESSION, 0, rating="up")

    assert transcript()[1]["feedback"] == "asked the wrong thing"
    assert transcript()[1]["feedback_rating"] == "up"


def test_a_note_alone_does_not_erase_the_thumb(fake_llm):
    two_replies(fake_llm)
    save_user_feedback(SESSION, 0, rating="down")

    save_user_feedback(SESSION, 0, "it missed the size I gave")

    assert transcript()[1]["feedback_rating"] == "down"
    assert transcript()[1]["feedback"] == "it missed the size I gave"


def test_a_verdict_on_a_reply_that_does_not_exist_is_dropped(fake_llm):
    two_replies(fake_llm)
    assert save_user_feedback(SESSION, 9, "???", rating="up") is False


def test_a_verdict_on_a_session_we_never_saw_is_dropped():
    assert save_user_feedback("nobody", 0, "hello", rating="up") is False


def test_only_a_real_rating_is_accepted(fake_llm):
    two_replies(fake_llm)
    with pytest.raises(ValueError):
        save_user_feedback(SESSION, 0, rating="sideways")


def test_a_verdict_survives_the_reload(fake_llm):
    two_replies(fake_llm)
    save_user_feedback(SESSION, 1, "good", rating="up")

    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn(SESSION, "gravel")

    assert transcript()[3]["feedback_rating"] == "up", "a later turn does not overwrite it"


# --------------------------------------------------------------------------- the API
def test_the_api_accepts_a_thumb(client, fake_llm, monkeypatch):
    """It used to be reachable only from app.py, in-process - so any other frontend had
    nowhere to send one."""
    recorded = []
    monkeypatch.setattr(
        conversation_store, "enqueue_save_user_feedback",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )

    response = client.post(f"/session/{SESSION}/feedback", json={"turn_idx": 1, "rating": "up"})

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert recorded[0][0][:2] == (SESSION, 1)
    assert recorded[0][1]["rating"] == "up"


def test_the_api_rejects_an_empty_verdict(client):
    response = client.post(f"/session/{SESSION}/feedback", json={"turn_idx": 0})
    assert response.status_code == 422


def test_the_api_rejects_a_rating_that_is_not_one(client):
    response = client.post(f"/session/{SESSION}/feedback", json={"turn_idx": 0, "rating": "meh"})
    assert response.status_code == 422


def test_the_api_rejects_a_session_id_that_is_not_one(client):
    response = client.post("/session/not-a-uuid/feedback", json={"turn_idx": 0, "rating": "up"})
    assert response.status_code == 422
