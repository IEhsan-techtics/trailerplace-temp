"""A conversation opened by our private reply to a public Facebook comment."""
from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config, conversation_store
from src.api import comment_handoff
from src.conversation_store import load_session, seed_from_comment, session_uuid_for
from src.followup.store import _is_customer_turn
from src.graph.build import run_turn
from src.graph.nodes import greeting
from src.graph.state import from_snapshot
from src.llm.prompt import state_block
from src.tools import contact_policy

from tests.factories import turn_output

PSID = "2468013579"
DM = "Hi there! Thanks for asking about the 18 ft version. Which type are you after?"


def seed(**overrides):
    kwargs = dict(platform="facebook", intent="product_interest",
                  comment_text="do you have the 18 ft version? I want that",
                  dm_text=DM, comment_id="c_1")
    kwargs.update(overrides)
    return seed_from_comment(PSID, **kwargs)


def state_now():
    session_id = session_uuid_for(PSID)
    snapshot, conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot), conversation


def test_a_new_customer_starts_with_their_comment_and_our_dm():
    result = seed()
    state, conversation = state_now()

    assert result["new_session"] is True
    assert result["known_contact"] is False
    assert [m["role"] for m in conversation] == ["user", "assistant"]
    assert "18 ft" in conversation[0]["content"]
    assert conversation[1]["content"] == DM
    # Their answer is turn 2, so it is never treated as a first contact.
    assert state["turn_index"] == 1
    assert state["comment_origin"]["reply_turn"] == 2
    assert state["comment_origin"]["intent"] == "product_interest"


def test_the_same_comment_seeds_only_once():
    seed()
    again = seed()
    _state, conversation = state_now()
    assert again["duplicate"] is True
    assert len(conversation) == 2


def test_a_returning_customer_keeps_their_history_and_details(fake_llm):
    session_id = session_uuid_for(PSID)
    fake_llm.push(turn_output(intent="contact_info_provided", name="Dave", email="dave@x.com"))
    run_turn(session_id, "I'm Dave, dave@x.com", channel_id=PSID)
    before, history = state_now()

    result = seed(comment_id="c_2")
    state, conversation = state_now()

    assert result["new_session"] is False
    assert result["known_contact"] is True
    assert conversation[: len(history)] == history
    assert state["comment_origin"]["reply_turn"] == before["turn_index"] + 1


def test_their_answer_is_not_greeted_as_a_first_message(fake_llm):
    seed()
    fake_llm.push(turn_output())
    run_turn(session_uuid_for(PSID), "same type please", channel_id=PSID)

    seen = fake_llm.states_seen[0]
    assert seen["turn_index"] == 2
    assert not greeting.is_first_turn(seen)
    _state, conversation = state_now()
    assert not conversation[-1]["content"].startswith(greeting.OPENING)


def test_their_answer_is_a_moment_to_ask_for_missing_details(monkeypatch):
    monkeypatch.setattr(config, "settings", replace(config.settings, llm_writes_reply=True))
    seed()
    state, _ = state_now()
    state["turn_index"] += 1  # what run_turn does before the model sees it

    assert contact_policy.moment(state, turn_output()) == "comment_handoff"
    block = state_block(state)
    assert "Do NOT greet them" in block
    assert "This turn IS a moment to ask" in block

    state["turn_index"] += 1  # a later turn is an ordinary one again
    assert contact_policy.moment(state, turn_output()) is None
    assert "Do NOT greet them" not in state_block(state)


def test_a_complaint_is_framed_as_one(monkeypatch):
    monkeypatch.setattr(config, "settings", replace(config.settings, llm_writes_reply=True))
    seed(intent="complaint", comment_text="nobody called me back", comment_id="c_3")
    state, _ = state_now()
    state["turn_index"] += 1
    assert "COMPLAINT" in state_block(state)


def test_the_followup_agent_does_not_count_the_handoff_as_the_customer():
    assert not _is_customer_turn(f"{conversation_store.COMMENT_HANDOFF_MARKER}: facebook comment) hi")
    assert _is_customer_turn("do you have a dump trailer?")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(comment_handoff, "settings",
                        replace(comment_handoff.settings, comment_handoff_token="s3cret"))
    app = FastAPI()
    app.include_router(comment_handoff.router)
    return TestClient(app)


BODY = {"psid": PSID, "intent": "product_interest", "comment_id": "c_9",
        "comment_text": "price?", "dm_text": DM}


def test_the_route_refuses_a_missing_or_wrong_token(client):
    assert client.post("/internal/comment-handoff", json=BODY).status_code == 401
    assert client.post("/internal/comment-handoff", json=BODY,
                       headers={"X-Comment-Handoff-Token": "nope"}).status_code == 401


def test_the_route_seeds_the_conversation(client):
    response = client.post("/internal/comment-handoff", json=BODY,
                           headers={"X-Comment-Handoff-Token": "s3cret"})
    assert response.status_code == 200
    assert response.json()["session_id"] == session_uuid_for(PSID)
    _state, conversation = state_now()
    assert conversation[-1]["content"] == DM
