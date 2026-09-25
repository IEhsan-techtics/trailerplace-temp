"""The 5-minute rule's timer: armed while a question waits, once per category.

A customer who has chosen a category and been asked a question, then goes quiet, is shown
that category's trailers after five minutes. Each turn arms, re-arms or cancels the timer,
and it is written with the turn itself.
"""
from __future__ import annotations

from dataclasses import replace
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src import config, idle_timer

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def on(monkeypatch):
    monkeypatch.setattr(config, "settings", replace(config.settings, llm_writes_reply=True, idle_results_minutes=5))


def _state(**overrides):
    state = {"category": "Dump", "pending_slot": "haul_item", "results_categories": []}
    state.update(overrides)
    return state


def test_a_question_waiting_on_a_chosen_category_arms_it_for_five_minutes():
    plan = idle_timer.plan(_state(), None, NOW)

    assert plan == {"channel": "web", "channel_id": None, "category": "Dump", "due_at": NOW + timedelta(minutes=5)}


def test_messenger_timers_carry_the_psid_to_send_to():
    plan = idle_timer.plan(_state(), "PSID-123", NOW)

    assert plan["channel"] == "messenger" and plan["channel_id"] == "PSID-123"


def test_no_category_no_timer():
    assert idle_timer.plan(_state(category=None), None, NOW) is None


def test_no_question_waiting_no_timer():
    assert idle_timer.plan(_state(pending_slot=None), None, NOW) is None


def test_one_of_our_confirmations_waiting_arms_it_too():
    state = _state(pending_slot=None, pending_category_switch={"suggested": "Equipment"})
    assert idle_timer.plan(state, None, NOW) is not None


def test_once_trailers_were_shown_for_a_category_it_never_arms_for_it_again():
    state = _state()
    idle_timer.note_results_shown(state)

    assert idle_timer.plan(state, None, NOW) is None


def test_a_new_category_arms_it_afresh():
    state = _state(results_categories=["Dump"], category="Utility")
    assert idle_timer.plan(state, None, NOW)["category"] == "Utility"


def test_off_with_the_switch_or_at_zero_minutes(monkeypatch):
    monkeypatch.setattr(config, "settings", replace(config.settings, llm_writes_reply=False))
    assert idle_timer.plan(_state(), None, NOW) is None
    monkeypatch.setattr(config, "settings", replace(config.settings, llm_writes_reply=True, idle_results_minutes=0))
    assert idle_timer.plan(_state(), None, NOW) is None


# ---- written with the turn ----


def test_every_turn_arms_re_arms_or_cancels_the_row(fake_llm, monkeypatch):
    from src import conversation_store
    from src.graph.build import run_turn
    from src.graph.nodes import compose
    from src.llm import client
    from tests.factories import turn_output

    # The whole turn on the new path; the scripted outputs carry no reply, so compose writes it.
    monkeypatch.setattr(compose, "settings", config.settings)
    monkeypatch.setattr(client, "rewrite_reply", lambda *args: None)

    fake_llm.push(turn_output(intent="category_selection", category_mentioned="Dump"))
    run_turn("s-idle", "I need a dump trailer")
    armed = conversation_store.armed_timer("s-idle")
    assert armed and armed["category"] == "Dump" and armed["channel"] == "web"

    fake_llm.push(turn_output(intent="skip_current"))
    fake_llm.push(turn_output(intent="skip_current"))
    run_turn("s-idle", "skip it")
    run_turn("s-idle", "skip that too")
    # Nothing left to ask, so nothing waiting: the timer is cancelled.
    assert conversation_store.armed_timer("s-idle") is None


def test_mid_switch_the_timer_is_for_the_category_they_are_moving_to():
    """Live: livestock shown, then "actually I need a utility trailer instead" - the keep
    question is waiting and Livestock is still set, but Utility is what they are after."""
    state = _state(category="Livestock", pending_slot=None, results_categories=["Livestock"],
                   pending_keep_filters={"new_category": "Utility", "filters": {"length": 24}})
    assert idle_timer.plan(state, None, NOW)["category"] == "Utility"


# ---- the sweep: a due timer shows the trailers, once ----


@pytest.fixture
def new_path(monkeypatch, fake_llm):
    """The whole turn on the new path, with the model scripted and the rewrite stubbed."""
    from src.graph.nodes import compose
    from src.llm import client

    monkeypatch.setattr(compose, "settings", config.settings)
    monkeypatch.setattr(client, "rewrite_reply", lambda *args: None)
    return fake_llm


def _quiet_dump_customer(fake_llm, session="s-sweep", channel_id=None):
    from src.graph.build import run_turn
    from tests.factories import turn_output

    fake_llm.push(turn_output(intent="category_selection", category_mentioned="Dump"))
    run_turn(session, "I need a dump trailer", channel_id=channel_id)


def _later():
    return datetime.now(timezone.utc) + timedelta(minutes=6)


def test_a_due_timer_shows_the_category_and_closes_the_question(new_path, no_search):
    from src import conversation_store, idle_sweep
    from src.graph.state import from_snapshot

    _quiet_dump_customer(new_path)
    results = idle_sweep.sweep(now=_later())

    assert [r["status"] for r in results] == ["fired"]
    assert no_search[-1]["category"] == "Dump"
    snapshot, conversation, _ = conversation_store.load_session("s-sweep")
    state = from_snapshot("s-sweep", snapshot)
    assert conversation[-1]["role"] == "assistant"
    assert state["results_categories"] == ["Dump"] and state["pending_slot"] is None
    assert conversation_store.armed_timer("s-sweep") is None


def test_it_is_not_due_before_its_time(new_path):
    from src import idle_sweep

    _quiet_dump_customer(new_path)
    assert idle_sweep.sweep(now=datetime.now(timezone.utc) + timedelta(minutes=4)) == []


def test_a_second_sweep_sends_nothing_more(new_path):
    from src import idle_sweep

    _quiet_dump_customer(new_path)
    idle_sweep.sweep(now=_later())
    assert idle_sweep.sweep(now=_later() + timedelta(minutes=10)) == []


def test_the_same_timer_twice_is_one_turn(new_path):
    """A retried sweep derives the same turn id, and the stored turn answers for it."""
    from src import conversation_store
    from src.graph.build import run_idle_turn

    _quiet_dump_customer(new_path)
    turn_id = uuid.uuid4()
    assert run_idle_turn("s-sweep", "Dump", turn_id=turn_id) is not None
    assert run_idle_turn("s-sweep", "Dump", turn_id=turn_id) is None
    _, conversation, _ = conversation_store.load_session("s-sweep")
    assert sum(1 for entry in conversation if entry["role"] == "assistant") == 2


def test_the_contact_request_follows_the_trailers(new_path):
    from src import conversation_store, idle_sweep

    _quiet_dump_customer(new_path)
    idle_sweep.sweep(now=_later())
    _, conversation, _ = conversation_store.load_session("s-sweep")
    assert "email" in conversation[-1]["content"].lower() or "phone" in conversation[-1]["content"].lower()


def test_messenger_gets_the_trailers_pushed(new_path):
    from src import conversation_store, idle_sweep

    class Transport:
        def __init__(self):
            self.sent = []

        def send_text(self, psid, text):
            self.sent.append(("text", psid))

        def send_card(self, psid, element):
            self.sent.append(("card", psid))

        def send_action(self, psid, action):
            pass

    psid = "PSID-42"
    session = conversation_store.session_uuid_for(psid)
    _quiet_dump_customer(new_path, session=session, channel_id=psid)
    transport = Transport()
    results = idle_sweep.sweep(now=_later(), transport=transport)

    assert results[0]["status"] == "fired" and results[0]["delivered"]
    assert transport.sent and all(recipient == psid for _, recipient in transport.sent)


def test_a_customer_who_came_back_is_not_shown_them(new_path):
    """They answered: the question is not waiting any more, so the timer has nothing to do."""
    from src import idle_sweep
    from src.graph.build import run_idle_turn
    from tests.factories import turn_output
    from src.graph.build import run_turn

    _quiet_dump_customer(new_path)
    new_path.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s-sweep", "gravel")
    # The timer re-armed for the next question; the stale one it replaced cannot fire.
    assert run_idle_turn("s-sweep", "Utility", turn_id=uuid.uuid4()) is None


def test_the_route_wants_its_token(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api import idle as idle_api

    monkeypatch.setattr(idle_api, "settings", replace(config.settings, idle_sweep_token="s3cret"))
    app = FastAPI()
    app.include_router(idle_api.router)
    client = TestClient(app)

    assert client.post("/internal/idle-sweep").status_code == 401
    assert client.post("/internal/idle-sweep", headers={"X-Idle-Sweep-Token": "wrong"}).status_code == 401
    assert client.post("/internal/idle-sweep", headers={"X-Idle-Sweep-Token": "s3cret"}).json() == {"swept": 0, "results": []}


def test_the_page_can_fetch_what_it_has_not_drawn_with_the_cards(new_path):
    """Web chat: the idle reply comes back from GET /session/{id}/messages with its listings."""
    from fastapi.testclient import TestClient

    import main
    from src import idle_sweep

    session = str(uuid.uuid4())
    _quiet_dump_customer(new_path, session=session)
    client = TestClient(main.app)
    before = client.get(f"/session/{session}/messages").json()["count"]

    idle_sweep.sweep(now=_later())
    after = client.get(f"/session/{session}/messages", params={"after": before}).json()

    assert after["count"] == before + 1
    [idle] = after["messages"]
    assert idle["role"] == "assistant" and idle["listings"]


# ---- the clock: in the API, every minute ----


def test_the_clock_fires_due_timers_by_itself(new_path, monkeypatch):
    """No job and no request: the background thread runs the sweep on its own."""
    import time as _time

    from src import conversation_store, idle_clock, idle_sweep

    monkeypatch.setattr(config, "settings", replace(config.settings, idle_sweep_interval_seconds=10))
    monkeypatch.setattr(idle_clock, "FIRST_TICK_SECONDS", 0.05)
    real_sweep = idle_sweep.sweep
    monkeypatch.setattr(idle_sweep, "sweep", lambda: real_sweep(now=_later()))

    _quiet_dump_customer(new_path, session="s-clock")
    assert idle_clock.start()
    try:
        deadline = _time.time() + 5
        while conversation_store.armed_timer("s-clock") is not None and _time.time() < deadline:
            _time.sleep(0.05)
    finally:
        idle_clock.stop()

    assert conversation_store.armed_timer("s-clock") is None
    _, conversation, _ = conversation_store.load_session("s-clock")
    assert conversation[-1]["role"] == "assistant" and conversation[-1].get("idle")


def test_the_clock_stays_off_without_the_rule(monkeypatch):
    from src import idle_clock

    monkeypatch.setattr(config, "settings", replace(config.settings, llm_writes_reply=False))
    assert idle_clock.start() is False
