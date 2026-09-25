"""The 5-minute rule's timer: armed while a question waits, once per category.

A customer who has chosen a category and been asked a question, then goes quiet, is shown
that category's trailers after five minutes. Each turn arms, re-arms or cancels the timer,
and it is written with the turn itself.
"""
from __future__ import annotations

from dataclasses import replace
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
