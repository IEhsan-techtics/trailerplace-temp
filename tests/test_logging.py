"""What a turn leaves behind in the logs: a JSON line for the machines, a block for a person.

The point of both is the support ticket three days later - "why did it ask that?" - which
can only be answered from what was written down at the time. So the test that matters most
here is the boring one: neither of them may ever raise.
"""
from __future__ import annotations

import json
import logging

import pytest

from src import conversation_log, turn_log
from src.graph.build import run_turn
from src.graph.state import new_state

from tests.factories import complete_welcome, turn_output


@pytest.fixture
def captured(caplog):
    caplog.set_level(logging.INFO)
    return caplog


# ------------------------------------------------------------------- the machine feed
def test_a_turn_records_what_it_cost_and_what_it_ran(fake_llm, captured):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    records = [rec for rec in captured.records if getattr(rec, "turn", None)]
    assert records, "every turn writes one"
    payload = records[-1].turn
    assert payload["event"] == "chat_turn"
    assert payload["session_id"] == "s1"
    assert payload["intent"] == "category_selection"
    assert payload["category"] == "Dump"
    assert payload["turn_id"] and payload["turn_index"] == 2
    assert payload["latency_ms"] > 0
    assert payload["llm_calls"]["chat_completions"] == 1


def test_the_tools_that_ran_are_named():
    assert turn_log.tools_fired({"search_ran": True, "result_count": 5}) == ["search"]
    assert turn_log.tools_fired({"inventory_lookup_ran": True}) == ["inventory_lookup"]
    assert turn_log.tools_fired({"outbox_events": [{}]}) == ["email"]
    assert turn_log.tools_fired({}) == []


def test_the_record_is_json(tmp_path, monkeypatch):
    """scripts/cost_report.py json.loads every line of the file, so nothing else may be on
    one."""
    from dataclasses import replace

    from src import config

    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr(config, "settings", replace(config.settings, turn_log_path=str(path)))
    turn_log.reset_for_tests()
    try:
        turn_log.log_turn(
            session_id="s1", turn_id="t1", intent="faq", category="Dump",
            latency_ms=12.3456, turn_outcome={"search_ran": True}, usage=None,
        )
    finally:
        turn_log.reset_for_tests()

    line = json.loads(path.read_text(encoding="utf-8").strip())
    assert line["latency_ms"] == 12.35 and line["tools_fired"] == ["search"]


def test_a_broken_record_costs_the_customer_nothing(captured):
    class Exploding:
        def as_dict(self):
            raise RuntimeError("nope")

    assert turn_log.log_turn(
        session_id="s1", turn_id="t1", intent=None, category=None,
        latency_ms=1.0, turn_outcome={}, usage=Exploding(),
    ) == {}


# ---------------------------------------------------------------------- the human block
def test_the_block_holds_what_a_support_ticket_would_ask(fake_llm, captured):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "I need a dump trailer")

    blocks = [rec.getMessage() for rec in captured.records
              if rec.name == conversation_log.CONVERSATION_LOGGER_NAME]
    assert blocks, "every turn writes one"
    block = blocks[-1]
    assert "USER: I need a dump trailer" in block
    assert "WHAT LUNA READ:" in block and "intent: category_selection" in block
    assert "STATE AFTER THE TURN:" in block and "category: Dump" in block
    assert "pending_question: haul_item" in block, "which question is open, and why"


def test_the_block_shows_an_axle_question_being_held_open():
    """The state lines have to name the open confirmations, or a held answer is invisible."""
    state = new_state("s1")
    state["pending_axle_basis"] = {"value": 14000.0, "asks": 1}
    lines = "\n".join(conversation_log._state_lines(state))

    assert "pending_axle_basis" in lines and "14000" in lines


def test_logging_never_raises_whatever_it_is_handed(captured):
    class Nonsense:
        def __getattr__(self, name):
            raise RuntimeError("no such thing")

    conversation_log.log_conversation_turn(
        session_id="s1", turn_id="t1", user_message="hi", output=Nonsense(),
        assistant_text=None, turn_outcome={}, state={},
    )
    # A turn whose model call failed has no analysis at all, and still logs.
    conversation_log.log_conversation_turn(
        session_id="s1", turn_id="t1", user_message="hi", output=None,
        assistant_text=None, turn_outcome={}, state={}, error="model timeout",
    )

    blocks = [rec.getMessage() for rec in captured.records
              if rec.name == conversation_log.CONVERSATION_LOGGER_NAME]
    assert "ERROR: model timeout" in blocks[-1]


def test_a_replayed_turn_says_so_rather_than_inventing_an_analysis():
    lines = conversation_log._analysis_lines(None)
    assert "replayed turn" in lines[0]
