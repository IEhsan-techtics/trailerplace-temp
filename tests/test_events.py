"""The grand-opening announcement: new customers only, its own message, gone after the event."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.domain import events
from src.domain.reply_chunks import split_reply_into_chunks
from src.graph.build import run_turn
from src.llm import prompt

from tests.factories import turn_output

BEFORE = datetime(2026, 10, 9, 15, 0, tzinfo=events.TEXAS)
ON_THE_DAY = datetime(2026, 10, 10, 11, 0, tzinfo=events.TEXAS)
OVER = datetime(2026, 10, 10, 14, 0, tzinfo=events.TEXAS)


@pytest.fixture
def clock(monkeypatch):
    def set_to(moment):
        monkeypatch.setattr(events, "_now", lambda now=None: (now or moment).astimezone(events.TEXAS))
    set_to(BEFORE)
    return set_to


def test_a_new_customer_gets_it_as_a_separate_message(fake_llm, clock):
    fake_llm.push(turn_output(intent="category_selection", category_mentioned="dump"))
    reply = run_turn("s1", "I need a dump trailer")["assistant_text"]

    chunks = split_reply_into_chunks(reply)
    assert chunks[-1].startswith("By the way, this Saturday, October 10, from 10 AM to 2 PM")
    assert len(chunks) >= 2, "the announcement never shares a bubble with our reply"


def test_it_is_said_once(fake_llm, clock):
    fake_llm.push(turn_output(intent="category_selection", category_mentioned="dump"))
    run_turn("s1", "I need a dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    assert "By the way" not in run_turn("s1", "gravel")["assistant_text"]


def test_on_the_day_it_says_today(clock):
    clock(ON_THE_DAY)
    assert events.announcement().startswith("By the way, today from 10 AM to 2 PM")


def test_it_is_gone_once_the_event_is_over(fake_llm, clock):
    clock(OVER)
    fake_llm.push(turn_output(intent="category_selection", category_mentioned="dump"))
    assert "By the way" not in run_turn("s1", "I need a dump trailer")["assistant_text"]
    assert events.prompt_block() == ""


def test_the_prompt_section_comes_and_goes_with_the_event(clock):
    assert "THIS WEEK'S EVENT" in prompt.system_prompt()
    assert "you don't know" in events.prompt_block()
    clock(OVER)
    assert "THIS WEEK'S EVENT" not in prompt.system_prompt(), "the cached prompt must drop it"


def test_the_new_address_and_hours_are_what_we_say():
    from src.domain import company

    block = company.company_facts_block()
    assert "3709 U.S. Hwy 59 S, Rosenberg, TX 77471" in block
    assert "Saturday, 8:00 AM to 2:00 PM" in block
    assert "Wharton" not in block
