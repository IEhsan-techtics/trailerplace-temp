"""When the results gate is open, Python runs the search - the reply model only presents it.

Live run, 7 of 13 "vague" conversations: qualification completed on a message that also asked
"what are your opening hours?". The reply model was offered search_inventory as an optional
tool, answered the hours, and never called it - the customer had to ask for the trailers
they had just qualified for.
"""
from __future__ import annotations

from src.graph.build import run_turn
from src.llm import respond
from src.llm.tools import ToolRunner

from tests.factories import complete_welcome, turn_output


def _qualify_dump(fake_llm, last_output):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "a dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")
    fake_llm.push(last_output)
    run_turn("s1", "last message")


# ---------------------------------------------------------------- the routing decision
def test_completing_qualification_alongside_a_question_prefetches(fake_llm, no_reply_pass, no_search):
    _qualify_dump(fake_llm, turn_output(
        slots={"payload_capacity": "7000 lbs"}, intent="faq", faq_key="store_info",
        user_question_to_answer="what are your opening hours?",
    ))
    assert no_reply_pass.prefetch_flags[-1] is True


def test_asking_to_skip_the_rest_mid_questions_prefetches(fake_llm, no_reply_pass, no_search):
    _qualify_dump(fake_llm, turn_output(intent="skip_all_show_results"))
    assert no_reply_pass.prefetch_flags[-1] is True


def test_no_prefetch_while_questions_remain(fake_llm, no_reply_pass, no_search):
    _qualify_dump(fake_llm, turn_output(intent="faq", faq_key="store_info"))
    assert True not in no_reply_pass.prefetch_flags
    assert no_search == []


def test_a_failed_reply_pass_still_searches_exactly_once(fake_llm, no_reply_pass, no_search):
    """The fake reply pass returns None, the "reply pass failed" signal: the backstop searches."""
    _qualify_dump(fake_llm, turn_output(slots={"payload_capacity": "7000 lbs"}))
    assert len(no_search) == 1


# ------------------------------------------------------------ what the agent is handed
def _capture_agent(monkeypatch):
    seen = {}

    def fake_run_agent(runner, system_prompt, messages):
        seen["messages"] = messages
        seen["runner"] = runner
        return "Here are the trailers."

    import src.graph.agent as agent_module

    monkeypatch.setattr(agent_module, "run_agent", fake_run_agent)
    return seen


def test_the_agent_starts_holding_the_search_results(monkeypatch):
    from langchain_core.messages import AIMessage, ToolMessage

    seen = _capture_agent(monkeypatch)
    monkeypatch.setattr(ToolRunner, "_search_inventory", lambda self: "1. A 14 ft dump trailer")
    state = {"session_id": "s1", "category": "Dump", "messages": [], "turn_outcome": {}}

    reply = respond.respond_with_tools(state, turn_output(), "what are your hours?", prefetch_search=True)

    assert reply.assistant_text == "Here are the trailers."
    call, result = seen["messages"][-2:]
    assert isinstance(call, AIMessage) and call.tool_calls[0]["name"] == "search_inventory"
    assert isinstance(result, ToolMessage) and result.tool_call_id == call.tool_calls[0]["id"]
    assert result.content == "1. A 14 ft dump trailer"
    assert seen["messages"][-3].content == "what are your hours?", "after the customer's message"


def test_without_prefetch_the_agent_decides(monkeypatch):
    seen = _capture_agent(monkeypatch)
    state = {"session_id": "s1", "category": "Dump", "messages": [], "turn_outcome": {}}
    respond.respond_with_tools(state, turn_output(), "hi")
    assert len(seen["messages"]) == 1


def test_a_refused_prefetch_reaches_the_agent_as_a_refusal(monkeypatch):
    """The ToolRunner's own preconditions still apply - no category means no search."""
    seen = _capture_agent(monkeypatch)
    state = {"session_id": "s1", "category": None, "messages": [], "turn_outcome": {}}
    respond.respond_with_tools(state, turn_output(), "show me", prefetch_search=True)
    assert seen["messages"][-1].content.startswith("NO SEARCH RAN")
