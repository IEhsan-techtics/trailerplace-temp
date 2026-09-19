"""The ReAct loop itself (src/graph/agent.py).

What is being pinned: the graph is agent -> tools -> agent with no third node, the decision
to keep looping is taken from the last message's tool_calls and nothing else, message state
accumulates across iterations, and every result of one tool call is handled by ONE following
agent invocation rather than one per result.
"""
from __future__ import annotations

import pytest
from factories import turn_output
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from src.graph.agent import (
    RECURSION_LIMIT,
    AgentState,
    build_agent_graph,
    build_tools,
    should_continue,
)
from src.llm.tools import ToolRunner


def _runner(**state_overrides):
    state = {"session_id": "s1", "category": "Dump", "slots": {},
             "contact": {"name": "D", "email": "d@x.ai", "declined": False}, "turn_outcome": {}}
    state.update(state_overrides)
    return ToolRunner(state, turn_output())


# ----------------------------------------------------------------- the routing decision
def test_routes_to_tools_when_the_agent_asked_for_one():
    message = AIMessage(
        content="",
        tool_calls=[{"name": "search_inventory", "args": {}, "id": "call_1"}],
    )
    assert should_continue(AgentState(messages=[message])) == "tools"


def test_routes_to_end_when_the_agent_answered():
    assert should_continue(AgentState(messages=[AIMessage(content="Here you go.")])) == END


def test_routing_ignores_how_many_rounds_have_happened():
    """The decision is the last message's tool_calls - never an iteration count."""
    history = [
        HumanMessage(content="show me trailers"),
        AIMessage(content="", tool_calls=[{"name": "search_inventory", "args": {}, "id": "1"}]),
        ToolMessage(content="1. TITLE: A | URL: https://x/1", tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"name": "search_inventory", "args": {}, "id": "2"}]),
    ]
    # Deep into the loop and still asking for a tool -> still routes to tools.
    assert should_continue(AgentState(messages=history)) == "tools"


# ------------------------------------------------------------------------- graph shape
def test_graph_is_agent_tools_agent_and_nothing_else():
    graph = build_agent_graph(_runner(), "system prompt").get_graph()
    assert sorted(n for n in graph.nodes if not n.startswith("__")) == ["agent", "tools"]

    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("__start__", "agent") in edges
    assert ("agent", "tools") in edges
    assert ("tools", "agent") in edges, "the loop back into the SAME agent"
    assert ("agent", "__end__") in edges


def test_no_summariser_or_formatter_node_exists():
    """The same agent handles deciding, reading results and writing the answer."""
    nodes = build_agent_graph(_runner(), "p").get_graph().nodes
    for banned in ("summarizer", "summariser", "formatter", "respond", "final"):
        assert not any(banned in name for name in nodes)


def test_recursion_limit_allows_several_rounds_but_is_finite():
    assert RECURSION_LIMIT >= 5, "must allow more than a single tool round"
    assert RECURSION_LIMIT < 50, "must still be a real ceiling"


# ---------------------------------------------------------------- state accumulates
def test_messages_accumulate_across_iterations(monkeypatch):
    """The second pass must see: user, its own tool call, and the tool's answer."""
    from src.graph import agent as agent_module
    from src.graph.nodes import search as search_module

    monkeypatch.setattr(
        search_module,
        "search_node",
        lambda state: state.setdefault("turn_outcome", {}).update(
            {"search_ran": True, "listings": [{"title": "A", "url": "https://x/1"}]}
        )
        or state,
    )

    seen: list[list] = []

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            seen.append(list(messages))
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "search_inventory", "args": {}, "id": "call_1"}],
                )
            return AIMessage(content="1. [A](https://x/1)")

    fake = FakeModel()
    monkeypatch.setattr(agent_module, "build_model", lambda tools: fake)

    graph = build_agent_graph(_runner(), "SYSTEM PROMPT HERE")
    result = graph.invoke({"messages": [HumanMessage(content="show me trailers")]})

    assert fake.calls == 2, "one call to ask for the tool, one to answer with its result"

    # The second invocation's message list is the whole story so far.
    second = seen[1]
    kinds = [type(m).__name__ for m in second]
    assert kinds[0] == "SystemMessage", "the prompt leads every iteration"
    assert "HumanMessage" in kinds
    assert "ToolMessage" in kinds, "the tool result stays in message state"
    assert any(getattr(m, "tool_calls", None) for m in second), "its own tool call is retained"

    # And nothing was lost from the first pass.
    assert len(second) > len(seen[0])
    assert result["messages"][-1].content == "1. [A](https://x/1)"


def test_past_bot_replies_in_the_history_are_not_counted_as_model_calls(monkeypatch):
    """The history we pass in contains earlier bot replies as AIMessages. They were counted
    as calls, so a live lookup turn that made 3 requests was reported as 4 - and the error
    grew with every reply in the window."""
    from src.graph import agent as agent_module
    from src.graph.nodes import inventory_lookup as lookup_module
    from src.llm import usage

    monkeypatch.setattr(
        lookup_module,
        "lookup_inventory",
        lambda **kw: {"match_status": "exact", "matches": [], "requested_label": "x"},
    )

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "lookup_inventory",
                                 "args": {"stock_number": "15300"}, "id": "call_1"}],
                )
            return AIMessage(content="Here it is.")

    fake = FakeModel()
    monkeypatch.setattr(agent_module, "build_model", lambda tools: fake)

    history = [
        HumanMessage(content="hi"),
        AIMessage(content="Thank you for contacting TrailerPlace."),
        HumanMessage(content="I'm Dave"),
        AIMessage(content="Great to have your details, Dave!"),
        HumanMessage(content="do you have stock 15300?"),
    ]
    with usage.usage_scope() as turn_usage:
        text = agent_module.run_agent(_runner(), "SYSTEM", history)

    assert text == "Here it is."
    assert fake.calls == 2
    assert turn_usage.chat_completions == 2, "two past replies in the history are not calls"


def test_the_same_system_prompt_leads_every_iteration(monkeypatch):
    from src.graph import agent as agent_module
    from src.graph.nodes import search as search_module

    monkeypatch.setattr(
        search_module,
        "search_node",
        lambda state: state.setdefault("turn_outcome", {}).update({"search_ran": True, "listings": []})
        or state,
    )
    prompts: list[str] = []

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            prompts.append(messages[0].content)
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{"name": "search_inventory", "args": {}, "id": "c1"}])
            return AIMessage(content="done")

    monkeypatch.setattr(agent_module, "build_model", lambda tools: FakeModel())
    build_agent_graph(_runner(), "THE ONE PROMPT").invoke(
        {"messages": [HumanMessage(content="hi")]}
    )

    assert prompts == ["THE ONE PROMPT", "THE ONE PROMPT"]


def test_all_results_of_one_tool_call_reach_a_single_agent_invocation(monkeypatch):
    """Ten results must be summarised by ONE following call, not ten."""
    from src.graph import agent as agent_module
    from src.graph.nodes import search as search_module

    listings = [{"title": f"T{i}", "url": f"https://x/{i}"} for i in range(10)]
    monkeypatch.setattr(
        search_module,
        "search_node",
        lambda state: state.setdefault("turn_outcome", {}).update(
            {"search_ran": True, "listings": listings}
        )
        or state,
    )

    class FakeModel:
        def __init__(self):
            self.calls = 0
            self.tool_payload = None

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{"name": "search_inventory", "args": {}, "id": "c1"}])
            self.tool_payload = "\n".join(
                m.content for m in messages if isinstance(m, ToolMessage)
            )
            return AIMessage(content="all ten")

    fake = FakeModel()
    monkeypatch.setattr(agent_module, "build_model", lambda tools: fake)
    build_agent_graph(_runner(), "p").invoke({"messages": [HumanMessage(content="show me")]})

    assert fake.calls == 2, "ten results, still only one follow-up invocation"
    for index in range(10):
        assert f"T{index}" in fake.tool_payload, "every result reached that one invocation"


# ------------------------------------------------------------------------ tool schemas
def test_tool_argument_schemas_are_preserved():
    tools = {t.name: t for t in build_tools(_runner())}
    assert set(tools) == {"search_inventory", "lookup_inventory", "escalate"}
    assert tools["search_inventory"].args == {}, "search takes no arguments by design"
    assert set(tools["lookup_inventory"].args) == {"year", "make", "model_text", "stock_number", "listing_url"}


def test_tools_still_go_through_the_runner_preconditions():
    tools = {t.name: t for t in build_tools(_runner(category=None))}
    assert "NO SEARCH RAN" in tools["search_inventory"].invoke({})


@pytest.mark.parametrize(
    "args",
    [{"make": "Diamond C", "year": None, "model_text": None, "stock_number": None}],
)
def test_lookup_tool_still_enforces_the_gate(args):
    tools = {t.name: t for t in build_tools(_runner())}
    assert "NO LOOKUP RAN" in tools["lookup_inventory"].invoke(args)


# --------------------------------------------------- a second search is the NEXT page
def test_a_second_search_in_one_loop_excludes_what_was_already_served(monkeypatch):
    """The ping-pong guard: state["shown_urls"] only grows in compose, after the loop, so
    the runner has to track what it served itself."""
    from src.graph.nodes import search as search_module

    catalogue = [{"title": f"T{i}", "url": f"https://x/{i}"} for i in range(6)]

    def _fake_search(state):
        excluded = set(state.get("shown_urls") or [])
        remaining = [row for row in catalogue if row["url"] not in excluded][:3]
        state.setdefault("turn_outcome", {}).update(
            {"search_ran": True, "listings": remaining, "result_count": len(remaining)}
        )
        return state

    monkeypatch.setattr(search_module, "search_node", _fake_search)
    runner = _runner()

    first = runner.call("search_inventory", "{}")
    second = runner.call("search_inventory", "{}")

    assert "T0" in first and "T2" in first
    assert "T0" not in second, "the second page must not repeat the first"
    assert "T3" in second and "T5" in second
    assert len(runner.served_listings) == 6, "both pages accumulate"


def test_running_out_tells_the_agent_to_stop_rather_than_returning_nothing(monkeypatch):
    from src.graph.nodes import search as search_module

    catalogue = [{"title": "T0", "url": "https://x/0"}]

    def _fake_search(state):
        excluded = set(state.get("shown_urls") or [])
        state.setdefault("turn_outcome", {}).update(
            {"search_ran": True, "listings": [r for r in catalogue if r["url"] not in excluded]}
        )
        return state

    monkeypatch.setattr(search_module, "search_node", _fake_search)
    runner = _runner()
    runner.call("search_inventory", "{}")
    assert "NO FURTHER MATCHES" in runner.call("search_inventory", "{}")


def test_the_persisted_shown_urls_are_not_mutated_by_a_search(monkeypatch):
    """Only what the REPLY cites counts as shown; compose records that."""
    from src.graph.nodes import search as search_module

    monkeypatch.setattr(
        search_module,
        "search_node",
        lambda state: state.setdefault("turn_outcome", {}).update(
            {"search_ran": True, "listings": [{"title": "A", "url": "https://x/1"}]}
        )
        or state,
    )
    runner = _runner(shown_urls=["https://old/1"])
    runner.call("search_inventory", "{}")
    assert runner.state["shown_urls"] == ["https://old/1"]


# --------------------------------------------------- category change drops the history
def test_changing_category_clears_the_shown_history():
    from src.tools.category import set_trailer_category

    state = {"category": "Dump", "required_slots": [], "shown_urls": ["https://x/1"],
             "results_shown": True, "asked_counts": {}, "declined_slots": []}
    set_trailer_category(state, "Utility")

    assert state["shown_urls"] == []
    assert state["results_shown"] is False


def test_staying_on_the_same_category_keeps_the_shown_history():
    """Re-selecting the category they are already on is not a change."""
    from src.tools.category import set_trailer_category

    state = {"category": "Dump", "required_slots": [], "shown_urls": ["https://x/1"],
             "results_shown": True, "asked_counts": {}, "declined_slots": []}
    set_trailer_category(state, "Dump")

    assert state["shown_urls"] == ["https://x/1"]
    assert state["results_shown"] is True
