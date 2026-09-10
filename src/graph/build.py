"""The LangGraph wiring and the public entry point.

    load_state -> analyze -> apply -> route -> [search | inventory_lookup | neither]
                                            -> compose -> persist -> END

``analyze`` holds the ONLY model call, and nothing downstream returns to it - the graph is
a straight line with one fan-out, never a loop (brief S31). The fan-out is how several
tools run in one turn: ``_route`` returns a LIST of node names and LangGraph runs them all
without re-invoking the model.
"""
from __future__ import annotations

import logging
from typing import Any

from src import conversation_store
from src.graph.nodes.apply import apply_node
from src.graph.nodes.compose import compose_node
from src.graph.nodes.inventory_lookup import inventory_lookup_node
from src.graph.nodes.search import search_node
from src.graph.state import STATE_SCHEMA_VERSION, from_snapshot, to_snapshot
from src.llm import usage
from src.llm.client import analyze_turn

logger = logging.getLogger(__name__)


def _route(state: dict, output: Any) -> list[str]:
    """Which tools this turn needs. May be several; may be none.

    A search needs a category. "Show me what you have" with no category ever chosen is NOT
    a search - there is nothing to narrow and the whole lot is not an answer - so compose
    sends them to the website instead (brief S25).
    """
    targets: list[str] = []

    if state.get("qualification_complete") and state.get("category"):
        targets.append("search")

    lookup = getattr(output, "inventory_lookup", None)
    if (
        lookup is not None
        and getattr(lookup, "is_lookup", False)
        and getattr(lookup, "confidence", "low") in {"medium", "high"}
    ):
        targets.append("inventory_lookup")

    return targets


def run_turn(session_id: str, user_message: str) -> dict[str, Any]:
    """One complete turn: load, analyze, apply, tools, compose, persist.

    Returns the payload the API and the CLI both render.
    """
    with usage.usage_scope() as turn_usage:
        lead_id = conversation_store.ensure_session(session_id)
        snapshot, conversation, stored_lead_id = conversation_store.load_session(session_id)

        state = from_snapshot(session_id, snapshot)
        state["lead_id"] = stored_lead_id or lead_id
        state["messages"] = conversation
        state["turn_outcome"] = {}
        state["turn_index"] = int(state.get("turn_index") or 0) + 1

        # ---- the single model call ----
        output = analyze_turn(state, user_message)

        # ---- deterministic ----
        apply_node(state, output, user_message)

        for target in _route(state, output):
            if target == "search":
                search_node(state)
            elif target == "inventory_lookup":
                state["turn"] = output
                try:
                    inventory_lookup_node(state)
                finally:
                    state.pop("turn", None)

        compose_node(state, output)

        assistant_text = state["turn_outcome"].get("assistant_text", "")
        state["messages"] = list(conversation) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_text},
        ]

        response = {
            "assistant_text": assistant_text,
            "listings": state["turn_outcome"].get("listings", []),
            "search_status_message": state["turn_outcome"].get("search_status_message"),
            "category": state.get("category"),
            "qualification_complete": bool(state.get("qualification_complete")),
        }

        conversation_store.save_turn(
            session_id,
            conversation=state["messages"],
            state_snapshot=to_snapshot(state),
            request_message=user_message,
            response=response,
            contact=state.get("contact"),
            item_of_interest=conversation_store.describe_interest(state),
        )

        logger.info(
            "TURN done: session=%s turn=%s calls=%d tokens=%d category=%s complete=%s",
            session_id, state["turn_index"], turn_usage.chat_completions,
            turn_usage.total_tokens, state.get("category"),
            state.get("qualification_complete"),
        )

        return {
            **response,
            "session_id": session_id,
            "slots": dict(state.get("slots") or {}),
            "state_schema_version": STATE_SCHEMA_VERSION,
            "usage": turn_usage.as_dict(),
        }


def build_graph():
    """The same pipeline as a compiled LangGraph.

    ``run_turn`` is the path the API uses - it is plain Python and therefore trivially
    testable. This exists so the pipeline is inspectable as a graph (and traceable in
    LangSmith) without the two ever diverging: both call the identical node functions in
    the identical order.
    """
    from langgraph.graph import END, START, StateGraph

    from src.graph.state import SessionState

    graph = StateGraph(SessionState)

    def _analyze(state: dict) -> dict:
        state["turn_outcome"] = state.get("turn_outcome") or {}
        output = analyze_turn(state, state["turn_outcome"].get("user_message", ""))
        state["turn_outcome"]["output"] = output
        return state

    def _apply(state: dict) -> dict:
        outcome = state.get("turn_outcome") or {}
        return apply_node(state, outcome.get("output"), outcome.get("user_message", ""))

    def _compose(state: dict) -> dict:
        return compose_node(state, (state.get("turn_outcome") or {}).get("output"))

    def _search(state: dict) -> dict:
        return search_node(state)

    def _lookup(state: dict) -> dict:
        state["turn"] = (state.get("turn_outcome") or {}).get("output")
        try:
            return inventory_lookup_node(state)
        finally:
            state.pop("turn", None)

    graph.add_node("analyze", _analyze)
    graph.add_node("apply", _apply)
    graph.add_node("search", _search)
    graph.add_node("inventory_lookup", _lookup)
    graph.add_node("compose", _compose)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "apply")
    graph.add_conditional_edges(
        "apply",
        lambda state: _route(state, (state.get("turn_outcome") or {}).get("output")) or ["compose"],
        ["search", "inventory_lookup", "compose"],
    )
    graph.add_edge("search", "compose")
    graph.add_edge("inventory_lookup", "compose")
    graph.add_edge("compose", END)
    return graph.compile()
