"""The LangGraph wiring and the public entry point.

    load_state -> analyze -> apply -> respond -> compose -> persist -> END
                                        |
                                        +-- tools: search_inventory / lookup_inventory

``analyze`` is one model call and reads the message; it never sees inventory, because it
runs before ``apply`` has folded this turn's answers into the slots.

``respond`` is the second, and it only happens on a turn the gate has opened - one with
inventory to talk about. It is where the model calls a tool, receives the top matches and
writes the cards. Every other turn costs exactly one model call, and compose assembles the
reply from the analysis pass's own pieces.

Search deliberately runs from INSIDE respond rather than as its own node: it has to happen
after ``apply``, or it filters on last turn's slots and misses the "make it 24 ft" the
customer just said.
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
from src.llm.respond import respond_with_tools
from src.tools.lookup_gate import lookup_requested

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

    # The structural gate, not just is_lookup+confidence: a bare make is a brand preference
    # and a category word is not a model, so neither may hijack the turn with a side query.
    if lookup_requested(output):
        targets.append("inventory_lookup")

    return targets


# Turns where the customer wants something only a person can do. They carry no listings, so
# the search gate never opens for them - but they still need the agent, because handing the
# request to a person through the escalate tool is exactly what it is for.
#
# Without this, a complaint fell through to the deterministic assembly and was answered with
# the next qualification question: "I'm sorry your order arrived damaged. What material will
# you be hauling?"
#
# `faq` is deliberately NOT here. The five FAQs have answers of their own, the analysis pass
# writes them into answer_to_customer_question, and compose then still asks the pending
# qualification question - so an interruption is answered without costing the customer their
# place in the flow. Routing those here threw that away and emailed the team about a question
# we can answer ourselves.
_NEEDS_A_PERSON_INTENTS = {"team_request_escalation", "listing_interest"}


def _needs_a_person(state: dict, output: Any) -> bool:
    """Deliberately NOT gated on the opening contact ask.

    It used to be, on the reasoning that turn one belongs to the greeting and "the analysis
    pass has recorded it, so it reaches the agent on the next turn". A live run disproved
    that: "I have a complaint about my last order, the trailer arrived damaged" was answered
    with the contact request and nothing else, the conversation moved on to financing, and
    the complaint was never raised again - no apology, no phone number, and no email to the
    team about a damaged trailer.

    Nothing carries an unhandled intent forward, so a suppressed escalation is a lost one.
    These turns now own themselves, and the contact request rides along with the canned
    answer instead of replacing it: ``ask_for_missing`` asks for exactly the same details the
    gate wanted, so nothing is given up by letting the escalation win the turn.
    """
    return getattr(output, "intent", "") in _NEEDS_A_PERSON_INTENTS


def _tools_and_reply(state: dict, output: Any, user_message: str) -> None:
    """Pull inventory and write the reply, on the turns that need it.

    One function, called by both ``run_turn`` and ``build_graph``, so the two can never
    describe different pipelines.

    The gate decides WHETHER we are in a position to show trailers; the model decides whether
    this customer wants to see them, by calling the tool or not. On every other turn there is
    no second model call at all and compose assembles the reply exactly as before.
    """
    outcome = state.setdefault("turn_outcome", {})
    if outcome.get("unavailable_type"):
        # A type we do not carry owns the turn: the team has been told and compose writes
        # the fixed reply. Running the reply pass too would risk a second escalation email.
        return
    targets = _route(state, output)
    needs_person = _needs_a_person(state, output)
    if not (targets or needs_person):
        return
    # Read by compose, so a failed reply pass on one of these turns falls back to the canned
    # line rather than to the next qualification question.
    outcome["needs_a_person"] = needs_person

    # The gate decides WHETHER trailers are shown; the reply pass only presents them. So when
    # the gate is open the search runs up front instead of being left to the model.
    reply = respond_with_tools(state, output, user_message, prefetch_search="search" in targets)
    if reply is not None:
        outcome["reply_text"] = reply.assistant_text
        outcome["cited_listing_urls"] = list(reply.cited_listing_urls or [])
        return

    # Backstop: the reply pass failed. Pull the inventory ourselves and let compose render it
    # deterministically - a model failure is a flatter reply, never a customer who asked to
    # see trailers and was shown none. Guarded on the *_ran flags because the pass may have
    # got as far as running a tool before it fell over.
    for target in targets:
        if target == "search" and not outcome.get("search_ran"):
            search_node(state)
        elif target == "inventory_lookup" and not outcome.get("inventory_lookup_ran"):
            state["turn"] = output
            try:
                inventory_lookup_node(state)
            finally:
                state.pop("turn", None)


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

        # ---- the reply pass, only when there is inventory to talk about ----
        # The gate decides WHETHER we are in a position to show trailers; the model decides
        # whether this customer wants to see them, by calling the tool or not. On every other
        # turn there is no second call at all and compose assembles the reply as before.
        _tools_and_reply(state, output, user_message)
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
            outbox_events=state["turn_outcome"].get("outbox_events"),
        )

        # After the commit and off the reply path: the customer must never wait on an SMTP
        # round trip, and a row that fails to send stays pending for the next drain.
        if state["turn_outcome"].get("outbox_events"):
            conversation_store.deliver_pending_outbox_async()

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
            "contact": dict(state.get("contact") or {}),
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

    def _respond(state: dict) -> dict:
        outcome = state.get("turn_outcome") or {}
        _tools_and_reply(state, outcome.get("output"), outcome.get("user_message", ""))
        return state

    graph.add_node("analyze", _analyze)
    graph.add_node("apply", _apply)
    graph.add_node("respond", _respond)
    graph.add_node("compose", _compose)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "apply")
    graph.add_edge("apply", "respond")
    graph.add_edge("respond", "compose")
    graph.add_edge("compose", END)
    return graph.compile()
