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
import functools
import time
import uuid
from typing import Any, Callable

from src import conversation_log, conversation_store, turn_log, turn_saver
from src.graph.nodes.apply import apply_node
from src.graph.nodes.compose import compose_node
from src.graph.nodes.inventory_lookup import inventory_lookup_node
from src.graph.nodes.search import search_node
from src.graph.state import STATE_SCHEMA_VERSION, from_snapshot, to_snapshot
from src.config import settings
from src.llm import usage
from src.llm.client import analyze_turn
from src.llm.respond import respond_with_tools
from src.tools.lookup_gate import lookup_requested, referenced_listing

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
    #
    # A trailer already on their screen is not a lookup either. "I like the 81419" is how
    # people pick one off a list, and it arrives with BOTH a listing_reference and a stock
    # number: looking it up fetched a trailer we had just shown and printed its card again.
    if lookup_requested(output) and referenced_listing(state, output) is None:
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
    if (state.get("turn_outcome") or {}).get("link_interest"):
        return True
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
    if outcome.get("off_topic"):
        # Nothing here is about trailers, so there is nothing to look up and nothing for the
        # reply pass to present. Returning is also what keeps the decline a decline: the
        # reply pass writes freely, and a model asked to answer a question it has been told
        # to refuse is the one place this could leak.
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


def _superseded(abandon_if: Callable[[], bool] | None) -> bool:
    """Has the customer moved on? A check that fails is read as 'no'.

    Deliberately: losing a finished reply because a health check blipped is worse than
    sending one the customer has already overtaken.
    """
    if abandon_if is None:
        return False
    try:
        return bool(abandon_if())
    except Exception:  # noqa: BLE001
        logger.exception("Abandon check failed; keeping the reply")
        return False


def run_turn(
    session_id: str,
    user_message: str,
    *,
    turn_id: Any = None,
    abandon_if: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """One complete turn: load, analyze, apply, tools, compose, persist.

    Returns the payload the API and the CLI both render.

    ``turn_id`` makes the turn idempotent. Give one that the caller can reproduce for a
    given customer message - the platform's own message id, or a hash of it - and a
    redelivery is answered from the stored reply instead of being run a second time.
    Without it every call is a new turn, which is what a browser holding its own request
    open actually wants.

    ``abandon_if`` is for a channel that can tell the customer has said more while we were
    answering. Checked once, just before the save: true and the turn is thrown away whole -
    nothing persisted, no transcript entry, no email to the team - and ``{"abandoned": True}``
    comes back. The caller then reruns with the newer message folded in, and only THAT reply
    is ever seen. Discarding before the save is what makes it complete: everything a turn
    writes goes in that one transaction, so not reaching it leaves no trace.
    """
    with usage.usage_scope() as turn_usage:
        # Their last turn commits in the background, so a message sent inside that window
        # would be answered from the session as it stood BEFORE it - the contact they just
        # gave us missing, their last answer gone. Normally nothing to wait for; when there
        # is, it is this customer's save only. See src/turn_saver.py.
        turn_saver.wait_for(session_id)

        # One round trip, not two: both halves come out of the same read of the same row.
        lead_id, snapshot, conversation = conversation_store.open_session(session_id)
        stored_lead_id = lead_id

        # Before the model call, because that is the expense being avoided. A turn row
        # exists only if that turn committed in full, so its reply is the whole answer.
        replay = conversation_store.stored_turn_response(session_id, turn_id)
        if replay is not None:
            logger.info("TURN replayed: session=%s turn=%s", session_id, turn_id)
            replayed_state = from_snapshot(session_id, snapshot)
            return {
                **replay,
                "session_id": session_id,
                "turn_id": str(turn_id),
                "slots": dict(replayed_state.get("slots") or {}),
                "contact": dict(replayed_state.get("contact") or {}),
                "state_schema_version": STATE_SCHEMA_VERSION,
                "usage": turn_usage.as_dict(),
                "replayed": True,
            }

        started = time.perf_counter()
        state = from_snapshot(session_id, snapshot)
        state["lead_id"] = stored_lead_id or lead_id
        state["messages"] = conversation
        state["turn_outcome"] = {}
        state["turn_index"] = int(state.get("turn_index") or 0) + 1
        # Logged and stored either way, so a turn can always be found by its id - it is only
        # the CALLER-supplied one that makes a turn replayable.
        turn_id = turn_id or uuid.uuid4()

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

        # The last moment a turn can be called off. After the save it is on the record and
        # its emails are queued, and calling it back would mean undoing both.
        if _superseded(abandon_if):
            logger.info(
                "TURN abandoned, the customer said more: session=%s turn=%s", session_id, turn_id
            )
            return {"abandoned": True, "session_id": session_id, "turn_id": str(turn_id),
                    "usage": turn_usage.as_dict()}

        outbox_events = state["turn_outcome"].get("outbox_events")
        commit = functools.partial(
            conversation_store.save_turn,
            session_id,
            conversation=state["messages"],
            state_snapshot=to_snapshot(state),
            request_message=user_message,
            response=response,
            contact=state.get("contact"),
            item_of_interest=conversation_store.describe_interest(state),
            outbox_events=outbox_events,
            turn_id=turn_id,
        )

        # The commit happens AFTER the reply is written, so the customer is waiting on a
        # result they never see - about 2.4 s of it. Queued instead, in order, never dropped,
        # and drained before the process exits (src/turn_saver.py). The in-memory store is
        # always saved inline: it is instant, and the tests read it straight back.
        if settings.background_turn_save and conversation_store.persistence_enabled():
            # Registered BEFORE it is queued: a redelivery arriving inside the save window
            # must still be answered from this reply rather than run the turn again.
            conversation_store.remember_inflight(session_id, turn_id, response)
            turn_saver.submit(
                commit, session_id=session_id, turn_id=turn_id, outbox_events=outbox_events
            )
        else:
            commit()
            # After the commit and off the reply path: the customer must never wait on an
            # SMTP round trip, and a row that fails to send stays pending for the next drain.
            if outbox_events:
                conversation_store.deliver_pending_outbox_async()

        logger.info(
            "TURN done: session=%s turn=%s calls=%d tokens=%d cached=%d category=%s complete=%s",
            session_id, state["turn_index"], turn_usage.chat_completions,
            turn_usage.total_tokens, turn_usage.cached_tokens, state.get("category"),
            state.get("qualification_complete"),
        )
        # After the save, so what is logged is what was persisted. Both are best-effort and
        # neither can raise - a logging failure must never cost the customer their reply.
        turn_log.log_turn(
            session_id=session_id,
            turn_id=str(turn_id),
            turn_index=state.get("turn_index"),
            intent=getattr(output, "intent", None),
            category=state.get("category"),
            latency_ms=(time.perf_counter() - started) * 1000,
            turn_outcome=state["turn_outcome"],
            usage=turn_usage,
        )
        conversation_log.log_conversation_turn(
            session_id=session_id,
            turn_id=str(turn_id),
            user_message=user_message,
            output=output,
            assistant_text=assistant_text,
            turn_outcome=state["turn_outcome"],
            state=state,
        )

        return {
            **response,
            "session_id": session_id,
            "turn_id": str(turn_id),
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
