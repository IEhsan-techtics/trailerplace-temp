"""The ReAct agent loop: one model, one prompt, looping through tools until it is done.

    START -> agent -> tools -> agent -> ... -> END
                 |                              ^
                 +------------------------------+
                        (no tool calls left)

The same node, the same model instance and the same system prompt handle every iteration.
Whether the agent is choosing a tool, reading what a tool returned, summarising ten results
at once, or writing the final reply is decided by the MESSAGES, not by routing it to a
different node. There is deliberately no summariser node and no formatter node.

Shaped like ``langgraph.prebuilt.create_react_agent``, but written out rather than called,
for two reasons this project needs:

* the tools close over the live session state, so ``search_inventory`` can take no arguments
  and still filter on what this customer told us (see src/llm/tools.py::ToolRunner);
* the system prompt is prepended on every iteration rather than baked into the message list
  once, so it cannot fall out of the window on a long tool loop.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.config import settings
from src.llm import usage
from src.tools.lookup_gate import nothing_left_to_look_up

logger = logging.getLogger(__name__)

# How many times the agent may go round the loop. Each round is two graph steps (agent then
# tools), plus the final agent step that answers - so the recursion limit is 2n + 1.
#
# This is a SAFETY NET, not the control flow. The loop ends when the agent stops asking for
# tools; this only stops a model that never stops asking. Nothing anywhere decides to finish
# because "this is iteration 2".
MAX_TOOL_ROUNDS = 3
RECURSION_LIMIT = 2 * MAX_TOOL_ROUNDS + 1

# One key for every reply pass, so they all hit the same prompt cache (see src/llm/client.py).
REPLY_CACHE_KEY = "luna-reply"


class AgentState(TypedDict):
    """``add_messages`` is what makes the history ACCUMULATE.

    Each node returns only the message(s) it produced and the reducer appends them, so by the
    second iteration the agent sees system + user + its own tool call + the ToolMessage that
    answered it. Returning a plain list here would overwrite the history instead, and the
    agent would meet the tool results with no memory of having asked for them.
    """

    messages: Annotated[list, add_messages]


def build_tools(runner: Any) -> list:
    """The agent's tools, bound to this turn's ToolRunner.

    Argument schemas are unchanged from the OpenAI specs they replace, and both still go
    through ``ToolRunner.call``, so the preconditions and the lookup gate apply exactly as
    before - the tools are re-described here, never reimplemented.
    """

    @tool
    def search_inventory() -> str:
        """Search our trailer inventory for the trailers matching what this customer has told
        us, and return the top matches, already filtered and ranked. Call this when the
        customer wants to see trailers. Takes no arguments - the requirements we have
        collected so far are applied automatically."""
        return runner.call("search_inventory", "{}")

    @tool
    def lookup_inventory(
        year: int | None = None,
        make: str | None = None,
        model_text: str | None = None,
        stock_number: str | None = None,
        listing_url: str | None = None,
    ) -> str:
        """Look up specific stock by identifier when the customer names one particular
        trailer: a stock number, a year plus a make, a make plus a model code, or a link to
        one of our listings (trailerplace.com/inventory/...). NOT for
        category shopping - a make on its own is a brand preference, and a trailer type
        ("dump trailer") is not a model.

        Args:
            year: Model year, never a stock number.
            make: Manufacturer, e.g. 'Diamond C'.
            model_text: Model code exactly as the customer typed it, e.g. 'LPX', 'fmax 212'.
            stock_number: Explicit stock/unit number only. Never a weight, size, price, year
                or phone number.
            listing_url: A trailerplace.com/inventory link they gave, copied exactly.
        """
        return runner.call(
            "lookup_inventory",
            json.dumps(
                {
                    "year": year,
                    "make": make,
                    "model_text": model_text,
                    "stock_number": stock_number,
                    "listing_url": listing_url,
                }
            ),
        )

    @tool
    def escalate(reason: str, summary: str) -> str:
        """Hand something to a human, by emailing our team. Call this whenever the customer
        needs something you cannot do yourself and it is not one of the FAQs you can already
        answer: a complaint or a problem with an order, a callback, a meeting or appointment,
        a quote or price negotiation, delivery scheduling, paperwork or titling, seeing a unit
        in person, a question about stock that is not on the lot today, a type of trailer we
        do not carry, or serious interest in one particular trailer.

        Do NOT call it for something you can already answer - financing, trade-ins, service
        and parts, where we are, or wanting our phone number all have answers in your
        instructions. Call it once per request, not once per message.

        Args:
            reason: One of complaint, callback, meeting, quote, pricing, delivery, paperwork,
                viewing, stock_question, unstocked_type, listing_interest, other.
            summary: AT MOST SEVEN WORDS naming what they want - a label, not a report. The
                email already says what kind of request this is and links to the whole
                conversation. "Wants best price on a dump trailer" or "Says last order
                arrived damaged". Anything longer is cut.
        """
        return runner.call("escalate", json.dumps({"reason": reason, "summary": summary}))

    if nothing_left_to_look_up(getattr(runner, "state", None), getattr(runner, "turn", None)):
        # Every trailer they named is already on their screen, and the card they are asking
        # about is in the messages with its full specs. Withheld rather than refused in the
        # handler: a refusal comes too late, because the model has already spent an agent
        # pass on the call by the time anything could turn it down.
        logger.info("AGENT lookup_inventory withheld: the trailer is already on their screen")
        return [search_inventory, escalate]

    return [search_inventory, lookup_inventory, escalate]


def build_model(tools: list):
    """The ONE model instance every iteration uses, with the tools bound to it."""
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.chat_model,
        "api_key": settings.openai_api_key,
        "timeout": settings.chat_timeout_seconds,
        "use_responses_api": True,
    }
    if settings.chat_reasoning_effort:
        kwargs["reasoning"] = {"effort": settings.chat_reasoning_effort}
    # Bound per call rather than set on the client: it is a request parameter, and binding
    # it here puts it on every iteration of the loop.
    return ChatOpenAI(**kwargs).bind_tools(tools, prompt_cache_key=REPLY_CACHE_KEY)


def should_continue(state: AgentState) -> str:
    """Loop or finish - decided ONLY by whether the agent just asked for a tool.

    No iteration counting, no "we have searched once so we must be done". An agent that
    wants a second lookup gets a second lookup; an agent that has written its answer ends.
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_agent_graph(runner: Any, system_prompt: str):
    """Compile the agent/tools loop for one turn.

    Compiled per turn because the tools close over that turn's session state. Compilation is
    pure Python assembly - no I/O - and costs nothing beside the model call it wraps.
    """
    tools = build_tools(runner)
    model = build_model(tools)

    def call_model(state: AgentState) -> dict:
        """The single agent node. Every iteration enters here."""
        # Prepended fresh each time, so iteration five is still bound by the same rules as
        # iteration one - including how to format listings the tools have just returned.
        started = time.perf_counter()
        try:
            response = model.invoke([SystemMessage(content=system_prompt), *state["messages"]])
        finally:
            usage.record_seconds("reply", time.perf_counter() - started)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile()


def run_agent(runner: Any, system_prompt: str, messages: list) -> str | None:
    """One graph invocation. LangGraph handles agent -> tools -> agent internally.

    Returns the agent's final text, or None when the loop could not produce one - which the
    caller reads as "fall back to the deterministic reply".
    """
    from langgraph.errors import GraphRecursionError

    graph = build_agent_graph(runner, system_prompt)
    try:
        result = graph.invoke(
            {"messages": messages},
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except GraphRecursionError:
        # The agent kept asking for tools. Whatever it wanted, it is not converging, and a
        # deterministic reply beats an endless one.
        logger.error("Agent loop hit the recursion limit of %d steps", RECURSION_LIMIT)
        return None
    except Exception:
        logger.exception("Agent loop failed")
        return None

    final = result["messages"][-1]
    if not isinstance(final, AIMessage):
        logger.error("Agent loop ended on %s, not an assistant message", type(final).__name__)
        return None

    # Read ``content`` rather than ``.text``: across langchain-core versions .text has been a
    # method, a property, and a deprecated callable wrapper, and touching it emits warnings on
    # some of them. content's shape is stable - a string, or Responses-API blocks.
    content = final.content
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text", None}
        )
    else:
        text = content
    text = str(text or "").strip()

    # Only what THIS invocation produced. result["messages"] opens with the chat history we
    # passed in, and earlier bot replies are AIMessages too - counting them billed a model
    # call for every past reply in the window. A live lookup turn made 3 requests and was
    # reported as 4. add_messages appends, so the new messages are everything past the input.
    produced = result["messages"][len(messages):]
    _record_usage(produced)

    tool_calls_made = sum(1 for message in produced if getattr(message, "tool_calls", None))
    logger.info(
        "AGENT loop done: messages=%d tool_rounds=%d chars=%d",
        len(result["messages"]), tool_calls_made, len(text),
    )
    return text or None


def _record_usage(messages: list) -> None:
    """One usage record per model response, counted from the messages themselves.

    The loop makes as many calls as it makes, so the count is read back off the transcript
    rather than incremented at a call site - which keeps ``usage.chat_completions`` an
    honest answer to "how many times did we invoke the model this turn?".
    """

    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        tokens = getattr(message, "usage_metadata", None) or {}
        usage.record_completion(
            settings.chat_model,
            prompt_tokens=tokens.get("input_tokens", 0) or 0,
            completion_tokens=tokens.get("output_tokens", 0) or 0,
            purpose="reply",
            cached_tokens=(tokens.get("input_token_details") or {}).get("cache_read", 0) or 0,
        )
