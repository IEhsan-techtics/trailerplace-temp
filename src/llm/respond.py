"""The reply-writing pass: the only place the model ever sees a trailer.

Runs AFTER ``apply_node``, and only on a turn the deterministic gate has opened - so at most
one of these happens per turn, and only when there is inventory to talk about. Every other
turn is still assembled by ``compose_node`` from the analysis pass's own pieces, exactly as
before.

Why a separate pass at all: the analysis call happens before apply, when ``state["slots"]``
still holds last turn's answers. A search fired from there filters on stale values. By the
time this runs, the turn has been folded in and the filters are current.

The model reaches the listings by CALLING A TOOL (src/llm/tools.py) rather than being handed
them, so a turn that needs no inventory costs no search - and the tool's own preconditions
are Python's, not the model's.
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.domain import brands, categories, company
from src.llm import usage
from src.llm.schemas import ReplyOutput
from src.llm.tools import TOOL_SPECS, ToolRunner

logger = logging.getLogger(__name__)

# One round is the normal case: the model calls search_inventory, reads the result, writes the
# cards. Two allows a follow-up (a lookup that came back ambiguous). Past that it is looping.
MAX_TOOL_ROUNDS = 2


_CARD_FORMAT = """
HOW TO PRESENT TRAILERS

Show EVERY listing the tool returned, in the exact order it returned them. They are already
filtered and ranked - do not drop one, add one, reorder them, or judge whether a size or
style "fits". A listing that looks like the odd one out still gets shown.

Each listing is ONE numbered card. Copy this shape exactly:

1. [2026 Iron Bull DTB - 15081](https://the-exact-url-from-the-tool)
   - Category: Utility
   - Make: Iron Bull Trailers
   - Price: $9,995
   - Length: 14 ft 0 in
   - Width: 6 ft 11 in
   - Payload: 4420 lbs
   - Axle capacity: 3500 lbs
   - Hitch type: Bumper Pull
   - Tandem 3,500 lb axles give you room for the skid steer you mentioned.

RULES FOR THE CARD - each one matters:
- The title is ALWAYS a markdown link to that listing's exact URL. A bare or bold title with
  no link is a failed card: the link is how the customer opens the trailer.
- Copy the title EXACTLY as it appears after "TITLE:", including the stock number on the end
  ("2026 Gooseneck Livestock - 91632", not "2026 Gooseneck Livestock"). Never read a spec out
  of a title - "15K" in a title is a model name, not a payload.
- EVERY field gets its own bullet on its own line. Never join fields with slashes or commas.
- A listing has ONLY the fields its tool line lists. If a field is absent, DELETE that bullet.
  Never write "None", "N/A", "Not specified", "unknown" or "Call for price", and never copy a
  value from another listing. A card with three bullets is correct if the tool gave three fields.
- Price is what customers care about most: when the tool gives a Price, its bullet is never omitted.
- The LAST BULLET of every card is a sales-pitch sentence of your own. It is a BULLET like
  all the others - same "   - " prefix, never a bare indented line - but it carries NO label
  in front of it (never "Pitch:", never "Description:"). Build it ONLY from that trailer's
  own fields and what the customer told us, and write a different one for every listing.
  Never invent a feature, spec, condition or price.

AFTER THE LAST CARD: one line offering a person, then ONE closing question, then STOP.
  "For a closer look at any of these, our sales team can walk you through them at 979-532-1486.
   Do any of these look like a fit, or would you like to see more options?"
Nothing else after the listings - no financing, delivery, trade-ins, visits, contact asks or
second questions.

IF THE TOOL RETURNED NO MATCHES, or said NO SEARCH RAN: show no cards, invent nothing, and say
nothing about our stock levels. Do what the tool's message tells you and ask the question it
names. "NO SEARCH RAN" means we have not looked yet - it is NOT an out-of-stock signal.
"""

_VOICE = """
VOICE
Professional, confident, helpful. 2-6 sentences outside the cards themselves.
Exactly ONE question mark in the whole reply.
NO exclamation marks, NO emojis, NO praise or filler ("Great choice", "Perfect", "Awesome",
"Excellent", "Thanks for sharing").
Never invent inventory, prices, specs or policies. A fact you were not given does not exist.
"""


def _state_line(state: dict, turn: Any) -> str:
    """What this customer has told us, so the pitch sentences can be about THEM."""
    slots = {
        name: value
        for name, value in (state.get("slots") or {}).items()
        if value not in (None, "", [])
    }
    contact = state.get("contact") or {}
    lines = [
        "WHAT WE KNOW ABOUT THIS CUSTOMER",
        f"- Trailer category: {state.get('category') or 'not chosen yet'}",
        f"- What they told us: {slots or 'nothing yet'}",
        f"- Their name: {contact.get('name') or 'unknown'}",
    ]
    if state.get("brand_preference"):
        lines.append(f"- Brand they asked for: {state['brand_preference']}")
    summary = (getattr(turn, "turn_summary", "") or "").strip()
    if summary:
        lines.append(f"- What they just said: {summary}")
    question = (getattr(turn, "user_question_to_answer", None) or "").strip()
    if question:
        lines.append(f'- Answer this first, in one or two sentences: "{question}"')
    return "\n".join(lines)


def build_system_prompt(state: dict, turn: Any) -> str:
    return "\n".join(
        [
            f"You are the sales assistant for {company.NAME}, a trailer dealership in "
            f"{company.LOCATION}. Write the next reply to this customer.",
            "",
            "You have tools. Call search_inventory when they want to see trailers that match "
            "what they have told us, and lookup_inventory when they name one specific trailer "
            "(a stock number, a year and make, or a make and model code). Call a tool BEFORE "
            "writing your reply - you have no other way to know what is on the lot.",
            "",
            _CARD_FORMAT.strip(),
            "",
            _VOICE.strip(),
            "",
            company.company_facts_block(),
            "",
            "OUR CATEGORIES:",
            categories.category_menu_block(),
            "",
            brands.make_prompt_block(),
            "",
            _state_line(state, turn),
        ]
    )


def _messages(state: dict, turn: Any, user_message: str) -> list[Any]:
    messages: list[Any] = [{"role": "system", "content": build_system_prompt(state, turn)}]
    for entry in (state.get("messages") or [])[-8:]:
        role = entry.get("role")
        content = entry.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": user_message})
    return messages


def _record(response: Any) -> None:
    tokens = getattr(response, "usage", None)
    usage.record_completion(
        settings.chat_model,
        prompt_tokens=getattr(tokens, "input_tokens", 0) or 0,
        completion_tokens=getattr(tokens, "output_tokens", 0) or 0,
        purpose="reply",
    )


def respond_with_tools(state: dict, turn: Any, user_message: str) -> ReplyOutput | None:
    """Write the reply, letting the model pull inventory through the tools.

    Returns None on any failure, which is the signal to fall back to the deterministic
    assembly in ``compose_node`` - a flat reply, never a broken turn.
    """
    from src.llm.client import get_client

    runner = ToolRunner(state, turn)
    messages = _messages(state, turn, user_message)

    try:
        response = get_client().responses.parse(
            model=settings.chat_model,
            input=messages,
            tools=TOOL_SPECS,
            text_format=ReplyOutput,
            reasoning={"effort": settings.chat_reasoning_effort},
        )
        _record(response)

        for _round in range(MAX_TOOL_ROUNDS):
            calls = [item for item in (response.output or []) if getattr(item, "type", "") == "function_call"]
            if not calls:
                break
            # Echo the model's own call items back before their results, which is what lets it
            # match each result to the call that asked for it.
            messages.extend(response.output)
            for call in calls:
                logger.info(
                    "TOOL call: session=%s name=%s args=%s",
                    state.get("session_id"), call.name, call.arguments,
                )
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": runner.call(call.name, call.arguments),
                    }
                )
            response = get_client().responses.parse(
                model=settings.chat_model,
                input=messages,
                tools=TOOL_SPECS,
                text_format=ReplyOutput,
                reasoning={"effort": settings.chat_reasoning_effort},
            )
            _record(response)
    except Exception:
        logger.exception("Reply pass failed: session=%s", state.get("session_id"))
        return None

    reply = getattr(response, "output_parsed", None)
    if reply is None or not (reply.assistant_text or "").strip():
        logger.error("Reply pass returned nothing usable: session=%s", state.get("session_id"))
        return None

    logger.info(
        "REPLY written: session=%s tools=%s cited=%d",
        state.get("session_id"), runner.ran, len(reply.cited_listing_urls or []),
    )
    return reply
