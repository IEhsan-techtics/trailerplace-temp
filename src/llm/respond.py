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

This module owns the PROMPT and the state translation. The loop itself is
``src/graph/agent.py``: one agent node that calls tools and reads their results until it has
an answer. There is no second model, and no separate node for summarising or formatting what
a tool returned - the same agent does all of it, deciding from the messages which job it is
on.
"""
from __future__ import annotations

import logging
from typing import Any

from src.domain import brands, categories, company
from src.llm.schemas import ReplyOutput
from src.llm.tools import ToolRunner

logger = logging.getLogger(__name__)


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
            "You have tools. Call search_inventory when they want to see trailers matching "
            "what they have told us, and lookup_inventory when they name one specific trailer "
            "(a stock number, a year and make, or a make and model code). Call a tool BEFORE "
            "writing your reply - you have no other way to know what is on the lot.",
            "",
            "WHEN A TOOL RETURNS RESULTS, you are shown them and you keep going in the same "
            "turn. Read ALL of them and write the whole reply at once - if it returns ten "
            "trailers, all ten get their card and their own one-line pitch in THIS reply. "
            "Never handle them one at a time and never ask for the same search twice. Call "
            "another tool ONLY if you genuinely still need something you were not given; "
            "otherwise write the final answer now.",
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


def _messages(state: dict, user_message: str) -> list:
    """The conversation the agent starts from.

    NO system message here - ``call_model`` prepends it on every iteration instead, so the
    prompt is as present on the agent's third pass as on its first.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    messages: list = []
    for entry in (state.get("messages") or [])[-8:]:
        role = entry.get("role")
        content = entry.get("content")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            messages.append(AIMessage(content=str(content)))
    messages.append(HumanMessage(content=user_message))
    return messages


def _cited_urls(listings: list, reply_text: str) -> list[str]:
    """Which of the trailers the tools returned actually made it into the reply.

    Taken from the tool results rather than asked of the model: the tools know exactly what
    they handed over, so a URL here can never be one the model invented, and a trailer the
    reply never mentioned is never marked as shown.

    Reads the runner's running total, not the last batch: an agent that searched twice cited
    trailers from both pages, and only the second would survive a last-batch check.
    """
    cited: list[str] = []
    for listing in listings:
        url = str((listing.get("url") if isinstance(listing, dict) else None) or "").strip()
        if url and url in reply_text and url not in cited:
            cited.append(url)
    return cited


def respond_with_tools(state: dict, turn: Any, user_message: str) -> ReplyOutput | None:
    """Write the reply, letting the agent pull inventory through the tools.

    One graph invocation. LangGraph runs agent -> tools -> agent internally for as many
    rounds as the agent asks for, and the loop ends when it stops asking.

    Returns None on any failure, which is the signal to fall back to the deterministic
    assembly in ``compose_node`` - a flat reply, never a broken turn.
    """
    from src.graph.agent import run_agent

    runner = ToolRunner(state, turn)
    text = run_agent(runner, build_system_prompt(state, turn), _messages(state, user_message))

    if not text:
        logger.error("Reply pass returned nothing usable: session=%s", state.get("session_id"))
        return None

    cited = _cited_urls(runner.served_listings, text)
    logger.info(
        "REPLY written: session=%s tools=%s cited=%d",
        state.get("session_id"), runner.ran, len(cited),
    )
    return ReplyOutput(assistant_text=text, cited_listing_urls=cited)
