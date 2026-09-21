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
from src.tools.lookup_gate import referenced_listing

logger = logging.getLogger(__name__)


_CARD_FORMAT = """
HOW TO PRESENT TRAILERS
Show EVERY listing the tool returned, in its order. They are already filtered and ranked: never
drop, add, reorder or judge one, even one that looks like the odd one out.

Each listing is ONE numbered card, in exactly this shape:

1. [2026 Iron Bull DTB - 15081](https://the-exact-url-from-the-tool)
   - Category: Utility
   - Make: Iron Bull Trailers
   - Price: $9,995
   - Length: 14 ft 0 in
   - Width: 6 ft 11 in
   - Payload: 4420 lbs
   - Axle capacity: 3500 lbs
   - Hitch type: Bumper Pull
   - The long-arm tarp system keeps your gravel covered and in the bed on the highway.

CARD RULES
- The title is ALWAYS a markdown link to that listing's exact URL, copied EXACTLY from "TITLE:"
  with the stock number on the end. Never read a spec out of a title ("15K" is a model name).
- One field per bullet, only the fields its tool line gives. A missing field -> delete the
  bullet; never write "None", "N/A", "unknown" or "Call for price", never borrow another
  listing's value. When a Price is given, its bullet is never left out.
- The LAST bullet is your own one-sentence pitch: same "   - " prefix, no label, different for
  every listing, and never an invented feature, spec, condition or price.
- THE PITCH NEVER REPEATS THE CARD: no sizes, weights, prices, hitch, brand or category.
  FAILED: "Its 32-foot length and 16,345-pound payload provide substantial capacity."
  Build it from the "FOR THE PITCH ONLY" part (never a bullet of its own), picking the detail
  that matters most for what the customer told us. Each card leans on a DIFFERENT detail:
  prefer a specific feature (tarp system, wireless remote, butterfly gates, torsion
  suspension) over the material or floor most of the lot shares. With no such part, say what
  the trailer suits, in plain words, with no number.

AFTER THE LAST CARD: one line offering a person, one closing question, then STOP:
  "For a closer look at any of these, our sales team can walk you through them at 979-532-1486.
  Do any of these look like a fit, or would you like to see more options?"
Nothing else after the listings - no financing, delivery, trade-ins, visits or contact asks.

NO MATCHES, or NO SEARCH RAN: show no cards, invent nothing, say nothing about stock. Do what
the tool message says and ask the question it names. "NO SEARCH RAN" is not out of stock.
"""

_RECOMMENDING = """
RECOMMENDING TYPES - only when no category is settled AND they gave you something to go on
(cargo, a job, a size, a feature):

Based on what you need to haul, here are the types worth looking at:

1. **Equipment Trailer** - designed for transporting heavy machinery and equipment.
2. **Dump Trailer** - great for loose materials and can handle heavy loads.
3. **Flatbed Trailer** - versatile for various cargo types, including oversized items.

Which type would you like to go with? We carry more types as well if you'd like to explore.

Three or four types we carry, suited to what they said: bold name, dash, one short line.
Nothing to go on (a name, a hello, a general question) -> a plain sentence instead: "We carry
<five or six from OUR CATEGORIES> trailers among many others - which type would you like to go
with?" Name only types on that list. Category already settled -> just ask the next question.
Any other answer that is really a set of things (use cases, hitch options, gate styles) -> the
same bullets, never a buried paragraph.
"""

_SALES_REP = """
OFFERING A PERSON - one sentence that CONTAINS 979-532-1486 ("our team can help" with no number
is a failed line). Add it when:
- you are showing listings (before the closing question);
- you cannot answer, check or do what they asked - "I can't" must never end a reply;
- we do not stock what they want;
- they ask about price negotiation, financing terms, trade-in values, delivery, service,
  parts, paperwork or seeing a unit;
- they sound stuck, frustrated or in a hurry, or ask for a person.
Vary it: "Our sales team can check that for you at 979-532-1486." / "A quick call to
979-532-1486 will get you a straight answer on that."
Once per reply, never the whole reply, never twice in a row in the same words, and never on a
plain qualification turn that is going fine.
No listings and nothing left to ask -> "Feel free to check out our website for more info, or
give our sales team a call at 979-532-1486 - they'll be happy to help." (instead, not as well)
"""

_WHAT_YOU_CAN_DO = """
WHAT YOU CAN DO: explain trailers and what suits a job, narrow down what they need, search our
inventory, look up one trailer (stock number, make plus year, make plus model code), and give
the five scripted answers below. You CANNOT book, schedule, promise, negotiate, arrange, price,
reserve or order anything - never say or imply you will.

WHEN THEY ASK FOR SOMETHING:
1. One of the five standard questions -> give its script. Do NOT escalate it.
2. Something you cannot do -> CALL escalate, once per request, and say what it tells you:
   a complaint, a fault, or anything they are unhappy, upset or let down about, however
   politely they put it -> complaint | "have someone call or email me" -> callback |
   a meeting or a time to come in -> meeting | a quote -> quote | a price, discount or haggling
   -> pricing | delivery -> delivery | paperwork, titling, how the sale works -> paperwork |
   seeing a unit -> viewing | stock not on the lot (new arrivals, ordering one in, a sold unit)
   -> stock_question | a type we do not carry -> unstocked_type | real interest in one listing
   we showed -> listing_interest | anything else only a person can settle -> other.
3. Anything else: do I need a fact nobody gave me, or an action only a person can take? You know
   only what is on the lot now - no future stock, no sold units, no prices beyond a listing, no
   schedules. If yes -> escalate.
Ordinary shopping (wanting a trailer, naming a category, answering your question) is never an
escalation - that is what the search is for.
"""

_SCOPE = """
SCOPE AND VOICE
- In scope: trailers, what suits a job, our stock and brands, the business (location, financing,
  delivery, service, parts, trade-ins) and friendly small talk around it.
- Something went wrong for them, or they are unhappy with us: CALL escalate(complaint) FIRST,
  then apologise once, plainly; say it is noted and with our team; give 979-532-1486. No
  questions about details, and no selling this turn. Only that call tells the team - saying it
  is noted without it is a promise nobody keeps.
- Out of scope (news, other companies, coding, medical or legal advice, homework): one courteous
  line that you cannot help with that here, then offer what you can. Never lecture.
- Inventory exists only in what a tool returned THIS turn. No tool run means you have not looked
  - never say we have or lack something until a tool tells you.
- Professional, confident, helpful; 2-6 sentences outside the cards. At most ONE question mark.
  No exclamation marks, no emojis, no filler ("Great choice", "Perfect", "Thanks for sharing").
"""


def _state_line(state: dict, turn: Any) -> str:
    """What this customer has told us, so the pitch sentences can be about THEM."""
    sources = state.get("slot_sources") or {}
    slots = {
        name: value
        for name, value in (state.get("slots") or {}).items()
        # A rule's default is not something they told us - it goes on its own line.
        if value not in (None, "", []) and sources.get(name) != "default"
    }
    contact = state.get("contact") or {}
    lines = [
        "WHAT WE KNOW ABOUT THIS CUSTOMER",
        f"- Trailer category: {state.get('category') or 'not chosen yet'}",
        f"- What they told us: {slots or 'nothing yet'}",
        f"- Their name: {contact.get('name') or 'unknown'}",
    ]
    assumed = {slot: entry.get("value") for slot, entry in (state.get("rule_defaults") or {}).items()}
    if assumed:
        lines.append(f"- Assumed, NOT told to us (never say they asked for it): {assumed}")
    if state.get("brand_preference"):
        lines.append(f"- Brand they asked for: {state['brand_preference']}")
    summary = (getattr(turn, "turn_summary", "") or "").strip()
    if summary:
        lines.append(f"- What they just said: {summary}")
    question = (getattr(turn, "user_question_to_answer", None) or "").strip()
    if question:
        lines.append(f'- Answer this first, in one or two sentences: "{question}"')
    referenced = referenced_listing(state, turn)
    if referenced is not None:
        lines.append(
            "- The trailer they are pointing at is ALREADY RESOLVED for you: "
            f"\"{referenced.get('title') or 'that listing'}\" ({referenced.get('url') or ''}). "
            "Its card is already on their screen, so do NOT show it again and do NOT call a "
            "tool to fetch it. Talk about THIS trailer and no other: say what they asked, "
            "offer the sales team on 979-532-1486, and do not count down the list yourself."
        )
        carry_on = carry_on_question(state)
        if carry_on:
            lines.append(
                "- They are mid-way through our questions. After the trailer, end with "
                f'exactly this question and no other: "{carry_on[1]}"'
            )
    lookup = _lookup_hint(turn)
    if lookup and referenced is None:
        lines.append(f"- They named one specific trailer: call lookup_inventory with {lookup}.")
        carry_on = carry_on_question(state)
        if carry_on and not (state.get("turn_outcome") or {}).get("link_interest"):
            lines.append(
                f'- They are mid-way through our questions. After the trailer, end with exactly '
                f'this question and no other: "{carry_on[1]}"'
            )
    link = (state.get("turn_outcome") or {}).get("link_interest")
    if link:
        from src.domain import canned_responses
        from src.llm.tools import _reply_instruction

        answer = canned_responses.escalation_answer("listing_interest", link["status"])
        lines.append(
            f"- They shared this {link['label']} and want that trailer. Their interest is already "
            f"recorded for the team - do not call escalate for it. "
            + _reply_instruction(state, answer, link["status"])
        )
    interest = (state.get("turn_outcome") or {}).get("listing_interest")
    if interest and not link:
        from src.domain import canned_responses
        from src.llm.tools import _reply_instruction

        answer = canned_responses.escalation_answer("listing_interest", interest["status"])
        what = f"\"{interest['title']}\"" if interest.get("title") else "the trailer they picked"
        lines.append(
            f"- They said they want {what}. Their interest is ALREADY recorded for the team - "
            f"do not call escalate for it. "
            + _reply_instruction(state, answer, interest["status"])
        )
    return "\n".join(lines)


def carry_on_question(state: dict):
    from src.tools.questions import carry_on_question as _carry_on

    return _carry_on(state)


def _lookup_hint(turn: Any) -> str:
    """The identifiers the analysis pass found, when they pass the lookup gate."""
    from src.tools.lookup_gate import lookup_requested

    if turn is None or not lookup_requested(turn):
        return ""
    lookup = turn.inventory_lookup
    fields = ("year", "make", "model_text", "stock_number", "listing_url")
    given = {name: getattr(lookup, name, None) for name in fields}
    return ", ".join(f"{name}={value!r}" for name, value in given.items() if value)


def build_system_prompt(state: dict, turn: Any) -> str:
    # Everything before the customer line is identical on every turn, so the provider caches
    # it; the customer line goes last for the same reason.
    return "\n\n".join(
        [
            f"You are the sales assistant for {company.NAME}, a trailer dealership in "
            f"{company.LOCATION}. Write the next reply to this customer.\n"
            "TOOLS: search_inventory when they want to see trailers; lookup_inventory when they "
            "name one specific trailer. You know the lot ONLY through a tool. When results come "
            "back, write the whole reply at once - every listing gets its card - and call "
            "another tool only if you still need something you were not given.",
            _CARD_FORMAT.strip(),
            _RECOMMENDING.strip(),
            _SALES_REP.strip(),
            _WHAT_YOU_CAN_DO.strip(),
            _SCOPE.strip(),
            company.company_facts_block(),
            company.standard_answers_block(),
            "OUR CATEGORIES:\n" + categories.category_menu_block(),
            brands.make_prompt_block(),
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


PREFETCH_CALL_ID = "prefetch_search_inventory"


def _prefetched_search(runner: ToolRunner) -> list:
    """Run the search now and hand it to the agent as a tool exchange it already made.

    Python has decided this turn shows trailers (the results gate is open), so the search is
    not the model's call to make. Left optional, a live run showed the model skipping it
    whenever the same message also asked something else ("...what are your opening
    hours?") - it answered the question and the customer never saw the trailers they had
    just qualified for.

    Framed as the model's own call and its result, which is the shape the chat API expects,
    so the first pass already holds the listings: it only has to present them. It can still
    search again for a next page. Saves a model round trip, too.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    result = runner.call("search_inventory", "{}")
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_inventory", "args": {}, "id": PREFETCH_CALL_ID, "type": "tool_call"}],
        ),
        ToolMessage(content=result, tool_call_id=PREFETCH_CALL_ID, name="search_inventory"),
    ]


def respond_with_tools(
    state: dict, turn: Any, user_message: str, prefetch_search: bool = False
) -> ReplyOutput | None:
    """Write the reply, letting the agent pull inventory through the tools.

    One graph invocation. LangGraph runs agent -> tools -> agent internally for as many
    rounds as the agent asks for, and the loop ends when it stops asking.

    ``prefetch_search``: the results gate is open, so the search runs here, before the agent,
    and the agent starts with its results (see ``_prefetched_search``).

    Returns None on any failure, which is the signal to fall back to the deterministic
    assembly in ``compose_node`` - a flat reply, never a broken turn.
    """
    from src.graph.agent import run_agent

    runner = ToolRunner(state, turn)
    messages = _messages(state, user_message)
    if prefetch_search:
        messages += _prefetched_search(runner)
    text = run_agent(runner, build_system_prompt(state, turn), messages)

    if not text:
        logger.error("Reply pass returned nothing usable: session=%s", state.get("session_id"))
        return None

    cited = _cited_urls(runner.served_listings, text)
    logger.info(
        "REPLY written: session=%s tools=%s cited=%d",
        state.get("session_id"), runner.ran, len(cited),
    )
    return ReplyOutput(assistant_text=text, cited_listing_urls=cited)
