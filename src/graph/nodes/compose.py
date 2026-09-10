"""Assemble the reply. No LLM calls - the prose was written by the single call already.

The model proposes a next question; this node decides. ``next_unanswered_slot`` picks the
slot independently, and the model's phrasing is used only when it names that same slot.
That is what makes "never re-ask an answered question" (brief S29) unbreakable by a bad
proposal rather than merely unlikely.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.domain import company
from src.domain import gooseneck as gooseneck_domain
from src.graph.nodes import greeting
from src.tools.questions import mark_asked, next_unanswered_slot, question_text

logger = logging.getLogger(__name__)

_RETRY_PREFIX = {
    "negative": "That came through as a negative number, which I don't think is what you meant.",
    "axle_range": "We carry trailers with one to four axles.",
}


def compose_node(state: dict, output: Any) -> dict:
    """Build ``turn_outcome["assistant_text"]`` and record what was asked."""
    outcome = state.setdefault("turn_outcome", {})

    # The reply pass already wrote this turn (it had listings to present, so the model saw
    # them through a tool and formatted the cards itself). Nothing here reassembles it - the
    # only job left is recording which trailers the customer was actually shown.
    if outcome.get("reply_text"):
        outcome["assistant_text"] = _with_handoff(outcome, outcome["reply_text"])
        outcome["asked_slot"] = None
        _record_shown(state, outcome.get("cited_listing_urls") or [])
        _note_contact_ask(state, outcome["assistant_text"])
        return state

    # The reply pass was meant to handle this turn and failed. Someone who has just told us
    # something went wrong must not be answered with the next qualification question, so the
    # fallback is the canned line rather than the flow.
    if outcome.get("needs_a_person"):
        outcome["assistant_text"] = _with_handoff(outcome, _person_fallback(state, output, outcome))
        outcome["asked_slot"] = None
        _note_contact_ask(state, outcome["assistant_text"])
        return state

    # The welcome turn is written entirely by greeting.py, not assembled from the model's
    # pieces: the wording is the dealership's, it has to read the same every time, and the
    # contact request must not have a qualification question competing with it.
    if greeting.contact_gate_applies(state):
        contact = state.setdefault("contact", {})
        outcome["assistant_text"] = _gate_text(state, output)
        outcome["asked_slot"] = None
        contact["asked"] = True
        contact["asks_without_progress"] = int(contact.get("asks_without_progress") or 0) + 1
        return state

    parts: list[str] = []
    contact = state.setdefault("contact", {})
    closing_the_gate = greeting.contact_is_complete(contact) and not contact.get("greeted")

    acknowledgement = (getattr(output, "acknowledgement", "") or "").strip()
    answer = (getattr(output, "answer_to_customer_question", None) or "").strip()

    if closing_the_gate:
        # The model writes this welcome too - it can use their name and react to what they
        # actually said. Checked, not trusted: the opening is the one line guaranteed to be
        # said, so if the model's words do not carry it, ours do.
        opening = _opening_text(state, acknowledgement)
        parts.append(opening)
        # The welcome is mandatory, so here it is the ANSWER that goes if they say the same
        # thing: "Great to have your contact info, Ibrahim! Thanks, Ibrahim - I've got your
        # contact information."
        if answer and _restates(answer, opening):
            logger.info(
                "COMPOSE dropped an answer restating the welcome: session=%s",
                state.get("session_id"),
            )
            answer = ""
    else:
        # Nothing is mandatory here, so the shorter, poorer line goes and the richer one
        # stays: an acknowledgement is by design a compressed version of the answer beside
        # it, which is exactly what makes it a duplicate when both go out.
        if acknowledgement and answer and _restates(acknowledgement, answer):
            logger.info(
                "COMPOSE dropped an acknowledgement restating the answer: session=%s",
                state.get("session_id"),
            )
            acknowledgement = ""
        if acknowledgement:
            parts.append(acknowledgement)

    if answer:
        parts.append(answer)

    if closing_the_gate:
        contact["greeted"] = True
        contact["asked"] = True

    # Their stashed request went out this turn. Say so before anything else we tack on: they
    # were asked for a name and a number several turns ago and told it was so the team could
    # follow up, and the turn that finally makes that true should not pass in silence.
    flushed = _handoff_line(int(outcome.get("emails_flushed") or 0))
    if flushed and not _MENTIONS_HANDOFF.search(" ".join(parts)):
        parts.append(flushed)

    closing, asked_slot = _closing_part(state, output)
    if closing and _repeats(parts, closing):
        # The model answered the question AND proposed the same question as its next one, so
        # the reply asked "what type of trailer are you looking for?" twice in a row. The
        # slot is still marked asked below - it WAS asked, once.
        logger.info(
            "COMPOSE dropped a closing question the reply already asked: session=%s",
            state.get("session_id"),
        )
        closing = ""
    if closing:
        parts.append(closing)

    # Counted here, where the question actually goes out, so the cap reflects what the
    # customer was shown rather than what a node intended to show them.
    if asked_slot:
        mark_asked(state, asked_slot)

    if _contact_ask_is_due(state):
        # The model usually asks for itself - the prompt tells it to, and its wording fits
        # the conversation better than a fixed line. Appended ONLY when it did not, so the
        # customer is never asked the same thing twice in one breath.
        if not _already_asks_for_contact(" ".join(parts)):
            parts.append(CONTACT_ASK)
        state.setdefault("contact", {})["asked"] = True

    text = " ".join(part for part in parts if part).strip()
    if not text:
        text = "Sorry, could you say that another way?"

    outcome["assistant_text"] = text
    outcome["asked_slot"] = asked_slot
    return state


# Ways a reply already says the request reached a person. Matched so the confirmation is not
# appended on top of one the agent wrote in its own words.
_MENTIONS_HANDOFF = re.compile(
    r"(passed (it|this|that|your \w+)?\s*(on\s*)?to (our|the) team|"
    r"(let|told) (our|the) team know|(our|the) team (has been|have been|is being) "
    r"(notified|told|informed)|notified (our|the) team|logged your (request|interest)|"
    r"(our|the) team will (be in touch|follow up|get back|reach out))",
    re.IGNORECASE,
)


def _handoff_line(count: int) -> str:
    """Confirmation that a stashed notification has now gone out."""
    if count <= 0:
        return ""
    what = "request" if count == 1 else "requests"
    return f"I've passed your {what} on to our team - they'll follow up with you shortly."


def _with_handoff(outcome: dict, text: str) -> str:
    line = _handoff_line(int(outcome.get("emails_flushed") or 0))
    if not line or _MENTIONS_HANDOFF.search(text or ""):
        return text
    return f"{text} {line}".strip()


def _note_contact_ask(state: dict, text: str) -> None:
    """Count a contact request this node did not write itself.

    ``asks_without_progress`` is what eventually stops us asking. A turn owned by the agent
    or by the escalation fallback can carry the very same request, so it has to count the
    same way - otherwise an escalation that asks for a name every turn never trips the cap.
    """
    contact = state.setdefault("contact", {})
    if contact.get("declined") or greeting.contact_is_complete(contact):
        return
    if not _already_asks_for_contact(text):
        return
    contact["asked"] = True
    contact["asks_without_progress"] = int(contact.get("asks_without_progress") or 0) + 1


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _repeats(parts: list[str], closing: str) -> bool:
    """Is this closing line already sitting in what the reply says?"""
    collapsed = _collapse(closing)
    return bool(collapsed) and collapsed in _collapse(" ".join(parts))


_FALLBACK_REASON = {"complaint": "Escalation", "team_request": "Team Request"}


# Words that carry no meaning of their own, so two sentences sharing only these are not
# saying the same thing. The politeness openers are in here on purpose: "Yes", "Thanks" and
# "Great" are exactly what a restatement is padded with.
_STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from get got had has
    have her here his i if in into is it its just like me my no not of on or our ours out over
    she so than that the their them then there these they this to too us was we were what when
    which who will with would you your yes yeah sure thanks thank great glad happy okay ok
    absolutely definitely certainly sorry also still now""".split()
)

# How much of the shorter line has to be present in the longer one before it is a restatement
# rather than a second point. Deliberately short of 1.0: "Yes, we do offer financing" against
# "We offer financing. Call 979-532-1486..." is a duplicate even though "yes" appears once.
_RESTATEMENT_OVERLAP = 0.7


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9:]+", str(text or "").lower())
        if word not in _STOPWORDS
    }


def _covered_by(word: str, body: set[str]) -> bool:
    """Is this word already in the other line, allowing for a shortened form of it?

    Prefix-matched because the restatement is rarely word-for-word - "your contact info"
    against "your contact information" is the same sentence twice.
    """
    if word in body:
        return True
    return any(
        len(other) >= 4 and len(word) >= 4 and (other.startswith(word) or word.startswith(other))
        for other in body
    )


def _restates(candidate: str, established: str) -> bool:
    """Does ``candidate`` say only what ``established`` already says?"""
    words = _content_words(candidate)
    body = _content_words(established)
    if not words or not body:
        return False
    covered = sum(1 for word in words if _covered_by(word, body))
    return covered / len(words) >= _RESTATEMENT_OVERLAP


def _person_fallback(state: dict, output: Any, outcome: dict) -> str:
    """What to say when the agent could not write the reply on a turn that needed a person.

    Never a qualification question: the customer asked for something we cannot do, and asking
    what they will be hauling reads as not having listened at all.

    It also RAISES the request. Only the agent's escalate tool used to do that, so on any turn
    the agent did not run - it failed, or it was never reached - the customer was told their
    request had been passed on and the team heard nothing. The canned line promises a
    follow-up, so something has to make that promise true.
    """
    from src.domain import canned_responses
    from src.tools import team_notify

    key = "complaint" if outcome.get("escalation_owns_turn") else "team_request"
    status = team_notify.record(
        state,
        reason=_FALLBACK_REASON[key],
        description=(getattr(output, "turn_summary", "") or "").strip()
        or "Customer asked for something the chatbot cannot do.",
    )

    answer = canned_responses.escalation_answer(key, status)
    if status != "stashed":
        return answer
    # Acknowledge first, ask second. A customer who has just told us something went wrong is
    # answered before they are asked for anything.
    return f"{answer} {team_notify.ask_for_missing(state)}"


def _record_shown(state: dict, urls: list[str]) -> None:
    """Mark the trailers the reply actually cited as shown.

    Recorded from what the model CITED, not from what the search returned: a "show me more"
    must exclude what the customer has seen, and locking out trailers that never reached the
    reply would hide them for the rest of the conversation.
    """
    if not urls:
        return
    state["results_shown"] = True
    shown = state.setdefault("shown_urls", [])
    for url in urls:
        cleaned = str(url or "").strip()
        if cleaned and cleaned not in shown:
            shown.append(cleaned)


CONTACT_ASK = "Also - who am I speaking with, and what's the best email or phone to reach you on?"

# Ways the model phrases the same request. Matched so the deterministic line is not appended
# on top of one the model already wrote.
_CONTACT_ASK_RE = re.compile(
    r"\b(your name|who am i speaking|who do i have|may i (get|have) your|"
    r"best (email|phone|number|way to reach)|reach you (on|at)|"
    r"email or phone|phone or email|contact (details|info))",
    re.IGNORECASE,
)


# What a welcome sounds like on a LATER turn - any warm acknowledgement will do, because the
# customer has already been properly greeted.
_WELCOME_RE = re.compile(
    r"\b(thank you for contacting|thanks for (contacting|reaching out)|"
    r"great to (hear from|have)|welcome|good to hear from|nice to meet)\b",
    re.IGNORECASE,
)

# What the FIRST reply must contain, no matter what the customer opened with. The thank-you
# for getting in touch is the one sentence every conversation is guaranteed to carry, so a
# merely friendly "Welcome, Ibrahim" is not enough to replace the written opening.
_OPENING_RE = re.compile(
    r"\bthank(s| you)? (you )?for (contacting|choosing|getting in touch)", re.IGNORECASE
)


def _already_asks_for_contact(text: str) -> bool:
    return bool(_CONTACT_ASK_RE.search(text or ""))


# Naming a few types helps; reading out all thirteen is a wall of names, not an answer.
# Six is the most a helpful sentence carries, so past that the written question stands in.
MAX_CATEGORIES_IN_A_QUESTION = 6


def _names_too_many_categories(text: str) -> bool:
    """A last-resort guard, not a policy: the prompt asks for four or five, and the model
    still recited all thirteen. Counted rather than rewritten, so the model keeps its own
    wording whenever it stays within the limit."""
    from src.domain.categories import CANONICAL_CATEGORIES

    low = (text or "").lower()
    return sum(1 for c in CANONICAL_CATEGORIES if c.lower() in low) > MAX_CATEGORIES_IN_A_QUESTION


def _welcomes_them(text: str, *, first_turn: bool) -> bool:
    """Does this text greet the customer well enough to stand as the opening?

    Stricter on the first turn: it must actually thank them for getting in touch. A model
    that opened with "Welcome, Ibrahim - we're glad to help" is friendly but not the
    opening the dealership asked for, and turn one is the only chance to say it.
    """
    if not text:
        return False
    if first_turn:
        return bool(_OPENING_RE.search(text))
    return bool(_WELCOME_RE.search(text))


def _opening_text(state: dict, acknowledgement: str) -> str:
    """The welcome on the turn contact becomes complete.

    On the FIRST turn the opening line is mandatory - "thank you for contacting us, I can
    help" is the one sentence every conversation is guaranteed to contain, and a customer
    who introduced themselves fully should not get a colder greeting than one who did not.
    So the model's version is used when it carries a welcome, and ours when it does not.

    On a later turn there is no such guarantee to keep: they have already been greeted, and
    a short "great to have your details" is the right size of acknowledgement.
    """
    written = (acknowledgement or "").strip()
    first_turn = _is_first_turn(state)
    if written and _welcomes_them(written, first_turn=first_turn):
        return written
    fallback = greeting.completion_greeting(state)
    if written and not first_turn:
        # Not a welcome, but still their acknowledgement - keep both.
        return f"{fallback} {written}"
    logger.info(
        "COMPOSE used the written welcome: session=%s model_welcomed=%s",
        state.get("session_id"), bool(written),
    )
    return fallback


def _is_first_turn(state: dict) -> bool:
    return int(state.get("turn_index") or 0) <= 1


def _gate_text(state: dict, output: Any) -> str:
    """The welcome reply: the model's words when they do the job, ours when they do not.

    The model writes a better greeting than a template does - it can answer their question,
    use their name, and react to what they actually said. But it is the one reply the
    conversation cannot afford to get wrong, so it is checked rather than trusted: if the
    model's text does not actually ASK for the missing details, the deterministic version
    goes out instead.

    Same shape as the next question - the model proposes, Python guarantees.
    """
    answer = (getattr(output, "answer_to_customer_question", None) or "").strip()
    written = " ".join(
        part
        for part in (
            (getattr(output, "acknowledgement", "") or "").strip(),
            answer,
            (getattr(output, "next_question_text", None) or "").strip(),
        )
        if part
    ).strip()

    if written and _already_asks_for_contact(written):
        return written

    logger.info(
        "COMPOSE fell back to the written greeting: session=%s model_asked=%s",
        state.get("session_id"), bool(written),
    )
    return greeting.gate_reply(state, answer)


def _contact_ask_is_due(state: dict) -> bool:
    """Whether the one-time request for a name and a way to reach them is due now.

    Appended to whatever else the reply says rather than competing with it. A customer whose
    first message is "looking for a 20ft livestock trailer" has answered every required
    question at once, so the reply is a set of listings - and if the opener only ever went
    out instead of a question, that customer would never be asked at all and the lead would
    be lost on the very turn we learned most about them.

    Asked once. Ignored, answered or refused, it is never raised again.
    """
    contact = state.get("contact") or {}
    if contact.get("asked") or contact.get("declined"):
        return False
    if any(contact.get(field) for field in ("name", "email", "phone")):
        return False
    # The opener is a first-message courtesy. Later on it reads as an interruption.
    return int(state.get("turn_index") or 0) <= 1


def _closing_part(state: dict, output: Any) -> tuple[str, str | None]:
    """The last thing the reply says: listings, a confirmation, or the next question."""
    outcome = state.get("turn_outcome") or {}

    # 1. A search ran. The listings ARE the answer; no question is appended to them.
    if outcome.get("search_ran"):
        return _render_listings(state, outcome), None

    # 2. They asked to see trailers with no category chosen: no search, the website (S25).
    #
    # Unless we just answered them. "Show me all your trailer types" reads to the model as a
    # results request, and the customer then got the answer, the whole category list and the
    # website line in one breath - three ways of saying the same thing. If a question was
    # answered this turn, the right next move is to ask which type fits, not to send them
    # away.
    if outcome.get("wants_results") and not state.get("category"):
        if (getattr(output, "answer_to_customer_question", None) or "").strip():
            return greeting.orientation_question(state), None
        return company.website_redirect_line(), None

    # 3. A pending confirmation outranks a new question - it is about what they just said.
    if state.get("pending_gooseneck_clarification"):
        return gooseneck_domain.clarification_question(), None

    switch = state.get("pending_category_switch")
    if switch:
        return _switch_question(switch), None

    keep = state.get("pending_keep_filters")
    if keep:
        return _keep_filters_question(keep), None

    # 4. No category yet. There is no slot to ask about, but there is still a conversation
    #    to carry: the contact opener on turn one, and "what will you be hauling?" after
    #    that. The model writes it - the whole situation is judgement, not bookkeeping - and
    #    a fallback covers a turn where it wrote nothing, so the reply is never a bare
    #    greeting that leaves the customer with nothing to answer.
    if not state.get("category"):
        # The most important question once we know who they are: WHICH TYPE. The model
        # writes it - it can pick the types that fit what they have already said, which a
        # fixed sentence cannot - and ours stands in when it writes nothing, or when it
        # tries to read the whole catalogue out.
        proposed = (getattr(output, "next_question_text", None) or "").strip()
        if proposed and not _names_too_many_categories(proposed):
            return proposed, None
        if proposed:
            logger.info(
                "COMPOSE replaced a question naming too many categories: session=%s",
                state.get("session_id"),
            )
        return greeting.orientation_question(state), None

    # 5. The next required question. Python picks the slot; the model may phrase it.
    slot = next_unanswered_slot(state)
    if not slot:
        return "", None

    proposed_slot = getattr(output, "next_question_slot", None)
    proposed_text = (getattr(output, "next_question_text", None) or "").strip()
    if proposed_slot == slot and proposed_text:
        question = proposed_text
    else:
        question = question_text(state, slot)
        if proposed_slot and proposed_slot != slot:
            logger.info(
                "COMPOSE overrode the model's question: session=%s proposed=%s using=%s",
                state.get("session_id"), proposed_slot, slot,
            )

    # A rejected value is re-asked with the reason attached, so the correction is in front
    # of them rather than implied by asking the same thing again.
    if state.get("invalid_retry_slot") == slot:
        prefix = _RETRY_PREFIX.get(str(state.get("invalid_retry_reason")), "")
        if prefix:
            question = f"{prefix} {question}"

    return question, slot


def _switch_question(switch: dict) -> str:
    """Offer the better-suited category, saying why."""
    suggested = switch.get("suggested")
    haul_item = switch.get("from_haul_item")
    return (
        f"For {haul_item}, an {suggested} trailer is usually the better fit - "
        f"would you like to switch to that instead?"
        if str(suggested or "")[:1].upper() in "AEIOU"
        else f"For {haul_item}, a {suggested} trailer is usually the better fit - "
        f"would you like to switch to that instead?"
    )


def _keep_filters_question(keep: dict) -> str:
    """Ask whether to carry the collected filters over.

    Only slots with real values are named - offering to keep a blank is noise (brief S12).
    """
    filters = keep.get("filters") or {}
    described = ", ".join(_describe(slot, value) for slot, value in filters.items())
    new_category = keep.get("new_category")
    return (
        f"Switching to {new_category}. You'd told me {described} - "
        f"should I keep those, or start fresh?"
    )


def _describe(slot: str, value: Any) -> str:
    label = slot.replace("_", " ")
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{label} {value}"


def _render_listings(state: dict, outcome: dict) -> str:
    """Deterministic listing cards.

    Written here rather than by the model so the trailers presented are exactly the
    trailers the search returned - the model never gets the chance to describe a listing
    it was not handed.
    """
    listings = outcome.get("listings") or []
    if not listings:
        return _no_results_line(state)

    state["results_shown"] = True
    lines: list[str] = ["Here's what fits:"]
    for index, listing in enumerate(listings, start=1):
        lines.append(_listing_line(index, listing))
        url = listing.get("url")
        if url:
            state.setdefault("shown_urls", [])
            if url not in state["shown_urls"]:
                state["shown_urls"].append(url)

    if outcome.get("filters_relaxed"):
        dropped = outcome.get("relaxed_filters_dropped") or []
        if dropped:
            lines.append(
                f"Nothing matched on {', '.join(dropped)}, so these are the closest I have."
            )
    elif outcome.get("brand_relaxed"):
        lines.append("None from that brand right now, so here are the closest matches.")

    return "\n".join(lines)


def _listing_line(index: int, listing: dict) -> str:
    """One card. Only fields the row actually carries - no placeholders, no invention."""
    title = listing.get("title") or "Trailer"
    bits = [f"{index}. {_unescape(title)}"]
    price = listing.get("price_display") or listing.get("price")
    if price:
        bits.append(f"- {price}")
    if listing.get("url"):
        bits.append(f"- {listing['url']}")
    return " ".join(bits)


def _unescape(text: Any) -> str:
    """Titles in trailer_listings still carry raw HTML entities on some rows
    ("2026 P&amp;C Car Hauler"). Unescaped at display time only: the column is owned by
    ingest and is not ours to rewrite."""
    import html

    return html.unescape(str(text or ""))


def _no_results_line(state: dict) -> str:
    category = state.get("category") or "trailer"
    return (
        f"I couldn't find a {category} matching all of that right now. "
        f"You can see everything we have at {company.website()}, "
        f"or give us a call on {company.PHONE} and the team can help."
    )
