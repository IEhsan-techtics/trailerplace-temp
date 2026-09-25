"""Send the model's own reply, when it passes Python's checks. LLM_WRITES_REPLY only.

With the flag on, the one call writes the whole message (``reply``) and names the question in
it (``asked_slots``). The model wrote that BEFORE ``apply_node`` ran, so it could not know what
Python then decided: a value rejected, a question skipped by a rule, a confirmation to ask. So
the reply is checked here against the state as it is now, and either sent as written or turned
down - and a turned-down reply costs nothing, because ``compose_node`` assembles the reply from
the model's pieces exactly as it does with the flag off.

Checked, never edited. Nothing in here rewrites the model's text: the regex surgery in compose
is what turned "You're welcome! What material will you be hauling?" into "You're welcome
material will you be hauling?" live, and a reply that needs cutting is a reply to turn down.

The rules held here are the same ones compose holds:

* one question per reply (the contact request rides along, as it does in compose);
* never a question that is answered, declined or already asked twice (brief S29, S23);
* the contact request only when the gate says it is due;
* Python's own questions - confirmations, a rejected value - are asked by Python.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.graph.nodes import greeting
from src.tools.questions import is_answered, is_resolved, mark_asked, required_remaining

logger = logging.getLogger(__name__)

# Questions Python asks in its own words. While one is open the reply is compose's: the
# model wrote before it knew one would be opened, and the customer must see the question the
# next turn is going to read their answer against.
_PYTHON_QUESTIONS = (
    "pending_gooseneck_clarification", "pending_category_switch", "pending_keep_filters",
    "pending_axle_basis", "pending_axle_count", "invalid_retry_slot",
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def use_model_reply(state: dict, output: Any, situation: str = "flow") -> bool:
    """Send ``output.reply`` as this turn's reply if it passes. True when it was used.

    ``situation`` is the kind of turn compose is on: "flow" for an ordinary one, or one of
    the turns that used to be wholly ours - "off_topic" - which add checks of their own.
    """
    reply = str(getattr(output, "reply", "") or "").strip()
    asked = [str(slot).strip() for slot in (getattr(output, "asked_slots", None) or []) if str(slot).strip()]

    problem = _problem(state, reply, asked, situation)
    if problem:
        logger.info(
            "COMPOSE fell back: session=%s situation=%s reason=%s asked_slots=%s reply=%r",
            state.get("session_id"), situation, problem, asked, reply,
        )
        return False

    from src.graph.nodes.compose import _contact_ask_is_due, _with_handoff

    outcome = state.setdefault("turn_outcome", {})
    contact = state.setdefault("contact", {})
    if _asks_for_contact(reply):
        greeting.note_asked(state)
    elif _contact_ask_is_due(state):
        # The one thing added rather than checked, and only on the end: the request is a
        # courtesy the gate has decided is due, and leaving the lead unasked is the costlier
        # mistake. Same as compose does when the model did not ask in its own words.
        reply = f"{reply} {greeting.gate_ask(state)}"
        greeting.note_asked(state)
    if greeting.contact_is_complete(contact) and not contact.get("greeted"):
        # The turn their details came in. compose would say its welcome here; the model's
        # reply has already acknowledged them, so it only has to be recorded as said.
        contact["greeted"] = True
        contact["asked"] = True

    slot = asked[0] if asked else None
    outcome["assistant_text"] = _with_handoff(outcome, reply)
    outcome["asked_slot"] = slot
    if slot:
        mark_asked(state, slot)
    logger.info(
        "COMPOSE used the model's reply: session=%s situation=%s asked_slot=%s",
        state.get("session_id"), situation, slot,
    )
    return True


def _problem(state: dict, reply: str, asked: list[str], situation: str = "flow") -> str | None:
    """Why this reply cannot go out as written, or None when it can."""
    from src.graph.nodes.compose import _contact_ask_is_due, _names_too_many_categories

    outcome = state.get("turn_outcome") or {}
    contact = state.get("contact") or {}

    if not reply:
        return "no reply written"
    if situation == "off_topic":
        problem = _off_topic_problem(state, reply)
        if problem:
            return problem

    # ---- turns that belong to Python whatever the model wrote ----
    for key in _PYTHON_QUESTIONS:
        if state.get(key):
            return f"python asks this turn ({key})"
    if outcome.get("search_ran"):
        return "a search ran"
    if outcome.get("wants_results") and not state.get("category"):
        return "results asked for with no category"
    if greeting.contact_is_complete(contact) and not contact.get("greeted") and greeting.is_first_turn(state):
        return "first-turn welcome"
    if not state.get("category") and state.get("listing_interest_logged"):
        return "they already picked a trailer"

    # ---- one question ----
    questions = [q for q in _questions(reply) if not _asks_for_contact(q)]
    if not state.get("category") and _one_category_question(questions):
        questions = [" ".join(questions)]
    if len(questions) > 1:
        return f"{len(questions)} questions"
    if len(asked) > 1:
        return f"{len(asked)} slots claimed"

    # ---- the question it asks is one it may ask ----
    remaining = required_remaining(state) if state.get("category") else []
    if asked:
        slot = asked[0]
        if slot not in remaining:
            if is_answered(state, slot):
                return f"{slot} is already answered"
            if is_resolved(state, slot):
                return f"{slot} is declined or asked twice"
            return f"{slot} is not a question to ask"
        if not questions:
            return f"claims {slot} but asks nothing"
        if not _is_about(questions[0], slot):
            return f"question is not about {slot}"
    elif questions:
        # A question with no slot claimed: fine when there is no slot to ask (the category
        # question, "anything else I can help with?"), not when it is really a slot question
        # the model forgot to name - that ask would never be counted.
        if remaining:
            return "asks a question but names no slot"
    elif remaining and not (_asks_for_contact(reply) and _contact_ask_is_due(state)):
        # Nothing asked with questions still open stalls the flow: compose would have asked.
        # Unless the one question it asks is for their details, when those are due - one
        # question per reply, and that one counts.
        return "asks nothing with questions left"

    # A question about something they already told us. Only when it is not also about the
    # slot it claims: the axle slots share one topic, and a question about the one being
    # asked is not a re-ask of its neighbour.
    if questions and not (asked and _is_about(questions[0], asked[0])):
        for known in _answered_slots(state):
            if _is_about(questions[0], known):
                return f"re-asks {known}"

    if not state.get("category") and questions and _names_too_many_categories(questions[0]):
        return "names too many categories"

    # ---- the contact request ----
    if _asks_for_contact(reply) and not _contact_ask_is_due(state):
        return "asks for contact details when it is not due"

    return None


def _off_topic_problem(state: dict, reply: str) -> str | None:
    """An off-topic reply declines in one short line and does not do what they asked.

    Measured on what is left once the questions and the opening are taken out - that is the
    decline. The cap is scope's own, and it is there because live the model opened "I mainly
    help with trailers, but here you go:" and gave the recipe; a real decline never runs long.
    """
    from src.graph.nodes.compose import _OPENING_RE
    from src.tools.scope import MAX_DECLINE_CHARS

    if "```" in reply:
        return "off-topic reply carries code"
    if greeting.is_first_turn(state) and not _OPENING_RE.search(reply):
        return "off-topic first reply has no welcome"
    decline = " ".join(
        s for s in _SENTENCE_END.split(reply.replace(greeting.OPENING, ""))
        if s.strip() and not s.strip().endswith("?") and not _asks_for_contact(s)
    ).strip()
    if not decline:
        return "off-topic reply does not decline"
    if len(decline) > MAX_DECLINE_CHARS:
        return f"off-topic decline runs {len(decline)} chars"
    return None


def _questions(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip().endswith("?")]


# How a request starts when it is not phrased as a question: "Please share your name...".
_REQUEST_OPENING = re.compile(r"^(please|kindly|feel free|when you (get|have) a (moment|chance)|share|send|drop|let me)\b", re.IGNORECASE)


def _asks_for_contact(text: str) -> bool:
    """Whether the text REQUESTS their details, sentence by sentence.

    compose's pattern matches the words, and "Understood - no contact details needed." has
    the words: live, that acknowledgement of a refusal read as asking again and the reply
    was turned down. A request is a question, or a sentence that opens like one.
    """
    from src.graph.nodes.compose import _already_asks_for_contact

    for sentence in _SENTENCE_END.split(text or ""):
        sentence = sentence.strip()
        if not _already_asks_for_contact(sentence):
            continue
        if sentence.endswith("?") or _REQUEST_OPENING.search(sentence):
            return True
    return False


def _one_category_question(questions: list[str]) -> bool:
    """The type question in the shape our own menu uses: "What type of trailer are you
    looking for? We have Utility, Dump, ... - which one fits what you need?" Two question
    marks, one question - it is word for word what greeting.orientation_question sends."""
    from src.graph.nodes.compose import _category_names_in

    return len(questions) == 2 and _is_about(questions[0], "base_category") and len(
        _category_names_in(questions[1])
    ) >= 2


def _is_about(question: str, slot: str) -> bool:
    """Whether a question is recognisably about this slot. A slot with no topic list (one
    added in the rules panel) cannot be told apart, so it is taken on trust."""
    from src.graph.nodes.compose import _SLOT_TOPICS, _collapse
    from src.graph.nodes.compose import _is_about as about

    if slot not in _SLOT_TOPICS:
        return True
    return about(_collapse(question), slot)


def _answered_slots(state: dict) -> list[str]:
    from src.graph.nodes.compose import _SLOT_TOPICS

    return [slot for slot in _SLOT_TOPICS if is_answered(state, slot)]
