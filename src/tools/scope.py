"""What this chatbot will and will not talk about. No LLM calls.

We sell trailers. A customer who wants a recipe, a python script or last night's score has
come to the wrong window, and answering them is worse than useless: it spends a turn, it
teaches them this is a general assistant, and the next question is harder to decline than
the first. So the reply says once that we only help with trailers, and goes straight back
to the question the conversation was on.

What is NOT off topic is deliberately broad, because the cost of the two mistakes is not
symmetric. Refusing a real customer loses a lead; answering a stray question costs one
turn. So "hello", "how are you", "thanks, you've been great" all pass, and so does anything
about us - hours, address, financing, delivery, a trailer, what it can haul.

Two decisions, and the model only makes one of them. It reads the message and says whether
it was off topic. Python decides what happens next, and holds the veto: a turn that carried
ANY trailer content is on topic whatever the model said, because that content could only
have come from a real customer answering us.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# A decline is one sentence: "I'm sorry, I only help with trailers here" is 46 characters.
# The cap is generous against that and still well under the shortest real answer to "how do
# I make a sandwich", so a proposed line over it is the model answering anyway behind an
# apology - live it opened "I mainly help with trailers, but here you go:" and gave the
# recipe. Ours goes instead.
MAX_DECLINE_CHARS = 120

DECLINE = "I'm sorry - I can only help with trailers and TrailerPlace here."

# The model may only call a turn off topic while it is also reading it as a question or as
# chat. Any other intent is it contradicting itself, and the intent is the field the rest of
# the turn is routed on.
_INTENTS_THAT_MAY_BE_OFF_TOPIC = {"general_question", "smalltalk_other"}


def _extracted_carries_anything(extracted: Any) -> bool:
    if extracted is None:
        return False
    for field in (
        "length", "width", "height", "payload_capacity", "axle_capacity",
        "total_axle_capacity_lbs", "axle_count", "hitch_type", "haul_item",
        "brand_preference", "non_metadata_features", "numeric_no_preference", "quantities",
    ):
        if getattr(extracted, field, None):
            return True
    return False


def _contact_carries_anything(contact: Any) -> bool:
    if contact is None:
        return False
    return bool(
        getattr(contact, "name", None)
        or getattr(contact, "email", None)
        or getattr(contact, "phone", None)
        or getattr(contact, "declined", False)
    )


def is_off_topic(output: Any) -> bool:
    """The model said so AND nothing in the turn contradicts it."""
    if not getattr(output, "off_topic", False):
        return False

    intent = str(getattr(output, "intent", "") or "")
    if intent not in _INTENTS_THAT_MAY_BE_OFF_TOPIC:
        logger.info("OFF-TOPIC ignored: intent=%r is not a question or chat", intent)
        return False

    # Every one of these is a customer telling us something about their trailer. A message
    # that carried one is on topic no matter what else was in it.
    on_topic_signals = (
        getattr(output, "category_mentioned", None),
        getattr(output, "unavailable_type_requested", None),
        getattr(output, "faq_key", None),
        getattr(output, "listing_reference", None),
        getattr(output, "shared_link_interest", False),
        getattr(output, "keep_fields_answer", None),
        getattr(output, "category_confirm_answer", None),
        getattr(output, "answered_current_question", False),
        getattr(output, "slot_answers", None),
        getattr(output, "dropped_fields", None),
        getattr(getattr(output, "inventory_lookup", None), "is_lookup", False),
    )
    if any(on_topic_signals):
        logger.info("OFF-TOPIC ignored: the turn also carried trailer content")
        return False
    if _extracted_carries_anything(getattr(output, "extracted", None)):
        logger.info("OFF-TOPIC ignored: the turn extracted a trailer field")
        return False
    if _contact_carries_anything(getattr(output, "contact", None)):
        logger.info("OFF-TOPIC ignored: the turn carried contact details")
        return False
    return True


def _decline_line(output: Any) -> str:
    """The model's wording when it is a decline; ours when it is anything else.

    A question mark disqualifies it: the steer below carries the turn's one question, and a
    reply that asks two gets neither answered.
    """
    proposed = " ".join(str(getattr(output, "acknowledgement", "") or "").split())
    if proposed and "?" not in proposed and len(proposed) <= MAX_DECLINE_CHARS:
        return proposed
    if proposed:
        logger.info("OFF-TOPIC replaced a decline that was not one: %r", proposed[:120])
    return DECLINE


def steer(state: dict) -> tuple[str, str | None]:
    """Where the conversation goes now: (question, the slot it asks about).

    Whatever the flow was already owed, so declining costs the customer nothing. The
    contact request outranks it for the same reason it does everywhere else.
    """
    from src.graph.nodes import greeting
    from src.tools import questions, team_notify

    if greeting.contact_gate_applies(state):
        missing = team_notify.missing_pieces(state)
        if missing == ["name"]:
            return "Could I take your name so our team can reach you?", None
        if missing == ["contact"]:
            return "Could I take an email or phone number so our team can reach you?", None
        return (
            "Could I take your name and an email or phone number so our team can reach you?",
            None,
        )

    carry_on = questions.carry_on_question(state)
    if carry_on:
        slot, wording = carry_on
        return wording, slot

    if not state.get("category"):
        return greeting.orientation_question(state), None
    return "Is there anything else about our trailers I can help with?", None


def reply(state: dict, output: Any, *, first_turn: bool = False) -> tuple[str, str | None]:
    """The whole off-topic turn: (text, the slot it asked about)."""
    from src.graph.nodes import greeting

    question, slot = steer(state)
    opener = f"{greeting.OPENING} " if first_turn else ""
    return f"{opener}{_decline_line(output)} {question}".strip(), slot
