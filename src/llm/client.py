"""The single conversational call per turn.

One call, one place. Every business decision downstream reads its output, so this module
guarantees two things:

* it is called **once** per turn - there is no tool-call loop back into the model, and
  ``usage.record_completion`` makes that assertable from the logs (brief S31);
* it **always returns a usable object**. An API error, a timeout or a schema the model
  could not satisfy degrades to an empty output, and the deterministic layer carries the
  turn on its own - the customer gets the canonical next question instead of a 500.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from src.config import settings
from src.llm import usage
from src.llm.prompt import build_messages
from src.llm.schemas import (
    ChatbotTurnOutput,
    ContactInfo,
    ExtractedFields,
    HaulClassification,
    InventoryLookup,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client():
    """The OpenAI client. Cached - building one per turn leaks connection pools."""
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key, timeout=settings.chat_timeout_seconds)


def empty_output(reason: str = "") -> ChatbotTurnOutput:
    """A turn output that asserts nothing.

    Every field is the value that means "the model told us nothing", so the deterministic
    layer stores nothing, decides nothing new, and falls back to the canonical question.
    A degraded turn is a slightly flat reply, never a wrong one.
    """
    return ChatbotTurnOutput(
        turn_summary=reason or "The assistant could not read this message.",
        intent="smalltalk_other",
        category_mentioned=None,
        is_category_info_only=False,
        extracted=ExtractedFields(
            length=None, width=None, height=None, payload_capacity=None,
            axle_capacity=None, total_axle_capacity_lbs=None, axle_count=None,
            axle_capacity_basis=None, hitch_type=None, haul_item=None,
            brand_preference=None, non_metadata_features=[], numeric_no_preference=[],
            raw_numeric_spans=[],
        ),
        slot_answers=[],
        contact=ContactInfo(name=None, email=None, phone=None, declined=False),
        inventory_lookup=InventoryLookup(
            is_lookup=False, year=None, make=None, model_text=None,
            stock_number=None, wants=None, confidence="low",
        ),
        haul_classification=HaulClassification(cargo_traits=[], haul_item_matched=None),
        keep_fields_answer=None,
        kept_fields=[],
        category_confirm_answer=None,
        answered_current_question=False,
        user_question_to_answer=None,
        faq_key=None,
        unavailable_type_requested=None,
        listing_reference=None,
        dropped_fields=[],
        acknowledgement="",
        answer_to_customer_question=None,
        next_question_slot=None,
        next_question_text=None,
    )


def analyze_turn(state: dict, user_message: str) -> ChatbotTurnOutput:
    """THE call. Exactly one per user turn.

    Nothing here decides anything - it reads the message and returns structured evidence.
    ``apply.py`` is what acts on it.
    """
    messages = build_messages(state, user_message)
    try:
        response = get_client().responses.parse(
            model=settings.chat_model,
            input=messages,
            reasoning={"effort": settings.chat_reasoning_effort},
            text_format=ChatbotTurnOutput,
        )
    except Exception as exc:  # noqa: BLE001 - any failure degrades the same way
        logger.exception(
            "LLM call failed: session=%s model=%s", state.get("session_id"), settings.chat_model
        )
        return empty_output(f"The model call failed: {type(exc).__name__}.")

    _record_usage(response)

    output = getattr(response, "output_parsed", None)
    if output is None:
        # A refusal or an unparseable body. Same handling as an exception: the turn
        # continues deterministically rather than failing.
        logger.error(
            "LLM returned no parsed output: session=%s status=%r",
            state.get("session_id"), getattr(response, "status", None),
        )
        return empty_output("The model returned nothing readable.")

    logger.info(
        "LLM turn: session=%s intent=%s category=%r answered=%s slots=%s",
        state.get("session_id"), output.intent, output.category_mentioned,
        output.answered_current_question,
        [answer.slot_name for answer in output.slot_answers],
    )
    return output


def _record_usage(response: Any) -> None:
    """Count the call so test_single_call can assert exactly one happened."""
    tokens = getattr(response, "usage", None)
    usage.record_completion(
        settings.chat_model,
        prompt_tokens=getattr(tokens, "input_tokens", 0) or 0,
        completion_tokens=getattr(tokens, "output_tokens", 0) or 0,
        purpose="chat_turn",
    )
