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
import time
from functools import lru_cache
from typing import Any

from src.config import settings
from src.llm import usage
from src.llm.prompt import build_messages
from src.llm.schemas import (
    ChatbotTurnOutput,
    ChatbotTurnReplyOutput,
    ReplyRewrite,
    ContactInfo,
    ExtractedFields,
    HaulClassification,
    InventoryLookup,
)

logger = logging.getLogger(__name__)

ANALYSIS_CACHE_KEY = "luna-analysis"


@lru_cache(maxsize=1)
def get_client():
    """The OpenAI client. Cached - building one per turn leaks connection pools."""
    from openai import OpenAI

    # Four retries rather than the SDK's two. gpt-6-luna's limit on this account is 200k
    # tokens a minute and a turn sends ~16k, so a handful of customers at once reaches it; the
    # 429 says to wait about a second, and two quick retries were live not always enough.
    return OpenAI(api_key=settings.openai_api_key, timeout=settings.chat_timeout_seconds, max_retries=4)


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
            quantities=[],
        ),
        slot_answers=[],
        contact=ContactInfo(name=None, email=None, phone=None, declined=False),
        inventory_lookup=InventoryLookup(
            is_lookup=False, year=None, make=None, model_text=None,
            stock_number=None, listing_url=None, wants=None, confidence="low",
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
        shared_link_interest=False,
        off_topic=False,
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
    started = time.perf_counter()
    try:
        response = get_client().responses.parse(
            model=settings.chat_model,
            input=messages,
            reasoning={"effort": settings.chat_reasoning_effort},
            text_format=ChatbotTurnReplyOutput if settings.llm_writes_reply else ChatbotTurnOutput,
            # Every session shares one static prefix; the key routes them all to the same
            # cache instead of leaving it to the provider's hashing.
            prompt_cache_key=ANALYSIS_CACHE_KEY,
        )
    except Exception as exc:  # noqa: BLE001 - any failure degrades the same way
        logger.exception(
            "LLM call failed: session=%s model=%s", state.get("session_id"), settings.chat_model
        )
        return empty_output(f"The model call failed: {type(exc).__name__}.")
    finally:
        usage.record_seconds("analysis", time.perf_counter() - started)

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


def rewrite_reply(state: dict, user_message: str, first_reply: str, problem: str, needs: str) -> ReplyRewrite | None:
    """One more go at the reply, after Python turned the first one down. None on any failure.

    The first reply was written BEFORE Python applied the message, so it could not know what
    Python then decided. This call is made after: the state block it reads is the state as it
    is now, and it is told what was wrong with the first reply and what this turn needs. The
    customer never sees the first one.
    """
    messages = build_messages(state, user_message)
    messages.append({"role": "assistant", "content": first_reply})
    messages.append({
        "role": "system",
        "content": (
            "That reply cannot be sent. What was wrong with it: " + problem + ".\n"
            "What this reply needs: " + needs + "\n"
            "The state block above is up to date with their latest message. Write the reply "
            "again, fixing that and keeping everything else it did well, and report on it "
            "truthfully in the other fields."
        ),
    })
    started = time.perf_counter()
    try:
        response = get_client().responses.parse(
            model=settings.chat_model,
            input=messages,
            reasoning={"effort": settings.chat_reasoning_effort},
            text_format=ReplyRewrite,
            prompt_cache_key=ANALYSIS_CACHE_KEY,
        )
    except Exception:  # noqa: BLE001 - the caller falls back
        logger.exception("LLM rewrite failed: session=%s", state.get("session_id"))
        return None
    finally:
        usage.record_seconds("reply", time.perf_counter() - started)
    _record_usage(response, purpose="rewrite")
    return getattr(response, "output_parsed", None)


def _record_usage(response: Any, purpose: str = "chat_turn") -> None:
    """Count the call so test_single_call can assert exactly one happened."""
    tokens = getattr(response, "usage", None)
    details = getattr(tokens, "input_tokens_details", None)
    usage.record_completion(
        settings.chat_model,
        prompt_tokens=getattr(tokens, "input_tokens", 0) or 0,
        completion_tokens=getattr(tokens, "output_tokens", 0) or 0,
        purpose=purpose,
        cached_tokens=getattr(details, "cached_tokens", 0) or 0,
    )
