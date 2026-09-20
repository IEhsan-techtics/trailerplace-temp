"""The readable per-turn block: what the customer said, what Luna made of it, what it did.

Separate from src/turn_log.py, which is a compact JSONL feed for the cost report. This one
is for a person to watch the bot think - the raw message, everything the single analysis
call extracted and decided, which tools fired and what they returned, the reply, and the
whole session state as it stood right after the turn. One block per turn, on the
``trailerplace.conversation`` logger, so it reaches the console and the daily log file that
src/log_setup.py already wires up.

Structured - fixed labels, grep-able section markers - but written for a terminal, not for
a parser. The machine-readable case is already covered next door.
"""
from __future__ import annotations

import logging
from typing import Any

CONVERSATION_LOGGER_NAME = "trailerplace.conversation"

logger = logging.getLogger(CONVERSATION_LOGGER_NAME)

_BAR = "=" * 100
_THIN = "-" * 100


def _fmt(value: Any) -> str:
    """One value for a ``label: value`` line. Anything empty renders as '-'."""
    if value is None or value == "":
        return "-"
    if isinstance(value, (list, tuple)):
        return "-" if not value else ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return "-" if not value else ", ".join(f"{key}={val}" for key, val in value.items())
    return str(value)


def _analysis_lines(output: Any) -> list[str]:
    """The single call's structured output - the bot's whole reasoning for the turn."""
    if output is None:
        return ["  (no analysis - a replayed turn, or the model call failed)"]

    extracted = getattr(output, "extracted", None)
    lookup = getattr(output, "inventory_lookup", None)
    contact = getattr(output, "contact", None)

    lines = [
        f"  summary: {_fmt(getattr(output, 'turn_summary', None))}",
        f"  intent: {getattr(output, 'intent', '-')}"
        f"   category_mentioned: {_fmt(getattr(output, 'category_mentioned', None))}"
        f" (info_only={getattr(output, 'is_category_info_only', False)})",
    ]
    if extracted is not None:
        lines += [
            "  extracted: "
            f"length={_fmt(extracted.length)} width={_fmt(extracted.width)} "
            f"height={_fmt(extracted.height)} payload={_fmt(extracted.payload_capacity)} "
            f"hitch={_fmt(extracted.hitch_type)} brand={_fmt(extracted.brand_preference)}",
            "  axles: "
            f"per_axle={_fmt(extracted.axle_capacity)} total={_fmt(extracted.total_axle_capacity_lbs)} "
            f"count={_fmt(extracted.axle_count)} basis={_fmt(extracted.axle_capacity_basis)}",
            f"  haul_item: {_fmt(extracted.haul_item)}   features: {_fmt(extracted.non_metadata_features)}"
            f"   no_preference: {_fmt(extracted.numeric_no_preference)}",
        ]
        if extracted.quantities:
            lines.append(
                "  quantities: " + _fmt([
                    f"{q.slot_name}={q.low}{'-' + str(q.high) if q.high else ''}{q.unit} ({q.raw_text!r})"
                    for q in extracted.quantities
                ])
            )
    answers = getattr(output, "slot_answers", None)
    if answers:
        lines.append(f"  slot_answers: {_fmt([f'{a.slot_name}={a.raw_answer!r}' for a in answers])}")
    haul = getattr(output, "haul_classification", None)
    if haul is not None and (haul.cargo_traits or haul.haul_item_matched):
        lines.append(
            f"  cargo: traits={_fmt(haul.cargo_traits)} matched={_fmt(haul.haul_item_matched)}"
        )
    if lookup is not None and (lookup.is_lookup or lookup.listing_url):
        lines.append(
            "  inventory_lookup: "
            f"year={_fmt(lookup.year)} make={_fmt(lookup.make)} model={_fmt(lookup.model_text)} "
            f"stock={_fmt(lookup.stock_number)} url={_fmt(lookup.listing_url)} "
            f"wants={_fmt(lookup.wants)} confidence={lookup.confidence}"
        )
    if contact is not None and (contact.name or contact.email or contact.phone or contact.declined):
        lines.append(
            f"  contact_in_message: name={_fmt(contact.name)} email={_fmt(contact.email)} "
            f"phone={_fmt(contact.phone)} declined={contact.declined}"
        )
    if getattr(output, "faq_key", None):
        lines.append(f"  faq: {output.faq_key}")
    if getattr(output, "unavailable_type_requested", None):
        lines.append(f"  unavailable_type_requested: {output.unavailable_type_requested}")
    if getattr(output, "listing_reference", None):
        lines.append(f"  listing_reference: #{output.listing_reference}")
    if getattr(output, "shared_link_interest", False):
        lines.append("  shared_link_interest: yes")
    if not getattr(output, "answered_current_question", True) and getattr(output, "user_question_to_answer", None):
        lines.append(
            f'  interruption: "{output.user_question_to_answer}" (the pending question went unanswered)'
        )
    if getattr(output, "dropped_fields", None) or getattr(output, "keep_fields_answer", None):
        lines.append(
            f"  requirement_change: dropped={_fmt(output.dropped_fields)} "
            f"keep_answer={_fmt(output.keep_fields_answer)} kept={_fmt(getattr(output, 'kept_fields', None))}"
        )
    if getattr(output, "category_confirm_answer", None):
        lines.append(f"  category_confirm_answer: {output.category_confirm_answer}")
    return lines


def _tool_lines(outcome: dict) -> list[str]:
    tools: list[str] = []
    if outcome.get("search_ran"):
        tools.append(
            f"search(results={outcome.get('result_count', 0)}, "
            f"brand_relaxed={outcome.get('brand_relaxed', False)}, "
            f"filters_relaxed={outcome.get('filters_relaxed', False)})"
        )
    if outcome.get("inventory_lookup_ran"):
        tools.append(f"inventory_lookup(status={outcome.get('inventory_match_status')})")
    if outcome.get("outbox_events"):
        tools.append(f"email({len(outcome['outbox_events'])} queued)")
    if outcome.get("escalated"):
        tools.append("escalate")
    lines = [f"TOOLS FIRED: {', '.join(tools) if tools else '-'}"]
    if outcome.get("email_status"):
        lines.append(f"EMAIL STATUS: {outcome['email_status']}")
    if outcome.get("link_interest"):
        lines.append(f"LINK INTEREST: {_fmt(outcome['link_interest'])}")
    if outcome.get("reply_text") is not None:
        lines.append("REPLY WRITTEN BY: the reply pass")
    return lines


def _state_lines(state: dict) -> list[str]:
    """Every field a support ticket needs to answer 'what does the bot think it knows?'."""
    contact = state.get("contact") or {}
    pending_slot = state.get("pending_slot")
    pending = (
        f"{pending_slot} (asked {(state.get('asked_counts') or {}).get(pending_slot, 0)}x)"
        if pending_slot else "-"
    )
    confirmations = {
        key: state.get(key) for key in (
            "pending_keep_filters", "pending_category_switch", "pending_gooseneck_clarification",
            "pending_axle_basis", "pending_axle_count",
        ) if state.get(key)
    }
    slots = state.get("slots") or {}
    declined = state.get("declined_slots") or []
    return [
        f"  category: {_fmt(state.get('category'))}   "
        f"qualification_complete: {state.get('qualification_complete', False)}   "
        f"results_shown: {state.get('results_shown', False)}",
        f"  slots: {_fmt(slots)}",
        f"  sources: {_fmt(state.get('slot_sources'))}",
        f"  still to ask: {_fmt([slot for slot in (state.get('required_slots') or []) if slot not in slots and slot not in declined])}",
        f"  declined: {_fmt(declined)}   rule_skipped: {_fmt(state.get('rule_skipped'))}   "
        f"rule_defaults: {_fmt(state.get('rule_defaults'))}",
        f"  pending_question: {pending}   retry: {_fmt(state.get('invalid_retry_slot'))}"
        f"/{_fmt(state.get('invalid_retry_reason'))}",
        f"  brand_preference: {_fmt(state.get('brand_preference'))}   "
        f"features: {_fmt(state.get('non_metadata_features'))}",
        f"  contact: name={_fmt(contact.get('name'))} email={_fmt(contact.get('email'))} "
        f"phone={_fmt(contact.get('phone'))} declined={contact.get('declined', False)}",
        f"  open confirmations: {_fmt(confirmations)}",
        f"  pending_email_actions: {len(state.get('pending_email_actions') or [])}   "
        f"waiting on: {_fmt(state.get('contact_followup_pending'))}",
        f"  shown_urls: {len(state.get('shown_urls') or [])}   "
        f"unavailable_requests: {_fmt(state.get('unavailable_requests'))}",
    ]


def log_conversation_turn(
    *,
    session_id: str,
    turn_id: str | None,
    user_message: str,
    output: Any,
    assistant_text: str | None,
    turn_outcome: dict,
    state: dict,
    error: str | None = None,
) -> None:
    """Log one full, readable turn block. Never raises - observability is not worth a 500."""
    try:
        outcome = turn_outcome or {}
        lines: list[str] = [
            _BAR,
            f"TURN  session={session_id}  turn={turn_id}  #{state.get('turn_index', '?')}",
            _THIN,
            f"USER: {user_message}",
            _THIN,
            "WHAT LUNA READ:",
            *_analysis_lines(output),
            _THIN,
            *_tool_lines(outcome),
            _THIN,
        ]
        if error:
            lines.append(f"ERROR: {error}")
        else:
            lines.append(f"LUNA: {assistant_text or ''}")
            if outcome.get("cited_listing_urls"):
                lines.append(f"  cited_listing_urls: {_fmt(outcome['cited_listing_urls'])}")
            lines.append(f"  listings_shown: {len(outcome.get('listings') or [])}")
            if outcome.get("asked_slot"):
                lines.append(f"  asked: {outcome['asked_slot']}")
        lines.extend([_THIN, "STATE AFTER THE TURN:", *_state_lines(state or {}), _BAR])
        logger.info("\n".join(lines))
    except Exception:  # noqa: BLE001 - logging must never break the reply
        logger.exception("Conversation logging failed: session=%s", session_id)
