"""Turn one structured output into stored slot values. No LLM calls, no I/O.

The rule that shapes this whole module: **raw customer text beats the model's arithmetic.**

``ExtractedFields`` numerics arrive already converted by the model, which makes any rule
that depends on the ORIGINAL WORDING unenforceable on them:

* brief S15 (a range resolves to its SMALLEST side) - a live probe had gpt-5.6-luna answer
  19.0 for "18-20 ft", the midpoint the brief forbids;
* brief S22 (a negative measurement is re-asked, never sign-flipped) - once "-500 lbs" has
  become the float 500.0 it is indistinguishable from an honest 500.

So whenever the customer's verbatim wording for a slot is available - from ``slot_answers``
or from ``extracted.raw_numeric_spans`` - that text is what gets parsed, and the model's
number is used only as a fallback for slots no raw text covers. Whatever the input form,
what lands in the state is a NUMBER (20.0, 18.5) and never a word.
"""
from __future__ import annotations

import logging
from typing import Any

from src.domain import slot_map
from src.domain.slot_map import (
    axle_count_out_of_range,
    is_impossible_measurement,
    normalize_answer_for_slot,
    sanitize_non_metadata_features,
)

logger = logging.getLogger(__name__)

# ExtractedFields attribute -> slot name. Identical for most, so this map exists for the
# ones that differ and to keep the set of fields we copy explicit rather than reflective.
_EXTRACTED_TO_SLOT: dict[str, str] = {
    "length": "length",
    "width": "width",
    "height": "height",
    "payload_capacity": "payload_capacity",
    "axle_capacity": "axle_capacity",
    "total_axle_capacity_lbs": "total_axle_capacity_lbs",
    "axle_count": "axle_count",
    "hitch_type": "hitch_type",
    "haul_item": "haul_item",
}

# Slots whose value is a measurement and can therefore be stated negatively.
_MEASUREMENT_SLOTS = frozenset(
    {"length", "width", "height", "payload_capacity", "axle_capacity", "total_axle_capacity_lbs"}
)

# Slots this module is willing to write. Anything else the model invents is ignored rather
# than stored, which is what stops a hallucinated slot name reaching the search filters.
_SLOT_TARGETS = frozenset(_EXTRACTED_TO_SLOT.values()) | {
    "payload_lbs", "bin_size", "cargo_size", "trailer_size", "base_category",
}


class FieldApplication:
    """What one pass of ``apply_extracted_fields`` did, for the reply and the logs."""

    def __init__(self) -> None:
        self.stored: dict[str, Any] = {}
        self.no_preference: list[str] = []
        self.invalid_slot: str | None = None
        self.invalid_reason: str | None = None
        self.invalid_raw: str | None = None

    @property
    def touched_slots(self) -> list[str]:
        """Every slot this turn resolved one way or the other - stored or declined."""
        return list(self.stored) + list(self.no_preference)


def raw_text_for_slots(output: Any) -> dict[str, str]:
    """Every slot the customer gave verbatim wording for this turn.

    ``raw_numeric_spans`` first, then ``slot_answers`` - the latter is the model's explicit
    "this message answers that question", so it wins when both mention a slot.
    """
    raw: dict[str, str] = {}
    spans = getattr(getattr(output, "extracted", None), "raw_numeric_spans", None) or []
    for span in spans:
        name = (getattr(span, "slot_name", "") or "").strip()
        text = getattr(span, "raw_answer", None)
        if name and text is not None and str(text).strip():
            raw[name] = str(text)
    for answer in getattr(output, "slot_answers", None) or []:
        name = (getattr(answer, "slot_name", "") or "").strip()
        text = getattr(answer, "raw_answer", None)
        if name and text is not None and str(text).strip():
            raw[name] = str(text)
    return raw


def _apply_one(state: dict, slot: str, raw: Any, category: str, result: FieldApplication) -> None:
    """Validate and store a single slot.

    ``raw`` is the customer's text when we have it, otherwise the model's converted value.
    """
    # Order matters: both checks read the RAW text, so they must run before any parsing
    # strips the sign or the count is clamped.
    if slot in _MEASUREMENT_SLOTS and is_impossible_measurement(category, slot, raw):
        # Brief S22. Not stored - storing 500 for "-500 lbs" would filter on the opposite
        # of what the customer said.
        result.invalid_slot = slot
        result.invalid_reason = "negative"
        result.invalid_raw = str(raw)
        logger.info("FILTER reject: slot=%s reason=negative raw=%r", slot, raw)
        return
    if slot == "axle_count" and axle_count_out_of_range(raw):
        result.invalid_slot = slot
        result.invalid_reason = "axle_range"
        result.invalid_raw = str(raw)
        logger.info("FILTER reject: slot=%s reason=axle_range raw=%r", slot, raw)
        return

    value = normalize_answer_for_slot(category, slot, raw)

    if value is None:
        # A vague answer, an explicit "no preference", or a stated 0 (brief S18). Recorded
        # so the question is never asked again, but never stored as a filter.
        if slot not in result.no_preference:
            result.no_preference.append(slot)
        return

    # Do not let a vaguer restatement of the same slot overwrite a clean value already held.
    #
    # Only for slots with a parsed KIND. is_recognized_slot_value asks "is this already in
    # the clean form for this slot's kind", and free text has no kind - so applying the
    # guard there rejected every update to haul_item once one existed, and a customer
    # correcting "actually it's dirt, not gravel" was silently ignored.
    existing = (state.get("slots") or {}).get(slot)
    if (
        existing is not None
        and slot_map.slot_value_kind(slot) is not None
        and not slot_map.is_recognized_slot_value(slot, value)
    ):
        return

    state.setdefault("slots", {})[slot] = value
    result.stored[slot] = value


def _log_divergence(slot: str, raw: str, model_value: Any, result: FieldApplication) -> None:
    """Say so when Python and the model disagree about a number.

    Not an error - Python is right by construction here - but silent disagreement is how
    prompt drift hides, so every instance is visible in the logs.
    """
    stored = result.stored.get(slot)
    if model_value is None or stored is None:
        return
    if isinstance(stored, (int, float)) and isinstance(model_value, (int, float)):
        if abs(float(stored) - float(model_value)) > 0.01:
            logger.warning(
                "FILTER divergence: slot=%s raw=%r parsed=%s model_said=%s (raw wins)",
                slot, raw, stored, model_value,
            )


def _apply_features(state: dict, extracted: Any) -> None:
    """Merge newly stated non-metadata features, lifting a hitch stated as one."""
    new_features = list(getattr(extracted, "non_metadata_features", None) or [])
    if not new_features:
        return
    merged = list(state.get("non_metadata_features") or []) + new_features
    kept, hitch = sanitize_non_metadata_features(merged)
    state["non_metadata_features"] = kept
    # "gooseneck" arriving as a feature is a hitch preference in disguise; only fill the
    # slot when it is still empty, so an explicit answer is never overwritten.
    if hitch and not (state.get("slots") or {}).get("hitch_type"):
        state.setdefault("slots", {})["hitch_type"] = hitch


def apply_extracted_fields(state: dict, output: Any) -> FieldApplication:
    """Apply everything the customer stated this turn to ``state["slots"]``.

    Runs whether or not a category has been chosen (brief S11): configuration volunteered
    before the category is settled is kept, not discarded.
    """
    result = FieldApplication()
    category = state.get("category") or ""
    extracted = getattr(output, "extracted", None)
    raw_by_slot = raw_text_for_slots(output)

    # 1. Raw text wins. Anything the customer worded themselves is parsed from their words.
    for slot, raw in raw_by_slot.items():
        if slot in _SLOT_TARGETS:
            _apply_one(state, slot, raw, category, result)

    # 2. The model's converted numbers fill only the gaps.
    if extracted is not None:
        for field, slot in _EXTRACTED_TO_SLOT.items():
            value = getattr(extracted, field, None)
            if slot in raw_by_slot:
                _log_divergence(slot, raw_by_slot[slot], value, result)
                continue
            if value is None or (isinstance(value, list) and not value):
                continue
            _apply_one(state, slot, value, category, result)

        # Slots the model itself flagged as "they stated no preference" (brief S18).
        for slot in getattr(extracted, "numeric_no_preference", None) or []:
            if slot in _SLOT_TARGETS and slot not in result.stored:
                if slot not in result.no_preference:
                    result.no_preference.append(slot)

        _apply_features(state, extracted)

    return result
