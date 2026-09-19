"""Turn one structured output into stored slot values. No LLM calls, no I/O.

Numbers come in three ways, in this order of trust:

1. ``extracted.quantities`` - the model READS the amount ("seven and a half feet" is 7.5 ft,
   a unitless "144 x 72" is inches) and ``src/domain/quantities`` does the arithmetic. A live
   run showed the regex parser misreading exactly this kind of everyday wording, so the
   language is the model's job and the conversion is ours.
2. The customer's raw words, parsed in Python - the fallback when no quantity came back.
3. The model's own converted number, for anything neither of the above covers.

Before the quantities existed, the rule was **raw customer text beats the model's
arithmetic**, and it still holds for the parts that are arithmetic:

``ExtractedFields`` numerics arrive already converted by the model, which makes any rule
that depends on the ORIGINAL WORDING unenforceable on them:

* brief S15 (a range resolves to its SMALLEST side) - a live probe had gpt-5.6-luna answer
  19.0 for "18-20 ft", the midpoint the brief forbids;
* brief S22 (a negative measurement is re-asked, never sign-flipped) - once "-500 lbs" has
  become the float 500.0 it is indistinguishable from an honest 500.

So a range is resolved to its smaller end in Python, never by the model, and the sign is
read off the customer's own words. Whatever the input form, what lands in the state is a
NUMBER (20.0, 18.5) and never a word.
"""
from __future__ import annotations

import logging
from typing import Any

from src.domain import quantities as quantity_math
from src.domain import slot_map
from src.rules.engine import is_default, mark_user_value
from src.rules.store import current_rules
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


def _writable_slots() -> frozenset[str]:
    """The fixed targets plus every slot the live question rules can ask about.

    A question added in the control panel has to be answerable: without this its answer
    would arrive in slot_answers and be thrown away as an unknown slot name. A name the
    rules do not ask about is still ignored, exactly as before.
    """
    return _SLOT_TARGETS | current_rules().question_slots()


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

    The quantities' ``raw_text`` first, then ``slot_answers`` - the latter is the model's
    explicit "this message answers that question", so it wins when both mention a slot.
    """
    raw: dict[str, str] = {}
    for quantity in getattr(getattr(output, "extracted", None), "quantities", None) or []:
        name = (getattr(quantity, "slot_name", "") or "").strip()
        text = getattr(quantity, "raw_text", None)
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
    # A rule's default is not the customer's answer, so anything they say replaces it.
    if (
        existing is not None
        and not is_default(state, slot)
        and slot_map.slot_value_kind(slot) is not None
        and not slot_map.is_recognized_slot_value(slot, value)
    ):
        return

    state.setdefault("slots", {})[slot] = value
    mark_user_value(state, slot)
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
    if hitch and (not (state.get("slots") or {}).get("hitch_type") or is_default(state, "hitch_type")):
        state.setdefault("slots", {})["hitch_type"] = hitch
        mark_user_value(state, "hitch_type")


def _apply_quantities(state: dict, output: Any, category: str, writable: frozenset[str],
                      raw_by_slot: dict[str, str], result: FieldApplication) -> set[str]:
    """Store every amount the model read. Returns the slots it settled, one way or another.

    A quantity whose unit does not fit its slot is not settled here, so the raw-text parse
    still gets its turn - a bad unit costs nothing but the shortcut.
    """
    settled: set[str] = set()
    for quantity in getattr(getattr(output, "extracted", None), "quantities", None) or []:
        slot = (getattr(quantity, "slot_name", "") or "").strip()
        if slot in settled or slot not in writable or not quantity_math.supports(slot):
            continue
        raw = str(getattr(quantity, "raw_text", "") or "") or raw_by_slot.get(slot, "")

        # The sign is the model's reading too (brief S22 still re-asks a negative). Python used
        # to overrule it from the raw text, and a regex cannot read people: it took the dash
        # in "10k-12k" for a minus and re-asked a perfectly good range. Typos, ranges and
        # dashes are language; the regex check now runs only on the fallback path below.
        negative = float(getattr(quantity, "low", 0) or 0) < 0
        if slot in _MEASUREMENT_SLOTS and negative:
            result.invalid_slot, result.invalid_reason, result.invalid_raw = slot, "negative", raw
            logger.info("FILTER reject: slot=%s reason=negative raw=%r", slot, raw)
            settled.add(slot)
            continue

        value = quantity_math.to_canonical(slot, quantity)
        if value is None:
            logger.info("QUANTITY unit does not fit: slot=%s unit=%s raw=%r", slot, getattr(quantity, "unit", None), raw)
            continue
        if quantity_math.check(slot, value) == "implausible":
            result.invalid_slot, result.invalid_reason, result.invalid_raw = slot, "implausible", raw
            logger.info("FILTER reject: slot=%s reason=implausible value=%s raw=%r", slot, value, raw)
            settled.add(slot)
            continue

        _log_parser_divergence(category, slot, raw, value)
        _apply_one(state, slot, value, category, result)
        settled.add(slot)
    return settled


def _log_parser_divergence(category: str, slot: str, raw: str, value: float) -> None:
    """Say so when the regex parser would have stored something else.

    Nothing is decided by it. It is how we can see, in the logs, every answer the old
    fallback would have got wrong - and how a model misreading shows up just as clearly.
    """
    if not raw:
        return
    parsed = normalize_answer_for_slot(category, slot, raw)
    if isinstance(parsed, (int, float)) and abs(float(parsed) - value) > max(0.1, value * 0.03):
        logger.warning(
            "QUANTITY divergence: slot=%s raw=%r model=%s python_parser=%s (model wins)",
            slot, raw, value, parsed,
        )


def apply_extracted_fields(state: dict, output: Any) -> FieldApplication:
    """Apply everything the customer stated this turn to ``state["slots"]``.

    Runs whether or not a category has been chosen (brief S11): configuration volunteered
    before the category is settled is kept, not discarded.
    """
    result = FieldApplication()
    category = state.get("category") or ""
    extracted = getattr(output, "extracted", None)
    raw_by_slot = raw_text_for_slots(output)
    writable = _writable_slots()

    # 1. The amounts the model read, converted here.
    settled = _apply_quantities(state, output, category, writable, raw_by_slot, result)

    # 2. Raw text for everything else. Anything the customer worded themselves is parsed
    #    from their words.
    for slot, raw in raw_by_slot.items():
        if slot in writable and slot not in settled:
            _apply_one(state, slot, raw, category, result)

    # 3. The model's converted numbers fill only the gaps.
    if extracted is not None:
        for field, slot in _EXTRACTED_TO_SLOT.items():
            value = getattr(extracted, field, None)
            if slot in settled:
                continue
            if slot in raw_by_slot:
                _log_divergence(slot, raw_by_slot[slot], value, result)
                continue
            if value is None or (isinstance(value, list) and not value):
                continue
            _apply_one(state, slot, value, category, result)

        # Slots the model itself flagged as "they stated no preference" (brief S18) - but
        # only a slot the customer was actually answering. A live run had "just some random
        # stuff", said about the cargo, also flag the weight, and a question never asked was
        # skipped. Volunteering "any size is fine" still counts: the model puts their words
        # for that slot in slot_answers, which is what raw_by_slot holds.
        addressed = set(raw_by_slot) | {state.get("pending_slot")}
        for slot in getattr(extracted, "numeric_no_preference", None) or []:
            if slot not in writable or slot in result.stored or slot in result.no_preference:
                continue
            if slot not in addressed:
                logger.info("FILTER ignored no-preference: slot=%s (not asked, not addressed)", slot)
                continue
            result.no_preference.append(slot)

        _apply_features(state, extracted)

    return result
