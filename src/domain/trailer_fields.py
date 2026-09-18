"""
trailer_fields.py – Tool that returns the required and optional qualification
slots for a given trailer type, as the question-rules document (src/rules) defines
them. Called by the specialist node once `trailer_type` is set in the session state.

Slot names map 1-to-1 to the keys used in SessionState["slots_collected"].

One name per concept, shared across every category: a length is `length` whether the
customer is describing a load, a vehicle or the trailer itself, and what they are hauling
is `haul_item` whether the category calls it material, vehicle type or use case. Categories
that ask a genuinely different question (hitch_type, deck_style, bin_size, ...) keep their
own names. See slot_map._SLOT_VALUE_KIND for how each name is parsed and filtered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Slot schema
# ---------------------------------------------------------------------------

@dataclass
class TrailerFieldSpec:
    """Required and optional slots for one trailer category."""
    category: str
    required: list[str]
    optional: list[str]
    # Human-readable question prompts keyed by slot name (used in dynamic prompt)
    questions: dict[str, str] = field(default_factory=dict)
    # Guidance keyed by slot name describing what counts as a valid customer answer.
    answer_guidance: dict[str, str] = field(default_factory=dict)
    # Brief notes injected into the specialist prompt
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-category field definitions
# ---------------------------------------------------------------------------
# They live in the question-rules document now (src/rules/seed.json, or the active version
# in chatbot_question_rules), so an admin can add, remove and reword questions without a
# code change. This module keeps its public API and builds each spec from that document.
# A category the document does not list gets its "Unknown" fallback spec.

# The shared measurement slots, plus the combined-size and one-off numeric questions that
# keep their own names. trailer_size/cargo_size/bin_size ask for SEVERAL numbers at once
# (or a yardage), so they stay separate questions - but their answers still map onto
# length/width/height targets via slot_map._SLOT_METADATA_FILTER_MAP.
_NUMERIC_OR_MEASUREMENT_SLOTS = {
    "length", "width", "height", "payload_capacity", "axle_capacity",
    "trailer_size", "cargo_size", "bin_size",
    "tank_capacity", "crew_size",
}
# Every "what are you hauling / what is it for" question now shares one name, so a customer
# who answers it once is never asked again under another category's wording.
_FREE_TEXT_SLOTS = {
    "haul_item",
}


def _loose_answer_guidance(slot: str) -> str:
    if slot in _NUMERIC_OR_MEASUREMENT_SLOTS:
        return (
            "Loose-answer rule: accept digits, number words, ranges, or approximations; for every range store only "
            "the smallest stated value (15–18 ft becomes 15 ft; 5,000–10,000 lbs becomes 5,000 lbs). "
            "If a cooperative answer has no usable numeric value and is not a counter-question or another-field answer, skip this field as no preference; never invent or retry a value."
        )
    if slot in _FREE_TEXT_SLOTS:
        return (
            "Loose-answer rule: store any substantive wording the user gives for this field, however broad or informal. "
            "Do not store it only when the user explicitly refuses/skips, asks a counter-question, or clearly answers another field."
        )
    return (
        "Loose-answer rule: store a recognizable value for this field. "
        "If the user is cooperative but vague/flexible and gives no usable field value, skip it as no preference; do not retry or invent a value."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_trailer_fields(trailer_type: str) -> TrailerFieldSpec:
    """
    Return the TrailerFieldSpec for the given trailer_type.
    Falls back to a generic spec if the type is not recognised.

    Usage:
        spec = get_trailer_fields("Dump")
        print(spec.required)   # ['haul_item', 'payload_capacity']
        print(spec.optional)   # ['dump_mechanism']
    """
    from src.rules.models import FALLBACK_CATEGORY
    from src.rules.store import current_rules

    rules = current_rules()
    name = trailer_type if trailer_type in rules.categories else FALLBACK_CATEGORY
    spec = rules.categories[name]
    return TrailerFieldSpec(
        category=name,
        required=list(spec.required),
        optional=list(spec.optional),
        questions=dict(spec.questions),
        answer_guidance=dict(spec.answer_guidance),
        notes=spec.notes,
    )


def get_trailer_fields_as_dict(trailer_type: str) -> dict:
    """
    Return a JSON-serialisable dict representation of the field spec.
    Suitable for injecting into LLM prompts or tool results.
    """
    spec = get_trailer_fields(trailer_type)
    answer_guidance = {
        slot: " ".join(
            part for part in (
                str(spec.answer_guidance.get(slot) or "").strip(),
                _loose_answer_guidance(slot),
            )
            if part
        )
        for slot in dict.fromkeys(spec.required + spec.optional)
    }
    return {
        "category": spec.category,
        "required_slots": spec.required,
        "optional_slots": spec.optional,
        "questions": spec.questions,
        "answer_guidance": answer_guidance,
        "notes": spec.notes,
    }


def list_all_categories() -> list[str]:
    """Return all supported trailer categories."""
    from src.rules.models import FALLBACK_CATEGORY
    from src.rules.store import current_rules

    return sorted(name for name in current_rules().categories if name != FALLBACK_CATEGORY)


# Optional slots whose answer is NOT a descriptive feature: a hitch is a hard search filter and
# base_category is the Aluminum subcategory filter. Both have a home of their own; copied into the
# feature list they would be matched as loose text instead of filtering the search.
_NON_FEATURE_OPTIONAL_SLOTS = {"hitch_type", "base_category"}


def feature_like_optional_slots(trailer_type: str) -> list[str]:
    """The optional slots for this category whose answer doubles as a non-metadata feature.

    An optional answer like "butterfly gates", "scissor lift", "drive-over fenders" or "lined walls"
    describes equipment we hold NO metadata field for, so a slot value alone can never reach the
    search: the only thing that acts on it is the non-metadata feature matcher. These slots are
    therefore mirrored into ``non_metadata_features`` as well as stored under their own name.
    Numeric/measurement slots (trailer_size, crew_size, ...) are excluded - they are sizes, not
    features, and they already have their own filters.
    """
    spec = get_trailer_fields(trailer_type)
    return [
        slot
        for slot in spec.optional
        if slot not in _NUMERIC_OR_MEASUREMENT_SLOTS and slot not in _NON_FEATURE_OPTIONAL_SLOTS
    ]
