"""The question-rules document: its shape, and everything that makes one valid.

One JSON document holds every question the bot may ask and every condition that changes
the set. It is edited as a whole (a control panel PUTs the full document back) and stored
as numbered versions, so the models here are deliberately flat - one rule carries exactly
one action, which is what keeps a panel form for it simple.

Validation is strict on purpose. A document that names a category we do not have, a slot
nothing can fill, or a trait the model is never told about would not fail loudly at
runtime - it would just quietly never fire. Catching it at save time is the only place the
admin can still be told.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.domain import slot_map
from src.domain.categories import CANONICAL_CATEGORIES

# The key a document uses for the spec that applies to a category it does not list.
FALLBACK_CATEGORY = "Unknown"

ACTIONS = ("skip_question", "ask_question", "set_default")
POSITIONS = ("next", "end")
CONDITION_KEYS = ("categories_in", "categories_not_in", "traits_any", "traits_none")

# Slot names the extraction step writes under fixed names (filters._EXTRACTED_TO_SLOT) and
# every slot with a parse kind. Together with the slots the document's own categories use,
# this is the set a rule may name.
_BUILT_IN_SLOTS = (
    "length", "width", "height", "payload_capacity", "axle_capacity",
    "total_axle_capacity_lbs", "axle_count", "hitch_type", "haul_item",
)

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CategorySpec(_Strict):
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)
    questions: dict[str, str] = Field(default_factory=dict)
    answer_guidance: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class CargoTrait(_Strict):
    key: str
    label: str
    definition: str
    examples: list[str] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def _key_shape(cls, value: str) -> str:
        if not _KEY_RE.match(value):
            raise ValueError("trait key must be lower_snake_case")
        return value


class When(_Strict):
    """Every condition given must hold. A condition left out matches anything."""

    categories_in: list[str] | None = None
    categories_not_in: list[str] | None = None
    traits_any: list[str] | None = None
    traits_none: list[str] | None = None


class Rule(_Strict):
    id: str
    enabled: bool = True
    action: Literal["skip_question", "ask_question", "set_default"]
    slot: str
    when: When = Field(default_factory=When)
    # ask_question only
    question: str | None = None
    position: Literal["next", "end"] = "next"
    # set_default only
    value: Any = None
    # Plain words for the admin and for the prompt ("light load - any utility trailer carries it").
    reason: str = ""


class RulesDocument(_Strict):
    categories: dict[str, CategorySpec]
    cargo_traits: list[CargoTrait] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)

    def trait_keys(self) -> set[str]:
        return {trait.key for trait in self.cargo_traits}

    def category_spec(self, category: str | None) -> CategorySpec:
        return self.categories.get(category or "") or self.categories[FALLBACK_CATEGORY]

    def known_slots(self) -> set[str]:
        """Every slot name a rule may refer to."""
        slots = set(_BUILT_IN_SLOTS) | set(slot_map._SLOT_VALUE_KIND)
        for spec in self.categories.values():
            slots.update(spec.required)
            slots.update(spec.optional)
        return slots

    def question_slots(self) -> set[str]:
        """Slots this document can actually ask about - the ones an answer may be stored under."""
        slots: set[str] = set()
        for spec in self.categories.values():
            slots.update(spec.required)
            slots.update(spec.optional)
        slots.update(rule.slot for rule in self.rules if rule.action == "ask_question")
        return slots


def normalize_default(slot: str, value: Any, category: str = "") -> Any:
    """The value a default is stored as, parsed exactly like a customer's answer. None if unusable."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return slot_map.normalize_answer_for_slot(category, slot, value)


def _check(doc: RulesDocument) -> list[str]:
    errors: list[str] = []
    canonical = set(CANONICAL_CATEGORIES)

    if FALLBACK_CATEGORY not in doc.categories:
        errors.append(f"categories: the fallback spec '{FALLBACK_CATEGORY}' is required")
    for name, spec in doc.categories.items():
        where = f"categories.{name}"
        if name != FALLBACK_CATEGORY and name not in canonical:
            errors.append(f"{where}: unknown category (known: {', '.join(CANONICAL_CATEGORIES)})")
        for slot in spec.required + spec.optional:
            if not _KEY_RE.match(slot):
                errors.append(f"{where}: slot '{slot}' must be lower_snake_case")
        overlap = set(spec.required) & set(spec.optional)
        if overlap:
            errors.append(f"{where}: {sorted(overlap)} listed as both required and optional")
        if len(set(spec.required)) != len(spec.required):
            errors.append(f"{where}: a required slot is listed twice")
        for slot in spec.required:
            if not (spec.questions.get(slot) or "").strip():
                errors.append(f"{where}: required slot '{slot}' has no question wording")

    trait_keys = [trait.key for trait in doc.cargo_traits]
    if len(set(trait_keys)) != len(trait_keys):
        errors.append("cargo_traits: duplicate trait key")

    known_slots = doc.known_slots()
    seen_ids: set[str] = set()
    for index, rule in enumerate(doc.rules):
        where = f"rules[{index}] ({rule.id})"
        if not rule.id.strip():
            errors.append(f"rules[{index}]: id is required")
        elif rule.id in seen_ids:
            errors.append(f"{where}: duplicate id")
        seen_ids.add(rule.id)

        if rule.action == "ask_question":
            if not _KEY_RE.match(rule.slot):
                errors.append(f"{where}: slot '{rule.slot}' must be lower_snake_case")
            if not (rule.question or "").strip():
                errors.append(f"{where}: ask_question needs question wording")
        elif rule.slot not in known_slots:
            errors.append(f"{where}: unknown slot '{rule.slot}'")

        if rule.action == "set_default" and normalize_default(rule.slot, rule.value) is None:
            errors.append(f"{where}: default {rule.value!r} is not a usable value for '{rule.slot}'")

        when = rule.when
        for key in ("categories_in", "categories_not_in"):
            for category in getattr(when, key) or []:
                if category not in canonical:
                    errors.append(f"{where}: when.{key} has unknown category '{category}'")
        for key in ("traits_any", "traits_none"):
            for trait in getattr(when, key) or []:
                if trait not in trait_keys:
                    errors.append(f"{where}: when.{key} has unknown trait '{trait}'")
    return errors


def validate_document(raw: Any) -> tuple[RulesDocument | None, list[str]]:
    """Parse and check a document. Returns (document, []) or (None, errors)."""
    try:
        doc = raw if isinstance(raw, RulesDocument) else RulesDocument.model_validate(raw)
    except ValidationError as exc:
        return None, [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
        ]
    errors = _check(doc)
    return (None, errors) if errors else (doc, [])
