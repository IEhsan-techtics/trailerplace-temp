"""Turn a rules document plus one session into the questions that session gets. No I/O.

``evaluate`` is the pure half: category and cargo traits in, effects out. ``apply_rules``
writes those effects into the session - the defaults it fills, the question list it
recomputes - and is run every turn, so a rule edited in the control panel reaches a
conversation already in progress on its next message.

Precedence, fixed here and nowhere else:

1. A value the customer gave beats everything. A default never overwrites it, and a skip or
   an ask about an answered slot has nothing left to do.
2. ``skip_question`` beats ``ask_question`` for the same slot.
3. A default counts as an answer, so its question is not asked.
4. A disabled rule does nothing.

When two rules of the same action name the same slot, the first one in the document wins.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.graph.state import MAX_ASKS_PER_SLOT
from src.rules.models import Rule, RulesDocument, normalize_default

logger = logging.getLogger(__name__)

SOURCE_DEFAULT = "default"
SOURCE_USER = "user"


@dataclass(frozen=True)
class Ask:
    slot: str
    question: str
    position: str
    rule_id: str
    reason: str = ""


@dataclass
class Effects:
    skip: dict[str, str] = field(default_factory=dict)          # slot -> reason
    ask: list[Ask] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)      # slot -> parsed value
    default_reasons: dict[str, str] = field(default_factory=dict)


def rule_matches(rule: Rule, category: str | None, traits: set[str]) -> bool:
    if not rule.enabled or not category:
        return False
    when = rule.when
    if when.categories_in is not None and category not in when.categories_in:
        return False
    if when.categories_not_in is not None and category in when.categories_not_in:
        return False
    if when.traits_any is not None and not traits & set(when.traits_any):
        return False
    if when.traits_none is not None and traits & set(when.traits_none):
        return False
    return True


def evaluate(category: str | None, traits: list[str] | set[str], rules: RulesDocument) -> Effects:
    """What the rules say for this category and cargo. Pure."""
    effects = Effects()
    active = set(traits) & rules.trait_keys()
    matched = [rule for rule in rules.rules if rule_matches(rule, category, active)]

    for rule in matched:
        if rule.action == "skip_question":
            effects.skip.setdefault(rule.slot, rule.reason or rule.id)
        elif rule.action == "set_default" and rule.slot not in effects.defaults:
            value = normalize_default(rule.slot, rule.value, category or "")
            if value is not None:
                effects.defaults[rule.slot] = value
                effects.default_reasons[rule.slot] = rule.reason or rule.id

    asked: set[str] = set()
    for rule in matched:
        if rule.action != "ask_question" or rule.slot in effects.skip or rule.slot in asked:
            continue
        asked.add(rule.slot)
        effects.ask.append(
            Ask(rule.slot, (rule.question or "").strip(), rule.position, rule.id, rule.reason)
        )
    return effects


# ------------------------------------------------------------------ applying to a session
def _is_answered(state: dict, slot: str) -> bool:
    value = (state.get("slots") or {}).get(slot)
    if value is None:
        return False
    return not (isinstance(value, (list, str)) and not value)


def _is_resolved(state: dict, slot: str) -> bool:
    # Mirrors src.tools.questions.is_resolved. Not imported from there: questions reads the
    # live rules for its wording, so importing it here would be a cycle.
    return (
        _is_answered(state, slot)
        or slot in (state.get("declined_slots") or [])
        or (state.get("asked_counts") or {}).get(slot, 0) >= MAX_ASKS_PER_SLOT
    )


def effective_required(state: dict, rules: RulesDocument, effects: Effects | None = None) -> list[str]:
    """The category's required list, minus skipped slots, plus asked ones at their anchor."""
    category = state.get("category")
    if not category:
        return []
    if effects is None:
        effects = evaluate(category, state.get("cargo_traits") or [], rules)
    required = [slot for slot in rules.category_spec(category).required if slot not in effects.skip]
    anchors = state.get("rule_ask_anchors") or {}
    for ask in effects.ask:
        if ask.slot in required:
            continue
        anchor = anchors.get(ask.slot) if ask.position == "next" else None
        if anchor in required:
            required.insert(required.index(anchor), ask.slot)
        else:
            required.append(ask.slot)
    return required


def _anchor_for(state: dict, required: list[str]) -> str | None:
    """The slot an injected "next" question goes in front of, fixed when the rule first fires.

    Straight after the question currently on the table if it is still open, otherwise in
    front of the next open one - i.e. it is the next thing asked. Stored rather than
    recomputed, so the question stays put as the conversation moves on.
    """
    remaining = [slot for slot in required if not _is_resolved(state, slot)]
    pending = state.get("pending_slot")
    if pending in remaining:
        after = remaining.index(pending) + 1
        return remaining[after] if after < len(remaining) else None
    return remaining[0] if remaining else None


def apply_rules(state: dict, rules: RulesDocument) -> Effects:
    """Fold the rules into the session: defaults, then the required list."""
    category = state.get("category")
    known = rules.trait_keys()
    state["cargo_traits"] = [trait for trait in (state.get("cargo_traits") or []) if trait in known]
    effects = evaluate(category, state["cargo_traits"], rules)

    _apply_defaults(state, effects)

    # Anchors for asks that no longer apply are forgotten, so the question lands in the
    # right place if the rule fires again later.
    anchors = {
        slot: anchor for slot, anchor in (state.get("rule_ask_anchors") or {}).items()
        if any(ask.slot == slot for ask in effects.ask)
    }
    state["rule_ask_anchors"] = anchors
    base = [slot for slot in rules.category_spec(category).required if slot not in effects.skip] if category else []
    for ask in effects.ask:
        if ask.position == "next" and ask.slot not in anchors and ask.slot not in base:
            anchors[ask.slot] = _anchor_for(state, base)
            logger.info(
                "RULES ask: session=%s rule=%s slot=%s before=%s",
                state.get("session_id"), ask.rule_id, ask.slot, anchors[ask.slot],
            )

    state["required_slots"] = effective_required(state, rules, effects)
    base_required = set(rules.category_spec(category).required) if category else set()
    state["rule_skipped"] = {slot: reason for slot, reason in effects.skip.items() if slot in base_required}
    return effects


def _apply_defaults(state: dict, effects: Effects) -> None:
    slots = state.setdefault("slots", {})
    sources = state.setdefault("slot_sources", {})
    applied = state.get("rule_defaults") or {}

    for slot, source in list(sources.items()):
        if source != SOURCE_DEFAULT:
            continue
        if slot in slots and slots[slot] != (applied.get(slot) or {}).get("value"):
            # Something wrote over the default without going through the filters hook -
            # whoever did, it was not us, so it is the customer's value now.
            sources[slot] = SOURCE_USER
        elif slot not in effects.defaults:
            # The rule was switched off, or no longer matches this category.
            slots.pop(slot, None)
            del sources[slot]

    current: dict[str, dict[str, Any]] = {}
    for slot, value in effects.defaults.items():
        ours = sources.get(slot) == SOURCE_DEFAULT
        declined = slot in (state.get("declined_slots") or [])
        if ours or (not _is_answered(state, slot) and not declined):
            slots[slot] = value
            sources[slot] = SOURCE_DEFAULT
            current[slot] = {"value": value, "reason": effects.default_reasons.get(slot, "")}
    state["rule_defaults"] = current


def is_default(state: dict, slot: str) -> bool:
    """True when the slot's value came from a rule, not from the customer."""
    return (state.get("slot_sources") or {}).get(slot) == SOURCE_DEFAULT


def mark_user_value(state: dict, slot: str) -> None:
    """The customer stated this slot: whatever a default put there is theirs now."""
    sources = state.setdefault("slot_sources", {})
    if sources.get(slot) == SOURCE_DEFAULT:
        logger.info("RULES default replaced by customer: session=%s slot=%s", state.get("session_id"), slot)
    sources[slot] = SOURCE_USER
