"""The category tool: choose one, change one, or suggest a better one. No LLM calls.

``set_trailer_category`` is the deterministic tool the brief asks for (S8): it validates
the category, stores it, and pulls that category's required questions out of the existing
``trailer_fields`` spec. It never invents a category and never invents a question - both
come from the single sources of truth already in ``src/domain``.
"""
from __future__ import annotations

import logging
from typing import Any

from src.domain import brands
from src.domain.categories import (
    CANONICAL_CATEGORIES,
    resolve_category_from_text,
    resolve_category_matches,
)
from src.domain.trailer_fields import get_trailer_fields_as_dict

logger = logging.getLogger(__name__)

# Answers that mean "I have no category in mind". They must resolve to None rather than to
# whatever category a synonym search happens to surface (brief S7).
_NO_CATEGORY_TERMS = (
    "not sure", "unsure", "no idea", "dont know", "don't know", "do not know",
    "any", "anything", "whatever", "no preference", "doesnt matter", "doesn't matter",
    "you tell me", "not certain", "undecided",
)


def normalize_category(text: Any) -> str | None:
    """Resolve free text to one canonical category, or None.

    Official name, alias, or natural language all work (brief S7); "not sure" / "any" /
    "I don't know" deliberately resolve to None, as does anything unrecognized. A category
    we do not stock resolves to None too, so qualification can never start down a path
    that has no inventory behind it.
    """
    raw = str(text or "").strip()
    if not raw:
        return None

    low = raw.lower()
    if any(term in low for term in _NO_CATEGORY_TERMS):
        return None

    # An exact canonical name, case-insensitively.
    for canonical in CANONICAL_CATEGORIES:
        if low == canonical.lower():
            return canonical if _is_stocked(canonical) else None

    resolution = resolve_category_from_text(raw)
    category = resolution.category
    if not category:
        return None
    return category if _is_stocked(category) else None


def _is_stocked(category: str) -> bool:
    stocked = brands.stocked_categories()
    # An empty inventory means the catalogue could not be read at all. Failing open here is
    # deliberate: rejecting every category would break the whole conversation, while the
    # prompt blocks that advertise categories already derive from the same empty list.
    return not stocked or category in stocked


def set_trailer_category(state: dict, category: str) -> dict[str, Any]:
    """Validate, store, and load the category's required questions into the session.

    Persist-safe: everything it writes is a plain JSON type, so the state snapshot round
    trips. No LLM call, no network.

    Returns a small report for the caller; raises nothing on a bad category, because a
    model that proposes one should degrade to "no category yet" rather than to a 500.
    """
    canonical = normalize_category(category)
    if not canonical:
        logger.info("CATEGORY rejected: %r is not a stocked canonical category", category)
        return {"ok": False, "category": None, "reason": "unrecognized"}

    previous = state.get("category")
    spec = get_trailer_fields_as_dict(canonical)

    state["category"] = canonical
    state["required_slots"] = list(spec.get("required_slots") or [])
    state["optional_slots"] = list(spec.get("optional_slots") or [])
    state["slot_questions"] = dict(spec.get("questions") or {})

    if previous and previous != canonical:
        _prune_for_new_category(state)

    logger.info(
        "TOOL set_trailer_category: session=%s category=%s required=%s",
        state.get("session_id"), canonical, state["required_slots"],
    )
    return {
        "ok": True,
        "category": canonical,
        "changed_from": previous if previous != canonical else None,
        "required_slots": list(state["required_slots"]),
    }


def _prune_for_new_category(state: dict) -> None:
    """Drop bookkeeping that belonged to the old category.

    The slots themselves are NOT dropped here - whether collected filters carry over is
    the customer's decision (brief S12), taken in apply.py. What must go is the per-slot
    ask/decline history: a question declined for a Dump trailer has never been put to them
    about an Equipment trailer, so it has to be askable again.

    The shown-listing history goes too. It exists to keep "show me more" showing something
    new, and it is only meaningful WITHIN a category: excluding the dump trailers they
    already saw from a search for utility trailers narrows nothing, and it would hide those
    trailers for good if they ever switched back. ``_apply_refined_search`` clears it the
    same way when a requirement changes, but returns early on the category-change path -
    which is exactly why it has to be cleared here.
    """
    required = set(state.get("required_slots") or [])
    state["asked_counts"] = {
        slot: count for slot, count in (state.get("asked_counts") or {}).items()
        if slot in required
    }
    state["declined_slots"] = [
        slot for slot in (state.get("declined_slots") or []) if slot in required
    ]
    state["pending_slot"] = None
    state["invalid_retry_slot"] = None
    state["invalid_retry_reason"] = None
    state["shown_urls"] = []
    # Nothing has been shown FOR THIS CATEGORY, so the results gate treats it as a fresh
    # search rather than a repeat, and asks the new category's questions first.
    state["results_shown"] = False


# The only slots the keep-or-drop question is ever about: the trailer's configuration.
# They describe the machine, so they carry across a category change and it is reasonable to
# ask whether to keep them.
#
# haul_item is deliberately absent. What they are hauling is what DEFINES the category, so
# on a change it is either restated in the same breath ("actually I need to move a tractor")
# or it no longer applies - "gravel" is not an answer to what an Equipment trailer will
# carry. Offering to keep it would invite a customer to preserve the very thing that just
# changed, and it is a required question for the new category anyway, so it gets asked
# properly rather than inherited.
KEEP_QUESTION_SLOTS: tuple[str, ...] = (
    "length", "width", "payload_capacity", "axle_capacity", "hitch_type",
)


def meaningful_filters(state: dict) -> dict[str, Any]:
    """The collected values worth mentioning in a keep-or-drop question (brief S12).

    Only configuration slots, and only those holding a real value. A None, an empty list
    and a declined slot are all "nothing was collected here", and offering to keep them
    would be noise.
    """
    slots = state.get("slots") or {}
    kept: dict[str, Any] = {}
    for slot in KEEP_QUESTION_SLOTS:
        value = slots.get(slot)
        if value is None:
            continue
        if isinstance(value, (list, str)) and not value:
            continue
        kept[slot] = value
    return kept


def suggest_category_from_haul_item(state: dict, haul_item: Any) -> dict[str, Any] | None:
    """A better-suited category implied by what they said they are hauling.

    Only fires on a CARGO-tier match: the customer described a load ("a tractor"), not a
    trailer type. A naming-tier match means they named a type outright, and that is a
    category selection to be handled as one rather than second-guessed.

    Returns None when there is nothing to suggest - no category set yet, the match is the
    category they are already on, or they have already turned this same suggestion down.
    """
    text = str(haul_item or "").strip()
    current = state.get("category")
    if not text or not current:
        return None

    for candidate, tier in resolve_category_matches(text):
        if tier != "cargo":
            continue
        if candidate == current or not _is_stocked(candidate):
            continue
        pair = f"{current}->{candidate}"
        if pair in (state.get("rejected_switches") or []):
            return None
        logger.info(
            "TOOL suggest_category: session=%s haul_item=%r %s -> %s",
            state.get("session_id"), text, current, candidate,
        )
        return {"suggested": candidate, "from_haul_item": text, "current": current, "pair": pair}
    return None
