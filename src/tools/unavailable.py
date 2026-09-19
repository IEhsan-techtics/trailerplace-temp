"""A customer asking for a trailer type we do not carry. No LLM calls.

Two kinds, handled the same way:

* a category we recognise but hold no stock in right now (Diesel Tank today) - Python can
  tell this on its own, from ``category_mentioned`` and the live catalogue;
* a type we never sell (a boat trailer, a camper) - only the model can spot it, so it
  reports it in ``unavailable_type_requested``.

Either way the team is told, so someone can help them find an alternative, and the reply
says what is true: we do not have it, this is what we do have, and the team will be in touch.
It is a lead worth following up, not a dead end.
"""
from __future__ import annotations

import logging
from typing import Any

from src.domain import brands
from src.domain.canned_responses import PHONE
from src.domain.categories import CANONICAL_CATEGORIES, resolve_category_matches
from src.tools import team_notify

logger = logging.getLogger(__name__)


def _stocked() -> tuple[str, ...]:
    # An empty tuple means the catalogue could not be read. Everything counts as stocked
    # then, exactly as normalize_category fails open - better to qualify them for a type we
    # may not have than to tell them we have nothing.
    return brands.stocked_categories() or tuple(CANONICAL_CATEGORIES)


def _named_category(text: str) -> str | None:
    """The canonical category the text NAMES outright ("a dump trailer"), if any.

    Naming tier only: cargo words merely suggest a category ("a trailer for my cattle"), and
    must not be allowed to overrule the model's reading of a type we do not sell.
    """
    low = text.strip().lower()
    for canonical in CANONICAL_CATEGORIES:
        if low == canonical.lower():
            return canonical
    for category, tier in resolve_category_matches(text):
        if tier == "naming":
            return category
    return None


def requested_type(output: Any) -> str | None:
    """The type they asked for that we cannot sell them right now, or None."""
    stocked = set(_stocked())

    reported = str(getattr(output, "unavailable_type_requested", None) or "").strip()
    if reported:
        named = _named_category(reported)
        if named in stocked:
            # The model flagged a type we DO carry ("toy hauler" is our Car Hauler). The
            # catalogue has the final say.
            logger.info("UNAVAILABLE ignored: %r names stocked %s", reported, named)
            return None
        return named or reported

    mentioned = str(getattr(output, "category_mentioned", None) or "").strip()
    if mentioned:
        named = _named_category(mentioned)
        if named and named not in stocked:
            return named
    return None


def _label(requested: str) -> str:
    """How the reply names it, in the plural.

    "Diesel Tank" -> "Diesel Tank trailers", "boat trailer" -> "boat trailers",
    "camper" / "campers" -> "campers" (a camper is not a kind of trailer to the customer).
    """
    text = requested.strip()
    low = text.lower()
    if text in CANONICAL_CATEGORIES:
        return text + " trailers"
    if low.endswith("trailer"):
        return text + "s"
    if "trailer" in low or low.endswith("s"):
        return text
    return text + "s"


def record(state: dict, output: Any, requested: str) -> str:
    """Tell the team, once per type per conversation. Returns the team_notify status."""
    seen = state.setdefault("unavailable_requests", [])
    # By the plural label, so "boat trailer" and "boat trailers" are the same request.
    key = _label(requested).lower()
    if key in seen:
        # Asked again. The team already has it (or will, once we can reach them) - a second
        # email about the same boat trailer helps no one.
        if team_notify.declined(state):
            return "dropped"
        return "sent" if team_notify.contact_complete(state) else "stashed"
    seen.append(key)
    summary = (getattr(output, "turn_summary", "") or "").strip()
    return team_notify.record(
        state,
        reason=f"Trailer type not in stock - {requested}",
        description=summary or f"Asked for {_label(requested)}, which we do not currently carry.",
    )


def reply(state: dict, requested: str, status: str, first_turn: bool = False) -> str:
    """Not available, what we do have (as bullets), and what happens next."""
    opener = "Thank you for contacting TrailerPlace. " if first_turn else ""
    head = (
        f"{opener}I'm sorry - we don't currently have {_label(requested)} available. "
        "Here's what we do carry right now:"
    )
    bullets = "\n".join(f"- {category}" for category in _stocked())

    if status == "sent":
        tail = (
            "I've passed this on to our team, and they'll contact you shortly to guide you "
            "to the right option."
        )
    elif status == "dropped":
        tail = f"If you'd like help finding an alternative, our team is on {PHONE}."
    else:
        tail = "Our team will contact you shortly to guide you to the right option. " + _ask(state)
    return f"{head}\n{bullets}\n\n{tail}".strip()


def _ask(state: dict) -> str:
    missing = team_notify.missing_pieces(state)
    if missing == ["name"]:
        return "Could I take your name so they can reach you?"
    if missing == ["contact"]:
        return "Could I take an email or phone number so they can reach you?"
    return "Could I take your name and an email or phone number so they can reach you?"
