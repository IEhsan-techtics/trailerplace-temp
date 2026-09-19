"""Per-axle or total: reading an axle capacity the way the customer meant it. Pure, no I/O.

"7,000 lb axles" and "14,000 lbs total" describe different trailers, and filing one as the
other ranks trailers rated at double or half what they need. The model is asked which one
the customer meant (``axle_capacity_basis``), but its confidence is not evidence: New Prompt
saw it read "14,000 lbs of axle capacity" as a confident total. So the customer's own
wording gets a second opinion here, and a capacity that says neither is held and asked
about rather than guessed.

Ported from New Prompt's ``src/graph/apply_analysis.py``; the wording and the patterns are
unchanged.
"""
from __future__ import annotations

import re

BASIS_QUESTION = (
    "Just so I match the right trailers - is that per axle, or the total across all the axles?"
)
COUNT_QUESTION = "And how many axles should the trailer have - single, tandem, or triple?"

# Each question is asked at most this many times, like every other question (brief S23).
MAX_CLARIFY_ASKS = 2

# How the customer answers "per axle, or total?". Deliberately narrow: anything that is not
# clearly one side leaves the question open rather than guessing again.
_PER_AXLE_REPLY_RE = re.compile(r"\b(per|each|every|apiece|a piece)\b|\beach axle\b|\bper axle\b", re.I)
_TOTAL_REPLY_RE = re.compile(
    r"\b(total|combined|altogether|all together|overall|between them|across)\b", re.I
)
# "7,000 lb axles" / "2-7,000# axles" - the rating is attached to the axles themselves, which
# by convention means EACH of them. Not ambiguous, and never questioned.
_PER_AXLE_WORDING_RE = re.compile(r"\d[\d,.]*\s*(?:#|k\b|lbs?\b|pounds?\b)?[\s-]*axles?\b", re.I)
# "14,000 lbs of axle capacity" - a quantity of capacity with nothing saying whose.
_BARE_CAPACITY_WORDING_RE = re.compile(r"axle\s+capacity|capacity\s+of\s+the\s+axles", re.I)

# "Not sure" to the count question is an answer: no preference.
_NO_PREFERENCE_RE = re.compile(
    r"\b(not sure|unsure|don'?t know|dunno|no idea|any|either|whatever|no preference|doesn'?t matter)\b",
    re.I,
)


def infer_basis(text: str, model_basis: str | None) -> str | None:
    """The model's basis, unless the wording is a bare "axle capacity" with no marker.

    Only ever promotes to "unclear": wording that genuinely says which is left to the model.
    """
    words = str(text or "")
    if _PER_AXLE_REPLY_RE.search(words) or _TOTAL_REPLY_RE.search(words):
        return model_basis
    if _PER_AXLE_WORDING_RE.search(words):
        return model_basis
    if _BARE_CAPACITY_WORDING_RE.search(words):
        return "unclear"
    return model_basis


def basis_from_reply(text: str, model_basis: str | None) -> str | None:
    """Their answer to BASIS_QUESTION: "per_axle", "total", or None when still unclear.

    Their words first; the model's field second, because a bare "per axle" carries no number
    for the model to reason about and its basis is usually null on this turn.
    """
    words = str(text or "")
    if _PER_AXLE_REPLY_RE.search(words):
        return "per_axle"
    if _TOTAL_REPLY_RE.search(words):
        return "total"
    if model_basis in {"per_axle", "total"}:
        return model_basis
    return None


# A number carrying a unit, or a big one, is a weight or a size - not a count of axles.
_MEASUREMENT_RE = re.compile(
    r"\d[\d,.]*\s*(?:k\b|lbs?\b|pounds?\b|tons?\b|kg\b|ft\b|foot\b|feet\b|inch|in\b|#|'|\")"
    r"|\d{3,}|\d,\d{3}",
    re.I,
)


def count_from_reply(text: str) -> int | None:
    """The axle count in a reply to COUNT_QUESTION, or None when it states none.

    Out-of-range counts ("5 axles") are returned as they are, so the range check can explain
    them. A weight or a size ("about 5,000 lbs") is not a count at all: New Prompt read that
    as 5,000 axles and told the customer we only carry one to four.
    """
    from src.domain.slot_map import parse_axle_count_answer

    words = str(text or "")
    if _MEASUREMENT_RE.search(words):
        return None
    return parse_axle_count_answer(words)


def is_no_preference(text: str) -> bool:
    """"Not sure", "any", "whatever works" - to the count question, no preference."""
    return bool(_NO_PREFERENCE_RE.search(str(text or "")))
