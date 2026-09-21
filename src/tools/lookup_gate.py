"""Whether this turn's identifiers really are an inventory lookup.

An inventory lookup is a SIDE QUERY: it pulls up specific stock by identifier and must
never touch category, brand preference or any collected slot. That makes a false positive
expensive - it hijacks the turn, returns one arbitrary trailer instead of qualifying the
customer, and suppresses the brand they actually named.

Four layers keep it honest. This module is layer 3; the other three live elsewhere:

1. The prompt tells the model what a stock number is and is not (src/llm/prompt.py).
2. ``confidence`` - a low-confidence lookup never fires (checked below).
3. THIS MODULE - the structural gate. A lookup needs a real identifier: a stock number, OR
   year + make, OR a genuine model code. A make on its own is a brand preference, and a
   category word is not a model.
4. ``inventory_matcher.match_inventory`` fails closed against the catalogue: a stock number
   matching no row never enters the exact-stock branch, so a bad identifier cannot return a
   WRONG trailer.

Layer 4 is what stops us showing the wrong trailer. This layer is what stops the lookup
running at all when the customer was only shopping.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.domain.categories import resolve_category_matches
from src.domain.links import normalize_listing_url
from src.domain.normalizer import normalize_make

logger = logging.getLogger(__name__)

# Words that make a number a MEASUREMENT rather than an identifier. "7000 lbs" is a load
# weight however confidently the model labelled it a stock number - nobody writes a unit
# after a stock number. Checked against the raw text the model handed us, before the digits
# are stripped out of it.
_MEASUREMENT_UNIT_RE = re.compile(
    r"""(lbs?\b|pounds?\b|kgs?\b|tons?\b|\#|ft\b|foot\b|feet\b|'|"|in\b|inch|yards?\b|\$)""",
    re.IGNORECASE,
)

# The numeric fields on ExtractedFields. A number that ALSO came back as one of these on the
# same turn is that measurement, not a stock number - the model filed one value under two
# names and only one of them can be true.
_MEASUREMENT_FIELDS = (
    "length",
    "width",
    "height",
    "payload_capacity",
    "axle_capacity",
    "total_axle_capacity_lbs",
    "axle_count",
)

# Stock numbers in trailer_listings are 4-6 digits. Necessary, not sufficient: plenty of
# weights are four digits too, which is why the two checks above carry the real weight.
_MIN_STOCK_DIGITS = 4
_MAX_STOCK_DIGITS = 6


def _digits(value: Any) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def _model_text_is_category_word(model_text: str | None) -> bool:
    """True when the "model" is really a trailer CATEGORY ("dump trailer", "utility").

    Ported from New Prompt, where it exists because of a live failure: "I want a Diamond C
    dump trailer, 7x14, ..." came back as a HIGH-confidence lookup with
    model_text="dump trailer". The lookup hijacked the turn - one arbitrary "exact" match
    instead of a qualified search - and suppressed the Diamond C brand preference as "just
    the lookup's make". A category word names what they are shopping for, never a unit.
    """
    text = str(model_text or "").strip().lower()
    if not text:
        return False
    stripped = re.sub(r"\b(trailers?)\b", " ", text).strip()
    if not stripped:
        return True  # "trailer" alone is not a model either
    # A digit-bearing token ("lpx14", "fhg 24k") is a real model code, category word or not.
    return any(
        tier == "naming" for _category, tier in resolve_category_matches(stripped) if _category
    ) and not re.search(r"[a-z]*\d", stripped)


def stock_number_is_plausible(raw: Any, extracted: Any = None) -> bool:
    """Whether this really is a stock number, rather than a measurement in its clothes.

    New Prompt has no equivalent: it relies on the prompt telling the model what a stock
    number is not, and on the matcher failing closed. That leaves a misread number joining
    the fuzzy query text, where it can still nudge make/model scoring.

    Deliberately NOT a length check alone - "7000" is four digits and so is a real stock
    number, so length can never separate the two. The signals that actually can:

    * a UNIT in the customer's wording ("7000 lbs", "20 ft", "$9,995");
    * the same number arriving on this turn as a measurement field as well.
    """
    text = str(raw or "").strip()
    if not text:
        return False

    if _MEASUREMENT_UNIT_RE.search(text):
        logger.info("lookup_gate | stock number %r carries a unit - it is a measurement", text)
        return False

    digits = _digits(text)
    if not (_MIN_STOCK_DIGITS <= len(digits) <= _MAX_STOCK_DIGITS):
        logger.info(
            "lookup_gate | stock number %r is %d digits, outside %d-%d",
            text, len(digits), _MIN_STOCK_DIGITS, _MAX_STOCK_DIGITS,
        )
        return False

    if extracted is not None:
        for field in _MEASUREMENT_FIELDS:
            value = getattr(extracted, field, None)
            if value is None:
                continue
            # 7000.0 and "7000" are the same number filed under two names.
            normalized = int(value) if isinstance(value, float) and value.is_integer() else value
            if _digits(normalized) == digits:
                logger.info(
                    "lookup_gate | stock number %r is also this turn's %s - dropping it",
                    text, field,
                )
                return False

    return True


def usable_stock_number(turn: Any) -> str | None:
    """This turn's stock number, or None when it does not survive the plausibility check.

    Dropping it does NOT cancel the lookup: a turn carrying year + make or a real model code
    still looks those up. Only the bad identifier is discarded.
    """
    lookup = getattr(turn, "inventory_lookup", None) if turn else None
    raw = getattr(lookup, "stock_number", None) if lookup else None
    if not raw:
        return None
    extracted = getattr(turn, "extracted", None)
    return str(raw).strip() if stock_number_is_plausible(raw, extracted) else None


def lookup_requested(turn: Any) -> bool:
    """A direct-identifier inventory lookup rides along with ANY intent, not just
    intent=inventory_lookup.

    The prompt deliberately keeps the dominant intent on the bigger action ("here's my
    email, I'm looking for a Diamond C LPX" -> contact_info_provided), so gating on the
    intent silently drops those lookups. The extracted identifier block is the signal.
    """
    lookup = getattr(turn, "inventory_lookup", None) if turn else None
    if not (lookup and getattr(lookup, "is_lookup", False)):
        return False
    if getattr(lookup, "confidence", "low") not in {"medium", "high"}:
        return False
    if usable_stock_number(turn) or (lookup.year and lookup.make):
        return True
    if normalize_listing_url(getattr(lookup, "listing_url", None)):
        # One of our own listing pages identifies exactly one trailer. A Facebook or
        # Instagram link does not - that goes to the team instead (apply._apply_shared_link).
        return True
    # Make alone is a brand preference, never a lookup - require a real identifier, and a
    # category word posing as the model is category shopping, not an identifier.
    return bool(lookup.model_text) and not _model_text_is_category_word(lookup.model_text)


def _shown_rows(state: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The batch in front of them, and everything shown so far."""
    state = state or {}
    batch = [row for row in (state.get("last_shown_listings") or []) if isinstance(row, dict)]
    running = [row for row in (state.get("shown_listings") or []) if isinstance(row, dict)]
    return batch, running


def identified_listing(state: Any, turn: Any) -> dict[str, Any] | None:
    """The trailer this turn's IDENTIFIERS name, if we have already shown it.

    Separate from the index, because the two answer different questions. This one asks "is
    the thing they named already on their screen?" - which is what decides whether a lookup
    has any work to do. A stock number that matches nothing we have shown is a real lookup
    and must still run.
    """
    batch, running = _shown_rows(state)
    lookup = getattr(turn, "inventory_lookup", None) if turn else None

    stock = _digits(usable_stock_number(turn))
    if stock:
        for row in running or batch:
            if _digits(row.get("stock_number")) == stock:
                return row

    url = normalize_listing_url(getattr(lookup, "listing_url", None)) if lookup else None
    if url:
        for row in running or batch:
            if normalize_listing_url(row.get("url")) == url:
                return row
    return None


def has_an_identifier(turn: Any) -> bool:
    """Does this turn name a trailer by something other than its place in a list?"""
    lookup = getattr(turn, "inventory_lookup", None) if turn else None
    if lookup is None:
        return False
    return bool(
        usable_stock_number(turn)
        or normalize_listing_url(getattr(lookup, "listing_url", None))
        or getattr(lookup, "model_text", None)
        or (getattr(lookup, "year", None) and getattr(lookup, "make", None))
    )


def referenced_listing(state: Any, turn: Any) -> dict[str, Any] | None:
    """The trailer they just pointed at, out of what we have already put on their screen.

    Resolved here rather than by the reply model, which was given only the numbering and
    guessed wrong, quoting a trailer nobody had picked.

    The IDENTIFIER wins over the index. "I like the 81419" arrives as both a stock number
    and a listing_reference, and the two disagree the moment a "show me more" has been
    through: the model numbers from the batch it happens to be looking at, while the stock
    number says exactly which trailer, however far back it was shown.

    An index counts into the LAST batch only - the list in front of them while they type.
    Out of range resolves to nothing rather than to a wrong trailer, and the turn falls
    back to an ordinary lookup.
    """
    identified = identified_listing(state, turn)
    if identified is not None:
        return identified

    batch, _running = _shown_rows(state)
    try:
        index = int(getattr(turn, "listing_reference", None) or 0)
    except (TypeError, ValueError):
        return None
    return batch[index - 1] if 1 <= index <= len(batch) else None


def nothing_left_to_look_up(state: Any, turn: Any) -> bool:
    """Is every trailer this turn names already on the customer's screen?

    The reply pass is told which trailer they mean and that its card is already up - and it
    called lookup_inventory anyway, live, spending a whole agent pass to be told "already
    shown". The tool is withheld for the turn rather than left to be refused, because by the
    time a handler could refuse it the round trip has already been paid for.

    Deliberately narrow. "I like the 81419, but is the 12345 available?" points at a trailer
    on screen AND names one we have never shown: the identifier does not match anything
    shown, so the tool stays.
    """
    if referenced_listing(state, turn) is None:
        return False
    if not has_an_identifier(turn):
        return True  # a bare "the 5th one" - the index resolved, there is nothing to fetch
    return identified_listing(state, turn) is not None


def brand_is_lookup_make(turn: Any, brand: str) -> bool:
    """True when the extracted brand is just the make half of this turn's lookup identifier.

    "I am looking for Iron Bull Dtb" is a request to pull up specific stock, not a standing
    instruction to filter every later search to Iron Bull - recording it as brand_preference
    also fires the which-category-for-that-brand question over the lookup's own results.
    """
    if not lookup_requested(turn):
        return False
    make = getattr(turn.inventory_lookup, "make", None) or ""
    if not make:
        return False
    # "Iron Bull" vs "Iron Bull Trailers": same make, different suffix - containment either way.
    brand_norm = normalize_make(brand).lower()
    make_norm = normalize_make(make).lower()
    return brand_norm in make_norm or make_norm in brand_norm
