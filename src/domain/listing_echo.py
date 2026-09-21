"""Telling a customer's own words apart from a trailer's spec sheet read back at us.

When someone picks a trailer off a list - "I like the 81419" - the analysis pass has the
whole card in its history, and it reads the card's specs into ``extracted`` as though the
customer had stated them. Seen live, from four words:

    extracted: length=32.0 width=5.0 payload=11470.0 hitch=Gooseneck brand=Gooseneck
    axles: per_axle=6000.0   features: adjustable coupler
    quantities: trailer_size=5.0ft ("5' x 32'"), length=32.0ft ("32'"), ...

Every one of those was stored with source=user. The customer's real requirement (24 ft)
was overwritten by the card's 32 ft, the shown-history was cleared as a "requirement
change", qualification reopened, and an axle-count question was asked off the card's own
axle rating.

The discriminator is deterministic and needs no model: **did those words appear in the
message they just sent?** "5' x 32'" is not in "I like the 81419"; "24 ft" is in "the
81419, but I need 24 ft". So a pointing turn keeps what they actually typed and drops what
was read off the card - which is the house rule either way: the model interprets the
customer, Python decides what is stored.

Only pointing turns are guarded. On an ordinary turn the model is the one that reads the
customer's words, typos, ranges and spelled-out numbers included, and a literal check
against the raw message would throw away perfectly good readings.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Words that carry no evidence about whether the customer said a thing: they turn up in
# almost any sentence, so requiring them to match proves nothing.
_NOISE_WORDS = frozenset(
    {"the", "and", "for", "with", "that", "this", "one", "its", "has", "lbs", "lb", "ft",
     "feet", "foot", "inch", "inches", "trailer", "trailers", "capacity", "type", "about"}
)

_DIGITS_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[a-z]{3,}")


def is_pointing_turn(output: Any) -> bool:
    """Are they pointing at a trailer already on their screen, rather than specifying one?

    Either signal counts. ``listing_reference`` is the model saying "this is listing #5 of
    the batch I showed", and ``listing_interest`` is the same thing when it could not pin
    down which one.
    """
    if getattr(output, "listing_reference", None) is not None:
        return True
    return str(getattr(output, "intent", "") or "") == "listing_interest"


def _as_text(text: Any) -> str:
    """The value as the customer might have typed it.

    ``32.0`` is written "32" by every human being alive, and the trailing ".0" would put a
    zero into the digit runs that their message could never match. A list is a multi-valued
    slot (``hitch_type``); each member is checked by the caller's ``all``.
    """
    if isinstance(text, bool) or text is None:
        return ""
    if isinstance(text, float) and text.is_integer():
        return str(int(text))
    if isinstance(text, (list, tuple, set)):
        return " ".join(_as_text(item) for item in text)
    return str(text)


def _said_in(message: str, text: Any) -> bool:
    """Is this value present in the customer's own message?

    Numbers first, because a spec is almost always a number: every digit run in the value
    has to appear in what they typed. "11470" against "I like the 81419" does not, so the
    payload was the card's, not theirs.

    Words are the fallback, for the requirement they spelled out ("twenty four feet") and
    for the values that are not numeric at all ("Gooseneck"). Noise words are ignored, so
    a match has to rest on something that means something.
    """
    body = _as_text(text).replace(",", "").casefold()
    if not body.strip():
        return False
    said = str(message or "").replace(",", "").casefold()

    numbers = _DIGITS_RE.findall(body)
    if numbers:
        # Whole runs, not substrings: "141" must not count as found because "81419"
        # happens to contain it.
        theirs = set(_DIGITS_RE.findall(said))
        return all(number in theirs for number in numbers)

    words = [word for word in _WORD_RE.findall(body) if word not in _NOISE_WORDS]
    return bool(words) and all(word in said for word in words)


def echo_guard(output: Any, user_message: str) -> Callable[[str, Any], bool]:
    """A predicate: should this value be dropped as the card's rather than theirs?

    Always False when the turn is not a pointing turn, so an ordinary turn is untouched.
    The slot name is taken only to say which value was dropped, in the log.
    """
    if not is_pointing_turn(output):
        return lambda _slot, _text: False

    def dropped(slot: str, text: Any) -> bool:
        if _said_in(user_message, text):
            return False
        logger.info(
            "LISTING echo ignored: slot=%s value=%r - they pointed at a listing, "
            "they did not state this",
            slot, text,
        )
        return True

    return dropped
