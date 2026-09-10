"""Deciding what a customer means by "gooseneck".

It is genuinely both things in our catalogue: a hitch type on every category, and a
manufacturer on four Livestock rows in ``trailer_listings``. Read the wrong way it is not a
near miss - as a make it narrows the search to one brand, as a hitch it filters a column
the customer actually cares about. The two readings return disjoint inventory.

``slot_map.brand_is_actually_a_hitch`` already guards the common case, but it decides
every time: no "brand"/"make" word in the message means hitch, full stop. That is right far
more often than not, and silently wrong for "I want the gooseneck trailer" - which really
is ambiguous, and is worth one short question rather than a guess.

So this module resolves what it can and admits what it cannot:

* answering the hitch question       -> hitch
* "gooseneck hitch", "gooseneck coupler" -> hitch
* the model read another make too   -> hitch (the other name is the brand)
* "the gooseneck brand", "made by gooseneck" -> brand
* anything else                      -> ambiguous, ask

No LLM call. All four resolutions are readable off the text itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

# The make as it appears in trailer_listings.
GOOSENECK_MAKE = "Gooseneck"

_GOOSENECK_RE = re.compile(r"\bgoose[\s-]?neck\b", re.IGNORECASE)

# "gooseneck hitch", "gooseneck coupler", "gooseneck setup" - the noun after it names the
# COUPLING, so they are describing how it attaches, not who built it.
_HITCH_CONTEXT_RE = re.compile(
    r"\bgoose[\s-]?neck\b[\s-]*(hitch|coupler|coupling|ball|setup|connection|style|type)\b"
    r"|\b(hitch|coupler|coupling)\b[\s-]*\bgoose[\s-]?neck\b",
    re.IGNORECASE,
)

# "the gooseneck brand", "made by gooseneck", "a gooseneck brand trailer".
_BRAND_CONTEXT_RE = re.compile(
    r"\b(brand|make|manufacturer|manufactured|made\s+by|built\s+by)\b[\s\w]{0,12}?"
    r"\bgoose[\s-]?neck\b"
    r"|\bgoose[\s-]?neck\b[\s-]*(brand|make|manufacturer)\b",
    re.IGNORECASE,
)

# Reasons, so the caller can log WHY and the tests can assert on the path taken rather than
# only on the answer.
HITCH = "hitch"
BRAND = "brand"
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class GooseneckReading:
    meaning: str                      # HITCH | BRAND | AMBIGUOUS
    reason: str
    other_brand: Optional[str] = None  # a different make named in the same message

    @property
    def needs_clarification(self) -> bool:
        return self.meaning == AMBIGUOUS


def mentions_gooseneck(text: object) -> bool:
    return bool(_GOOSENECK_RE.search(str(text or "")))


def resolve_gooseneck_mention(
    text: object,
    *,
    pending_slot: str | None = None,
    extracted_brand: str | None = None,
    known_makes: Iterable[str] | None = None,
    gooseneck_is_a_stocked_make: bool | None = None,
) -> GooseneckReading:
    """What did they mean by "gooseneck"?

    ``pending_slot`` is the question we asked last turn - answering the hitch question with
    "gooseneck" is the least ambiguous case there is, so it is checked first.

    ``extracted_brand`` is the make the MODEL read out of the message. Brand recognition is
    the model's job: it has the full make list in its prompt and it copes with the misspellings
    and part-names customers actually type. Matching make names against the text here instead
    meant fuzzy-matching, and "trailer" scores 0.83 against the "Trail" in a name like Load
    Trail - so the commonest word in these conversations resolved the ambiguity by naming a
    brand nobody had mentioned.

    ``gooseneck_is_a_stocked_make`` defaults to reading the catalogue. When Gooseneck is not
    a make we stock, the word has only one possible meaning and there is nothing to ask
    about - the ambiguity is a property of the inventory, not of the language.
    """
    message = str(text or "")
    if not mentions_gooseneck(message):
        return GooseneckReading(HITCH, reason="not_mentioned")

    # 1. They are answering "bumper pull or gooseneck?". Nothing to disambiguate.
    if pending_slot == "hitch_type":
        return GooseneckReading(HITCH, reason="answering_hitch_question")

    # 2. They said what kind of thing it is: "gooseneck hitch", "gooseneck coupler".
    if _HITCH_CONTEXT_RE.search(message):
        return GooseneckReading(HITCH, reason="hitch_word_nearby")

    # 3. They named it as a manufacturer outright.
    if _BRAND_CONTEXT_RE.search(message):
        return GooseneckReading(BRAND, reason="brand_word_nearby")

    # 4. The model read another make out of the message, so that one is the manufacturer
    #    and gooseneck describes the hitch.
    other = (extracted_brand or "").strip()
    if other and other.lower() != GOOSENECK_MAKE.lower():
        return GooseneckReading(HITCH, reason="another_make_named", other_brand=other)

    # 5. Genuinely ambiguous - but only if both readings are real. If we do not stock the
    #    make, there is only one thing it can mean.
    if gooseneck_is_a_stocked_make is None:
        if known_makes is None:
            from src.domain.brands import known_makes as _known_makes

            known_makes = _known_makes()
        gooseneck_is_a_stocked_make = GOOSENECK_MAKE in set(known_makes or ())
    if not gooseneck_is_a_stocked_make:
        return GooseneckReading(HITCH, reason="make_not_stocked")

    return GooseneckReading(AMBIGUOUS, reason="both_readings_possible")


def clarification_question() -> str:
    """One short question. Deliberately offers the two readings in plain words rather than
    the words "hitch type" and "make", which is what the customer was ambiguous about in
    the first place."""
    return (
        "Quick check - do you mean a gooseneck hitch, or Gooseneck the trailer brand? "
        "We carry both."
    )


def apply_clarification_answer(text: object) -> Optional[str]:
    """Read their reply to the clarification question: HITCH, BRAND, or None if unclear."""
    low = str(text or "").lower()
    if not low.strip():
        return None
    # Checked before the hitch words: "the brand" is decisive even in a sentence that also
    # says "hitch", as in "the brand, not the hitch".
    if re.search(r"\b(brand|make|manufacturer|company)\b", low):
        return BRAND
    if re.search(r"\b(hitch|coupler|coupling|tow\w*|attach\w*|pull\w*)\b", low):
        return HITCH
    return None
