from __future__ import annotations

import re
from typing import Any

from src.domain.categories import resolve_categories_from_text
from src.domain.normalizer import normalize_category
from src.domain.units import (
    _range_candidates,
    parse_dimensions,
    parse_length_ft,
    parse_length_ft_loose,
    parse_weight_lbs,
    parse_weight_lbs_loose,
)

_ALLOWED_HITCH_TYPES = {"Gooseneck", "Bumper Pull"}

# What KIND of value each slot itself holds. `_SLOT_METADATA_FILTER_MAP` says where a
# slot's answer gets copied for filtering; this says how the slot's OWN value should be
# parsed before storing, so a numeric slot never ends up holding raw text like "8x25".
_SLOT_VALUE_KIND = {
    # One slot per measured concept. The per-category aliases these replaced
    # (haul_length_ft / vehicle_length_ft / trailer_length_ft, haul_weight_lbs /
    # payload_need / total_weight, ...) all asked the same question under a different
    # name; sharing one name is what makes an answer given once count everywhere.
    "length": "length_ft",
    "width": "width_ft",
    "height": "height_ft",
    "payload_capacity": "payload_lbs",
    "payload_lbs": "payload_lbs",
    # A kind of its own, NOT "payload_lbs". Kinds define the alias families that let one
    # slot answer another (see _ALIAS_GROUPS/can_autofill_slot): filed under payload, the
    # load weight the customer already gave would silently auto-fill the axle rating and
    # the question would never be asked. A lone kind also falls outside can_autofill_slot's
    # whitelist, so it never receives a value from anywhere else.
    "axle_capacity": "axle_capacity_lbs",
    # Two more lone kinds, for the same reason and one more besides: per-axle and total are
    # the SAME question asked two ways, so sharing a kind would let one auto-fill the other
    # and quietly answer the very ambiguity we now stop to ask about. "7,000 lb axles" and
    # "7,000 lbs across the axles" are different trailers.
    "total_axle_capacity_lbs": "total_axle_capacity_lbs",
    # Not a weight at all - a plain count, so it parses differently from every kind above.
    "axle_count": "axle_count",
    "bin_size": "length_ft",
    # Combined size questions ("size preference (length / width)?"). Their own value is the
    # LENGTH in feet — the width/height they also carry are stored under their own slots.
    # Without a kind here they kept the raw sentence ("I'd rather keep it 18ft"), which then
    # travelled verbatim into the search query text.
    "cargo_size": "length_ft",
    "trailer_size": "length_ft",
    "hitch_type": "hitch_type",
    "base_category": "subcategory",
}

# Real answers are phrases ("goose neck please"), not the exact HITCH_MAP keys - match the
# known terms as a word-bounded substring, longest first so "bumper pull" outranks a
# coincidental shorter match.
_HITCH_TERM_ORDER = sorted(
    {
        "bumper pull": "Bumper Pull", "bumperpull": "Bumper Pull", "bumper-pull": "Bumper Pull",
        "tag along": "Bumper Pull", "tag-along": "Bumper Pull",
        "gooseneck": "Gooseneck", "goose neck": "Gooseneck", "goose-neck": "Gooseneck",
    }.items(),
    key=lambda pair: -len(pair[0]),
)


def normalize_hitch_answer(raw_answer: Any) -> Any:
    """Canonicalize a free-text hitch answer to the same ``["Gooseneck"]``-style
    single-item list the LLM's constrained extraction produces for a clear preference -
    so a raw slot_answers entry can never downgrade that clean value to unnormalized text.

    "either"/"any"/"no preference", a genuine typo the term list doesn't catch, or any
    other unrecognized text all mean the same thing here: no specific hitch preference
    was given, so the slot is null - never invented, never a middle-ground list of both.
    """
    text = str(raw_answer or "").strip().lower()
    if not text:
        return None
    for term, canonical in _HITCH_TERM_ORDER:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
            return [canonical]
    return None


def normalize_subcategory_answer(raw_answer: Any) -> Any:
    """Only a recognizable trailer-type name counts as an Aluminum ``base_category``
    answer. The question is open-ended ("utility, equipment, enclosed, or something
    else?"), but a vague reply to that ("not sure", "whatever's in stock") is not a real
    preference, so it resolves to null rather than being stored as junk text. Reuses
    categories.py's own term list so this can never recognize a name the category
    resolver itself wouldn't.
    """
    text = str(raw_answer or "").strip()
    if not text:
        return None
    matches = resolve_categories_from_text(text)
    return matches[0] if matches else None


# "I want it in Gooseneck" is a hitch. "I want a Gooseneck" is... also usually a hitch —
# even though Gooseneck is a real make we carry. We only read it as the make when the
# customer frames it as one.
_EXPLICIT_BRAND_RE = re.compile(r"\b(brand|make|manufacturer|manufactured|made\s+by)\b", re.IGNORECASE)


def brand_is_actually_a_hitch(brand: Any, user_text: str) -> bool:
    """True when a 'brand' the extractor reported is really a hitch preference.

    Gooseneck is both a hitch type and a make in our catalogue. Read as a make it silently
    narrows the search to one manufacturer; read as a hitch it filters on the thing the
    customer actually cares about. So it is only a make when they say so.
    """
    return normalize_hitch_answer(brand) is not None and not _EXPLICIT_BRAND_RE.search(user_text or "")


# A "feature" is something we CANNOT filter on. Sizes, weights, hitch type, the Aluminum
# sub-category and price all have homes of their own; parked in non_metadata_features they
# are silently dropped from the search instead of filtering it.
_PRICE_RE = re.compile(
    r"\$|\b(price[ds]?|pricing|budget|cost|costs|afford\w*)\b|\b(?:under|below|over|up\s+to)\s*\d",
    re.IGNORECASE,
)
_MEASUREMENT_WORDS = re.compile(
    r"\b(ft|foot|feet|in|inch|inches|lb|lbs|pound|pounds|ton|tons|kg|"
    r"long|length|wide|width|tall|height|high|weigh\w*|weight|payload|capacity|gvwr|"
    r"trailer|trailers|about|around|roughly|approx\w*|max|maximum|min|minimum|at|least|"
    r"under|below|over|up|to|less|more|than|"
    r"a|an|the|of|is|be|should|only|please|prefer\w*|need|want|k)\b",
    re.IGNORECASE,
)

# Defensive cleanup for occasional extractor leakage. The Analyze prompt is the
# primary classifier, but category/make words must not become feature requirements
# even when the model returns a surrounding noun phrase such as "insulated enclosed
# trailer". Longest phrases are removed first.
_NON_FEATURE_IDENTITY_PHRASES = (
    "race trailer", "car hauler", "toy hauler", "roll off", "roll-off",
    "diesel tank", "fuel tank", "tank trailer", "box trailer", "v nose", "v-nose",
    "command trailer", "flat bed", "flatbed", "dump trailer", "full tilt",
    "enclosed", "equipment", "utility", "fiber", "livestock", "tilt", "dump",
    "aluminum", "subcategory", "category", "trailer", "trailers",
)
_NON_FEATURE_COLOURS = (
    "black", "white", "gray", "grey", "silver", "red", "blue", "green",
    "yellow", "orange", "brown", "tan", "beige", "charcoal", "bronze",
    "gold", "maroon", "burgundy", "purple",
)
_FEATURE_FILLER_RE = re.compile(
    r"\b(?:a|an|the|with|and|or|in|on|of|by|from|made|year|model|stock|number|"
    r"hitch|only|preferred|preference|please)\b",
    re.IGNORECASE,
)
_FEATURE_DIMENSION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ft|foot|feet|in|inch|inches|lb|lbs|pound|pounds|tons?|kg)\b",
    re.IGNORECASE,
)


def _remove_phrase(text: str, phrase: str) -> str:
    words = [re.escape(word) for word in re.findall(r"\w+", phrase)]
    if not words:
        return text
    pattern = r"(?<!\w)" + r"[\s-]+".join(words) + r"(?!\w)"
    return re.sub(pattern, " ", text, flags=re.IGNORECASE)


def _clean_feature_only_value(value: Any) -> str:
    """Remove searchable trailer identity/metadata from a feature noun phrase."""
    text = str(value or "").strip()
    if not text:
        return ""

    # Lazy import avoids making brand inventory loading part of module import.
    from src.domain.brands import known_makes

    removable = sorted(
        (*known_makes(), *_NON_FEATURE_IDENTITY_PHRASES, *_NON_FEATURE_COLOURS),
        key=len,
        reverse=True,
    )
    for phrase in removable:
        text = _remove_phrase(text, str(phrase))
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = _FEATURE_DIMENSION_RE.sub(" ", text)
    text = _FEATURE_FILLER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" ,;:-")


# Anything naming an axle belongs to the axle_capacity slot, never to the feature list. The number
# is the whole point of an axle phrase, and the feature path throws it away: "10k axles" survives
# _is_only_a_measurement_or_price (the noun "axles" outlives the digits) and reaches the reranker
# as the bare string "axles", which matches nearly every tandem trailer. Worse, feature_ranker
# STRIPS the "Axle Capacity" label from the evidence it shows the model, so an axle feature can
# only ever score 0 and drag the coverage average down. Seen live: "a trailer with 10k axles"
# correctly stored axle_capacity=10000 and ALSO kept the feature "10k axles".
_AXLE_FEATURE_RE = re.compile(r"\baxles?\b", re.IGNORECASE)


def mentions_an_axle(value: Any) -> bool:
    """True when a phrase names an axle in any form ("10k axles", "axle capacity", "7000 lb axle")."""
    return bool(_AXLE_FEATURE_RE.search(str(value or "")))


# An axle phrase is junk-as-a-feature UNLESS it names the axle's TYPE or construction.
# Counts and capacities have fields of their own (axle_count, axle_capacity) and a
# duplicate feature string can only score 0 in the ranker - feature_ranker strips the Axle
# Capacity label from the evidence it shows the model. But the axle's TYPE has no field
# anywhere, so "torsion axles" is a real requirement and the feature list is the only place
# it can do any work; the old guard dropped every phrase containing "axle" and lost it.
# A whitelist, not "anything without a number": "heavy duty axles" is a vague quality that
# matches almost any tandem trailer and scores 0 just like a bare count would.
_AXLE_TYPE_WORD_RE = re.compile(
    r"\b(torsion|spring|leaf|drop|straight|lift|idler|greaseable|oil[- ]?bath|rubber(?:[- ]ride)?|electric|hydraulic|disc|drum|self[- ]adjusting)\b",
    re.IGNORECASE,
)


def axle_phrase_is_count_or_capacity(value: Any) -> bool:
    """True for an axle phrase we must NOT keep as a feature.

    False only when the phrase names the axle's type or construction ("torsion axles",
    "spring axles with electric brakes") - that is a requirement we hold no metadata for.
    """
    text = str(value or "")
    if not mentions_an_axle(text):
        return False
    return not _AXLE_TYPE_WORD_RE.search(text)


def _is_only_a_measurement_or_price(feature: str) -> bool:
    """True when the phrase says nothing beyond a size, a weight or a price.

    "18 ft long" and "$20,000 budget" are requirements, not features. "16 ft ramps" IS a
    feature â€” after the number and the filler words, a real noun ("ramps") survives.
    """
    text = str(feature or "").strip().lower()
    if not text:
        return True
    is_measurement = parse_length_ft(text) is not None or parse_weight_lbs(text) is not None
    is_price = bool(_PRICE_RE.search(text))
    if not (is_measurement or is_price):
        return False
    remainder = _PRICE_RE.sub(" ", text)
    remainder = re.sub(r"[\d.,'\"Ã—x-]+", " ", remainder)
    remainder = _MEASUREMENT_WORDS.sub(" ", remainder)
    return not re.search(r"[a-z]", remainder)


def sanitize_non_metadata_features(features: Any) -> tuple[list[str], Any]:
    """Split the extractor's feature list into real features and a hitch preference.

    Returns ``(features_to_keep, hitch_value_or_None)``. A hitch stated as a feature
    ("gooseneck hitch only") is a filter we own, not a nice-to-have, so it is lifted out;
    a bare size/weight/price is dropped (its value is already in its own slot), and so is
    anything naming an axle, which belongs to axle_capacity.
    """
    kept: list[str] = []
    hitch: Any = None
    for feature in features or []:
        found = normalize_hitch_answer(feature)
        if found:
            hitch = hitch or found
        # After the hitch lift, so "gooseneck with 10k axles" still yields the hitch.
        if axle_phrase_is_count_or_capacity(feature):
            continue
        if _is_only_a_measurement_or_price(feature):
            continue
        cleaned = _clean_feature_only_value(feature)
        if (
            cleaned
            and not axle_phrase_is_count_or_capacity(cleaned)
            and not _is_only_a_measurement_or_price(cleaned)
            and cleaned.casefold() not in {value.casefold() for value in kept}
        ):
            kept.append(cleaned)
    return kept, hitch


# Where each slot's answer is copied for filtering. Slot names on the left are the shared
# ones; the targets on the right are the search-side/DB-facing names and are NOT renamed.
# The combined-size questions keep their own slot names but still land on the dimension
# targets - that is how "8x25" answering trailer_size still reaches the length/width filters.
_SLOT_METADATA_FILTER_MAP = {
    "axle_capacity": ("axle_capacity_lbs",),
    "total_axle_capacity_lbs": ("total_axle_capacity_lbs",),
    "axle_count": ("axle_count",),
    "base_category": ("subcategory",),
    "bin_size": ("length_ft",),
    "cargo_size": ("length_ft", "width_ft", "height_ft"),
    "length": ("length_ft",),
    "width": ("width_ft",),
    "height": ("height_ft",),
    "payload_capacity": ("payload_lbs",),
    "trailer_size": ("length_ft", "width_ft"),
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_roll_off_bin_size_as_length(value: Any) -> float | None:
    """Bin size is yardage, mapped directly to a length_ft NUMBER (15 yd -> 15, not a unit
    conversion) - so it follows the same range/vague rules as any other measurement: a
    range ("15-20 yd") takes the smallest side, and a vague answer with no number at all
    ("as big as you have") is null.
    """
    if _is_number(value):
        return float(value)
    text = str(value or "").strip().lower().replace(",", "")
    if not text:
        return None
    bounds = _range_candidates(text)
    if bounds:
        nums = [float(m.group(0)) for c in bounds if (m := re.search(r"\d+(?:\.\d+)?", c))]
        if nums:
            return min(nums)
    matches = re.findall(r"\d+(?:\.\d+)?", text)
    return float(matches[0]) if matches else None


def normalize_slot_value(category: str, key: str, value: Any) -> Any:
    if _is_number(value):
        return float(value)

    if key in {"length_ft", "width_ft", "height_ft"}:
        if key == "length_ft" and normalize_category(category) == "Roll Off":
            return _normalize_roll_off_bin_size_as_length(value)
        # "8x25"/"8x25x6.5" carries multiple dimensions: pull out the one this key asks for.
        dims = parse_dimensions(value)
        if dims:
            width_ft, length_ft, height_ft = dims
            return {"width_ft": width_ft, "length_ft": length_ft, "height_ft": height_ft}[key]
        return parse_length_ft_loose(value)

    if key == "axle_count":
        # A count, not a measurement: no units to convert and no range to collapse. Parsed
        # here rather than with the weights so "2" never picks up a "lbs" reading.
        return parse_axle_count_answer(value)

    if key in {"payload_lbs", "axle_capacity_lbs", "total_axle_capacity_lbs"}:
        # Resolves a range to its smallest side ("5,000-7,000 lbs" -> 5000) and returns
        # None for an answer carrying no usable number, which normalize_answer_for_slot
        # stores as "asked, no preference".
        return parse_weight_lbs_loose(value)

    return value


_DIMENSION_TARGETS = {"length_ft", "width_ft", "height_ft"}


def normalize_slot_targets(category: str, slot_name: str, value: Any) -> dict[str, Any]:
    """Every search filter target this one answer fills, with its parsed number.

    A combined size question ("Do you have a size preference (length / width)?") maps to
    several dimension targets, but a lone number answering it states ONE of them - and it
    is the length. Copying that number into width_ft (and height_ft) too would invent a
    requirement the customer never gave and wreck the fit rerank, so a single number only
    spreads across the targets when the answer really does carry several dimensions
    ("8x25", "8x20x7").
    """
    targets = _SLOT_METADATA_FILTER_MAP.get(slot_name, ())
    dimension_targets = set(targets) & _DIMENSION_TARGETS
    one_number_only = len(dimension_targets) > 1 and parse_dimensions(value) is None
    filled: dict[str, Any] = {}
    for target in targets:
        if one_number_only and target in {"width_ft", "height_ft"}:
            continue
        parsed = normalize_slot_value(category, target, value)
        if parsed is not None:
            filled[target] = float(parsed) if _is_number(parsed) else parsed
    return filled


def slot_value_kind(slot_name: str) -> str | None:
    """The parse kind for a slot's own value ("length_ft"/"width_ft"/...), or None if free text."""
    return _SLOT_VALUE_KIND.get(slot_name)


# Categories used to ask for the same measurement under a different name each time - a
# trailer length was `trailer_length_ft` on Livestock, `haul_length_ft` on Equipment and
# `vehicle_length_ft` on Car Hauler. They are one fact, so they are now one slot, `length`.
# The kind grouping survives the rename because it still does real work: it carries the
# combined-size questions (trailer_size, cargo_size, bin_size), which keep their own names,
# into the same family as `length`, so a size the customer already gave answers the length
# question instead of us asking for the same number twice.
_SLOTS_BY_KIND: dict[str, tuple[str, ...]] = {}
for _slot, _kind in _SLOT_VALUE_KIND.items():
    _SLOTS_BY_KIND.setdefault(_kind, ())
    _SLOTS_BY_KIND[_kind] += (_slot,)
# The generic keys search writes its parsed values under are part of the group too.
for _kind in ("length_ft", "width_ft", "height_ft", "payload_lbs"):
    _SLOTS_BY_KIND.setdefault(_kind, ())
    if _kind not in _SLOTS_BY_KIND[_kind]:
        _SLOTS_BY_KIND[_kind] += (_kind,)

# The "what are you hauling?" question went the same way: `haul_material` (Dump),
# `vehicle_type` (Car Hauler/Race), `use_case` (Enclosed), `fiber_use_case` (Fiber) and
# `equipment_list` (Welding) were one fact asked five ways, and are now all `haul_item`.
# The customer only ever tells us once ("random things, wood to pipes to furniture").
# Kept as a group of its own - NOT in _SLOT_VALUE_KIND - because it is free text: giving it
# a parse kind would run it through the numeric normalizer and null it out.
_CARGO_SLOTS: tuple[str, ...] = (
    "haul_item",
)

# Slots that may DONATE a value to a sibling but must never RECEIVE one:
#   trailer_size / cargo_size ask for several numbers at once, so a lone length does not
#     answer them — we would skip a question the customer never got.
#   bin_size is a yardage, which only coincides with a length by a business rule; a trailer
#     length carried in from another category is not a bin size the customer chose.
_NO_AUTOFILL_SLOTS = frozenset({"trailer_size", "cargo_size", "bin_size"})

# One slot can stand in for another only within its own group.
_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = tuple(_SLOTS_BY_KIND.values()) + (_CARGO_SLOTS,)


def slots_of_kind(kind: str) -> tuple[str, ...]:
    """Every slot name that holds a value of this kind, generic keys included."""
    return _SLOTS_BY_KIND.get(kind, ())


def equivalent_slots(slot_name: str) -> tuple[str, ...]:
    """The other slot names that hold the same fact as ``slot_name``."""
    siblings: list[str] = []
    for group in _ALIAS_GROUPS:
        if slot_name in group:
            siblings.extend(name for name in group if name != slot_name)
    return tuple(dict.fromkeys(siblings))


def can_autofill_slot(slot_name: str) -> bool:
    if slot_name in _NO_AUTOFILL_SLOTS:
        return False
    if slot_name in _CARGO_SLOTS:
        return True
    return _SLOT_VALUE_KIND.get(slot_name) in {"length_ft", "width_ft", "height_ft", "payload_lbs"}


def is_recognized_slot_value(slot_name: str, value: Any) -> bool:
    """True when ``value`` is already in the clean, parsed form for this slot's kind
    (a number for a dimension/payload slot, a single-item canonical list for hitch_type,
    a resolved category name for subcategory) rather than raw, unnormalized text. Used to
    stop a vague answer in the SAME slot from clobbering a good value another part of this
    same turn already produced."""
    kind = _SLOT_VALUE_KIND.get(slot_name)
    if kind is None:
        return False
    if kind == "hitch_type":
        return isinstance(value, list) and len(value) == 1 and value[0] in _ALLOWED_HITCH_TYPES
    if kind == "subcategory":
        return isinstance(value, str) and bool(value)
    return _is_number(value)


MIN_AXLE_COUNT = 1
MAX_AXLE_COUNT = 4

_AXLE_COUNT_WORDS = {
    "single": 1, "one": 1, "1": 1,
    "tandem": 2, "dual": 2, "double": 2, "two": 2, "2": 2,
    "tri": 3, "triple": 3, "three": 3, "3": 3,
    "quad": 4, "quadruple": 4, "four": 4, "4": 4,
    # Above the range on purpose: spelled out, these must PARSE so the range check can
    # refuse them. Left out, "seven axles" would return None and be filed as "no
    # preference" - the customer's answer silently discarded instead of questioned.
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_AXLE_COUNT_NUMBER_RE = re.compile(r"(?<![\d.])(\d+)")


def parse_axle_count_answer(value: Any) -> Any:
    """How many axles the customer asked for, as a whole number.

    Words as well as digits: "tandem" and "2" are the same answer, and customers say both.
    A number outside the stocked range is returned AS IS rather than as None - the range
    check is a separate decision (see axle_count_out_of_range), because a 0 must be refused
    and re-asked, while a None would be filed as "asked, no preference" and never raised
    again.
    """
    if _is_number(value):
        return int(round(float(value)))
    text = str(value or "").strip().lower()
    if not text:
        return None
    for word, count in _AXLE_COUNT_WORDS.items():
        if re.search(r"\b{}\b".format(re.escape(word)), text):
            return count
    match = _AXLE_COUNT_NUMBER_RE.search(text.replace(",", ""))
    return int(match.group(1)) if match else None


def axle_count_out_of_range(value: Any) -> bool:
    """True for a stated count we do not stock: zero, negative, or above four."""
    if value is None:
        return False
    count = parse_axle_count_answer(value)
    if count is None:
        return False
    return not (MIN_AXLE_COUNT <= count <= MAX_AXLE_COUNT)


def is_impossible_measurement(category: str, slot_name: str, value: Any) -> bool:
    """True when a measurement answer states a NEGATIVE number.

    Negative only. A zero is handled as no preference (see normalize_answer_for_slot) and a
    vague answer keeps its existing meaning - neither is re-asked. A negative is different in
    kind: the parsers silently DROP the sign, so "-500 lbs" is recorded as 500 lbs, the
    opposite of what the customer said, and the search then filters on it. That is worth one
    short re-ask.

    Only measurement kinds can be negative in a way that matters. Free text, hitch and
    subcategory have no ordering, so nothing about them is out of range.
    """
    kind = _SLOT_VALUE_KIND.get(slot_name)
    if kind not in {
        "length_ft", "width_ft", "height_ft", "payload_lbs",
        "axle_capacity_lbs", "total_axle_capacity_lbs",
    }:
        return False
    # The RAW text, not the parsed value: by the time it is parsed the sign is already gone.
    text = str(value or "").replace(",", "")
    # A minus is a SIGN only when no digit precedes it. Between digits it is the range dash,
    # and ranges are supported everywhere ("5000-7000 lbs" is a legitimate answer, not -7000).
    return bool(re.search(r"(?<![\d.])-\s*\d", text))


# Words that say "I don't know / don't mind" and nothing else. A free-text answer made of
# ONLY these is no answer at all: a live run stored "Hmm, not sure." as the cargo and the
# question could never be skipped. Deliberately a whole-answer test, not a keyword hit -
# "not sure, probably gravel" still stores, because what is left ("probably gravel") says
# something. And a BROAD answer is not a non-answer: "just random stuff" tells us they haul
# a mix (brief S20) and is kept, where "not sure" tells us nothing.
_NON_ANSWER_PHRASES = tuple(sorted((
    "not sure", "not really sure", "i'm not sure", "im not sure", "i am not sure", "unsure",
    "no idea", "i have no idea", "no clue", "not a clue", "dunno", "idk", "i don't know",
    "i dont know", "don't know", "dont know", "not certain", "can't say", "cant say",
    "no preference", "doesn't matter", "doesnt matter", "does not matter", "don't mind",
    "dont mind", "i don't mind", "whatever", "anything", "anything really", "whatever works",
    "haven't decided", "havent decided", "not decided yet", "not yet", "n/a", "na", "none",
    "nothing specific", "nothing in particular",
), key=len, reverse=True))
_FILLER = ("hmm", "hm", "um", "umm", "uh", "er", "well", "honestly", "really", "to be honest",
           "sorry", "yet", "right now", "at the moment", "just", "i guess", "tbh", "oh", "ok", "okay", "lol")


def _whole_words(phrases: tuple[str, ...]) -> re.Pattern[str]:
    ordered = sorted(phrases, key=len, reverse=True)
    return re.compile(r"(?<![\w'])(?:" + "|".join(re.escape(p) for p in ordered) + r")(?![\w'])")


_NON_ANSWER_RE = _whole_words(_NON_ANSWER_PHRASES)
_FILLER_RE = _whole_words(_FILLER)


def is_non_answer(value: Any) -> bool:
    """True when a free-text answer says nothing beyond "not sure" / "no idea" / "whatever"."""
    if not isinstance(value, str):
        return False
    text = re.sub(r"[^\w\s'/]", " ", value.lower().replace("’", "'"))
    text, hits = _NON_ANSWER_RE.subn(" ", text)
    return hits > 0 and not _FILLER_RE.sub(" ", text).strip()


def normalize_answer_for_slot(category: str, slot_name: str, value: Any) -> Any:
    """The value to STORE under ``slot_name`` itself.

    Numeric slots (length/width/height/payload) keep numbers, resolving a range to its
    smallest side and falling back to any plain number in the text as a last resort; a
    truly vague answer with no extractable number is null (no preference), never raw
    text. hitch_type and subcategory (Aluminum's base_category) work the same way: a
    recognized answer is canonicalized, anything else - including "either"/"any" for
    hitch_type, or a vague reply for subcategory - is null. Free-text slots (haul_item and
    similar) have no kind here and are stored completely as-is, however vague.
    """
    kind = _SLOT_VALUE_KIND.get(slot_name)
    if kind is None:
        return None if is_non_answer(value) else value
    if kind == "hitch_type":
        return normalize_hitch_answer(value)
    if kind == "subcategory":
        return normalize_subcategory_answer(value)
    normalized = normalize_slot_value(category, kind, value)
    if not _is_number(normalized):
        return None
    # A stated zero means no preference, exactly like a vague answer: stored as null, never
    # re-asked. Keeping the 0 itself would be worse than useless - length_ft=0 filters the
    # whole catalogue out - and most parsers already collapse "0 lbs" to None anyway, so this
    # only makes the odd one out ("0" for a length, which parses to 0.0) behave the same.
    return None if normalized == 0 else normalized
