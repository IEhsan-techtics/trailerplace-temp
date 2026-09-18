from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

CANONICAL_CATEGORIES = [
    "Aluminum",
    "Car Hauler",
    "Equipment",
    "Enclosed",
    "Utility",
    "Fiber",
    "Race Trailer",
    "Roll Off",
    "Diesel Tank",
    "Flatbed",
    "Dump",
    "Tilt",
    "Livestock",
    # Now stocked. The units were once filed under Enclosed, and a _CATEGORY_ALIASES entry
    # folded them there; the workbook re-tagged them in August 2026, which is exactly the
    # condition that turns this into a stocked category, so the alias was dropped.
    # Whether it is advertised is decided by the catalogue, never by this list - if the last
    # concession trailer sells, stocked_categories() drops it again on its own.
    "Concession",
]

# Synonym terms are split into two salience tiers so that an explicitly *named*
# trailer type always outranks a mere *cargo/brand* word that only implies a type.
# Without this split the resolver returned the first dict-order match, so a cargo
# word ("tractor") could beat a named type ("tilt") in "tilt trailer to haul a
# tractor" and put the customer on the wrong qualification track.
#
# _NAMING_TERMS: the category word itself or an unambiguous style name for it.
# _CARGO_TERMS: cargo items, brands, or attributes that only *suggest* the category.
_NAMING_TERMS: dict[str, list[str]] = {
    "Aluminum": ["aluminum"],
    "Car Hauler": ["car hauler", "toy hauler"],
    "Equipment": ["equipment", "lowboy", "low profile", "deckover", "deck over"],
    "Enclosed": ["enclosed", "box trailer", "v-nose", "v nose", "command trailer"],
    "Utility": ["utility"],
    "Fiber": ["fiber", "splicing trailer", "fiber optic "],
    "Race Trailer": ["race trailer", "enclosed car hauler"],
    "Roll Off": ["roll off", "roll-off", "dumpster","trailer with bins","bin trailer", "3 bins"],
    "Diesel Tank": ["diesel tank", "fuel tank", "tank trailer"],
    "Flatbed": ["flatbed", "hotshot", "step deck", "dovetail", "platform", "flat bed", "dove tail"],
    "Dump": ["dump", "dump trailer"],
    "Tilt": ["tilt", "full tilt", "gravity dampened tilt", "hydraulic dampened tilt"],
    "Livestock": ["livestock", "live stock"],
    # "food trailers" is listed alongside the singular because terms match on a hard word
    # boundary ((?!\w) in _ranked_category_matches), so "food trailer" does NOT match "food
    # trailers" - and the plural is how customers actually ask. "concession" needs no plural:
    # the bare word already matches inside "concession trailers".
    "Concession": ["concession", "concession trailer", "food trailer", "food trailers"],
}

_CARGO_TERMS: dict[str, list[str]] = {
    "Aluminum": ["lightweight", "won't rust", "wont rust", "will not rust"],
    "Car Hauler": ["trailer without sides","car"],
    "Equipment": ["skid steer", "mini ex", "mini excavator", "mini excuvator", "tractor"],
    "Enclosed": ["cargo"],
    "Utility": ["landscape", "lawnmower", "atv", "bike", "motorcycle", "landscaping", "lawn mower", "land scaping"],
    "Dump": ["scissor lift", "hoist", "telescopic", "front lift"],
    "Livestock": ["galyean", "star trailer", "calico trailer", "goats", "hogs", "cattle"],
    "Roll Off": ["roll off", "roll-off", "dumpster","trailer with bins","bin trailer","3 bins"],
}

# Backward-compatible merged view (naming terms first) for callers that only need
# "every term for this category" — e.g. category_prompt_block().
_SYNONYMS: dict[str, list[str]] = {
    category: _NAMING_TERMS.get(category, []) + _CARGO_TERMS.get(category, [])
    for category in CANONICAL_CATEGORIES
    if _NAMING_TERMS.get(category) or _CARGO_TERMS.get(category)
}

_OFFICE_TERMS = ("office trailer", "cooldown trailer", "cool down trailer")
_FIBER_CONTEXT_TERMS = ("fiber", "telecom", "splicing", "fiber optic")
_GENERAL_OFFICE_TERMS = ("general office", "office only", "jobsite office", "site office")


@dataclass(frozen=True)
class CategoryResolution:
    category: Optional[str]
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_key: Optional[str] = None
    # "naming" when the resolved category came from an explicit type word,
    # "cargo" when it came only from a cargo/brand implication, None when
    # unresolved. Consumers (the mind node) use this to decide how much to
    # trust the deterministic hint versus the LLM's own category proposal.
    match_tier: Optional[str] = None


_CATEGORY_CLARIFICATION_RULES: dict[str, dict[str, object]] = {
    "office_trailer_use": {
        "trigger_terms": _OFFICE_TERMS,
        "question": "Will this be for fiber/telecom work specifically, or a more general office trailer?",
        "options": {
            "Fiber": _FIBER_CONTEXT_TERMS,
            "Enclosed": _GENERAL_OFFICE_TERMS,
        },
    },
}


def _matches_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


# Words that may sit directly in front of "equipment" while it still names the TRAILER TYPE
# ("an equipment trailer", "want equipment", "go with equipment"). Anything else in front of
# it makes it the head of a CARGO phrase instead — "lawn equipment", "landscaping equipment",
# "heavy equipment" — which must not count as the customer naming the Equipment category.
_EQUIPMENT_NAMING_PRECEDERS = frozenset({
    "a", "an", "the", "some", "any", "that", "this", "my", "your", "our", "another", "one",
    "in", "for", "to", "with", "on", "of", "want", "wants", "need", "needs", "like", "prefer",
    "choose", "chose", "pick", "picked", "go", "get", "getting", "buy", "buying", "and", "or",
    "maybe", "probably", "actually", "instead",
})


def _equipment_term_is_cargo_usage(low: str, match: "re.Match[str]") -> bool:
    """True when this "equipment" match is cargo ("lawn equipment"), not the trailer type.

    Seen live: "I want one for hauling lawn equipment" resolved as NAMING the Equipment
    category — an explicit type choice the customer never made — because the bare naming
    term "equipment" matched inside the cargo phrase. Followed by "trailer" it is always
    the type; preceded by a modifier word that is not a determiner/verb it is the cargo.
    """
    after = low[match.end():].lstrip()
    if after.startswith("trailer"):
        return False
    before = low[: match.start()].rstrip()
    preceding = re.findall(r"[a-z'\-]+$", before)
    return bool(preceding) and preceding[0] not in _EQUIPMENT_NAMING_PRECEDERS


def _ranked_category_matches(text: str) -> list[tuple[str, str, int, int]]:
    """Return category matches ranked by salience.

    Each entry is (category, tier, sort_key_position, sort_key_neg_len). Results
    are ordered so the best match is first, using: naming terms before cargo
    terms; within a tier, the earliest position in the text; then the longest
    (most specific) matching term. Only the strongest match per category is kept.
    """
    low = text or ""
    best_per_category: dict[str, tuple[int, int, int]] = {}
    for tier_rank, term_map in ((0, _NAMING_TERMS), (1, _CARGO_TERMS)):
        for category, terms in term_map.items():
            for term in terms:
                match = re.search(rf"(?<!\w){re.escape(term.strip())}(?!\w)", low)
                if not match:
                    continue
                if tier_rank == 0 and term == "equipment" and _equipment_term_is_cargo_usage(low, match):
                    continue
                candidate = (tier_rank, match.start(), -len(term.strip()))
                existing = best_per_category.get(category)
                if existing is None or candidate < existing:
                    best_per_category[category] = candidate

    ranked = sorted(best_per_category.items(), key=lambda item: item[1])
    return [
        (category, "naming" if key[0] == 0 else "cargo", key[1], key[2])
        for category, key in ranked
    ]


def _direct_category_from_text(text: str) -> Optional[str]:
    matches = _ranked_category_matches((text or "").lower())
    return matches[0][0] if matches else None


def _direct_category_tier_from_text(text: str) -> Optional[str]:
    matches = _ranked_category_matches((text or "").lower())
    return matches[0][1] if matches else None


def resolve_categories_from_text(text: str) -> list[str]:
    return [category for category, _tier, _pos, _len in _ranked_category_matches((text or "").lower())]


def resolve_category_matches(text: str) -> list[tuple[str, str]]:
    """Ranked (category, tier) pairs for a message; tier is "naming" or "cargo".

    "naming" means the user said the trailer type itself ("tilt trailer"); "cargo"
    means they only named a load/job that implies the type ("haul a tractor").
    Callers need the tier to tell an explicit choice apart from an implied one.
    """
    return [(category, tier) for category, tier, _pos, _len in _ranked_category_matches((text or "").lower())]


def _clarification_rule(key: str) -> dict[str, object] | None:
    return _CATEGORY_CLARIFICATION_RULES.get(str(key or "").strip())


def category_clarification_question(key: str | None) -> Optional[str]:
    rule = _clarification_rule(str(key or "").strip())
    if not rule:
        return None
    return str(rule.get("question") or "").strip() or None


def resolve_category_clarification_answer(text: str, clarification_key: str | None) -> CategoryResolution:
    low = (text or "").lower()
    rule = _clarification_rule(str(clarification_key or "").strip())
    if not rule:
        return CategoryResolution(_direct_category_from_text(low))

    options = dict(rule.get("options") or {})
    for category, terms in options.items():
        if _matches_any_term(low, tuple(terms)):
            return CategoryResolution(str(category), clarification_key=str(clarification_key or "").strip())

    direct_category = _direct_category_from_text(low)
    if direct_category:
        return CategoryResolution(direct_category, clarification_key=str(clarification_key or "").strip())

    return CategoryResolution(
        None,
        True,
        str(rule.get("question") or "").strip() or None,
        clarification_key=str(clarification_key or "").strip() or None,
    )


def _advertised_categories() -> tuple[str, ...]:
    """The categories the prompts may name: canonical AND currently in stock.

    The vocabulary is hand-written above — the type/cargo terms encode judgment
    no database column supplies — but availability comes from the catalogue, so
    we never advertise a category whose search returns nothing. Falls back to
    every canonical category when the catalogue cannot be read.
    """
    from src.domain.brands import stocked_categories

    return stocked_categories()


def unstocked_categories() -> tuple[str, ...]:
    """Canonical categories we recognise but currently hold NO stock in.

    The complement of _advertised_categories(). It exists so the reply can say "we do not
    have those right now" as a FACT it was given, rather than by noticing a category is
    absent from a list - an inference small models make unreliably, and one that fails
    silently in the direction that matters (claiming stock we do not have).
    """
    stocked = set(_advertised_categories())
    return tuple(category for category in CANONICAL_CATEGORIES if category not in stocked)


def unstocked_categories_line() -> str:
    """Comma-joined not-in-stock categories, or "" when we stock everything we recognise."""
    return ", ".join(unstocked_categories())


def category_terms_line(category: str) -> str:
    """Every term that names one category, for quoting back what the customer called it."""
    return ", ".join(_NAMING_TERMS.get(category, ()))


def unstocked_categories_block() -> str:
    """The not-in-stock categories WITH their terms, for the Respond prompt.

    The terms matter as much as the names: a customer asks for a "food trailer", never for a
    "Concession". Without the terms the model has to know they are the same thing.
    """
    missing = unstocked_categories()
    if not missing:
        return "None - we currently hold stock in every category we recognise."
    lines = []
    for category in missing:
        terms = category_terms_line(category)
        lines.append(f"- {category}" + (f" (they may call it: {terms})" if terms else ""))
    return "\n".join(lines)


def advertised_categories_line() -> str:
    """Comma-joined stocked categories for the 'what we carry' catalogue line.

    Single source of truth so the customer-facing catalogue can never drift from
    the categories the system actually supports (previously the KNOWLEDGE block
    advertised 10 types while 13 were qualifiable).
    """
    return ", ".join(_advertised_categories())


def category_prompt_block() -> str:
    """Full category catalogue for the Analyze prompt, with the two term tiers split apart.

    TYPE TERMS name the trailer itself ("tilt trailer"); CARGO/TASK TERMS only *imply*
    the category ("haul a tractor" -> Equipment). Analyze must treat the two very
    differently, so they are labelled separately rather than merged into one list.
    """
    advertised = _advertised_categories()
    lines = [
        "These are the ONLY trailer categories we carry. There are exactly "
        f"{len(advertised)}: {', '.join(advertised)}.",
        "",
        "For each category below:",
        '  TYPE TERMS  = the user NAMED this trailer type ("I want a tilt trailer") -> an EXPLICIT choice.',
        '  CARGO TERMS = the user named a load or job this category is best suited for ("haul a tractor")',
        "                -> an IMPLIED choice only. It suggests the category; it does not name it.",
        "",
        "Gooseneck and Bumper Pull are strictly hitch types, never trailer categories. "
        "Do not infer, recommend, or return either one as a category.",
        "",
    ]
    for category in advertised:
        naming = ", ".join(_NAMING_TERMS.get(category, [])) or "(none)"
        cargo = ", ".join(_CARGO_TERMS.get(category, [])) or "(none)"
        lines.append(f"- {category}")
        lines.append(f"    TYPE TERMS: {naming}")
        lines.append(f"    CARGO TERMS (this category is best suited to haul these): {cargo}")
    for rule in _CATEGORY_CLARIFICATION_RULES.values():
        trigger_terms = ", ".join(rule.get("trigger_terms") or [])
        options = []
        for category, terms in dict(rule.get("options") or {}).items():
            options.append(f"{category} ({', '.join(terms)})")
        lines.append(
            f"- AMBIGUOUS: {trigger_terms} -> ask a clarification question first. Resolve using: {', '.join(options)}."
        )
    return "\n".join(lines)


def category_reference_block() -> str:
    """Customer-facing catalogue for the Respond prompt: what each category is good for.

    Respond needs this to answer "which trailer is best for X?" accurately and to
    explain WHY a suggested switch makes sense - without seeing Analyze's tier logic.
    """
    lines = []
    for category in _advertised_categories():
        cargo = ", ".join(_CARGO_TERMS.get(category, []))
        aliases = ", ".join(_NAMING_TERMS.get(category, []))
        detail = f" (also called: {aliases})" if aliases else ""
        suited = f" - best suited for: {cargo}" if cargo else ""
        lines.append(f"- {category}{detail}{suited}")
    return "\n".join(lines)


# One short use-case line per canonical category, so the assistant can present the menu as
# "here is what each is for" rather than as a bare list of names (brief S5).
#
# Authored text ABOUT the existing categories - it invents no category and no capability.
# category_reference_block() alone cannot carry this: several categories have no _CARGO_TERMS
# at all, so their line would read as a name and nothing else.
CATEGORY_BLURBS: dict[str, str] = {
    "Aluminum": "lightweight and rust-resistant, for anyone towing often or near salt water",
    "Car Hauler": "for moving cars, trucks and other vehicles",
    "Equipment": "heavy-duty decks for skid steers, mini excavators and tractors",
    "Enclosed": "lockable and weatherproof, for tools, cargo and anything that must stay dry",
    "Utility": "open general-purpose hauling - mowers, ATVs, furniture, yard work",
    "Fiber": "purpose-built for fiber and telecom crews and their reels",
    "Race Trailer": "enclosed haulers set up for race cars and track weekends",
    "Roll Off": "swappable bins for waste, demolition and clean-up work",
    "Diesel Tank": "fuel transport and on-site refuelling",
    "Flatbed": "open decks for long, wide or awkward loads",
    "Dump": "hydraulic beds for gravel, dirt, mulch and debris",
    "Tilt": "the deck tilts to load low-clearance equipment without ramps",
    "Livestock": "ventilated and partitioned for cattle, horses and other animals",
    "Concession": "built out for food service and mobile vending",
}


def category_menu_block() -> str:
    """The customer-facing menu: each advertised category with what it is good for.

    Built from _advertised_categories(), so a category we have sold out of never appears -
    the menu can only ever offer trailers the search can actually return.
    """
    lines = []
    for category in _advertised_categories():
        blurb = CATEGORY_BLURBS.get(category)
        lines.append(f"- {category}: {blurb}" if blurb else f"- {category}")
    return "\n".join(lines)


def make_prompt_block() -> str:
    from src.domain.brands import make_prompt_block as _make_prompt_block

    return _make_prompt_block()


def resolve_category_from_text(text: str) -> CategoryResolution:
    low = (text or "").lower()
    for key, rule in _CATEGORY_CLARIFICATION_RULES.items():
        trigger_terms = tuple(rule.get("trigger_terms") or ())
        if _matches_any_term(low, trigger_terms):
            return resolve_category_clarification_answer(low, key)

    return CategoryResolution(
        _direct_category_from_text(low),
        match_tier=_direct_category_tier_from_text(low),
    )
