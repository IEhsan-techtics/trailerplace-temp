"""
trailer_fields.py – Tool that returns the required and optional qualification
slots for a given trailer type.  Called by the specialist node once
`trailer_type` is set in the session state.

Slot names map 1-to-1 to the keys used in SessionState["slots_collected"].

One name per concept, shared across every category: a length is `length` whether the
customer is describing a load, a vehicle or the trailer itself, and what they are hauling
is `haul_item` whether the category calls it material, vehicle type or use case. Categories
that ask a genuinely different question (hitch_type, deck_style, bin_size, ...) keep their
own names. See slot_map._SLOT_VALUE_KIND for how each name is parsed and filtered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Slot schema
# ---------------------------------------------------------------------------

@dataclass
class TrailerFieldSpec:
    """Required and optional slots for one trailer category."""
    category: str
    required: list[str]
    optional: list[str]
    # Human-readable question prompts keyed by slot name (used in dynamic prompt)
    questions: dict[str, str] = field(default_factory=dict)
    # Guidance keyed by slot name describing what counts as a valid customer answer.
    answer_guidance: dict[str, str] = field(default_factory=dict)
    # Brief notes injected into the specialist prompt
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-category field definitions
# ---------------------------------------------------------------------------

_SPECS: dict[str, TrailerFieldSpec] = {

    "Equipment": TrailerFieldSpec(
        category="Equipment",
        required=["haul_item", "payload_capacity", "length", "hitch_type"],
        optional=["loading_style"],
        questions={
            "haul_item":       "What equipment will you be hauling (e.g. skid steer, mini excavator, tractor)?",
            "payload_capacity": "What's the rough total weight of the load?",
            "length":  "About how long is the load (or what deck length do you need)?",
            "hitch_type":      "Do you prefer a bumper pull or gooseneck hitch?",
            "loading_style":   "How will you load it — ramps, deckover, or drive-over fenders?",
        },
        answer_guidance={
            "haul_item": "Store the equipment or machinery the customer says they are hauling. Accept short noun phrases or free-form item descriptions.",
            "payload_capacity": "Store the rough total load weight or payload requirement. Accept pounds, lbs, tons, or equivalent weight wording only.",
            "length": "Store the load length or desired deck length. Accept feet, inches, or clear size shorthand when length is being answered.",
            "hitch_type": "Store only bumper pull or gooseneck when the customer explicitly chooses a hitch preference.",
            "loading_style": "Store how the customer wants to load the equipment, such as ramps, deckover, or drive-over fenders.",
        },
        notes="",
    ),

    "Car Hauler": TrailerFieldSpec(
        category="Car Hauler",
        required=["haul_item", "payload_capacity", "length"],
        optional=["open_vs_covered"],
        questions={
            "haul_item":      "What type of vehicle will you be hauling (make/model or class)?",
            "payload_capacity":   "What's the approximate weight of the vehicle?",
            "length": "About how long is the vehicle?",
            "open_vs_covered":   "Are you looking for an open car hauler or a covered/enclosed option?",
        },
        answer_guidance={
            "haul_item": "Store the vehicle make/model, class, or type the customer wants to haul.",
            "payload_capacity": "Store the approximate vehicle weight or payload requirement using weight units only.",
            "length": "Store the vehicle length or required deck length. Accept feet, inches, or clear size shorthand when answering length.",
            "open_vs_covered": "Store whether the customer wants an open car hauler or a covered/enclosed option.",
        },
    ),

    "Utility": TrailerFieldSpec(
        category="Utility",
        required=["haul_item", "payload_capacity"],
        optional=["trailer_size"],
        questions={
            "haul_item":           "What will you be hauling on the utility trailer?",
            "payload_capacity":     "What's the rough total weight of your load?",
            "trailer_size":        "Do you have a size preference (length / width)?",
            #"sides_gate_storage":  "Will you need side rails, a rear gate, or tool storage?",
        },
        answer_guidance={
            "haul_item": "Store the cargo, equipment, or use case the customer says they need the utility trailer for.",
            "payload_capacity": "Store the rough total load weight or payload requirement using weight units only.",
            "trailer_size": "Store the preferred trailer dimensions. Accept length, width, or combined size notation such as AxB or AxBxC when clearly giving size.",
            #"sides_gate_storage": "Store requested utility-trailer features such as side rails, rear gate, or tool storage.",
        },
        notes="Utility-only: lightweight haul handling is decided by the assistant agent (not used for other categories). "
              "Never ask the customer for an axle capacity here - it is not a qualification question for utility "
              "trailers; if they volunteer one it is still stored and used for ranking.",
    ),

    "Dump": TrailerFieldSpec(
        category="Dump",
        required=["haul_item", "payload_capacity"],
        optional=["dump_mechanism"],
        questions={
            "haul_item":   "What material will you be hauling (dirt, gravel, debris, etc.)?",
            "payload_capacity": "What's the rough haul weight per load?",
            "dump_mechanism":  "Do you have a preference for the dump mechanism — scissor lift, telescopic, or standard?",
        },
        answer_guidance={
            "haul_item": "Store the material or debris the customer says they will haul, such as dirt, gravel, rock, mulch, or construction debris.",
            "payload_capacity": "Store the rough haul weight per load or payload requirement using weight units only.",
            "dump_mechanism": "Store the preferred dump mechanism such as scissor lift, telescopic, or standard.",
        },
    ),

    "Tilt": TrailerFieldSpec(
        category="Tilt",
        required=["haul_item", "payload_capacity"],
        optional=["tilt_style"],
        questions={
            "haul_item":     "What will you be hauling on the tilt trailer?",
            "payload_capacity": "What's the approximate weight of the load?",
            "tilt_style":    "Would you prefer a full-tilt deck or one with a stationary front section?",
        },
        answer_guidance={
            "haul_item": "Store what the customer plans to haul on the tilt trailer.",
            "payload_capacity": "Store the approximate load weight or payload requirement using weight units only.",
            "tilt_style": "Store the preferred tilt configuration such as full tilt or stationary front section.",
        },
    ),

    "Enclosed": TrailerFieldSpec(
        category="Enclosed",
        required=["haul_item", "cargo_size"],
        optional=["ac_windows_cabinets", "finished_interior"],
        questions={
            "haul_item":              "What will you be using the enclosed trailer for (cargo hauling, mobile workshop, etc.)?",
            "cargo_size":            "What's the rough size of the cargo you need to fit (length × width × height)?",
            "ac_windows_cabinets":   "Will you need AC, windows, or cabinets inside?",
            "finished_interior":     "Does the interior need to be finished (e.g. lined walls, flooring)?",
        },
        answer_guidance={
            "haul_item": "Store the enclosed-trailer use case such as cargo hauling, workshop, vending, or similar.",
            "cargo_size": "Store the cargo dimensions or required interior fit. Accept length, width, height, inches, or size shorthand such as AxB or AxBxC.",
            "ac_windows_cabinets": "Store whether the customer wants interior amenities like AC, windows, cabinets, or similar.",
            "finished_interior": "Store whether the customer wants a finished interior such as lined walls or finished flooring.",
        },
    ),

    # STOCKED and advertised (see categories.CANONICAL_CATEGORIES; only Diesel Tank is
    # unstocked now). This spec was written while the category had no stock, so it used to
    # say the reply would tell the customer we did not have one - it qualifies and searches
    # like any other category today. Slot NAMES are reused from Enclosed on purpose: they
    # already have normalizers and metadata-filter targets behind them.
    "Concession": TrailerFieldSpec(
        category="Concession",
        # `length`, not cargo_size: a concession trailer is bought by its length - an 8.5x14
        # and an 8.5x22 are the same width and a different business. cargo_size asks for
        # length x width and stored the length anyway, so this asks the question we mean.
        required=["length"],
        optional=["ac_windows_cabinets", "finished_interior"],
        questions={
            "length":   "What length of concession trailer are you looking for?",
            "ac_windows_cabinets": "Will you need a serving window, AC, or cabinets fitted?",
            "finished_interior":   "Does the interior need to be finished out (lined walls, flooring, counters)?",
        },
        answer_guidance={
            "length": "Store the trailer length in feet. Accept a range or a size shorthand such as 8.5x14 and take the LENGTH from it.",
            "ac_windows_cabinets": "Store requested fit-out items such as a serving window, AC, or cabinets.",
            "finished_interior": "Store whether the customer wants a finished interior such as lined walls, flooring, or counters.",
        },
        notes="Concession trailers are an Enclosed body fitted out for vending.",
    ),

    "Livestock": TrailerFieldSpec(
        category="Livestock",
        required=["length"],
        optional=["gate_preferences"],
        questions={
            "length": "What trailer length are you looking for?",
            "gate_preferences":  "Any preference on gate style — butterfly, swing, or slant load?",
        },
        answer_guidance={
            "length": "Store the requested trailer length. Accept feet, inches, or clear length shorthand when length is being answered.",
            "gate_preferences": "Store any gate-style preference the customer explicitly requests.",
        },
        notes="Do NOT ask about animal type or count. Only ask about trailer length.",
    ),

    "Roll Off": TrailerFieldSpec(
        category="Roll Off",
        required=[ "bin_size"], #"package_scope",
        optional=["deck_style", "cdl_concern"],
        questions={
            #"package_scope": "Are you looking for just the trailer, just bins, or the trailer-and-bins package?",
            "bin_size":      "What size bins are you needing (e.g. 10 yd, 20 yd)?",
            "deck_style":    "Do you prefer a step-deck or standard deck?",
            "cdl_concern":   "Is staying under CDL weight thresholds a concern for you?",
        },
        answer_guidance={
            #"package_scope": "Store whether the customer wants only the trailer, only bins, or the full trailer-and-bins package.",
            "bin_size": (
                "Store the requested bin size, including cubic-yard wording such as 10 yd or 20 yd. "
                "For search, map its numeric value directly to trailer length_ft: 15 yd means length_ft='15 ft', "
                "not 45 ft; for a range, use the smallest value."
            ),
            "deck_style": "Store the preferred deck style such as step deck or standard deck.",
            "cdl_concern": "Store whether staying under CDL-related limits matters to the customer.",
        },
    ),

    "Diesel Tank": TrailerFieldSpec(
        category="Diesel Tank",
        required=["fuel_type", "tank_capacity"],
        optional=["deck_style", "cdl_concern"],
        questions={
            "fuel_type":     "Will you be transporting diesel, gasoline, or another fuel?",
            "tank_capacity": "What tank capacity are you looking for (in gallons)?",
            "deck_style":    "Do you prefer a step-deck or standard deck?",
            "cdl_concern":   "Is staying under CDL weight thresholds a concern for you?",
        },
        answer_guidance={
            "fuel_type": "Store the fuel type the customer needs to transport, such as diesel or gasoline.",
            "tank_capacity": "Store the requested tank capacity, typically in gallons.",
            "deck_style": "Store the preferred deck style such as step deck or standard deck.",
            "cdl_concern": "Store whether staying under CDL-related limits matters to the customer.",
        },
    ),

    "Flatbed": TrailerFieldSpec(
        category="Flatbed",
        required=["haul_item", "payload_capacity"],
        optional=["deck_style", "cdl_concern"],
        questions={
            "haul_item":     "What will you be hauling on the flatbed?",
            "payload_capacity": "What's the approximate load weight?",
            "deck_style":    "Do you prefer a step deck or standard deck?",
            "cdl_concern":   "Is staying under CDL weight thresholds a concern for you?",
        },
        answer_guidance={
            "haul_item": "Store what the customer plans to haul on the flatbed.",
            "payload_capacity": "Store the approximate load weight or payload requirement using weight units only.",
            "deck_style": "Store the preferred flatbed deck style such as step deck or standard deck.",
            "cdl_concern": "Store whether staying under CDL-related limits matters to the customer.",
        },
    ),

    "Fiber": TrailerFieldSpec(
        category="Fiber",
        required=["haul_item"],
        optional=["crew_size", "fiber_amenities"],
        questions={
            "haul_item":  "Will this be used for splicing, as an office trailer, or as a cooldown trailer?",
            "crew_size":       "How many crew members need to use it at once?",
            "fiber_amenities": "What amenities do you need — AC, workbench, generator hookup?",
        },
        answer_guidance={
            "haul_item": "Store the primary fiber-trailer use case such as splicing, office, or cooldown.",
            "crew_size": "Store how many crew members need to use the trailer at once.",
            "fiber_amenities": "Store requested amenities such as AC, workbench, generator hookup, or similar.",
        },
    ),

    "Race Trailer": TrailerFieldSpec(
        category="Race Trailer",
        required=["haul_item", "length"],
        optional=["race_amenities"],
        questions={
            "haul_item":      "What type of race vehicle will you be hauling?",
            "length": "What length trailer are you looking for?",
            "race_amenities":    "Will you need cabinets, a workspace, or living quarters?",
        },
        answer_guidance={
            "haul_item": "Store what race vehicle the customer wants to haul.",
            "length": "Store the requested trailer length. Accept feet, inches, or clear length shorthand when length is being answered.",
            "race_amenities": "Store requested race-trailer amenities such as cabinets, workspace, or living quarters.",
        },
    ),

    # NOTE: "Welding" is NOT a canonical category (see categories.CANONICAL_CATEGORIES)
    # and is never advertised or resolved to by the category resolver. This spec is
    # currently unreachable via normal qualification; kept only for the ingest-side
    # normalizer mapping. Promote it into CANONICAL_CATEGORIES before relying on it.
    "Welding": TrailerFieldSpec(
        category="Welding",
        required=["haul_item"],
        optional=["payload_capacity"],
        questions={
            "haul_item": "What welding equipment will you be mounting or carrying (welder, generator, gas bottles, etc.)?",
            "payload_capacity":   "Do you have a rough estimate of the total equipment weight?",
        },
        answer_guidance={
            "haul_item": "Store the welding equipment or components the customer plans to mount or carry.",
            "payload_capacity": "Store the rough total equipment weight using weight units only.",
        },
    ),

    "Aluminum": TrailerFieldSpec(
        category="Aluminum",
        required=["base_category", "payload_capacity"],
        optional=["sleeping_need"],
        questions={
            "base_category": (
                "What type of trailer are you looking for in aluminum — utility, equipment, enclosed, "
                "or something else?"
            ),
            "payload_capacity": "What's the rough total weight of the load?",
            "sleeping_need": "Will you need sleeping accommodations in the trailer?",
        },
        answer_guidance={
            "base_category": "Store the underlying trailer type the customer wants in aluminum, such as utility, equipment, enclosed, or similar.",
            "payload_capacity": "Store the rough total load weight or payload requirement using weight units only.",
            "sleeping_need": "Store whether the customer needs sleeping accommodations.",
        },
        notes="Aluminum is the primary inventory category. Store the underlying trailer type in base_category and map it to the subcategory filter.",
    ),
}

# Fallback for unknown categories
_DEFAULT_SPEC = TrailerFieldSpec(
    category="Unknown",
    required=["haul_item"],
    optional=["payload_capacity", "hitch_type"],
    questions={
        "haul_item":       "What will you be hauling or using this trailer for?",
        "payload_capacity": "What's the rough weight of the load?",
        "hitch_type":      "Do you prefer a bumper pull or gooseneck hitch?",
    },
    answer_guidance={
        "haul_item": "Store what the customer plans to haul or use the trailer for.",
        "payload_capacity": "Store the rough load weight or payload requirement using weight units only.",
        "hitch_type": "Store only bumper pull or gooseneck when the customer explicitly chooses a hitch preference.",
    },
)

# The shared measurement slots, plus the combined-size and one-off numeric questions that
# keep their own names. trailer_size/cargo_size/bin_size ask for SEVERAL numbers at once
# (or a yardage), so they stay separate questions - but their answers still map onto
# length/width/height targets via slot_map._SLOT_METADATA_FILTER_MAP.
_NUMERIC_OR_MEASUREMENT_SLOTS = {
    "length", "width", "height", "payload_capacity", "axle_capacity",
    "trailer_size", "cargo_size", "bin_size",
    "tank_capacity", "crew_size",
}
# Every "what are you hauling / what is it for" question now shares one name, so a customer
# who answers it once is never asked again under another category's wording.
_FREE_TEXT_SLOTS = {
    "haul_item",
}


def _loose_answer_guidance(slot: str) -> str:
    if slot in _NUMERIC_OR_MEASUREMENT_SLOTS:
        return (
            "Loose-answer rule: accept digits, number words, ranges, or approximations; for every range store only "
            "the smallest stated value (15–18 ft becomes 15 ft; 5,000–10,000 lbs becomes 5,000 lbs). "
            "If a cooperative answer has no usable numeric value and is not a counter-question or another-field answer, skip this field as no preference; never invent or retry a value."
        )
    if slot in _FREE_TEXT_SLOTS:
        return (
            "Loose-answer rule: store any substantive wording the user gives for this field, however broad or informal. "
            "Do not store it only when the user explicitly refuses/skips, asks a counter-question, or clearly answers another field."
        )
    return (
        "Loose-answer rule: store a recognizable value for this field. "
        "If the user is cooperative but vague/flexible and gives no usable field value, skip it as no preference; do not retry or invent a value."
    )


# Runtime note:
# The LangGraph specialist may inject the `width` slot as an additional required
# slot when heavy-duty/large-dimension hauling is detected.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_trailer_fields(trailer_type: str) -> TrailerFieldSpec:
    """
    Return the TrailerFieldSpec for the given trailer_type.
    Falls back to a generic spec if the type is not recognised.

    Usage:
        spec = get_trailer_fields("Dump")
        print(spec.required)   # ['haul_item', 'payload_capacity']
        print(spec.optional)   # ['dump_mechanism']
    """
    return _SPECS.get(trailer_type, _DEFAULT_SPEC)


def get_trailer_fields_as_dict(trailer_type: str) -> dict:
    """
    Return a JSON-serialisable dict representation of the field spec.
    Suitable for injecting into LLM prompts or tool results.
    """
    spec = get_trailer_fields(trailer_type)
    answer_guidance = {
        slot: " ".join(
            part for part in (
                str(spec.answer_guidance.get(slot) or "").strip(),
                _loose_answer_guidance(slot),
            )
            if part
        )
        for slot in dict.fromkeys(spec.required + spec.optional)
    }
    return {
        "category": spec.category,
        "required_slots": spec.required,
        "optional_slots": spec.optional,
        "questions": spec.questions,
        "answer_guidance": answer_guidance,
        "notes": spec.notes,
    }


def list_all_categories() -> list[str]:
    """Return all supported trailer categories."""
    return sorted(_SPECS.keys())


# Optional slots whose answer is NOT a descriptive feature: a hitch is a hard search filter and
# base_category is the Aluminum subcategory filter. Both have a home of their own; copied into the
# feature list they would be matched as loose text instead of filtering the search.
_NON_FEATURE_OPTIONAL_SLOTS = {"hitch_type", "base_category"}


def feature_like_optional_slots(trailer_type: str) -> list[str]:
    """The optional slots for this category whose answer doubles as a non-metadata feature.

    An optional answer like "butterfly gates", "scissor lift", "drive-over fenders" or "lined walls"
    describes equipment we hold NO metadata field for, so a slot value alone can never reach the
    search: the only thing that acts on it is the non-metadata feature matcher. These slots are
    therefore mirrored into ``non_metadata_features`` as well as stored under their own name.
    Numeric/measurement slots (trailer_size, crew_size, ...) are excluded - they are sizes, not
    features, and they already have their own filters.
    """
    spec = get_trailer_fields(trailer_type)
    return [
        slot
        for slot in spec.optional
        if slot not in _NUMERIC_OR_MEASUREMENT_SLOTS and slot not in _NON_FEATURE_OPTIONAL_SLOTS
    ]
