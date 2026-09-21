"""set_trailer_category and the haul-item switch suggestion."""
from __future__ import annotations

import pytest

from src.domain.trailer_fields import get_trailer_fields_as_dict
from src.graph.state import new_state
from src.tools.category import (
    meaningful_filters,
    normalize_category,
    set_trailer_category,
    suggest_category_from_haul_item,
)


# ------------------------------------------------------------------- recognition (S7)
@pytest.mark.parametrize("text", ["Dump", "dump", "DUMP", "Car Hauler", "car hauler"])
def test_official_names_resolve(text):
    assert normalize_category(text) is not None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("dump trailer", "Dump"),
        ("I need something to haul my skid steer", "Equipment"),
        ("a trailer for my car", "Car Hauler"),
        ("enclosed trailer", "Enclosed"),
    ],
)
def test_aliases_and_natural_language_resolve(text, expected):
    assert normalize_category(text) == expected


@pytest.mark.parametrize(
    "text", ["not sure", "I don't know", "any", "anything", "no preference", "you tell me"]
)
def test_no_preference_answers_resolve_to_none(text):
    """Brief S7: these must never be coerced into a category."""
    assert normalize_category(text) is None


@pytest.mark.parametrize("text", ["", "   ", "banana trailer", "spaceship"])
def test_unresolvable_text_resolves_to_none(text):
    assert normalize_category(text) is None


def test_a_category_we_do_not_stock_is_rejected(monkeypatch):
    from src.tools import category as category_tool

    monkeypatch.setattr(category_tool.brands, "stocked_categories", lambda: ("Dump", "Utility"))
    assert normalize_category("Dump") == "Dump"
    assert normalize_category("Concession") is None


# --------------------------------------------------------------- the tool itself (S8)
def test_setting_a_category_loads_that_categorys_required_questions():
    state = new_state("s1")
    report = set_trailer_category(state, "Equipment")

    assert report["ok"] is True
    assert state["category"] == "Equipment"

    spec = get_trailer_fields_as_dict("Equipment")
    assert state["required_slots"] == list(spec["required_slots"])
    assert state["slot_questions"] == dict(spec["questions"])
    # Never invented: every required slot has wording from the spec (brief S13).
    for slot in state["required_slots"]:
        assert slot in state["slot_questions"]


def test_an_unrecognized_category_is_refused_without_raising():
    state = new_state("s1")
    report = set_trailer_category(state, "spaceship")
    assert report["ok"] is False
    assert state["category"] is None


def test_the_tool_writes_only_json_safe_types():
    """It has to survive the state_snapshot round trip."""
    import json

    state = new_state("s1")
    set_trailer_category(state, "Dump")
    json.dumps({k: v for k, v in state.items() if k != "turn_outcome"})


def test_changing_category_resets_the_ask_history_but_not_the_values():
    """A question declined for a Dump has never been put to them about an Equipment."""
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    state["slots"] = {"length": 20.0, "payload_capacity": 6000.0}
    state["asked_counts"] = {"length": 2}
    state["declined_slots"] = ["length"]
    state["pending_slot"] = "length"

    set_trailer_category(state, "Equipment")

    assert state["category"] == "Equipment"
    assert state["slots"]["length"] == 20.0          # the customer's answer survives
    assert state["pending_slot"] is None
    # length is required for Equipment too, so its history is kept; a slot that is not
    # required for the new category loses its history entirely.
    assert "payload_capacity" in state["slots"]


def test_history_for_a_slot_the_new_category_does_not_use_is_dropped():
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    state["asked_counts"] = {"a_slot_no_category_requires": 2}
    state["declined_slots"] = ["a_slot_no_category_requires"]

    set_trailer_category(state, "Utility")

    assert "a_slot_no_category_requires" not in state["asked_counts"]
    assert "a_slot_no_category_requires" not in state["declined_slots"]


# ------------------------------------------------------------- keep-filters shortlist (S12)
def test_only_slots_with_real_values_are_offered_to_keep():
    state = new_state("s1")
    state["slots"] = {"length": 20.0, "width": None, "haul_item": "", "hitch_type": ["Gooseneck"]}
    assert meaningful_filters(state) == {"length": 20.0, "hitch_type": ["Gooseneck"]}


# ------------------------------------------------ haul-item cross-category suggestion
def test_a_haul_item_implying_another_category_produces_a_suggestion():
    state = new_state("s1")
    set_trailer_category(state, "Utility")
    suggestion = suggest_category_from_haul_item(state, "a skid steer")
    assert suggestion is not None
    assert suggestion["suggested"] == "Equipment"


def test_no_suggestion_when_the_haul_item_matches_the_current_category():
    state = new_state("s1")
    set_trailer_category(state, "Equipment")
    assert suggest_category_from_haul_item(state, "a skid steer") is None


def test_no_suggestion_before_a_category_is_chosen():
    """With no category there is nothing to switch away from - it is a selection instead."""
    state = new_state("s1")
    assert suggest_category_from_haul_item(state, "a skid steer") is None


def test_a_rejected_suggestion_is_never_raised_again():
    state = new_state("s1")
    set_trailer_category(state, "Utility")
    suggestion = suggest_category_from_haul_item(state, "a skid steer")
    state["rejected_switches"] = [suggestion["pair"]]
    assert suggest_category_from_haul_item(state, "a skid steer") is None


def test_naming_tier_matches_do_not_trigger_a_suggestion():
    """"I want a dump trailer" is a category selection, not a haul item to second-guess."""
    state = new_state("s1")
    set_trailer_category(state, "Utility")
    assert suggest_category_from_haul_item(state, "dump trailer") is None


# --------------------------------------------- cargo that ALSO suits where they already are
# Live, on a Dump trailer: "gravel and dirt for a landscaping job" was answered with "That
# sounds like a good fit for a Dump trailer. For gravel and dirt for a landscaping job, an
# Utility trailer is usually the better fit - would you like to switch to that instead?"
# Gravel and dirt is dump cargo. We asked what material they were hauling and then argued
# with the answer.
def test_cargo_that_suits_the_current_category_never_offers_a_switch():
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    assert suggest_category_from_haul_item(state, "gravel and dirt") is None


def test_a_second_category_in_the_same_phrase_does_not_win():
    """"landscaping" is a Utility word, but they named dump cargo first and are on a Dump."""
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    assert suggest_category_from_haul_item(state, "gravel and dirt for a landscaping job") is None


@pytest.mark.parametrize(
    "cargo_text", ["gravel", "dirt", "mulch", "debris", "topsoil", "sand", "rubble"]
)
def test_the_material_a_dump_customer_is_asked_for_keeps_them_on_dump(cargo_text):
    """rules/seed.json asks Dump customers "What material will you be hauling (dirt, gravel,
    debris, etc.)?" - every answer it invites must be an answer we accept."""
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    assert suggest_category_from_haul_item(state, cargo_text) is None


def test_cargo_for_a_genuinely_different_category_still_offers_the_switch():
    """The rule narrows when we speak up; it must not silence us altogether."""
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    suggestion = suggest_category_from_haul_item(state, "cattle")
    assert suggestion is not None and suggestion["suggested"] == "Livestock"


def test_a_backhoe_on_a_dump_trailer_still_offers_equipment():
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    suggestion = suggest_category_from_haul_item(state, "a backhoe")
    assert suggestion is not None and suggestion["suggested"] == "Equipment"


def test_a_dirt_bike_is_a_utility_load_not_dump_cargo():
    """"dirt" is a Dump term and "bike" a Utility one; the longer term must win the tie."""
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    suggestion = suggest_category_from_haul_item(state, "a dirt bike")
    assert suggestion is not None and suggestion["suggested"] == "Utility"


def test_cargo_that_settles_nothing_offers_nothing():
    """Pallets ride on a flatbed and on a utility trailer equally happily."""
    state = new_state("s1")
    set_trailer_category(state, "Utility")
    assert suggest_category_from_haul_item(state, "pallets of brick") is None


def test_every_stocked_category_can_hold_its_own_customer():
    """A category with no cargo terms can never be the match that keeps someone where they
    are, so every other category's words pull them away from it."""
    from src.domain import categories

    without_cargo = [
        category
        for category in categories._advertised_categories()
        if not categories._CARGO_TERMS.get(category)
    ]
    assert without_cargo == [], f"these can only ever lose a customer: {without_cargo}"


def test_cargo_terms_agree_with_what_we_tell_the_customer():
    """CATEGORY_BLURBS is the promise; the cargo terms are how we keep it. Dump's blurb said
    "gravel, dirt, mulch and debris" while its terms said "scissor lift, hoist, telescopic"."""
    from src.domain import categories

    blurb = categories.CATEGORY_BLURBS["Dump"].lower()
    for word in ("gravel", "dirt", "mulch", "debris"):
        assert word in blurb
        assert word in categories._CARGO_TERMS["Dump"], f"{word} is promised but not matched"
