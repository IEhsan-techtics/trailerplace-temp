""""An aluminum utility trailer" is one choice, not two.

Live, it cost a whole turn: the model read ``category_mentioned = "Aluminum utility
trailer"``, Python stored Aluminum, and the next reply asked "What type of trailer are you
looking for in aluminum - utility, equipment, enclosed, or something else?" - which they
had just answered. A second session read the SAME sentence as ``category_mentioned =
"Utility"``, so the reading cannot be left to the model: the customer's own words decide.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.graph.nodes.apply import _apply_category
from src.tools import category as category_tool
from src.tools.category import aluminum_base_category


@pytest.fixture(autouse=True)
def stocks_aluminum(monkeypatch):
    """The shared make fixture carries no Aluma, so Aluminum is not a stocked category
    there and every selection would be rejected. On the real lot it is the biggest one."""
    monkeypatch.setattr(
        category_tool.brands, "stocked_categories", lambda: ("Aluminum", "Utility", "Enclosed", "Equipment", "Dump", "Car Hauler", "Tilt"),
    )


def _said(mentioned: str) -> SimpleNamespace:
    return SimpleNamespace(category_mentioned=mentioned, is_category_info_only=False)


@pytest.mark.parametrize("said, base", [
    ("I am looking for an Aluminum utility trailer", "Utility"),
    ("a utility aluminum trailer", "Utility"),           # either order
    ("I want a utility trailer and it should be aluminum", "Utility"),  # two clauses
    ("looking for aluminum, utility type", "Utility"),
    ("do you have aluminum enclosed trailers?", "Enclosed"),
    ("I need an aluminium dump trailer", "Dump"),        # the other spelling
    ("aluminum car hauler", "Car Hauler"),               # a two-word type
    ("I'd like a tilt trailer in aluminum", "Tilt"),
])
def test_the_type_beside_aluminum_is_read_off_the_sentence(said, base):
    assert aluminum_base_category(said) == base


@pytest.mark.parametrize("said", [
    "aluminum trailer",                  # aluminum alone - the type is still to ask
    "I need a utility trailer",          # no aluminum - Utility is the category itself
    "an aluminum trailer for my tractor",  # a load, not a second type
])
def test_nothing_is_paired_when_they_named_one_thing(said):
    assert aluminum_base_category(said) is None


def test_both_land_in_one_turn_whichever_way_the_model_read_it():
    for mentioned in ("Aluminum utility trailer", "Utility", "Aluminum"):
        state: dict = {"session_id": "t"}
        _apply_category(state, _said(mentioned), "I am looking for an Aluminum utility trailer")

        assert state["category"] == "Aluminum", mentioned
        assert state["slots"]["base_category"] == "Utility", mentioned
        assert state["slot_sources"]["base_category"] == "user"


def test_the_question_is_no_longer_outstanding():
    from src.tools.questions import next_unanswered_slot

    state: dict = {"session_id": "t"}
    _apply_category(state, _said("Aluminum utility trailer"), "an aluminum utility trailer")

    assert next_unanswered_slot(state) != "base_category"


def test_a_later_change_of_mind_still_goes_through_the_normal_path():
    """Only a blank base_category is filled here. "Make it an enclosed one" arrives as a
    slot answer and must be allowed to overwrite."""
    state: dict = {"session_id": "t", "category": "Aluminum", "slots": {"base_category": "Utility"}}
    _apply_category(state, _said("Aluminum enclosed trailer"), "an aluminum enclosed trailer")

    assert state["slots"]["base_category"] == "Utility", "not clobbered from here"
