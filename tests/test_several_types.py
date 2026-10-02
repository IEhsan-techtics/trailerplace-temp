"""Wanting more than one trailer type before choosing one.

Live, a customer said "I'm towing heavy equipment and cars", was asked "Equipment or Car
Hauler?", and answered "Both". One type was all the session could hold, so nothing was set,
no search ran, and the reply just repeated him back. Every type they point at is now kept,
each is searched and ranked on its own, and the best few of each are shown.
"""
from __future__ import annotations

import pytest

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.nodes import search as search_module
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def say(fake_llm, text, **output):
    fake_llm.push(turn_output(**output))
    return run_turn("s1", text)


def heavy_equipment_and_cars(fake_llm):
    complete_welcome(fake_llm)
    say(fake_llm, "36 ft", intent="feature_request_no_category", extracted={"length": 36.0})
    return say(fake_llm, "I'm towing heavy equipment and cars", intent="recommendation_request",
               category_mentioned="Equipment", more_categories=["Car Hauler"],
               extracted={"haul_item": "heavy equipment and cars"})


# ------------------------------------------------------------------ the conversation level
def test_two_types_are_kept_and_searched_at_once(fake_llm, no_search):
    heavy_equipment_and_cars(fake_llm)

    state = state_after()
    assert state["category"] is None
    assert state["candidate_categories"] == ["Equipment", "Car Hauler"]
    assert len(no_search) == 1, "searched on the information they gave, with no questions first"
    assert no_search[0]["slots"]["length"] == 36.0


def test_both_after_we_offered_two_types_searches_them(fake_llm, no_search):
    """The model reads "Both" against the question it answers and names both types."""
    complete_welcome(fake_llm)
    say(fake_llm, "36 ft, 14k GVWR", intent="feature_request_no_category",
        extracted={"length": 36.0, "total_axle_capacity_lbs": 14000.0, "axle_capacity_basis": "total"})
    say(fake_llm, "Both", intent="qualification_answer",
        category_mentioned="Equipment", more_categories=["Car Hauler"])

    assert state_after()["candidate_categories"] == ["Equipment", "Car Hauler"]
    assert len(no_search) == 1


def test_naming_the_same_types_again_does_not_search_again(fake_llm, no_search):
    heavy_equipment_and_cars(fake_llm)
    say(fake_llm, "yeah equipment and car haulers", intent="qualification_answer",
        category_mentioned="Equipment", more_categories=["Car Hauler"])

    assert len(no_search) == 1


def test_a_new_detail_after_the_trailers_searches_the_types_again(fake_llm, no_search):
    heavy_equipment_and_cars(fake_llm)
    say(fake_llm, "I need 14k GVWR", intent="requirement_change",
        extracted={"total_axle_capacity_lbs": 14000.0, "axle_capacity_basis": "total"})

    assert len(no_search) == 2
    assert no_search[-1]["slots"]["total_axle_capacity_lbs"] == 14000.0


def test_a_plain_message_after_the_trailers_does_not_search_again(fake_llm, no_search):
    heavy_equipment_and_cars(fake_llm)
    say(fake_llm, "ok thanks", intent="smalltalk_other")

    assert len(no_search) == 1


def test_choosing_one_type_afterwards_starts_the_normal_flow(fake_llm, no_search):
    heavy_equipment_and_cars(fake_llm)
    say(fake_llm, "the equipment one", intent="category_selection", category_mentioned="Equipment")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["candidate_categories"] == []


def test_one_type_is_the_normal_flow(fake_llm, no_search):
    complete_welcome(fake_llm)
    say(fake_llm, "I haul gravel", intent="category_selection", category_mentioned="Dump")

    state = state_after()
    assert state["category"] == "Dump"
    assert state["candidate_categories"] == []
    assert no_search == [], "a single type asks its questions first, as before"


def test_a_type_already_chosen_is_not_replaced_by_several(fake_llm, no_search):
    complete_welcome(fake_llm)
    say(fake_llm, "dump trailer", intent="category_selection", category_mentioned="Dump")
    say(fake_llm, "gravel and cars", slots={"haul_item": "gravel and cars"},
        category_mentioned="Dump", more_categories=["Car Hauler"])

    state = state_after()
    assert state["category"] == "Dump"
    assert state["candidate_categories"] == []


def test_types_we_do_not_recognise_do_not_count(fake_llm, no_search):
    complete_welcome(fake_llm)
    say(fake_llm, "a spaceship or a dump", intent="category_selection",
        category_mentioned="Dump", more_categories=["Spaceship"])

    state = state_after()
    assert state["category"] == "Dump"
    assert state["candidate_categories"] == []


# ---------------------------------------------------------------------- the search itself
@pytest.fixture
def catalogue(monkeypatch):
    """search_listings stubbed: three trailers per type, and a record of every call."""
    calls = []

    def fake_search_listings(*, category, max_recommendations, **kwargs):
        calls.append({"category": category, "max": max_recommendations, **kwargs})
        return [
            {"title": f"{category} {i}", "url": f"https://x/{category}/{i}", "category": category}
            for i in range(1, 7)
        ][:max_recommendations]

    monkeypatch.setattr(search_module, "search_listings", fake_search_listings)
    return calls


def _searched(state):
    state.setdefault("turn_outcome", {})
    state["qualification_complete"] = True
    state.setdefault("slots", {"length": 36.0})
    return search_module.search_node(state)["turn_outcome"]


def test_two_types_show_the_top_three_of_each_ranked_separately(catalogue):
    outcome = _searched({"session_id": "s1", "candidate_categories": ["Equipment", "Car Hauler"]})

    assert [call["category"] for call in catalogue] == ["Equipment", "Car Hauler"]
    assert [call["max"] for call in catalogue] == [3, 3]
    assert [listing["title"] for listing in outcome["listings"]] == [
        "Equipment 1", "Equipment 2", "Equipment 3", "Car Hauler 1", "Car Hauler 2", "Car Hauler 3",
    ]
    assert outcome["several_types"] == ["Equipment", "Car Hauler"]


def test_three_types_show_the_top_two_of_each(catalogue):
    outcome = _searched({"session_id": "s1", "candidate_categories": ["Equipment", "Car Hauler", "Flatbed"]})

    assert [call["max"] for call in catalogue] == [2, 2, 2]
    assert len(outcome["listings"]) == 6


def test_more_than_three_types_is_the_every_type_search(catalogue):
    outcome = _searched({
        "session_id": "s1", "candidate_categories": ["Equipment", "Car Hauler", "Flatbed", "Dump"],
    })

    assert [call["category"] for call in catalogue] == [None]
    assert outcome["all_types"] is True
    assert "several_types" not in outcome


def test_each_type_gets_its_own_filters(catalogue):
    _searched({"session_id": "s1", "candidate_categories": ["Equipment", "Car Hauler"],
               "slots": {"length": 36.0, "hitch_type": ["Gooseneck"]}})

    assert all(call["metadata_filters"]["hitch_type"] == "Gooseneck" for call in catalogue)


def test_the_team_email_names_every_type_searched(catalogue):
    outcome = _searched({"session_id": "s1", "candidate_categories": ["Equipment", "Car Hauler"]})

    descriptions = [trigger["description"] for trigger in outcome["system_email_triggers"]]
    assert any("Equipment, Car Hauler" in text for text in descriptions)


def test_a_chosen_type_still_searches_only_that_type(catalogue):
    _searched({"session_id": "s1", "category": "Dump", "candidate_categories": []})

    assert [call["category"] for call in catalogue] == ["Dump"]
    assert catalogue[0]["max"] > 3


def test_the_search_tool_runs_for_several_types():
    from src.llm.tools import ToolRunner

    runner = ToolRunner({"session_id": "s1", "category": None,
                         "candidate_categories": ["Equipment", "Car Hauler"],
                         "contact": {"name": "Dave", "phone": "1", "email": None, "declined": False},
                         "turn_outcome": {}}, turn_output())
    assert runner._search_refusal() is None
