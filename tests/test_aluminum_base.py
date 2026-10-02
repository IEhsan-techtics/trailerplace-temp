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


# ------------------------------------------------- aluminum is never "several types"
@pytest.mark.parametrize("mentioned, more", [
    ("Aluminum", ["Utility"]),
    ("Utility", ["Aluminum"]),
])
def test_aluminum_with_another_type_is_one_choice_not_two(fake_llm, mentioned, more):
    from src.conversation_store import load_session
    from src.graph.build import run_turn
    from src.graph.state import from_snapshot
    from tests.factories import complete_welcome, turn_output

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(intent="category_selection", category_mentioned=mentioned, more_categories=more))
    run_turn("s1", "I want an aluminum utility trailer")

    state = from_snapshot("s1", load_session("s1")[0])
    assert state["category"] == "Aluminum"
    assert state["slots"]["base_category"] == "Utility"
    assert state["candidate_categories"] == []


# ------------------------------------------- no aluminum of that type: search the type
@pytest.fixture
def catalogue(monkeypatch):
    """No aluminum utility trailers; plenty of steel utility ones. Records every call."""
    from src.graph.nodes import search as search_module

    calls = []

    def fake_search_listings(*, category, metadata_filters, max_recommendations, **kwargs):
        calls.append({"category": category, "filters": dict(metadata_filters or {}), **kwargs})
        if category == "Aluminum":
            return []
        return [{"title": f"{category} {i}", "url": f"https://x/{i}", "category": category} for i in range(1, 4)]

    monkeypatch.setattr(search_module, "search_listings", fake_search_listings)
    return calls


def _aluminum_utility_search(**slots):
    from src.graph.nodes import search as search_module

    state = {"session_id": "s1", "category": "Aluminum", "qualification_complete": True,
             "slots": {"base_category": "Utility", **slots}, "slot_sources": {"base_category": "user"}}
    search_module.search_node(state)
    return state


def test_no_aluminum_of_that_type_searches_the_type_itself(catalogue):
    state = _aluminum_utility_search(length=16.0)

    assert [call["category"] for call in catalogue][-1] == "Utility"
    assert state["category"] == "Utility"
    assert "base_category" not in state["slots"]
    assert state["slots"]["length"] == 16.0, "everything else they told us is kept"
    assert state["turn_outcome"]["aluminum_dropped"] == "Utility"
    assert [l["category"] for l in state["turn_outcome"]["listings"]] == ["Utility"] * 3


def test_aluminum_of_every_kind_is_never_shown_in_its_place(catalogue):
    """The usual last resort keeps the category and drops the rest - for Aluminum that is
    aluminum trailers of any kind, which is not what they asked for."""
    _aluminum_utility_search(length=16.0)

    assert not any(call["category"] == "Aluminum" and call.get("category_only_filters") for call in catalogue)


def test_aluminum_of_that_type_in_stock_stays_aluminum(monkeypatch):
    from src.graph.nodes import search as search_module

    monkeypatch.setattr(search_module, "search_listings", lambda **kw: [{"title": "Alu", "url": "https://x/a"}])
    state = _aluminum_utility_search()

    assert state["category"] == "Aluminum" and state["slots"]["base_category"] == "Utility"
    assert "aluminum_dropped" not in state["turn_outcome"]


def test_the_reply_is_told_these_are_not_aluminum(catalogue):
    from src.llm.respond import _state_line
    from tests.factories import turn_output

    state = _aluminum_utility_search()
    text = _state_line(state, turn_output())
    assert "NO aluminum Utility" in text


def test_the_backstop_says_so_too(catalogue):
    from src.graph.nodes.compose import _render_listings

    state = _aluminum_utility_search()
    assert _render_listings(state, state["turn_outcome"]).startswith("We don't have an aluminum utility trailer")


@pytest.mark.parametrize("raw, base", [
    ("aluminum utility trailer", "Utility"),     # live: Aluminum, the first match, won
    ("utility", "Utility"),
    ("alluminum dump", "Dump"),
    ("aluminum", None),                          # the material alone answers nothing
])
def test_the_aluminum_sub_type_is_never_aluminum(raw, base):
    from src.domain.slot_map import normalize_subcategory_answer

    assert normalize_subcategory_answer(raw) == base
