"""The structural gate on inventory lookups (src/tools/lookup_gate.py).

A lookup is a side query that must never touch category, brand or a collected slot, so a
false positive is expensive: it hijacks the turn and answers a shopper with one arbitrary
trailer. These tests pin the two failures that motivated the gate - a bare make read as a
lookup, and a category word read as a model - plus the stock-number plausibility check.
"""
from __future__ import annotations

import pytest
from factories import complete_welcome, turn_output

from src.llm.schemas import InventoryLookup
from src.tools.lookup_gate import (
    brand_is_lookup_make,
    lookup_requested,
    stock_number_is_plausible,
    usable_stock_number,
)


def _turn(confidence="high", extracted=None, **identifiers):
    """A turn output carrying one inventory_lookup block."""
    output = turn_output()
    output.inventory_lookup = InventoryLookup(
        is_lookup=True,
        year=identifiers.get("year"),
        make=identifiers.get("make"),
        model_text=identifiers.get("model_text"),
        stock_number=identifiers.get("stock_number"),
        wants=None,
        confidence=confidence,
    )
    for field, value in (extracted or {}).items():
        setattr(output.extracted, field, value)
    return output


# --------------------------------------------------------------------------- the gate
@pytest.mark.parametrize(
    "label, turn, fires",
    [
        ("stock number alone", _turn(stock_number="81382"), True),
        ("year + make", _turn(year=2025, make="Diamond C"), True),
        ("make + model code", _turn(make="Iron Bull", model_text="DTB"), True),
        ("model code alone", _turn(model_text="lpx14"), True),
        # The two live regressions this module exists for.
        ("make alone is a brand preference", _turn(make="Diamond C"), False),
        ("category word is not a model", _turn(make="Diamond C", model_text="dump trailer"), False),
        # Layer 2.
        ("low confidence never fires", _turn(confidence="low", stock_number="81382"), False),
    ],
)
def test_lookup_requested(label, turn, fires):
    assert lookup_requested(turn) is fires, label


def test_no_lookup_block_at_all():
    assert lookup_requested(turn_output()) is False
    assert lookup_requested(None) is False


# ------------------------------------------------------- stock-number plausibility
@pytest.mark.parametrize(
    "raw, plausible",
    [
        ("81382", True),
        ("12914", True),
        ("02570", True),      # Excel drops the leading zero; the title keeps it
        ("7000", True),       # bare and ambiguous - only the cross-field check can judge it
        ("7000 lbs", False),  # a unit means a measurement, whatever the model labelled it
        ("20 ft", False),
        ("$9,995", False),
        ("5.2k axles", False),
        ("7", False),         # too short
        ("1234567", False),   # too long - a phone number, most likely
        ("", False),
        (None, False),
    ],
)
def test_stock_number_plausibility(raw, plausible):
    assert stock_number_is_plausible(raw) is plausible


def test_stock_number_that_is_also_this_turns_payload_is_dropped():
    """The model filed one number under two names; only one of them can be true."""
    turn = _turn(stock_number="7000", extracted={"payload_capacity": 7000.0})
    assert usable_stock_number(turn) is None
    assert lookup_requested(turn) is False


def test_bad_stock_number_does_not_cancel_a_lookup_with_other_identifiers():
    """Drop the bad identifier, not the customer's request."""
    turn = _turn(stock_number="7000 lbs", year=2025, make="Big Tex")
    assert usable_stock_number(turn) is None
    assert lookup_requested(turn) is True


# ------------------------------------------------------------------ brand vs lookup make
def test_lookup_make_is_not_a_standing_brand_preference():
    turn = _turn(make="Iron Bull", model_text="DTB")
    assert brand_is_lookup_make(turn, "Iron Bull Trailers") is True


def test_unrelated_brand_survives():
    turn = _turn(stock_number="81382")
    assert brand_is_lookup_make(turn, "Diamond C") is False


# ------------------------------------------------------------------------- end to end
@pytest.fixture
def no_lookup(monkeypatch):
    """Keep the matcher off the live catalogue, recording the identifiers it was handed.

    Patched at ``lookup_inventory`` rather than at the node, so the REAL node runs and its
    own call to ``usable_stock_number`` is part of what these tests exercise.
    """
    from src.graph.nodes import inventory_lookup as lookup_module

    calls = []

    def _fake_lookup_inventory(*, year, make, model_text, stock_number, limit):
        calls.append(
            {"year": year, "make": make, "model_text": model_text, "stock_number": stock_number}
        )
        return {"match_status": "none", "matches": [], "requested_label": "that exact trailer"}

    monkeypatch.setattr(lookup_module, "lookup_inventory", _fake_lookup_inventory)
    return calls


def test_bare_make_does_not_fire_a_lookup(fake_llm, no_lookup):
    """"I want a Diamond C" is shopping. It must reach qualification, not a side query."""
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(_turn(make="Diamond C"))
    run_turn("s1", "I want a Diamond C")

    assert no_lookup == [], "a bare make must not trigger an inventory lookup"


def test_bare_make_is_still_recorded_as_a_brand_preference(fake_llm, no_lookup):
    """The guard drops the LOOKUP, never the customer's stated brand."""
    from src.graph.build import run_turn
    from src.graph.state import from_snapshot
    from src import conversation_store

    complete_welcome(fake_llm)
    output = _turn(make="Diamond C")
    output.extracted.brand_preference = "Diamond C"
    fake_llm.push(output)
    run_turn("s1", "I want a Diamond C")

    snapshot, _conversation, _lead = conversation_store.load_session("s1")
    state = from_snapshot("s1", snapshot)
    assert state.get("brand_preference") == "Diamond C"


def test_make_plus_model_fires_and_suppresses_the_brand(fake_llm, no_lookup):
    """A lookup identifier's make is not a standing filter on every later search."""
    from src.graph.build import run_turn
    from src.graph.state import from_snapshot
    from src import conversation_store

    complete_welcome(fake_llm)
    output = _turn(make="Iron Bull", model_text="DTB")
    output.extracted.brand_preference = "Iron Bull"
    fake_llm.push(output)
    run_turn("s1", "do you have the Iron Bull DTB?")

    assert len(no_lookup) == 1, "make + model code is a real lookup"
    snapshot, _conversation, _lead = conversation_store.load_session("s1")
    state = from_snapshot("s1", snapshot)
    assert state.get("brand_preference") is None


def test_implausible_stock_number_never_reaches_the_matcher(fake_llm, no_lookup):
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(_turn(stock_number="7000 lbs", year=2025, make="Big Tex"))
    run_turn("s1", "a 2025 Big Tex, my load is 7000 lbs")

    assert len(no_lookup) == 1
    assert no_lookup[0]["stock_number"] is None, "the weight must be dropped, not searched on"
