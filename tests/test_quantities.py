"""The model reads the amount; Python converts it. Every phrasing here came up in a live run
or is its close sibling - the four the old regex parser got wrong are marked."""
from __future__ import annotations

import pytest

from src.domain import quantities
from src.llm.schemas import Quantity
from src.tools.filters import apply_extracted_fields

from tests.test_filters import Output, make_extracted, state_with


def q(slot, low, unit, high=None, raw="x"):
    return Quantity(slot_name=slot, raw_text=raw, low=low, high=high, unit=unit)


# What the model is expected to hand over for each phrasing, and what must be stored.
CORPUS = [
    # slot, what they said, model's reading (low, high, unit), stored
    ("width", "seven and a half feet", (7.5, None, "ft"), 7.5),                  # regex: 0.5
    ("payload_capacity", "three and a half thousand pounds", (3500, None, "lb"), 3500.0),  # regex: 1000
    ("cargo_size", "144 x 72 inches", (144, None, "in"), 12.0),                  # regex: 144
    ("width", "144 x 72 inches", (72, None, "in"), 6.0),
    ("bin_size", "twenty yard containers", (20, None, "yd"), 20.0),             # regex: nothing
    ("bin_size", "15 cubic yards", (15, None, "cu_yd"), 15.0),
    ("payload_capacity", "a ton and a half", (1.5, None, "ton"), 3000.0),
    ("payload_capacity", "half a ton", (0.5, None, "ton"), 1000.0),
    ("width", "7 and a half feet", (7.5, None, "ft"), 7.5),
    ("length", "144 x 72 (no unit)", (144, None, "in"), 12.0),
    ("length", "about 6 meters", (6, None, "m"), 19.69),
    ("length", "240 inches", (240, None, "in"), 20.0),
    ("width", "2.2 m", (2.2, None, "m"), 7.22),
    ("payload_capacity", "roughly 2,000 kg", (2000, None, "kg"), 4409.25),
    ("payload_capacity", "1.5 tonnes", (1.5, None, "tonne"), 3306.93),
    ("payload_capacity", "around 2.5 tons", (2.5, None, "ton"), 5000.0),
    ("length", "18 to 20 foot", (18, 20, "ft"), 18.0),
    ("length", "20 down to 18 ft", (20, 18, "ft"), 18.0),                       # ends swapped
    ("payload_capacity", "between 4000 and 6000 lbs", (4000, 6000, "lb"), 4000.0),
    ("tank_capacity", "about 1,900 litres", (1900, None, "l"), 501.93),
]


@pytest.mark.parametrize("slot, said, reading, stored", CORPUS, ids=[c[1] for c in CORPUS])
def test_what_they_said_is_stored_correctly(slot, said, reading, stored):
    low, high, unit = reading
    assert quantities.to_canonical(slot, q(slot, low, unit, high)) == pytest.approx(stored, abs=0.01)


@pytest.mark.parametrize("slot, unit", [
    ("length", "lb"),        # a weight for a length
    ("payload_capacity", "ft"),
    ("bin_size", "ft"),      # bins are yardage
    ("tank_capacity", "kg"),
])
def test_a_unit_that_does_not_fit_the_slot_is_refused(slot, unit):
    assert quantities.to_canonical(slot, q(slot, 10, unit)) is None


def test_slots_without_a_quantity_kind_are_not_handled():
    assert not quantities.supports("haul_item")
    assert not quantities.supports("hitch_type")
    assert quantities.supports("cargo_size") and quantities.supports("bin_size")


@pytest.mark.parametrize("slot, value, verdict", [
    ("length", 144.0, "implausible"),           # 144 inches taken as feet
    ("length", 20.0, "ok"),
    ("width", 72.0, "implausible"),
    ("width", 8.5, "ok"),
    ("payload_capacity", 3500.0, "ok"),
    ("payload_capacity", 5.0, "implausible"),
    ("bin_size", 20.0, "ok"),
    ("bin_size", 600.0, "implausible"),
    ("length", 0.0, "ok"),                      # zero is "no preference", judged elsewhere
])
def test_plausibility(slot, value, verdict):
    assert quantities.check(slot, value) == verdict


# ------------------------------------------------------------- through apply_extracted_fields
def test_an_implausible_value_is_not_stored_and_is_asked_about_again():
    state = state_with()
    output = Output(make_extracted(quantities=[q("length", 144, "ft", raw="144 x 72")]))
    result = apply_extracted_fields(state, output)
    assert result.invalid_slot == "length" and result.invalid_reason == "implausible"
    assert "length" not in state["slots"]


def test_a_misfit_unit_falls_back_to_the_raw_text_parse():
    state = state_with()
    output = Output(make_extracted(quantities=[q("length", 20, "lb", raw="20 ft")]))
    apply_extracted_fields(state, output)
    assert state["slots"]["length"] == 20.0


def test_a_whole_size_answer_fills_every_dimension():
    state = state_with("Enclosed")
    output = Output(make_extracted(quantities=[
        q("cargo_size", 144, "in", raw="144 x 72 inches"),
        q("length", 144, "in", raw="144 x 72 inches"),
        q("width", 72, "in", raw="144 x 72 inches"),
    ]))
    apply_extracted_fields(state, output)
    assert state["slots"]["cargo_size"] == 12.0
    assert state["slots"]["length"] == 12.0
    assert state["slots"]["width"] == 6.0


def test_roll_off_bins_keep_their_yardage():
    state = state_with("Roll Off")
    output = Output(make_extracted(quantities=[q("bin_size", 20, "yd", raw="twenty yard containers")]))
    apply_extracted_fields(state, output)
    assert state["slots"]["bin_size"] == 20.0


def test_a_stated_zero_is_still_no_preference():
    state = state_with()
    output = Output(make_extracted(quantities=[q("payload_capacity", 0, "lb", raw="0 lbs")]))
    result = apply_extracted_fields(state, output)
    assert "payload_capacity" not in state["slots"]
    assert "payload_capacity" in result.no_preference


def test_the_implausible_retry_is_explained_to_the_customer_and_the_model():
    from src.graph.nodes.compose import _RETRY_PREFIX
    from src.graph.state import new_state
    from src.llm.prompt import state_block
    from src.tools.category import set_trailer_category

    state = new_state("s1")
    set_trailer_category(state, "Dump")
    state.update(invalid_retry_slot="payload_capacity", invalid_retry_reason="implausible")
    assert "unit was probably misread" in state_block(state)
    assert "implausible" in _RETRY_PREFIX
