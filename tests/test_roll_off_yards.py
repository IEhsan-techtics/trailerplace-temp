"""A Roll Off given in yards is a bin size, not three feet to the yard.

"I need a 20 yard roll off" names the bin - that is how the trade talks - and we store the
yardage straight into length_ft by a business rule (slot_map). Read as a measurement it
becomes a 60 ft trailer: longer than anything on the lot, so the search returns nothing.

The raw-text parser always knew this. The quantity path - the model reporting "20 yd" as a
number and a unit - did not, so the answer depended on which of the two read it.
"""
from __future__ import annotations

from src.domain import quantities as quantity_math
from src.domain.slot_map import normalize_answer_for_slot


def _q(low: float, unit: str, high: float | None = None) -> dict:
    return {"low": low, "high": high, "unit": unit, "slot_name": "length", "raw_text": ""}


def test_yards_on_a_roll_off_stay_yards():
    assert quantity_math.to_canonical("length", _q(20, "yd"), "Roll Off") == 20.0
    assert quantity_math.to_canonical("length", _q(20, "cu_yd"), "Roll Off") == 20.0


def test_a_range_of_bins_takes_the_smaller_one():
    assert quantity_math.to_canonical("length", _q(15, "yd", 20), "Roll Off") == 15.0


def test_feet_on_a_roll_off_are_still_feet():
    assert quantity_math.to_canonical("length", _q(20, "ft"), "Roll Off") == 20.0


def test_yards_anywhere_else_are_a_measurement():
    """Only Roll Off carries the yardage rule. "20 yards long" about a Flatbed means 60 ft."""
    assert quantity_math.to_canonical("length", _q(20, "yd"), "Flatbed") == 60.0
    assert quantity_math.to_canonical("length", _q(20, "yd"), None) == 60.0


def test_the_bin_size_slot_itself_is_unchanged():
    assert quantity_math.to_canonical("bin_size", {"low": 20, "high": None, "unit": "cu_yd"}) == 20.0


def test_both_readers_now_agree():
    """The regex path and the quantity path land on the same number, which is the point."""
    assert normalize_answer_for_slot("Roll Off", "length", "20 yd") == 20.0
    assert quantity_math.to_canonical("length", _q(20, "yd"), "Roll Off") == 20.0
