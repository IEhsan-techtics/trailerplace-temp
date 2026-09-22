"""A roll-off bin and the trailer under it are one fact, off by one.

A 10 yd bin rides a 9 ft trailer, a 20 yd bin a 19 ft one. So either number tells us the
other: the bin size is what we store, the length is what we search on, and a customer who
gives one is never asked for the other.

It is not a unit conversion. Three feet to the yard makes a 20 yd bin a 60 ft trailer -
longer than anything built - which is what the quantity path was doing while the regex
parser did something else, so the answer depended on which of the two read it.
"""
from __future__ import annotations

from src.domain import quantities as quantity_math
from src.domain.slot_map import (
    bin_yards_to_length_ft,
    length_ft_to_bin_yards,
    normalize_answer_for_slot,
    normalize_slot_targets,
)
from src.graph.nodes.apply import _apply_roll_off_bin


def _q(low: float, unit: str, high: float | None = None, slot: str = "length") -> dict:
    return {"low": low, "high": high, "unit": unit, "slot_name": slot, "raw_text": ""}


def test_the_two_numbers_are_one_apart():
    assert bin_yards_to_length_ft(10) == 9.0
    assert length_ft_to_bin_yards(9) == 10.0
    assert bin_yards_to_length_ft(20) == 19.0


def test_a_bin_is_stored_in_yards_and_searched_in_feet():
    assert normalize_answer_for_slot("Roll Off", "bin_size", "20 yd") == 20.0
    assert normalize_slot_targets("Roll Off", "bin_size", "20 yd") == {"length_ft": 19.0}


def test_a_bare_number_answering_the_bin_question_is_yards():
    assert normalize_answer_for_slot("Roll Off", "bin_size", "20") == 20.0
    assert normalize_slot_targets("Roll Off", "bin_size", "20") == {"length_ft": 19.0}


def test_a_length_answers_the_bin_question_too():
    """"I need a 19 ft one" is a 20 yd bin - the same choice, said the other way."""
    assert normalize_answer_for_slot("Roll Off", "bin_size", "19 ft") == 20.0


def test_a_range_of_bins_takes_the_smaller_one():
    assert normalize_answer_for_slot("Roll Off", "bin_size", "15-20 yd") == 15.0
    assert quantity_math.to_canonical("bin_size", _q(15, "yd", 20, slot="bin_size")) == 15.0


def test_a_vague_answer_is_no_preference():
    assert normalize_answer_for_slot("Roll Off", "bin_size", "as big as you have") is None


def test_the_quantity_path_and_the_regex_path_agree():
    """The whole point: which reader got there first used to decide what we searched for."""
    assert quantity_math.to_canonical("length", _q(20, "yd"), "Roll Off") == 19.0
    assert normalize_answer_for_slot("Roll Off", "length", "20 yd") == 19.0

    assert quantity_math.to_canonical("bin_size", _q(9, "ft", slot="bin_size")) == 10.0
    assert normalize_answer_for_slot("Roll Off", "bin_size", "9 ft") == 10.0


def test_feet_on_a_roll_off_are_still_feet():
    assert quantity_math.to_canonical("length", _q(19, "ft"), "Roll Off") == 19.0


def test_yards_anywhere_else_are_a_measurement():
    """Only Roll Off carries the rule. "20 yards long" about a Flatbed is 60 ft."""
    assert quantity_math.to_canonical("length", _q(20, "yd"), "Flatbed") == 60.0
    assert quantity_math.to_canonical("length", _q(20, "yd"), None) == 60.0


def test_a_length_they_gave_answers_the_only_question_the_category_asks():
    """bin_size is Roll Off's one required question. Asking it of a customer who opened
    with "a 19 ft roll off" is asking them to say the same thing twice."""
    state = {"session_id": "t", "category": "Roll Off", "slots": {"length": 19.0}}

    _apply_roll_off_bin(state)

    assert state["slots"]["bin_size"] == 20.0
    assert state["slot_sources"]["bin_size"] == "user"


def test_a_bin_they_gave_is_left_alone():
    state = {"session_id": "t", "category": "Roll Off", "slots": {"length": 19.0, "bin_size": 30.0}}

    _apply_roll_off_bin(state)

    assert state["slots"]["bin_size"] == 30.0


def test_no_other_category_pairs_the_two():
    state = {"session_id": "t", "category": "Dump", "slots": {"length": 19.0}}

    _apply_roll_off_bin(state)

    assert "bin_size" not in state["slots"]
