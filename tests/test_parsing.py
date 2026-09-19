"""Brief sections 15-22: how a raw answer becomes a stored value.

Everything asserted here is deterministic Python. The model never picks a number.
"""
from __future__ import annotations

import pytest

from src.domain.slot_map import (
    axle_count_out_of_range,
    is_impossible_measurement,
    normalize_answer_for_slot,
    parse_axle_count_answer,
)
from src.domain.units import (
    normalize_number_words,
    parse_length_ft_loose,
    parse_weight_lbs_loose,
)


# ---------------------------------------------------------------- ranges -> smallest (S15)
@pytest.mark.parametrize(
    "text,expected",
    [
        ("18-20 ft", 18.0),
        ("15 to 18 feet", 15.0),
        ("between 20 and 24 ft", 20.0),
        ("20 ft", 20.0),
    ],
)
def test_length_range_takes_the_smallest_side(text, expected):
    assert parse_length_ft_loose(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5000-10000 lbs", 5000.0),
        ("5k-10k", 5000.0),
        ("7000 to 9000 pounds", 7000.0),
    ],
)
def test_weight_range_takes_the_smallest_side(text, expected):
    assert parse_weight_lbs_loose(text) == pytest.approx(expected)


def test_range_is_never_the_midpoint():
    """The live probe had gpt-5.6-luna answer 19.0 for "18-20 ft". Python must not."""
    assert parse_length_ft_loose("around 18-20 ft") == 18.0


# ------------------------------------------------------------------ unit conversion (S17)
@pytest.mark.parametrize(
    "text,expected",
    [("3 tons", 6000.0), ("1.5 tons", 3000.0), ("2 ton", 4000.0), ("5k lbs", 5000.0)],
)
def test_tons_convert_to_pounds(text, expected):
    assert parse_weight_lbs_loose(text) == pytest.approx(expected)


# ----------------------------------------------------------------- approximations (S16)
@pytest.mark.parametrize("text", ["about 20 ft", "roughly 20 feet", "~20ft", "20 ft or so"])
def test_approximations_are_valid_answers(text):
    assert parse_length_ft_loose(text) == pytest.approx(20.0)


# ------------------------------------------------------------------- number words (new)
@pytest.mark.parametrize(
    "text,expected",
    [
        ("twenty feet", 20.0),
        ("twenty four feet", 24.0),
        ("twenty-four ft", 24.0),
        ("sixteen foot", 16.0),
    ],
)
def test_spelled_out_lengths_parse(text, expected):
    assert parse_length_ft_loose(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("two thousand pounds", 2000.0),
        ("two thousand five hundred lbs", 2500.0),
        ("ten thousand lbs", 10000.0),
        ("a couple of tons", 4000.0),
    ],
)
def test_spelled_out_weights_parse(text, expected):
    assert parse_weight_lbs_loose(text) == pytest.approx(expected)


def test_number_word_rewrite_leaves_ordinary_words_alone():
    assert "gooseneck" in normalize_number_words("gooseneck hitch please")
    assert normalize_number_words("bumper pull") == "bumper pull"


# ------------------------------------------------------ vague / zero -> None (S18, S19)
@pytest.mark.parametrize(
    "text", ["whatever works", "not sure", "no preference", "doesn't matter", "any"]
)
def test_vague_numeric_answers_store_none(text):
    assert normalize_answer_for_slot("Dump", "length", text) is None


@pytest.mark.parametrize("text", ["0", "0 lbs", "zero"])
def test_zero_is_never_stored_as_a_filter(text):
    """length_ft=0 would filter the entire catalogue out (S18)."""
    assert normalize_answer_for_slot("Dump", "length", text) is None
    assert normalize_answer_for_slot("Dump", "payload_capacity", text) is None


# ------------------------------------------------------------ negatives re-ask (S22)
@pytest.mark.parametrize("text", ["-500 lbs", "-500", "minus 500 lbs"])
def test_negative_measurement_is_flagged_not_sign_flipped(text):
    if text.startswith("-"):
        assert is_impossible_measurement("Dump", "payload_capacity", text) is True


@pytest.mark.parametrize("text", ["5000-7000 lbs", "10k-12k", "5k - 10k lbs", "8ft-10ft", "3.5t-4t"])
def test_negative_is_not_confused_with_a_range_dash(text):
    """"10k-12k" was once read as -12k: the "k" before the dash is not a digit."""
    assert is_impossible_measurement("Dump", "payload_capacity", text) is False


@pytest.mark.parametrize("text", ["about -500 lbs", "(-500)", "-.5 ton"])
def test_a_sign_after_a_space_or_bracket_is_still_negative(text):
    assert is_impossible_measurement("Dump", "payload_capacity", text) is True


def test_a_k_range_stores_its_smaller_end():
    assert normalize_answer_for_slot("Dump", "payload_capacity", "10k-12k") == 10000.0


# --------------------------------------------------------------- axle count 1-4 (S22)
@pytest.mark.parametrize("text,expected", [("2", 2), ("tandem", 2), ("single", 1), ("tri", 3)])
def test_axle_count_parses(text, expected):
    assert parse_axle_count_answer(text) == expected


@pytest.mark.parametrize("text", ["7", "0", "12 axles"])
def test_axle_count_outside_one_to_four_is_rejected(text):
    assert axle_count_out_of_range(text) is True


@pytest.mark.parametrize("text", ["1", "2", "3", "4"])
def test_axle_count_inside_range_is_accepted(text):
    assert axle_count_out_of_range(text) is False


# ------------------------------------------------------------------ hitch type (S21)
@pytest.mark.parametrize(
    "text,expected",
    [
        ("gooseneck", ["Gooseneck"]),
        ("bumper pull", ["Bumper Pull"]),
        ("goose neck please", ["Gooseneck"]),
    ],
)
def test_hitch_answers_canonicalize_to_a_single_value(text, expected):
    assert normalize_answer_for_slot("Dump", "hitch_type", text) == expected


@pytest.mark.parametrize("text", ["either", "any", "no preference", "doesn't matter"])
def test_hitch_either_stores_none_never_both(text):
    """S21 forbids storing ["Bumper Pull", "Gooseneck"]."""
    assert normalize_answer_for_slot("Dump", "hitch_type", text) is None


# --------------------------------------------------------------------- free text (S20)
@pytest.mark.parametrize(
    "text", ["just random stuff", "a bit of everything", "gravel", "my skid steer"]
)
def test_broad_but_substantive_free_text_is_kept_verbatim(text):
    assert normalize_answer_for_slot("Dump", "haul_item", text) == text


# ------------------------------------------------- stored values are numbers, never words
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("twenty two feet", 22.0),
        ("22ft", 22.0),
        ("18.5'", 18.5),
        ("18.5 ft", 18.5),
        ("about twenty ft", 20.0),
    ],
)
def test_length_is_stored_as_a_number_not_text(raw, expected):
    """Words are an INPUT form only. What lands in session state is always numeric."""
    stored = normalize_answer_for_slot("Dump", "length", raw)
    assert isinstance(stored, (int, float)) and not isinstance(stored, bool)
    assert stored == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw,expected", [("two thousand lbs", 2000.0), ("2000 lbs", 2000.0), ("3 tons", 6000.0)]
)
def test_payload_is_stored_as_a_number_not_text(raw, expected):
    stored = normalize_answer_for_slot("Dump", "payload_capacity", raw)
    assert isinstance(stored, (int, float)) and not isinstance(stored, bool)
    assert stored == pytest.approx(expected)


# ------------------------------------- the catalog parsers must stay digit-only (ingest parity)
def test_catalog_parsers_are_untouched_by_the_number_word_change():
    """parse_length_ft/parse_weight_lbs wrote the numeric columns in trailer_listings.
    Teaching them number words would silently change indexed values, so they stay strict."""
    from src.domain.units import parse_length_ft, parse_weight_lbs

    assert parse_length_ft("twenty feet") is None
    assert parse_weight_lbs("two thousand pounds") is None
    assert parse_length_ft("24 ft 0 in") == pytest.approx(24.0)
    assert parse_weight_lbs("9990 lbs") == pytest.approx(9990.0)
