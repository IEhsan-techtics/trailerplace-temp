"""The deterministic filter layer: raw text beats the model's arithmetic.

No API calls. A test builds the structured output the model would have returned and
asserts on the state that results.
"""
from __future__ import annotations

import logging

import pytest

from src.graph.state import new_state
from src.llm.schemas import ExtractedFields, SlotAnswer
from src.tools.filters import apply_extracted_fields, raw_text_for_slots


def make_extracted(**kwargs):
    """An ExtractedFields with every field defaulted, so a test states only what matters."""
    defaults = dict(
        length=None, width=None, height=None, payload_capacity=None,
        axle_capacity=None, total_axle_capacity_lbs=None, axle_count=None,
        axle_capacity_basis=None, hitch_type=None, haul_item=None,
        brand_preference=None, non_metadata_features=[], numeric_no_preference=[],
        raw_numeric_spans=[],
    )
    defaults.update(kwargs)
    return ExtractedFields(**defaults)


class Output:
    """The two attributes apply_extracted_fields reads off a ChatbotTurnOutput."""

    def __init__(self, extracted=None, slot_answers=None):
        self.extracted = extracted if extracted is not None else make_extracted()
        self.slot_answers = slot_answers or []


def state_with(category="Dump", **kwargs):
    state = new_state("s1")
    state["category"] = category
    state.update(kwargs)
    return state


# ------------------------------------------------------------ the live-probe regression
def test_raw_range_overrides_the_models_midpoint():
    """gpt-5.6-luna really did answer 19.0 for "18-20 ft". Brief S15 says 18."""
    state = state_with()
    output = Output(
        make_extracted(
            length=19.0,
            raw_numeric_spans=[SlotAnswer(slot_name="length", raw_answer="18-20 ft")],
        )
    )
    apply_extracted_fields(state, output)
    assert state["slots"]["length"] == 18.0


def test_divergence_between_python_and_the_model_is_logged(caplog):
    state = state_with()
    output = Output(
        make_extracted(
            length=19.0,
            raw_numeric_spans=[SlotAnswer(slot_name="length", raw_answer="18-20 ft")],
        )
    )
    with caplog.at_level(logging.WARNING, logger="src.tools.filters"):
        apply_extracted_fields(state, output)
    assert any("divergence" in r.message for r in caplog.records)


def test_raw_negative_overrides_a_sign_flipped_model_number():
    """The model hands back +500.0 for "-500 lbs"; the raw text must still trigger a re-ask."""
    state = state_with()
    output = Output(
        make_extracted(
            payload_capacity=500.0,
            raw_numeric_spans=[
                SlotAnswer(slot_name="payload_capacity", raw_answer="-500 lbs")
            ],
        )
    )
    result = apply_extracted_fields(state, output)
    assert result.invalid_slot == "payload_capacity"
    assert result.invalid_reason == "negative"
    assert "payload_capacity" not in state["slots"]


def test_slot_answers_outrank_raw_numeric_spans():
    state = state_with()
    output = Output(
        make_extracted(
            length=30.0,
            raw_numeric_spans=[SlotAnswer(slot_name="length", raw_answer="30 ft")],
        ),
        slot_answers=[SlotAnswer(slot_name="length", raw_answer="18-20 ft")],
    )
    apply_extracted_fields(state, output)
    assert state["slots"]["length"] == 18.0


def test_model_number_is_used_when_no_raw_text_covers_the_slot():
    state = state_with()
    apply_extracted_fields(state, Output(make_extracted(width=8.5)))
    assert state["slots"]["width"] == 8.5


# ---------------------------------------------------------------- multi-field answers (S14)
def test_every_field_in_one_message_is_extracted():
    state = state_with()
    output = Output(
        make_extracted(length=20.0, payload_capacity=6000.0, hitch_type=["Gooseneck"]),
        slot_answers=[
            SlotAnswer(slot_name="length", raw_answer="20 ft"),
            SlotAnswer(slot_name="payload_capacity", raw_answer="3 tons"),
            SlotAnswer(slot_name="hitch_type", raw_answer="gooseneck"),
        ],
    )
    apply_extracted_fields(state, output)
    assert state["slots"]["length"] == 20.0
    assert state["slots"]["payload_capacity"] == 6000.0
    assert state["slots"]["hitch_type"] == ["Gooseneck"]


# ------------------------------------------------- captured before a category exists (S11)
def test_configuration_is_kept_when_no_category_is_selected_yet():
    state = new_state("s1")
    assert state["category"] is None
    output = Output(slot_answers=[SlotAnswer(slot_name="length", raw_answer="24 ft")])
    apply_extracted_fields(state, output)
    assert state["slots"]["length"] == 24.0


# -------------------------------------------------------------- zero and vague (S18, S19)
@pytest.mark.parametrize("raw", ["0", "0 lbs"])
def test_zero_is_recorded_as_no_preference_never_stored(raw):
    state = state_with()
    output = Output(slot_answers=[SlotAnswer(slot_name="payload_capacity", raw_answer=raw)])
    result = apply_extracted_fields(state, output)
    assert "payload_capacity" not in state["slots"]
    assert "payload_capacity" in result.no_preference


def test_vague_answer_is_recorded_as_no_preference():
    state = state_with()
    output = Output(slot_answers=[SlotAnswer(slot_name="length", raw_answer="whatever works")])
    result = apply_extracted_fields(state, output)
    assert "length" not in state["slots"]
    assert "length" in result.no_preference


def test_numeric_no_preference_from_the_model_is_honoured():
    state = state_with()
    result = apply_extracted_fields(state, Output(make_extracted(numeric_no_preference=["width"])))
    assert "width" in result.no_preference


# ---------------------------------------------------------------------- hitch type (S21)
def test_either_hitch_never_stores_both_values():
    state = state_with()
    output = Output(slot_answers=[SlotAnswer(slot_name="hitch_type", raw_answer="either is fine")])
    apply_extracted_fields(state, output)
    assert state["slots"].get("hitch_type") is None


def test_hitch_stated_as_a_feature_is_lifted_into_its_slot():
    state = state_with()
    apply_extracted_fields(state, Output(make_extracted(non_metadata_features=["gooseneck"])))
    assert state["slots"].get("hitch_type") == ["Gooseneck"]


# --------------------------------------------------------------------- axle count (S22)
def test_axle_count_outside_one_to_four_is_rejected_not_stored():
    state = state_with()
    output = Output(slot_answers=[SlotAnswer(slot_name="axle_count", raw_answer="7")])
    result = apply_extracted_fields(state, output)
    assert result.invalid_slot == "axle_count"
    assert result.invalid_reason == "axle_range"
    assert "axle_count" not in state["slots"]


# ---------------------------------------------------------------------- free text (S20)
def test_broad_haul_item_is_kept_verbatim():
    state = state_with()
    output = Output(slot_answers=[SlotAnswer(slot_name="haul_item", raw_answer="just random stuff")])
    apply_extracted_fields(state, output)
    assert state["slots"]["haul_item"] == "just random stuff"


# ------------------------------------------------------------------ invented slot names
def test_a_slot_name_the_model_invents_is_ignored():
    state = state_with()
    output = Output(slot_answers=[SlotAnswer(slot_name="colour_preference", raw_answer="red")])
    apply_extracted_fields(state, output)
    assert "colour_preference" not in state["slots"]


# ------------------------------------------------------------------------- helper itself
def test_raw_text_helper_merges_both_sources_with_slot_answers_winning():
    output = Output(
        make_extracted(raw_numeric_spans=[SlotAnswer(slot_name="length", raw_answer="A"),
                                          SlotAnswer(slot_name="width", raw_answer="B")]),
        slot_answers=[SlotAnswer(slot_name="length", raw_answer="C")],
    )
    assert raw_text_for_slots(output) == {"length": "C", "width": "B"}


def test_free_text_can_be_corrected_by_a_later_message():
    """"actually it's dirt, not gravel" must replace the stored value, not be ignored."""
    state = state_with()
    apply_extracted_fields(
        state, Output(slot_answers=[SlotAnswer(slot_name="haul_item", raw_answer="gravel")])
    )
    assert state["slots"]["haul_item"] == "gravel"

    apply_extracted_fields(
        state, Output(slot_answers=[SlotAnswer(slot_name="haul_item", raw_answer="dirt")])
    )
    assert state["slots"]["haul_item"] == "dirt"


def test_a_vague_numeric_restatement_still_cannot_clobber_a_good_value():
    """The guard the fix above narrowed is still doing its job for parsed slots."""
    state = state_with()
    apply_extracted_fields(
        state, Output(slot_answers=[SlotAnswer(slot_name="length", raw_answer="20 ft")])
    )
    apply_extracted_fields(
        state, Output(slot_answers=[SlotAnswer(slot_name="length", raw_answer="whatever")])
    )
    assert state["slots"]["length"] == 20.0
