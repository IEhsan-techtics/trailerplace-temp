"""Pointing at a trailer we already showed: "I like the 81419".

Everything here is taken from one live conversation. The customer asked for a 24 ft
livestock trailer, was shown eight, and said four words about one of them. What came back
from the analysis pass was that trailer's whole card, read as if he had specified it:

    extracted: length=32.0 width=5.0 payload=11470.0 hitch=Gooseneck brand=Gooseneck
    axles: per_axle=6000.0   features: adjustable coupler
    quantities: trailer_size=5.0ft ("5' x 32'"), length=32.0ft ("32'"), ...

All of it was stored as source=user, which overwrote his real 24 ft, cleared the shown
history as a "requirement change", reopened qualification and asked him how many axles he
wanted. On top of that the lookup tool ran and printed the same card a second time.
"""
from __future__ import annotations

import pytest

from src.conversation_store import load_session
from src.domain import listing_echo
from src.graph.build import run_turn
from src.graph.state import from_snapshot
from src.llm.schemas import Quantity
from src.tools.questions import all_required_resolved
from src.tools.filters import apply_extracted_fields

from tests.factories import complete_welcome, turn_output
from tests.test_filters import Output, make_extracted, state_with


THE_MESSAGE = "I like the 81419"


def q(slot, raw, low, unit="ft"):
    return Quantity(slot_name=slot, raw_text=raw, low=low, high=None, unit=unit)


def the_card_read_back(**extra):
    """The output the live model actually returned for those four words."""
    return turn_output(
        intent="listing_interest",
        listing_reference=5,
        category_mentioned="Livestock",
        extracted={
            "length": 32.0,
            "width": 5.0,
            "payload_capacity": 11470.0,
            "axle_capacity": 6000.0,
            "axle_capacity_basis": "per_axle",
            "hitch_type": ["Gooseneck"],
            "brand_preference": "Gooseneck",
            "non_metadata_features": ["adjustable coupler"],
            "quantities": [
                q("trailer_size", "5' x 32'", 5.0),
                q("length", "32'", 32.0),
                q("width", "5'", 5.0),
                q("payload_capacity", "11470 lbs", 11470.0, unit="lb"),
                q("axle_capacity", "6000 lbs", 6000.0, unit="lb"),
            ],
        },
        **extra,
    )


# --------------------------------------------------------------- which turns are guarded
def test_a_listing_reference_makes_it_a_pointing_turn():
    assert listing_echo.is_pointing_turn(turn_output(listing_reference=5))


def test_so_does_listing_interest_without_a_number():
    assert listing_echo.is_pointing_turn(turn_output(intent="listing_interest"))


def test_an_ordinary_turn_is_not_guarded():
    assert not listing_echo.is_pointing_turn(turn_output(intent="qualification_answer"))


# -------------------------------------------------------------- 1. the card is not theirs
def test_the_cards_specs_are_not_stored_as_their_requirements():
    state = state_with(category="Livestock", slots={"length": 24.0})
    apply_extracted_fields(state, the_card_read_back(), THE_MESSAGE)

    assert state["slots"]["length"] == 24.0, "their own 24 ft must survive the card's 32 ft"
    for slot in ("width", "payload_capacity", "axle_capacity", "trailer_size", "hitch_type"):
        assert slot not in state["slots"], f"{slot} came off the card, not out of their message"


def test_nothing_is_recorded_as_answered_either():
    state = state_with(category="Livestock")
    result = apply_extracted_fields(state, the_card_read_back(), THE_MESSAGE)
    assert result.stored == {}


def test_a_feature_off_the_card_is_not_a_feature_they_asked_for():
    state = state_with(category="Livestock")
    apply_extracted_fields(state, the_card_read_back(), THE_MESSAGE)
    assert not state.get("non_metadata_features")


def test_what_they_did_type_is_still_stored():
    """The guard drops the echo, not the turn. A requirement in their own words stands."""
    state = state_with(category="Livestock", slots={"length": 24.0})
    output = the_card_read_back()
    output.extracted.quantities.append(q("length", "24 ft", 24.0))
    apply_extracted_fields(state, output, "I like the 81419 but I need 24 ft")
    assert state["slots"]["length"] == 24.0


def test_a_number_they_spelled_out_is_still_stored():
    state = state_with(category="Livestock")
    output = turn_output(
        intent="listing_interest",
        listing_reference=5,
        extracted={"quantities": [q("width", "eight and a half feet", 8.5)]},
    )
    apply_extracted_fields(state, output, "I like that one, eight and a half feet wide")
    assert state["slots"]["width"] == 8.5


def test_an_ordinary_turn_still_stores_everything_the_model_read():
    """The check is literal, so it must never run outside a pointing turn: on an ordinary
    turn the model is the one reading their words, typos and all."""
    state = state_with(category="Dump")
    output = Output(make_extracted(length=20.0, quantities=[q("width", "8 ft", 8.0)]))
    apply_extracted_fields(state, output, "")
    assert state["slots"]["length"] == 20.0
    assert state["slots"]["width"] == 8.0


@pytest.mark.parametrize("text, said", [
    ("11470 lbs", False),
    ("5' x 32'", False),
    ("Gooseneck", False),
    ("adjustable coupler", False),
    ("81419", True),
])
def test_the_discriminator_is_their_own_message(text, said):
    guard = listing_echo.echo_guard(turn_output(listing_reference=5), THE_MESSAGE)
    assert guard("slot", text) is not said


def test_a_digit_run_is_matched_whole_not_as_a_substring():
    """"141" must not count as stated because "81419" happens to contain it."""
    guard = listing_echo.echo_guard(turn_output(listing_reference=5), THE_MESSAGE)
    assert guard("payload_capacity", 141)


# ----------------------------------------------------------------- the brand is a finger
def test_the_make_of_the_trailer_they_like_is_not_a_brand_preference(fake_llm, no_search):
    """"I like the Gooseneck one" names a make, but it is a finger, not a filter."""
    complete_welcome(fake_llm)
    fake_llm.push(the_card_read_back())
    run_turn("s1", THE_MESSAGE)
    assert state_after()["brand_preference"] is None


# --------------------------------------------------------- the whole turn, through the graph
def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def livestock_customer(fake_llm):
    """Where the live conversation was: eight trailers shown, 24 ft on file, qualified."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="livestock", intent="category_selection"))
    run_turn("s1", "Hi, looking for a 24ft trailer for my livestock")
    fake_llm.push(turn_output(slots={"length": "24 ft"}))
    run_turn("s1", "24 ft")


def test_pointing_at_a_trailer_does_not_undo_their_answers(fake_llm, no_search):
    """Their 24 ft used to be overwritten by the 32 ft printed on the card they liked."""
    livestock_customer(fake_llm)
    assert state_after()["slots"]["length"] == 24.0

    fake_llm.push(the_card_read_back())
    run_turn("s1", THE_MESSAGE)

    after = state_after()
    assert after["slots"]["length"] == 24.0
    assert all_required_resolved(after), "every question they answered is still answered"


def test_pointing_at_a_trailer_does_not_run_a_new_search(fake_llm, no_search):
    livestock_customer(fake_llm)
    searches = len(no_search)

    fake_llm.push(the_card_read_back())
    run_turn("s1", THE_MESSAGE)

    assert len(no_search) == searches, "they picked one off the list; nothing changed to search on"


def test_pointing_at_a_trailer_does_not_clear_the_shown_history(fake_llm, no_search):
    livestock_customer(fake_llm)
    shown = list(state_after()["shown_urls"])
    assert shown, "the fixture has to start from a customer who has seen trailers"

    fake_llm.push(the_card_read_back())
    run_turn("s1", THE_MESSAGE)

    assert state_after()["shown_urls"] == shown


def test_pointing_at_a_trailer_does_not_open_an_axle_question(fake_llm, no_search):
    """The card's own 6,000 lb axle rating asked the customer how many axles he wanted."""
    livestock_customer(fake_llm)
    fake_llm.push(the_card_read_back())
    run_turn("s1", THE_MESSAGE)
    assert state_after().get("pending_axle_count") is None
