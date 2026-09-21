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
from src.graph.state import from_snapshot, new_state
from src.llm.schemas import Quantity
from src.tools.lookup_gate import referenced_listing
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


# ------------------------------------------------- 2. the batch on their screen is kept
GALYEAN = "https://www.trailerplace.com/inventory/2026-galyean-32-cattle-trailer-015087/"
GOOSENECK = "https://www.trailerplace.com/inventory/2026-gooseneck-5-x-32-5-bale-hay-81419/"


def a_batch_of_two():
    return [
        {"title": "2026 Galyean 32' Cattle Trailer - 015087", "url": GALYEAN,
         "stock_number": "15087", "price_display": "$32,250", "make": "Galyean",
         "description": "a paragraph nobody needs two turns later"},
        {"title": "2026 Gooseneck 5' x 32' 5 Bale Hay Trailer - 81419", "url": GOOSENECK,
         "stock_number": "81419", "price_display": "$10,495", "make": "Gooseneck"},
    ]


def test_the_batch_is_recorded_in_the_order_it_was_shown():
    from src.graph.nodes.compose import _record_shown

    state = new_state("s1")
    _record_shown(state, [GOOSENECK, GALYEAN], a_batch_of_two())

    assert [row["stock_number"] for row in state["last_shown_listings"]] == ["81419", "15087"], (
        "the order they were CITED in is the order on the customer's screen"
    )


def test_only_the_identifying_fields_are_kept():
    """A state_snapshot is written every turn; whole listing rows would bloat it."""
    from src.graph.nodes.compose import _KEPT_LISTING_FIELDS, _record_shown

    state = new_state("s1")
    _record_shown(state, [GALYEAN], a_batch_of_two())
    assert set(state["last_shown_listings"][0]) <= set(_KEPT_LISTING_FIELDS)
    assert "description" not in state["last_shown_listings"][0]


def test_a_new_batch_replaces_the_old_one():
    """"The second one" means the second thing in front of them now."""
    from src.graph.nodes.compose import _record_shown

    state = new_state("s1")
    _record_shown(state, [GALYEAN], a_batch_of_two())
    _record_shown(state, [GOOSENECK], a_batch_of_two())

    assert [row["stock_number"] for row in state["last_shown_listings"]] == ["81419"]
    assert len(state["shown_urls"]) == 2, "shown_urls still accumulates; only the batch resets"


# --------------------------------------------------- 3 & 4. the reference resolves itself
def state_showing_the_batch():
    state = new_state("s1")
    state["last_shown_listings"] = [
        {k: v for k, v in row.items() if k != "description"} for row in a_batch_of_two()
    ]
    state["shown_urls"] = [GALYEAN, GOOSENECK]
    return state


def test_the_reference_resolves_to_that_trailer():
    state = state_showing_the_batch()
    listing = referenced_listing(state, turn_output(listing_reference=2))
    assert listing["stock_number"] == "81419"


@pytest.mark.parametrize("ref", [0, 3, None, "fifth"])
def test_a_reference_we_cannot_count_to_resolves_to_nothing(ref):
    """Rather than to a wrong trailer: out of range means it counted into a list we do not
    have, and the turn falls back to an ordinary lookup."""
    assert referenced_listing(state_showing_the_batch(), turn_output(listing_reference=ref)) is None


def test_a_trailer_on_their_screen_is_not_looked_up_again():
    """The live failure: listing_reference #5 AND stock 81419, so the lookup ran anyway."""
    from src.graph.build import _route

    output = the_card_read_back()
    output.listing_reference = 2
    output.inventory_lookup.is_lookup = True
    output.inventory_lookup.confidence = "high"
    output.inventory_lookup.stock_number = "81419"

    assert "inventory_lookup" not in _route(state_showing_the_batch(), output)


def test_a_trailer_that_is_not_on_their_screen_still_is():
    from src.graph.build import _route

    output = turn_output(intent="inventory_lookup")
    output.inventory_lookup.is_lookup = True
    output.inventory_lookup.confidence = "high"
    output.inventory_lookup.stock_number = "99999"

    assert "inventory_lookup" in _route(state_showing_the_batch(), output)


def test_the_reply_pass_is_told_which_trailer_and_not_to_fetch_it():
    from src.llm.respond import build_system_prompt

    prompt = build_system_prompt(state_showing_the_batch(), turn_output(listing_reference=2))
    assert "ALREADY RESOLVED" in prompt
    assert "5 Bale Hay Trailer - 81419" in prompt
    assert "do NOT call a" in prompt


def test_without_a_batch_the_reply_pass_is_told_to_look_it_up():
    from src.llm.respond import build_system_prompt

    output = turn_output(intent="inventory_lookup")
    output.inventory_lookup.is_lookup = True
    output.inventory_lookup.confidence = "high"
    output.inventory_lookup.stock_number = "99999"

    prompt = build_system_prompt(new_state("s1"), output)
    assert "call lookup_inventory" in prompt
    assert "ALREADY RESOLVED" not in prompt


def showing_two_batches():
    """The live shape: six trailers, then "show me more", then two more."""
    state = state_showing_the_batch()
    state["shown_listings"] = [
        {k: v for k, v in row.items() if k != "description"} for row in a_batch_of_two()
    ]
    state["last_shown_listings"] = [
        {"title": "2026 Galyean 32' - 15086", "url": GALYEAN.replace("015087", "15086"),
         "stock_number": "15086", "make": "Galyean"},
    ]
    state["shown_listings"] += state["last_shown_listings"]
    return state


def test_a_stock_number_beats_the_index():
    """Live: the customer said "the 81419" and the model numbered it #4, counting into a
    batch that by then held two trailers. The number says which trailer; the index does not."""
    output = turn_output(intent="listing_interest", listing_reference=4)
    output.inventory_lookup.is_lookup = True
    output.inventory_lookup.confidence = "high"
    output.inventory_lookup.stock_number = "81419"

    assert referenced_listing(showing_two_batches(), output)["stock_number"] == "81419"


def test_an_index_counts_into_the_batch_in_front_of_them():
    """With no identifier there is nothing but the index, and it means the newest batch."""
    listing = referenced_listing(showing_two_batches(), turn_output(listing_reference=1))
    assert listing["stock_number"] == "15086"


def test_a_running_list_is_capped():
    from src.graph.nodes.compose import _MAX_REMEMBERED_LISTINGS, _record_shown

    state = new_state("s1")
    for batch in range(_MAX_REMEMBERED_LISTINGS + 5):
        url = f"https://www.trailerplace.com/inventory/t-{batch}/"
        _record_shown(state, [url], [{"title": f"T{batch}", "url": url, "stock_number": str(batch)}])

    assert len(state["shown_listings"]) == _MAX_REMEMBERED_LISTINGS
    assert state["shown_listings"][-1]["stock_number"] == str(_MAX_REMEMBERED_LISTINGS + 4)


def test_showing_a_trailer_again_does_not_duplicate_it():
    from src.graph.nodes.compose import _record_shown

    state = new_state("s1")
    _record_shown(state, [GALYEAN], a_batch_of_two())
    _record_shown(state, [GALYEAN, GOOSENECK], a_batch_of_two())

    assert [row["stock_number"] for row in state["shown_listings"]] == ["15087", "81419"]
