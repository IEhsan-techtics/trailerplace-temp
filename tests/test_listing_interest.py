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


# ------------------------------------ the tool is withheld, not left to be refused later
def tool_names(state, turn):
    from src.graph.agent import build_tools
    from src.llm.tools import ToolRunner

    return [t.name for t in build_tools(ToolRunner(state, turn))]


def pointing_turn(stock=None, reference=2):
    output = turn_output(intent="listing_interest", listing_reference=reference)
    if stock:
        output.inventory_lookup.is_lookup = True
        output.inventory_lookup.confidence = "high"
        output.inventory_lookup.stock_number = stock
    return output


def test_the_lookup_tool_is_withheld_when_the_trailer_is_on_screen():
    """Live: the reply pass was told the listing was already resolved and called the tool
    anyway, spending an agent pass to be told "already shown". A handler refusal comes too
    late - the round trip is paid for the moment the model emits the call."""
    assert "lookup_inventory" not in tool_names(showing_two_batches(), pointing_turn("81419"))


def test_a_bare_reference_withholds_it_too():
    """"I like the first one" - the index resolved, so there is nothing to fetch."""
    assert "lookup_inventory" not in tool_names(showing_two_batches(), pointing_turn(reference=1))


def test_a_trailer_they_have_never_seen_keeps_the_tool():
    """"I like the 81419, but is the 12345 available?" points at one on screen AND names one
    we have never shown. Withholding the tool there would cost them a real answer."""
    assert "lookup_inventory" in tool_names(showing_two_batches(), pointing_turn("12345"))


def test_an_ordinary_turn_keeps_the_tool():
    assert "lookup_inventory" in tool_names(new_state("s1"), turn_output(intent="qualification_answer"))


def test_the_other_tools_are_never_withheld():
    names = tool_names(showing_two_batches(), pointing_turn("81419"))
    assert "search_inventory" in names and "escalate" in names


# ------------------------------------------- 4. saying yes is a lead, whatever the agent does
#
# Live, from a conversation where every detail was already in front of the reply pass:
#
#     > I like the 6th one
#     "The 2026 Calico Trailers HOGPEN/LIVESTOCK TRAILER 16' - 00564 sounds like the one
#      that fits your livestock-hauling needs. Our sales team can help..."
#
# The model simply did not call escalate, so no email went out on the one turn in the whole
# conversation where the customer said yes. Whether they expressed interest is the model's
# call; whether the dealership hears about it is not.
@pytest.fixture
def mail(monkeypatch):
    """Nothing leaves the machine; record what would have."""
    from src.tools import email_sender

    sent = []
    monkeypatch.setattr(
        email_sender, "send_email", lambda subject, body: sent.append((subject, body)) or True
    )
    return sent


def listing_interest(reference=1, **extra):
    return turn_output(intent="listing_interest", listing_reference=reference, **extra)


def bodies(mail, reason="Listing Interest"):
    return [body for subject, body in mail if reason in subject]


def test_picking_a_trailer_off_the_list_tells_the_team(fake_llm, no_search, mail):
    livestock_customer(fake_llm)
    fake_llm.push(listing_interest())
    run_turn("s1", "I like the first one")

    sent = bodies(mail)
    assert len(sent) == 1, "saying yes to a trailer is a lead, with or without the escalate tool"
    # The listing URL, not the title: one click to the exact trailer, and one "word" rather
    # than eleven.
    assert "[Listing Interest] Customer is interested in https://x/1" in sent[0]


def test_the_same_trailer_twice_is_one_email(fake_llm, no_search, mail):
    livestock_customer(fake_llm)
    fake_llm.push(listing_interest())
    run_turn("s1", "I like the first one")
    fake_llm.push(listing_interest())
    run_turn("s1", "yeah that one")

    assert len(bodies(mail)) == 1, "people say it twice; the team hears it once"


def test_a_different_trailer_is_a_second_email(fake_llm, no_search, mail):
    livestock_customer(fake_llm)
    fake_llm.push(listing_interest(reference=1))
    run_turn("s1", "I like the first one")
    fake_llm.push(listing_interest(reference=2))
    run_turn("s1", "actually the second one")

    assert len(bodies(mail)) == 2


def test_interest_we_cannot_pin_down_is_still_a_lead(fake_llm, no_search, mail):
    livestock_customer(fake_llm)
    fake_llm.push(turn_output(intent="listing_interest", turn_summary="They want one of them."))
    run_turn("s1", "yeah I am")

    assert "[Listing Interest] Interested in a trailer we showed" in bodies(mail)[0]


def test_without_contact_details_the_lead_waits_and_then_goes(fake_llm, no_search, mail):
    fake_llm.push(turn_output(intent="general_question"))
    run_turn("s2", "hi")
    fake_llm.push(turn_output(intent="listing_interest", turn_summary="Wants that trailer."))
    run_turn("s2", "yeah I am")
    assert bodies(mail) == [], "no name, no number - nothing to send it about"

    fake_llm.push(turn_output(intent="contact_info_provided", name="Ibrahim", email="i@x.ai"))
    run_turn("s2", "my name is Ibrahim and email is i@x.ai")

    assert len(bodies(mail)) == 1, "the stashed lead goes out the moment we can reach them"
    assert "Ibrahim" in bodies(mail)[0]


def test_the_escalate_tool_does_not_send_a_second_one():
    """apply records it; the agent is told so. A tool call anyway must not double up."""
    import json

    from src.llm.tools import ToolRunner

    state = {
        "session_id": "s1",
        "contact": {"name": "Dave", "email": "d@x.ai", "phone": None, "declined": False},
        "turn_outcome": {"listing_interest": {"status": "sent", "title": "A trailer", "url": "u"}},
    }
    runner = ToolRunner(state, turn_output())
    runner.call("escalate", json.dumps({"reason": "listing_interest", "summary": "wants it"}))

    assert not state["turn_outcome"].get("outbox_events"), "already recorded by apply"


def test_the_reply_pass_is_told_the_interest_is_already_recorded():
    from src.llm.respond import _state_line

    state = dict(
        new_state("s1"),
        turn_outcome={"listing_interest": {"status": "sent", "title": "A Trailer", "url": "u"}},
    )
    line = _state_line(state, turn_output(intent="listing_interest"))
    assert "ALREADY recorded" in line and "do not call escalate" in line


# ------------------------------- 5. once they have picked one, stop reading out the catalogue
def test_the_catalogue_is_not_read_out_after_their_lead_is_logged(fake_llm, no_search, mail):
    """Live: the turn that finally captured the lead ended with "What type of trailer are you
    looking for? We have Utility, Enclosed, Equipment..." - to a customer who had picked a
    trailer two messages earlier."""
    fake_llm.push(turn_output(intent="general_question"))
    run_turn("s3", "is the trailer 15131 still available?")
    fake_llm.push(turn_output(intent="listing_interest", turn_summary="Wants trailer 15131."))
    run_turn("s3", "yeah I am")

    # The live question was the MODEL's, word for word - so it is pushed as the model's, and
    # held all the same. Which type they want is a question for a customer with nothing on
    # the table.
    fake_llm.push(
        turn_output(
            intent="contact_info_provided", name="Ibrahim", email="i@x.ai",
            next_question_text=(
                "What type of trailer are you looking for? We have Utility, Enclosed, "
                "Equipment, Dump, Flatbed and many more - which one fits what you need?"
            ),
        )
    )
    reply = run_turn("s3", "my name is Ibrahim and email is i@x.ai")["assistant_text"]

    assert "What type of trailer are you looking for" not in reply
    assert "passed your request on to our team" in reply


def test_the_lead_is_named_after_the_trailer_they_picked(fake_llm, no_search):
    """The lead row's item_of_interest is what the team opens it to see. A customer who
    asked about one trailer by stock number and said yes to it was filed under "no details
    yet", because nothing he had said was a category or a slot."""
    from src.conversation_store import describe_interest

    livestock_customer(fake_llm)
    fake_llm.push(listing_interest())
    run_turn("s1", "I like the first one")

    assert describe_interest(state_after()) == "2026 P&amp;C Car Hauler"


def test_two_trailers_from_one_make_are_two_emails(fake_llm, no_search, mail):
    """Listing URLs differ only in the slug on the end. Keyed on the first 60 characters,
    two trailers from the same make normalised to the same string and the team heard about
    one of them."""
    from src.tools.team_notify import _key, build_event

    base = "https://www.trailerplace.com/inventory/2026-galyean-32-cattle-trailer-w-butterfly-gates-"
    events = [
        build_event({}, reason="Listing Interest", description=f"Customer is interested in {base}{n}/")
        for n in ("015087", "015086")
    ]
    assert _key(events[0]) != _key(events[1])


def test_the_agent_raising_it_names_the_trailer_too():
    """The agent raises this itself when the analysis pass did not read the turn as
    listing_interest. Same email, so it says the same thing."""
    import json

    from src.llm.tools import ToolRunner

    state = dict(
        showing_two_batches(),
        session_id="s1",
        contact={"name": "Dave", "email": "d@x.ai", "phone": None, "declined": False},
        turn_outcome={},
    )
    turn = turn_output(intent="team_request_escalation", listing_reference=1)
    runner = ToolRunner(state, turn)
    runner.call("escalate", json.dumps({"reason": "listing_interest", "summary": "wants it"}))

    body = state["turn_outcome"]["outbox_events"][0]["payload"]["body"]
    assert "[Listing Interest] Customer is interested in " in body
