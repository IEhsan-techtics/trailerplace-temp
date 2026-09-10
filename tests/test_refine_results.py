"""After results have been shown: more of them, or different ones.

Three things a customer does once listings are on screen, and they need different handling:
"show me more" (same criteria, next page), "make it 24 ft" (new criteria, search again),
and "actually I need an equipment trailer" (the keep-or-drop question first).
"""
from __future__ import annotations

from src.conversation_store import load_lead, load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def shown_results(fake_llm):
    """A qualified Dump session that has already seen listings."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel", "payload_capacity": "6000 lbs"}))
    run_turn("s1", "gravel, 6000 lbs")


# ------------------------------------------------------------------------ show me more
def test_show_more_searches_again(fake_llm, no_search):
    shown_results(fake_llm)
    before = len(no_search)

    fake_llm.push(turn_output(intent="show_more_results"))
    result = run_turn("s1", "show me more")

    assert len(no_search) == before + 1
    assert result["listings"]


def test_show_more_keeps_the_already_shown_list_so_the_next_page_is_new(fake_llm, no_search):
    shown_results(fake_llm)
    seen = list(state_after()["shown_urls"])
    assert seen

    fake_llm.push(turn_output(intent="show_more_results"))
    run_turn("s1", "show me more")

    assert set(seen).issubset(set(state_after()["shown_urls"])), "history kept for exclusion"


# ------------------------------------------------------------------ changing a requirement
def test_changing_a_measurement_after_results_searches_again(fake_llm, no_search):
    shown_results(fake_llm)
    before = len(no_search)

    fake_llm.push(turn_output(slots={"length": "24 ft"}, intent="requirement_change"))
    result = run_turn("s1", "actually make it 24 ft")

    assert len(no_search) == before + 1
    assert no_search[-1]["slots"]["length"] == 24.0, "the search used the new value"
    assert result["listings"]


def test_a_refinement_clears_the_shown_history(fake_llm, no_search):
    """The old list was built under requirements that no longer apply, so the best match
    for the new ones may well be a trailer they were shown before."""
    shown_results(fake_llm)

    fake_llm.push(turn_output(slots={"length": "24 ft"}, intent="requirement_change"))
    run_turn("s1", "make it 24 ft")

    # Cleared, then repopulated by the fresh search - so it holds only this turn's results.
    assert state_after()["shown_urls"] == ["https://x/1", "https://x/2"]


def test_changing_the_hitch_after_results_searches_again(fake_llm, no_search):
    shown_results(fake_llm)
    before = len(no_search)

    fake_llm.push(turn_output(slots={"hitch_type": "bumper pull"}, intent="requirement_change"))
    run_turn("s1", "bumper pull please")

    assert len(no_search) == before + 1
    assert no_search[-1]["slots"]["hitch_type"] == ["Bumper Pull"]


def test_a_bare_gooseneck_after_results_asks_which_one_before_searching(fake_llm, no_search):
    """It is both a hitch type and a make we stock, so it is a question, not a filter."""
    shown_results(fake_llm)
    before = len(no_search)

    fake_llm.push(turn_output(intent="requirement_change"))
    result = run_turn("s1", "gooseneck please")

    assert len(no_search) == before, "no search while the ambiguity is open"
    assert "gooseneck hitch" in result["assistant_text"]


def test_dropping_a_requirement_after_results_searches_again(fake_llm, no_search):
    shown_results(fake_llm)
    before = len(no_search)

    fake_llm.push(turn_output(intent="drop_requirements", dropped_fields=["payload_capacity"]))
    run_turn("s1", "forget the weight")

    assert len(no_search) == before + 1
    assert "payload_capacity" not in no_search[-1]["slots"]


def test_a_feature_request_after_results_searches_again(fake_llm, no_search):
    """Features rank rather than filter, but they still change what comes back first."""
    shown_results(fake_llm)
    before = len(no_search)

    fake_llm.push(
        turn_output(intent="requirement_change", extracted={"non_metadata_features": ["ramps"]})
    )
    run_turn("s1", "does it come with ramps?")

    assert len(no_search) == before + 1
    assert "ramps" in state_after()["non_metadata_features"]


def test_a_plain_question_after_results_does_not_search_again(fake_llm, no_search):
    """Nothing changed, so there is nothing to look up again."""
    shown_results(fake_llm)
    before = len(no_search)

    fake_llm.push(
        turn_output(intent="general_question", user_question_to_answer="where are you based?",
                    answer_to_customer_question="Wharton, TX.")
    )
    run_turn("s1", "where are you based?")

    assert len(no_search) == before


def test_nothing_re_searches_before_the_first_results(fake_llm, no_search):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(slots={"length": "24 ft"}, intent="requirement_change"))
    run_turn("s1", "make it 24 ft")

    assert no_search == [], "still qualifying - haul_item is unanswered"


# --------------------------------------------------------- a category change is different
def test_a_new_category_asks_the_keep_question_instead_of_searching(fake_llm, no_search):
    shown_results(fake_llm)
    before = len(no_search)

    fake_llm.push(turn_output(category_mentioned="equipment trailer", intent="category_change"))
    result = run_turn("s1", "actually I need an equipment trailer")

    assert len(no_search) == before, "no search while the question is outstanding"
    assert state_after()["pending_keep_filters"] is not None
    assert "keep those" in result["assistant_text"]


def test_a_mapping_term_for_another_category_is_treated_as_a_category_change(
    fake_llm, no_search
):
    """"a skid steer" is a cargo term for Equipment, not a new filter for Dump."""
    shown_results(fake_llm)

    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    result = run_turn("s1", "I'll be hauling a skid steer")

    assert state_after()["pending_category_switch"]["suggested"] == "Equipment"
    assert "switch" in result["assistant_text"].lower()


# ------------------------------------------------------ contact given late in the chat
def test_contact_given_after_results_is_still_captured(fake_llm, no_search):
    """The gate is long closed, but a customer who volunteers their details later must
    still become a lead."""
    complete_welcome(fake_llm, session_id="s2")
    # A session that declined at the gate.
    fake_llm.push(turn_output(intent="contact_declined", declined=True))
    run_turn("s1", "not right now thanks")
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(
        turn_output(intent="contact_info_provided", name="Ibrahim", phone="03304388550")
    )
    run_turn("s1", "actually, I'm Ibrahim - 03304388550")

    state = state_after()
    assert state["contact"]["name"] == "Ibrahim"
    assert state["contact"]["phone"] == "03304388550"

    lead = load_lead("s1")
    assert lead["lead_type"] == "hard"
    assert lead["contact_status"] == "complete"


def test_details_volunteered_mid_qualification_do_not_interrupt_the_flow(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(
        turn_output(slots={"haul_item": "gravel"}, email="ibrahim@example.com")
    )
    run_turn("s1", "gravel, and my email is ibrahim@example.com")

    state = state_after()
    assert state["contact"]["email"] == "ibrahim@example.com"
    assert state["slots"]["haul_item"] == "gravel", "the answer landed too"
