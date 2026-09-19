"""Axle capacity: per-axle or total, how many axles, and what an axle rating is not.

Whole turns through run_turn() with a scripted model. The model's output is set to what a
real one returns, including its known mistakes: a confident "total" for a bare "axle
capacity", a halved re-read on the follow-up, a bare "Tandem." left unconverted.
"""
from __future__ import annotations

from src.conversation_store import load_session
from src.domain import axles
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def dump_trailer(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "I need a dump trailer")


def say(fake_llm, text, **output):
    fake_llm.push(turn_output(**output))
    return run_turn("s1", text)


# ------------------------------------------------------------------- the wording, unit-level
def test_bare_axle_capacity_is_unclear_whatever_the_model_says():
    assert axles.infer_basis("I need 14,000 lbs of axle capacity", "total") == "unclear"
    assert axles.infer_basis("I need 14,000 lbs of axle capacity", "per_axle") == "unclear"


def test_wording_that_says_which_is_left_to_the_model():
    assert axles.infer_basis("7,000 lb axles", "per_axle") == "per_axle"
    assert axles.infer_basis("14,000 lbs of axle capacity in total", "total") == "total"
    assert axles.infer_basis("14k combined across the axles", "total") == "total"


def test_the_reply_to_the_basis_question_is_read_from_their_words():
    assert axles.basis_from_reply("per axle", None) == "per_axle"
    assert axles.basis_from_reply("each one", None) == "per_axle"
    assert axles.basis_from_reply("that's the total", None) == "total"
    assert axles.basis_from_reply("combined", None) == "total"
    assert axles.basis_from_reply("hmm, not sure", None) is None


# ------------------------------------------------------------- 1. per axle or total?
def test_an_unclear_capacity_is_held_and_asked_about_not_stored(fake_llm):
    dump_trailer(fake_llm)
    result = say(fake_llm, "I need 14,000 lbs of axle capacity",
                 extracted={"axle_capacity": 14000.0, "axle_capacity_basis": "total"})

    state = state_after()
    assert state["pending_axle_basis"]["value"] == 14000.0
    assert "axle_capacity" not in state["slots"]
    assert "total_axle_capacity_lbs" not in state["slots"]
    assert axles.BASIS_QUESTION in result["assistant_text"]


def test_per_axle_files_the_number_they_said_not_the_models_re_read(fake_llm):
    dump_trailer(fake_llm)
    say(fake_llm, "I need 14,000 lbs of axle capacity", extracted={"axle_capacity": 14000.0})
    say(fake_llm, "per axle", extracted={"axle_capacity": 7000.0, "axle_capacity_basis": "per_axle"})

    state = state_after()
    assert state["slots"]["axle_capacity"] == 14000.0
    assert "total_axle_capacity_lbs" not in state["slots"]
    assert state["pending_axle_basis"] is None


def test_total_files_it_as_the_total(fake_llm):
    dump_trailer(fake_llm)
    say(fake_llm, "I need 14,000 lbs of axle capacity", extracted={"axle_capacity": 14000.0})
    say(fake_llm, "that's the total")

    state = state_after()
    assert state["slots"]["total_axle_capacity_lbs"] == 14000.0
    assert "axle_capacity" not in state["slots"]


def test_two_unclear_replies_drop_the_number_and_it_is_never_asked_again(fake_llm):
    dump_trailer(fake_llm)
    say(fake_llm, "I need 14,000 lbs of axle capacity", extracted={"axle_capacity": 14000.0})
    second = say(fake_llm, "hmm")
    assert axles.BASIS_QUESTION in second["assistant_text"], "asked a second time"
    third = say(fake_llm, "not sure really")

    state = state_after()
    assert state["pending_axle_basis"] is None
    assert "axle_capacity" not in state["slots"] and "total_axle_capacity_lbs" not in state["slots"]
    assert axles.BASIS_QUESTION not in third["assistant_text"], "never a third time"


# ----------------------------------------------------------------- 2. how many axles?
def test_a_per_axle_rating_is_stored_at_once_then_the_count_is_asked(fake_llm):
    dump_trailer(fake_llm)
    result = say(fake_llm, "I want 7,000 lb axles",
                 extracted={"axle_capacity": 7000.0, "axle_capacity_basis": "per_axle"})

    state = state_after()
    assert state["slots"]["axle_capacity"] == 7000.0
    assert state["pending_axle_basis"] is None
    assert axles.COUNT_QUESTION in result["assistant_text"]


def test_a_count_given_with_the_rating_is_never_asked(fake_llm):
    dump_trailer(fake_llm)
    result = say(fake_llm, "2-7,000# axles",
                 extracted={"axle_capacity": 7000.0, "axle_count": 2, "axle_capacity_basis": "per_axle"})

    assert state_after()["slots"]["axle_count"] == 2
    assert axles.COUNT_QUESTION not in result["assistant_text"]


def _count_question_open(fake_llm):
    dump_trailer(fake_llm)
    say(fake_llm, "I want 7,000 lb axles",
        extracted={"axle_capacity": 7000.0, "axle_capacity_basis": "per_axle"})


def test_a_bare_tandem_is_read_as_two_even_when_the_model_misses_it(fake_llm):
    _count_question_open(fake_llm)
    say(fake_llm, "Tandem.")
    state = state_after()
    assert state["slots"]["axle_count"] == 2
    assert state["pending_axle_count"] is None


def test_quadruple_is_four(fake_llm):
    _count_question_open(fake_llm)
    say(fake_llm, "quadruple")
    assert state_after()["slots"]["axle_count"] == 4


def test_a_count_we_do_not_stock_is_explained_and_asked_again(fake_llm):
    _count_question_open(fake_llm)
    result = say(fake_llm, "5", extracted={"axle_count": 5})

    state = state_after()
    assert "axle_count" not in state["slots"]
    assert "one to four axles" in result["assistant_text"]
    assert "how many axles" in result["assistant_text"].lower()


def test_nonsense_is_questioned_once_then_taken_as_no_preference(fake_llm):
    _count_question_open(fake_llm)
    second = say(fake_llm, "banana")
    assert "didn't catch that" in second["assistant_text"]
    third = say(fake_llm, "banana")

    state = state_after()
    assert "axle_count" in state["declined_slots"]
    assert state["pending_axle_count"] is None
    assert "how many axles" not in third["assistant_text"].lower()


def test_not_sure_is_no_preference_and_never_asked_again(fake_llm):
    _count_question_open(fake_llm)
    result = say(fake_llm, "not sure, whatever works")

    state = state_after()
    assert "axle_count" in state["declined_slots"]
    assert "how many axles" not in result["assistant_text"].lower()


# ------------------------------------------------------ 3. the load-weight question
def test_an_axle_rating_skips_the_load_weight_question(fake_llm):
    dump_trailer(fake_llm)
    say(fake_llm, "gravel, and I want 7,000 lb axles",
        slots={"haul_item": "gravel"},
        extracted={"axle_capacity": 7000.0, "axle_capacity_basis": "per_axle"})

    state = state_after()
    assert "payload_capacity" in state["rule_skipped"]
    assert "payload_capacity" not in state["required_slots"]


def test_a_total_rating_skips_it_too(fake_llm):
    dump_trailer(fake_llm)
    say(fake_llm, "gravel, 14k combined across the axles",
        slots={"haul_item": "gravel"},
        extracted={"total_axle_capacity_lbs": 14000.0, "axle_capacity_basis": "total"})

    assert "payload_capacity" in state_after()["rule_skipped"]


def test_an_axle_rating_is_never_the_load_weight(fake_llm):
    """New Prompt, live: "5k axles" came back as the load weight and invented a 5,000 lb load."""
    dump_trailer(fake_llm)
    say(fake_llm, "gravel", slots={"haul_item": "gravel"})
    say(fake_llm, "5k axles", slots={"payload_capacity": "5k axles"})

    assert "payload_capacity" not in state_after()["slots"]


# ------------------------------------------------------------- one question at a time
def test_no_search_while_an_axle_question_is_open(fake_llm, no_search):
    dump_trailer(fake_llm)
    say(fake_llm, "gravel, and I want 7,000 lb axles",
        slots={"haul_item": "gravel"},
        extracted={"axle_capacity": 7000.0, "axle_capacity_basis": "per_axle"})
    assert not no_search, "every required question is done, but the count question is open"

    say(fake_llm, "tandem")
    assert len(no_search) == 1, "answered - now it searches"


# -------------------------------------------------------------------- axle type is a feature
def test_the_axle_type_is_a_feature_and_the_count_is_not(fake_llm):
    dump_trailer(fake_llm)
    say(fake_llm, "tandem 5200 lb torsion axles",
        extracted={"axle_capacity": 5200.0, "axle_count": 2, "axle_capacity_basis": "per_axle",
                   "non_metadata_features": ["torsion axles", "tandem axles"]})

    state = state_after()
    assert state["non_metadata_features"] == ["torsion axles"]
    assert state["slots"]["axle_count"] == 2


def test_the_halved_guess_is_not_the_number_held(fake_llm):
    """New Prompt, live: for "14,000 lbs of axle capacity" the model filled BOTH fields - a
    per-axle 7,000 it made up by assuming two axles, and the 14,000 they said."""
    dump_trailer(fake_llm)
    say(fake_llm, "I need 14,000 lbs of axle capacity",
        extracted={"axle_capacity": 7000.0, "total_axle_capacity_lbs": 14000.0})

    assert state_after()["pending_axle_basis"]["value"] == 14000.0


def test_a_weight_or_size_is_not_an_axle_count():
    assert axles.count_from_reply("about 5,000 lbs") is None
    assert axles.count_from_reply("20 ft") is None
    assert axles.count_from_reply("7000") is None
    assert axles.count_from_reply("single axle please") == 1
    assert axles.count_from_reply("5 axles") == 5, "out of range, returned for the range check"


def test_answering_the_load_instead_is_not_scolded_as_a_wrong_count(fake_llm):
    """Live: "about 5,000 lbs" to "how many axles?" was read as 5,000 axles."""
    _count_question_open(fake_llm)
    result = say(fake_llm, "about 5,000 lbs", slots={"payload_capacity": "about 5,000 lbs"})

    assert "one to four" not in result["assistant_text"]
    assert "axle_count" not in state_after()["slots"]


def test_the_models_own_count_is_not_a_weight_either(fake_llm):
    """Live: the model filed "about 5,000 lbs" as axle_count too, and Python's range check
    told the customer we only carry one to four."""
    _count_question_open(fake_llm)
    result = say(fake_llm, "about 5,000 lbs",
                 slots={"payload_capacity": "about 5,000 lbs", "axle_count": "about 5,000 lbs"})

    assert "one to four" not in result["assistant_text"]
    assert "didn't catch" not in result["assistant_text"].lower()
    assert "axle_count" not in state_after()["slots"]
    assert state_after()["pending_axle_count"], "the question still stands"


def test_a_count_out_of_range_is_still_explained():
    from src.tools.filters import FieldApplication, _apply_one

    result = FieldApplication()
    _apply_one({"slots": {}}, "axle_count", "5 axles", "Dump", result)
    assert result.invalid_reason == "axle_range"


def test_the_axle_question_replaces_the_pending_slot_question(fake_llm):
    dump_trailer(fake_llm)
    say(fake_llm, "gravel", slots={"haul_item": "gravel"})
    assert state_after()["pending_slot"] == "payload_capacity"
    say(fake_llm, "5k axles", extracted={"axle_capacity": 5000.0, "axle_capacity_basis": "per_axle"})

    assert state_after()["pending_slot"] is None, "the count question went out, not the weight"


def test_show_me_skips_the_open_count_question_and_shows_results(fake_llm, no_search):
    """Live: "Can you show me what you have?" got "Sorry, I didn't catch that" and the
    count question again."""
    dump_trailer(fake_llm)
    say(fake_llm, "gravel, and I want 7,000 lb axles",
        slots={"haul_item": "gravel"},
        extracted={"axle_capacity": 7000.0, "axle_capacity_basis": "per_axle"})
    result = say(fake_llm, "Can you show me what you have?", intent="skip_all_show_results")

    state = state_after()
    assert "axle_count" in state["declined_slots"]
    assert "didn't catch" not in result["assistant_text"]
    assert len(no_search) == 1


def test_show_me_lets_go_of_a_held_capacity(fake_llm):
    dump_trailer(fake_llm)
    say(fake_llm, "I need 14,000 lbs of axle capacity", extracted={"axle_capacity": 14000.0})
    say(fake_llm, "just show me what you have", intent="skip_all_show_results")

    state = state_after()
    assert state["pending_axle_basis"] is None
    assert "axle_capacity" not in state["slots"]


def test_conventional_per_axle_wording_is_per_axle_even_when_the_model_says_unclear():
    """Live: "5k axles" came back unclear and was held for a question it never needed."""
    assert axles.infer_basis("5k axles", "unclear") == "per_axle"
    assert axles.infer_basis("7,000 lb axles", None) == "per_axle"
