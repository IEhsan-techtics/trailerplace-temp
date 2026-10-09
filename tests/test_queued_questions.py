"""Questions of ours that are queued but not yet asked.

Several of our own questions can be waiting at once (gooseneck hitch-or-brand, switch
category, keep earlier answers, per-axle-or-total, how many axles) and only the first goes
out in a reply. A queued one has never been seen by the customer, so their next message is
never judged as an answer to it.

Live, 7 Oct: "22 or 24 gooseneck ... 10k torsion axles", then "Equipment". The axle-count
question was queued behind the gooseneck one. The customer answered the gooseneck question
with "Hitch", which was judged as an unclear axle count: the model's correct reply was
thrown away for "Sorry, I didn't catch that. How many axles...?".
"""
from __future__ import annotations

from dataclasses import replace

from src import config
from src.conversation_store import load_session
from src.domain import axles
from src.graph.build import run_turn
from src.graph.state import from_snapshot
from src.llm.prompt import state_block

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def say(fake_llm, text, **output):
    fake_llm.push(turn_output(**output))
    return run_turn("s1", text)


def gooseneck_ahead_of_axle_count(fake_llm):
    """Beny's first three turns: the axle-count question ends up queued, never asked."""
    complete_welcome(fake_llm)
    say(fake_llm, "Looking for a 22 or 24 gooseneck trailer with 10k lb torsion axles",
        intent="feature_request_no_category",
        extracted={"length": 22.0, "axle_capacity": 10000.0, "axle_capacity_basis": "per_axle",
                   "hitch_type": ["Gooseneck"]})
    say(fake_llm, "Equipment heavy duty fenders on outside",
        intent="category_selection", category_mentioned="equipment")
    state = state_after()
    assert state["pending_gooseneck_clarification"], "the gooseneck question goes first"
    assert state["pending_axle_count"] == {"asks": 0}, "the axle question is queued, not asked"


def test_answering_the_gooseneck_question_is_not_an_unclear_axle_count(fake_llm):
    gooseneck_ahead_of_axle_count(fake_llm)
    result = say(fake_llm, "Hitch", intent="qualification_answer",
                 extracted={"hitch_type": ["Gooseneck"]})

    state = state_after()
    assert state["slots"]["hitch_type"] == ["Gooseneck"]
    assert state["pending_gooseneck_clarification"] is None
    assert state["invalid_retry_slot"] is None
    assert "didn't catch that" not in result["assistant_text"]
    # Now it is first in line, so it goes out - as a plain question.
    assert axles.COUNT_QUESTION.removeprefix("And ").lower()[:20] in result["assistant_text"].lower()


def test_a_count_volunteered_before_it_is_asked_is_still_taken(fake_llm):
    gooseneck_ahead_of_axle_count(fake_llm)
    say(fake_llm, "the hitch, and two axles", intent="qualification_answer",
        extracted={"hitch_type": ["Gooseneck"], "axle_count": 2})

    state = state_after()
    assert state["slots"]["axle_count"] == 2
    assert state["pending_axle_count"] is None


def test_an_asked_axle_question_still_gets_the_retry(fake_llm):
    """The apology is right when we DID ask and could not read the answer."""
    complete_welcome(fake_llm)
    say(fake_llm, "I need a dump trailer", intent="category_selection", category_mentioned="dump")
    say(fake_llm, "I want 7,000 lb axles",
        extracted={"axle_capacity": 7000.0, "axle_capacity_basis": "per_axle"})
    assert state_after()["pending_axle_count"]["asks"] == 1

    result = say(fake_llm, "purple")
    assert state_after()["invalid_retry_slot"] == "axle_count"
    assert "didn't catch that" in result["assistant_text"]


def test_the_model_is_not_told_it_asked_a_queued_question(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "settings", replace(config.settings, llm_writes_reply=True))
    state = {"category": "Equipment", "pending_axle_count": {"asks": 0},
             "pending_axle_basis": {"value": 14000.0, "asks": 0},
             "pending_category_switch": {"suggested": "Equipment", "from_haul_item": "a skid steer",
                                         "asks": 0},
             "pending_keep_filters": {"new_category": "Equipment", "filters": {}, "asks": 0}}
    block = state_block(state)
    assert "You asked how many axles" not in block
    assert "You asked whether" not in block
    assert "You suggested switching" not in block
    assert "keep_fields_answer" not in block

    for key in ("pending_axle_count", "pending_axle_basis", "pending_category_switch",
                "pending_keep_filters"):
        state[key]["asks"] = 1
    block = state_block(state)
    assert "You asked how many axles" in block
    assert "You asked whether" in block
    assert "You suggested switching" in block
    assert "keep_fields_answer" in block


def test_an_unasked_switch_is_not_answered_by_a_stray_yes(fake_llm):
    from src.graph.nodes.apply import _apply_category_switch_answer

    state = {"pending_category_switch": {"suggested": "Equipment", "pair": ["Utility", "Equipment"],
                                         "asks": 0}, "category": "Utility"}
    assert _apply_category_switch_answer(state, turn_output(category_confirm_answer="yes")) is False
    assert state["category"] == "Utility"
    assert state["pending_category_switch"]


def test_the_exact_live_case_flags_nothing():
    """The live turn as the log recorded it: axle count queued (asks 0), "Hitch" carrying no
    count and storing nothing new. Before the fix this set retry axle_count/unclear, which
    is what threw the model's reply away for the apology."""
    from types import SimpleNamespace

    from src.graph.nodes.apply import _apply_axle_count

    state = {"session_id": "s1", "category": "Equipment", "turn_index": 4,
             "slots": {"length": 22.0, "hitch_type": ["Gooseneck"], "axle_capacity": 10000.0},
             "slot_sources": {"axle_capacity": "user"}, "declined_slots": [],
             "pending_axle_count": {"asks": 0},
             "invalid_retry_slot": None, "invalid_retry_reason": None}
    result = SimpleNamespace(stored={}, no_preference=[])
    _apply_axle_count(state, turn_output(intent="qualification_answer"), "Hitch", result)

    assert state["invalid_retry_slot"] is None
    assert state["pending_axle_count"] == {"asks": 0}, "still waiting to be asked"
