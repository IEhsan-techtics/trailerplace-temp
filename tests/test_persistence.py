"""State survives across turns and across a restart (brief S9)."""
from __future__ import annotations

from src import conversation_store
from src.conversation_store import describe_interest, load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot, new_state, to_snapshot

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def test_everything_the_conversation_learned_survives_a_reload(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection",
                              name="Dave", email="dave@x.com"))
    run_turn("s1", "dump trailer, I'm Dave dave@x.com")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")

    # A fresh read, exactly as a new process would do it.
    state = state_after()
    assert state["category"] == "Dump"
    assert state["slots"]["haul_item"] == "gravel"
    assert state["asked_counts"]
    assert state["contact"]["name"] == "Dave"
    assert state["required_slots"], "the category's question list came back too"


def test_a_conversation_resumes_mid_question(fake_llm):
    """The pending question is state, not memory, so a restart does not lose the thread."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    assert state_after()["pending_slot"] == "haul_item"

    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")
    assert state_after()["slots"]["haul_item"] == "gravel"


def test_declines_and_attempt_counts_survive(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(intent="skip_current", answered_current_question=False))
    run_turn("s1", "skip it")

    state = state_after()
    assert "haul_item" in state["declined_slots"]
    assert state["asked_counts"]["haul_item"] == 1


def test_pending_confirmations_survive(fake_llm):
    fake_llm.push(turn_output(category_mentioned="utility", intent="category_selection"))
    run_turn("s1", "utility trailer")
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    run_turn("s1", "a skid steer")

    restored = state_after()
    assert restored["pending_category_switch"]["suggested"] == "Equipment"


def test_the_transcript_is_persisted_in_order(fake_llm):
    """The welcome reply is written by greeting.py, so the assistant line is that text
    rather than anything the model produced."""
    fake_llm.push(turn_output(intent="smalltalk_other", acknowledgement="Hi there!"))
    run_turn("s1", "hi")

    _snapshot, conversation, _lead = load_session("s1")
    assert [m["role"] for m in conversation] == ["user", "assistant"]
    assert conversation[0]["content"] == "hi"
    assert "Thank you for contacting TrailerPlace" in conversation[1]["content"]


def test_two_sessions_do_not_share_state(fake_llm):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(category_mentioned="utility", intent="category_selection"))
    run_turn("s2", "utility trailer")

    assert state_after("s1")["category"] == "Dump"
    assert state_after("s2")["category"] == "Utility"


def test_the_snapshot_carries_no_listing_payloads(fake_llm, no_search):
    """turn_outcome is scratch. Persisting it would bloat the row with listing dicts."""
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    run_turn("s1", "show me")

    snapshot, _conversation, _lead = load_session("s1")
    assert "turn_outcome" not in snapshot
    assert "listings" not in str(snapshot)


def test_the_snapshot_is_json_serialisable(fake_llm):
    import json

    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection",
                              name="Dave", phone="979-555-0100"))
    run_turn("s1", "dump trailer")
    snapshot, _conversation, _lead = load_session("s1")
    json.dumps(snapshot)


def test_the_version_counter_advances_with_each_turn(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hi")
    first = conversation_store._MEMORY["s1"]["state_version"]

    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hello again")
    assert conversation_store._MEMORY["s1"]["state_version"] == first + 1


# ------------------------------------------------------------------------- interest summary
def test_the_interest_summary_describes_the_search():
    state = new_state("s1")
    state["category"] = "Dump"
    state["slots"] = {"haul_item": "gravel", "length": 20.0}
    summary = describe_interest(state)
    assert "Dump" in summary and "gravel" in summary
    assert "20.0" not in summary, "rendered as a plain number"


def test_the_interest_summary_is_never_empty():
    assert describe_interest(new_state("s1")) == "Unspecified"


def test_round_tripping_an_empty_session_is_stable():
    state = new_state("s1")
    assert from_snapshot("s1", to_snapshot(state)) == state
