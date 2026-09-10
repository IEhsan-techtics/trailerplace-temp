"""SessionState round-trips, and keeps the contract search_node depends on."""
from __future__ import annotations

from src.graph.state import (
    MAX_ASKS_PER_SLOT,
    SessionState,
    from_snapshot,
    new_state,
    to_snapshot,
)

# Read straight out of src/graph/nodes/search.py. If search_node starts reading a new key,
# this list is where that has to be noticed.
SEARCH_NODE_KEYS = (
    "session_id", "category", "slots", "brand_preference",
    "non_metadata_features", "shown_urls", "qualification_complete", "turn_outcome",
)


def test_new_state_supplies_every_key_search_node_reads():
    state = new_state("s1")
    for key in SEARCH_NODE_KEYS:
        assert key in state, f"search_node reads {key!r} and it is missing"


def test_collections_are_materialised_so_nodes_need_no_none_guards():
    state = new_state("s1")
    assert state["slots"] == {}
    assert state["declined_slots"] == []
    assert state["asked_counts"] == {}
    assert state["non_metadata_features"] == []


def test_snapshot_excludes_transient_turn_outcome():
    state = new_state("s1")
    state["turn_outcome"] = {"listings": [{"url": "x"}]}
    assert "turn_outcome" not in to_snapshot(state)


def test_state_survives_a_save_load_round_trip():
    state = new_state("s1")
    state["category"] = "Dump"
    state["slots"] = {"length": 20.0, "payload_capacity": 6000.0}
    state["asked_counts"] = {"length": 2}
    state["declined_slots"] = ["width"]
    state["contact"] = {"name": "Dave", "email": "d@x.com", "phone": None,
                        "asked": True, "declined": False}
    state["pending_keep_filters"] = {"new_category": "Equipment", "filters": {"length": 20.0}}

    restored = from_snapshot("s1", to_snapshot(state))

    assert restored["category"] == "Dump"
    assert restored["slots"] == {"length": 20.0, "payload_capacity": 6000.0}
    assert restored["asked_counts"] == {"length": 2}
    assert restored["declined_slots"] == ["width"]
    assert restored["contact"]["name"] == "Dave"
    assert restored["pending_keep_filters"]["new_category"] == "Equipment"


def test_an_older_snapshot_missing_new_keys_still_loads():
    """Forward compatibility: a row written before a key existed must not crash a node."""
    restored = from_snapshot("s1", {"category": "Dump", "slots": {"length": 12.0}})
    assert restored["category"] == "Dump"
    assert restored["declined_slots"] == []
    assert restored["contact"]["asked"] is False


def test_unknown_snapshot_keys_are_dropped():
    restored = from_snapshot("s1", {"category": "Dump", "some_removed_key": 1})
    assert "some_removed_key" not in restored


def test_empty_snapshot_yields_a_fresh_session():
    assert from_snapshot("s1", None)["category"] is None


def test_attempt_cap_is_two():
    """Brief S23. Hard-coded here so a change to the constant is a deliberate act."""
    assert MAX_ASKS_PER_SLOT == 2
