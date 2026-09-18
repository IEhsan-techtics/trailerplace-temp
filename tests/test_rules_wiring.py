"""The rules reach the bot: question lists, wording and completion all read from them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.trailer_fields import get_trailer_fields_as_dict, list_all_categories
from src.graph.state import from_snapshot, new_state, to_snapshot
from src.rules import store
from src.rules.models import validate_document
from src.rules.store import SEED_PATH
from src.tools.category import set_trailer_category
from src.tools.questions import all_required_resolved, next_unanswered_slot, question_text

BEFORE = json.loads(
    (Path(__file__).with_name("fixtures_trailer_fields_before_rules.json")).read_text(encoding="utf-8")
)


def _serve(mutate):
    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    doc, errors = validate_document(raw)
    assert doc is not None, errors
    store.set_override(doc, version=7)


@pytest.mark.parametrize("category", sorted(set(BEFORE) - {"Welding", "NoSuchCategory"}))
def test_every_category_asks_exactly_what_it_asked_before_the_rules(category):
    """Captured from trailer_fields._SPECS immediately before it was deleted."""
    before = dict(BEFORE[category])
    after = get_trailer_fields_as_dict(category)
    if category == "Utility":
        # The lightweight-handling note described a behaviour that did not exist; the skip
        # rule is that behaviour now, so the note went.
        assert "lightweight" in before.pop("notes") and "lightweight" not in after.pop("notes")
    assert after == before


def test_an_unknown_category_gets_the_fallback_spec_it_always_got():
    before = dict(BEFORE["NoSuchCategory"])
    assert get_trailer_fields_as_dict("NoSuchCategory") == before


def test_welding_is_no_longer_listed():
    """It was never canonical, never advertised and never reachable - the rules document
    only accepts canonical categories, so its spec did not carry over."""
    assert "Welding" not in list_all_categories()
    assert "Utility" in list_all_categories()


def test_choosing_a_category_applies_the_rules():
    state = new_state("s1")
    set_trailer_category(state, "Flatbed")
    assert state["slots"]["width"] == 8.0
    assert state["slot_sources"]["width"] == "default"


def test_a_light_utility_load_is_done_once_the_cargo_is_known():
    state = new_state("s1")
    state["cargo_traits"] = ["lightweight"]
    set_trailer_category(state, "Utility")
    assert state["required_slots"] == ["haul_item"]
    state["slots"]["haul_item"] = "golf cart"
    assert all_required_resolved(state)


def test_every_question_skipped_still_counts_as_done():
    def skip_all(raw):
        raw["rules"].append({"id": "skip-haul", "action": "skip_question", "slot": "haul_item",
                             "when": {"categories_in": ["Utility"], "traits_any": ["lightweight"]}})
    _serve(skip_all)
    state = new_state("s1")
    state["cargo_traits"] = ["lightweight"]
    set_trailer_category(state, "Utility")
    assert state["required_slots"] == []
    assert all_required_resolved(state)


def test_an_empty_list_that_no_rule_emptied_is_not_done():
    state = new_state("s1")
    state["category"] = "Utility"
    assert not all_required_resolved(state)


def test_a_reworded_question_reaches_a_conversation_already_under_way():
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    assert question_text(state, "payload_capacity") == "What's the rough haul weight per load?"
    _serve(lambda raw: raw["categories"]["Dump"]["questions"].update(payload_capacity="How heavy is a load?"))
    assert question_text(state, "payload_capacity") == "How heavy is a load?"


def test_an_injected_question_is_worded_by_its_rule():
    state = new_state("s1")
    set_trailer_category(state, "Equipment")
    assert question_text(state, "width") == "About how wide is that item or trailer you need to haul?"


def test_the_injected_width_question_is_asked_next():
    state = new_state("s1")
    set_trailer_category(state, "Equipment")
    state["slots"]["haul_item"] = "skid steer"
    state["cargo_traits"] = ["large_or_heavy"]
    state["pending_slot"] = "haul_item"
    from src.rules.engine import apply_rules
    apply_rules(state, store.current_rules())
    assert next_unanswered_slot(state) == "width"


def test_rule_bookkeeping_survives_a_snapshot_round_trip():
    state = new_state("s1")
    state["cargo_traits"] = ["lightweight"]
    set_trailer_category(state, "Flatbed")
    restored = from_snapshot("s1", json.loads(json.dumps(to_snapshot(state))))
    for key in ("cargo_traits", "slot_sources", "rule_defaults", "rule_skipped", "rule_ask_anchors"):
        assert restored[key] == state[key], key
