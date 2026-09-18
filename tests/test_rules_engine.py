"""The question rules: what a document is allowed to say, and what it does to a session."""
from __future__ import annotations

import copy
import json

import pytest

from src.graph.state import new_state
from src.rules import engine
from src.rules.models import validate_document
from src.rules.store import SEED_PATH, seed_document


def seed_raw() -> dict:
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


def doc_with(mutate) -> object:
    raw = seed_raw()
    mutate(raw)
    doc, errors = validate_document(raw)
    assert doc is not None, errors
    return doc


def session(category: str, traits=(), slots=None, pending=None):
    state = new_state("s1")
    state["category"] = category
    state["cargo_traits"] = list(traits)
    state["slots"] = dict(slots or {})
    state["pending_slot"] = pending
    return state


def rule(raw: dict, rule_id: str) -> dict:
    return next(r for r in raw["rules"] if r["id"] == rule_id)


# ------------------------------------------------------------------------------ the seed
def test_the_seed_is_valid():
    doc, errors = validate_document(seed_raw())
    assert errors == []
    assert doc is not None


def test_the_seed_keeps_the_question_lists_it_was_generated_from():
    """Spot checks. The full comparison against the pre-rules output is in
    test_rules_wiring, once trailer_fields reads from here. The Utility note about
    lightweight handling is gone on purpose: the skip rule now does that for real."""
    doc = seed_document()
    assert doc.categories["Utility"].required == ["haul_item", "payload_capacity"]
    assert doc.categories["Equipment"].required == ["haul_item", "payload_capacity", "length", "hitch_type"]
    assert "lightweight" not in doc.categories["Utility"].notes


# ------------------------------------------------------------------------- the engine
@pytest.mark.parametrize(
    "category, traits, expected",
    [
        ("Utility", ["lightweight"], ["haul_item"]),
        ("Utility", [], ["haul_item", "payload_capacity"]),
        ("Dump", ["lightweight"], ["haul_item", "payload_capacity"]),       # Utility-only rule
        ("Flatbed", ["large_or_heavy"], ["haul_item", "payload_capacity"]),  # width-exempt
        ("Car Hauler", ["large_or_heavy"], ["haul_item", "width", "payload_capacity", "length"]),
    ],
)
def test_required_questions_by_category_and_cargo(category, traits, expected):
    state = session(category, traits, slots={"haul_item": "x"}, pending="haul_item")
    engine.apply_rules(state, seed_document())
    assert state["required_slots"] == expected


def test_the_width_question_goes_straight_after_the_question_on_the_table():
    state = session("Equipment", ["large_or_heavy"], pending="payload_capacity",
                    slots={"haul_item": "skid steer"})
    engine.apply_rules(state, seed_document())
    assert state["required_slots"] == ["haul_item", "payload_capacity", "width", "length", "hitch_type"]


def test_the_width_question_stays_where_it_was_put_as_the_conversation_moves_on():
    state = session("Equipment", ["large_or_heavy"], pending="haul_item", slots={"haul_item": "skid steer"})
    engine.apply_rules(state, seed_document())
    first = list(state["required_slots"])
    state["slots"]["width"] = 7.0
    state["slots"]["payload_capacity"] = 5000.0
    state["pending_slot"] = "length"
    engine.apply_rules(state, seed_document())
    assert state["required_slots"] == first


def test_a_width_already_given_is_not_asked_for():
    from src.tools.questions import next_unanswered_slot

    state = session("Equipment", ["large_or_heavy"], slots={"haul_item": "skid steer", "width": 8.5})
    engine.apply_rules(state, seed_document())
    assert next_unanswered_slot(state) == "payload_capacity"


def test_flatbed_width_defaults_to_eight_feet_and_is_never_asked():
    state = session("Flatbed")
    engine.apply_rules(state, seed_document())
    assert state["slots"]["width"] == 8.0
    assert engine.is_default(state, "width")
    assert "width" not in state["required_slots"]


def test_a_width_the_customer_gave_beats_the_default():
    state = session("Flatbed", slots={"width": 10.0})
    engine.apply_rules(state, seed_document())
    assert state["slots"]["width"] == 10.0
    assert not engine.is_default(state, "width")


def test_the_customer_replacing_a_default_makes_it_theirs():
    state = session("Flatbed")
    engine.apply_rules(state, seed_document())
    state["slots"]["width"] = 10.0
    engine.mark_user_value(state, "width")
    engine.apply_rules(state, seed_document())
    assert state["slots"]["width"] == 10.0
    assert not engine.is_default(state, "width")


def test_a_default_leaves_with_its_category():
    state = session("Flatbed")
    engine.apply_rules(state, seed_document())
    state["category"] = "Dump"
    engine.apply_rules(state, seed_document())
    assert "width" not in state["slots"]
    assert "width" not in state["slot_sources"]


def test_a_default_declined_by_the_customer_is_not_forced_on_them():
    state = session("Flatbed")
    state["declined_slots"] = ["width"]
    engine.apply_rules(state, seed_document())
    assert "width" not in state["slots"]


def test_an_edited_default_value_reaches_a_session_already_holding_the_old_one():
    state = session("Flatbed")
    engine.apply_rules(state, seed_document())
    wider = doc_with(lambda raw: rule(raw, "flatbed-default-width").update(value=8.5))
    engine.apply_rules(state, wider)
    assert state["slots"]["width"] == 8.5


def test_a_disabled_rule_does_nothing():
    off = doc_with(lambda raw: [r.update(enabled=False) for r in raw["rules"]])
    for category, traits in (("Utility", ["lightweight"]), ("Equipment", ["large_or_heavy"]), ("Flatbed", [])):
        state = session(category, traits)
        engine.apply_rules(state, off)
        assert state["required_slots"] == off.categories[category].required
        assert "width" not in state["slots"]


def test_skip_beats_ask_for_the_same_slot():
    def both(raw):
        raw["rules"].append({
            "id": "skip-width", "action": "skip_question", "slot": "width",
            "when": {"categories_in": ["Equipment"]},
        })
    state = session("Equipment", ["large_or_heavy"])
    engine.apply_rules(state, doc_with(both))
    assert "width" not in state["required_slots"]


def test_a_trait_the_rules_do_not_define_is_dropped():
    state = session("Utility", ["lightweight", "made_up"])
    engine.apply_rules(state, seed_document())
    assert state["cargo_traits"] == ["lightweight"]


def test_no_category_means_no_effects():
    state = new_state("s1")
    state["cargo_traits"] = ["large_or_heavy"]
    engine.apply_rules(state, seed_document())
    assert state["required_slots"] == []
    assert state["slots"] == {}


def test_skipped_questions_are_recorded_with_their_reason():
    state = session("Utility", ["lightweight"])
    engine.apply_rules(state, seed_document())
    assert state["rule_skipped"] == {"payload_capacity": "light load - any utility trailer carries it"}


def test_a_new_question_added_to_a_category_is_asked():
    def add(raw):
        raw["categories"]["Dump"]["optional"].remove("dump_mechanism")
        raw["categories"]["Dump"]["required"].append("dump_mechanism")
    state = session("Dump")
    engine.apply_rules(state, doc_with(add))
    assert state["required_slots"] == ["haul_item", "payload_capacity", "dump_mechanism"]


# ------------------------------------------------------------------------- validation
def _errors(mutate) -> list[str]:
    raw = copy.deepcopy(seed_raw())
    mutate(raw)
    doc, errors = validate_document(raw)
    assert doc is None
    return errors


def test_an_unknown_category_is_rejected():
    assert any("unknown category" in e for e in _errors(lambda r: r["categories"].update(Boat={"required": []})))
    assert any("unknown category" in e for e in _errors(
        lambda r: rule(r, "flatbed-default-width")["when"].update(categories_in=["Boat"])))


def test_an_unknown_slot_is_rejected():
    assert any("unknown slot" in e for e in _errors(lambda r: rule(r, "skip-weight-light-utility").update(slot="colour")))


def test_an_undefined_trait_is_rejected():
    assert any("unknown trait" in e for e in _errors(
        lambda r: rule(r, "ask-width-large-cargo")["when"].update(traits_any=["enormous"])))


def test_a_default_that_does_not_parse_is_rejected():
    assert any("not a usable value" in e for e in _errors(lambda r: rule(r, "flatbed-default-width").update(value="wide")))


def test_duplicate_rule_ids_are_rejected():
    def dup(raw):
        raw["rules"].append(copy.deepcopy(raw["rules"][0]))
    assert any("duplicate id" in e for e in _errors(dup))


def test_an_ask_rule_needs_wording():
    assert any("needs question wording" in e for e in _errors(lambda r: rule(r, "ask-width-large-cargo").update(question="")))


def test_a_required_slot_needs_wording():
    assert any("no question wording" in e for e in _errors(lambda r: r["categories"]["Dump"]["required"].append("dump_mechanism_x")))


def test_the_fallback_spec_is_required():
    assert any("fallback" in e for e in _errors(lambda r: r["categories"].pop("Unknown")))


def test_a_misspelt_field_is_rejected_rather_than_ignored():
    assert _errors(lambda r: rule(r, "flatbed-default-width").update(valeu=9))
