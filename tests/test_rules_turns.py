"""The three New Prompt behaviours, end to end through run_turn(), driven by the rules."""
from __future__ import annotations

import json

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot
from src.llm.prompt import state_block
from src.llm.schemas import HaulClassification
from src.rules import store
from src.rules.models import validate_document

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def cargo(item, *traits):
    return HaulClassification(cargo_traits=list(traits), haul_item_matched=item)


def pick(fake_llm, category):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned=category, intent="category_selection"))
    run_turn("s1", f"a {category} trailer")


# ------------------------------------------------------------- 1. light utility load
def test_a_light_utility_load_is_never_asked_its_weight(fake_llm, no_search):
    pick(fake_llm, "utility")
    fake_llm.push(turn_output(slots={"haul_item": "a golf cart"},
                              haul_classification=cargo("golf cart", "lightweight")))
    result = run_turn("s1", "a golf cart")

    state = state_after()
    assert "payload_capacity" not in state["required_slots"]
    assert state["asked_counts"].get("payload_capacity", 0) == 0
    assert result["qualification_complete"] is True
    assert len(no_search) == 1
    assert "payload_capacity - light load" in state_block(state)


def test_a_heavy_utility_load_is_still_asked_its_weight(fake_llm):
    pick(fake_llm, "utility")
    fake_llm.push(turn_output(slots={"haul_item": "pallets of brick"},
                              haul_classification=cargo("pallets of brick")))
    run_turn("s1", "pallets of brick")
    assert state_after()["pending_slot"] == "payload_capacity"


def test_the_skip_goes_when_the_cargo_changes(fake_llm):
    pick(fake_llm, "utility")
    fake_llm.push(turn_output(slots={"haul_item": "a kayak"},
                              haul_classification=cargo("kayak", "lightweight"),
                              intent="skip_current"))
    run_turn("s1", "a kayak")
    fake_llm.push(turn_output(slots={"haul_item": "a pallet of pavers"}, intent="requirement_change",
                              haul_classification=cargo("pallet of pavers")))
    run_turn("s1", "actually a pallet of pavers")
    assert "payload_capacity" in state_after()["required_slots"]


# ------------------------------------------------------------------ 2. width for big cargo
def test_large_cargo_on_an_equipment_trailer_is_asked_its_width_next(fake_llm):
    pick(fake_llm, "equipment")
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"},
                              haul_classification=cargo("skid steer", "large_or_heavy")))
    run_turn("s1", "a skid steer")

    state = state_after()
    assert state["pending_slot"] == "width"
    assert state["required_slots"][:2] == ["haul_item", "width"]


def test_large_cargo_on_a_width_exempt_category_is_not_asked_width(fake_llm):
    pick(fake_llm, "dump")
    # Cargo that belongs on a dump trailer, so the width rule is what is under test and not
    # the switch question: "a backhoe" or "a skid steer" would both rightly offer Equipment.
    fake_llm.push(turn_output(slots={"haul_item": "a load of rubble"},
                              haul_classification=cargo("rubble", "large_or_heavy")))
    run_turn("s1", "a load of rubble")
    state = state_after()
    assert "width" not in state["required_slots"]
    assert state["pending_slot"] == "payload_capacity"


def test_a_width_already_stated_is_not_asked(fake_llm):
    pick(fake_llm, "equipment")
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer", "width": "7 ft wide"},
                              haul_classification=cargo("skid steer", "large_or_heavy")))
    run_turn("s1", "a skid steer, 7 ft wide")
    assert state_after()["pending_slot"] == "payload_capacity"


def test_turning_the_width_rule_off_takes_effect_mid_conversation(fake_llm):
    pick(fake_llm, "equipment")
    raw = json.loads(store.SEED_PATH.read_text(encoding="utf-8"))
    next(r for r in raw["rules"] if r["id"] == "ask-width-large-cargo")["enabled"] = False
    doc, errors = validate_document(raw)
    assert doc is not None, errors
    store.set_override(doc, version=5)

    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"},
                              haul_classification=cargo("skid steer", "large_or_heavy")))
    run_turn("s1", "a skid steer")
    state = state_after()
    assert "width" not in state["required_slots"]
    assert state["pending_slot"] == "payload_capacity"


# ------------------------------------------------------------------- 3. flatbed default
def test_a_flatbed_assumes_eight_feet_wide_without_asking(fake_llm, no_search):
    pick(fake_llm, "flatbed")
    state = state_after()
    assert state["slots"]["width"] == 8.0
    assert state["slot_sources"]["width"] == "default"
    block = state_block(state)
    assert "Assumed unless they say otherwise: width = 8" in block
    assert "    width = 8" not in block.split("Still to ask")[0], "not listed as something they told us"

    fake_llm.push(turn_output(slots={"haul_item": "steel pipe", "payload_capacity": "6000 lbs"}))
    run_turn("s1", "steel pipe, 6000 lbs")
    assert no_search[-1]["slots"]["width"] == 8.0, "the default reaches the fit rerank"


def test_the_customer_saying_a_width_replaces_the_default(fake_llm):
    pick(fake_llm, "flatbed")
    fake_llm.push(turn_output(slots={"width": "about 10 ft wide"}, intent="requirement_change"))
    run_turn("s1", "actually it needs to be 10 ft wide")
    state = state_after()
    assert state["slots"]["width"] == 10.0
    assert state["slot_sources"]["width"] == "user"
    assert "rule_defaults" in state and "width" not in state["rule_defaults"]


def test_leaving_flatbed_does_not_offer_to_keep_a_width_they_never_gave(fake_llm):
    pick(fake_llm, "flatbed")
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_change"))
    run_turn("s1", "actually I want a dump trailer")
    state = state_after()
    assert state["category"] == "Dump", "no keep-or-drop question over a default"
    assert state["pending_keep_filters"] is None
    assert "width" not in state["slots"]


# ------------------------------------------------------------------- added questions
def test_an_answer_to_a_question_added_in_the_rules_is_stored(fake_llm):
    raw = json.loads(store.SEED_PATH.read_text(encoding="utf-8"))
    dump = raw["categories"]["Dump"]
    dump["optional"].remove("dump_mechanism")
    dump["required"].append("dump_mechanism")
    doc, errors = validate_document(raw)
    assert doc is not None, errors
    store.set_override(doc, version=9)

    pick(fake_llm, "dump")
    fake_llm.push(turn_output(slots={"haul_item": "gravel", "payload_capacity": "3 tons"}))
    run_turn("s1", "gravel, 3 tons")
    assert state_after()["pending_slot"] == "dump_mechanism"
    fake_llm.push(turn_output(slots={"dump_mechanism": "scissor lift"}))
    result = run_turn("s1", "scissor lift")
    assert state_after()["slots"]["dump_mechanism"] == "scissor lift"
    assert result["qualification_complete"] is True


def test_a_slot_no_rule_asks_about_is_still_ignored(fake_llm):
    pick(fake_llm, "dump")
    fake_llm.push(turn_output(slots={"favourite_colour": "red"}))
    run_turn("s1", "red")
    assert "favourite_colour" not in state_after()["slots"]
