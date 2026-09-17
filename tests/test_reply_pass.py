"""The reply pass and its tools (src/llm/respond.py, src/llm/tools.py).

Two things are being pinned. First, the reply pass costs a second model call, so it must
run ONLY on turns that have inventory to talk about. Second, the tool's preconditions are
Python's call - the model decides whether the customer wants to see trailers, never whether
we are in a position to show them any.
"""
from __future__ import annotations

import pytest
from factories import complete_welcome, turn_output

from src.llm.schemas import ReplyOutput
from src.llm.tools import ToolRunner, listing_block, listing_line


def _reply(text="Here are the trailers.", urls=None):
    return ReplyOutput(assistant_text=text, cited_listing_urls=urls or [])


def _qualified_state(**overrides):
    state = {
        "session_id": "s1",
        "category": "Dump",
        "slots": {"haul_material": "gravel"},
        "contact": {"name": "Dave", "email": "d@x.ai", "phone": None, "declined": False},
        "turn_outcome": {},
    }
    state.update(overrides)
    return state


# ------------------------------------------------------------------ the tool result block
def test_listing_line_omits_fields_the_row_does_not_have():
    """A missing field must never reach the model as "None" - it copies it onto the card."""
    line = listing_line(1, {"title": "2026 Iron Bull DTB - 15081", "price_display": "$9,995",
                            "length": None, "width": "", "url": "https://x/1"})
    assert "TITLE: 2026 Iron Bull DTB - 15081" in line
    assert "Price: $9,995" in line
    assert "Length" not in line
    assert "Width" not in line
    assert line.endswith("URL: https://x/1")


def test_empty_listing_block_says_so_without_claiming_anything_about_stock():
    block = listing_block([])
    assert "NO MATCHES" in block
    assert "stock levels" in block


# ------------------------------------------------------------------ tool preconditions
def test_search_tool_refuses_without_a_category():
    runner = ToolRunner(_qualified_state(category=None), turn_output())
    result = runner.call("search_inventory", "{}")
    assert "NO SEARCH RAN" in result
    assert runner.ran == []


def test_search_tool_refuses_while_the_contact_gate_is_open():
    state = _qualified_state(
        contact={"name": None, "email": None, "phone": None, "declined": False,
                 "asks_without_progress": 0}
    )
    runner = ToolRunner(state, turn_output())
    result = runner.call("search_inventory", "{}")
    assert "NO SEARCH RAN" in result
    assert runner.ran == []


@pytest.mark.parametrize(
    "pending",
    ["pending_keep_filters", "pending_category_switch", "pending_gooseneck_clarification"],
)
def test_search_tool_refuses_while_a_question_of_ours_is_outstanding(pending):
    runner = ToolRunner(_qualified_state(**{pending: {"anything": True}}), turn_output())
    assert "NO SEARCH RAN" in runner.call("search_inventory", "{}")


def test_search_tool_runs_when_every_precondition_is_met(monkeypatch):
    from src.graph.nodes import search as search_module

    def _fake_search_node(state):
        state.setdefault("turn_outcome", {}).update(
            {"search_ran": True, "listings": [{"title": "A", "url": "https://x/1"}], "result_count": 1}
        )
        return state

    monkeypatch.setattr(search_module, "search_node", _fake_search_node)
    runner = ToolRunner(_qualified_state(), turn_output())
    result = runner.call("search_inventory", "{}")
    assert "TITLE: A" in result
    assert runner.ran == ["search_inventory"]


def test_lookup_tool_refuses_a_bare_make():
    """The same gate as the deterministic path - a make alone is a brand preference."""
    runner = ToolRunner(_qualified_state(), turn_output())
    result = runner.call("lookup_inventory", '{"make": "Diamond C", "year": null, "model_text": null, "stock_number": null}')
    assert "NO LOOKUP RAN" in result
    assert runner.ran == []


def test_unknown_tool_does_not_raise():
    runner = ToolRunner(_qualified_state(), turn_output())
    assert "No such tool" in runner.call("nonsense", "{}")


def test_broken_arguments_do_not_raise():
    runner = ToolRunner(_qualified_state(category=None), turn_output())
    assert "NO SEARCH RAN" in runner.call("search_inventory", "{not json")


# ------------------------------------------------------------------ when the pass runs
def test_reply_pass_does_not_run_on_a_plain_qualification_turn(fake_llm, no_reply_pass):
    """It costs a second model call. A turn with no inventory must not pay it."""
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="Dump"))
    run_turn("s1", "I need a dump trailer")

    assert no_reply_pass.calls == 0


def test_reply_pass_runs_when_the_gate_opens(fake_llm, no_reply_pass):
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="Dump"))
    run_turn("s1", "I need a dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    no_reply_pass.push(_reply())
    run_turn("s1", "just show me what you have")

    assert no_reply_pass.calls == 1


def test_the_models_reply_is_used_verbatim(fake_llm, no_reply_pass):
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="Dump"))
    run_turn("s1", "I need a dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    no_reply_pass.push(_reply("1. [A](https://x/1)\n   - Price: $1", ["https://x/1"]))
    result = run_turn("s1", "just show me what you have")

    assert result["assistant_text"] == "1. [A](https://x/1)\n   - Price: $1"


def test_cited_urls_are_recorded_as_shown(fake_llm, no_reply_pass):
    """"Show me more" excludes what they have seen, so this is what makes paging work."""
    from src import conversation_store
    from src.graph.build import run_turn
    from src.graph.state import from_snapshot

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="Dump"))
    run_turn("s1", "I need a dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    no_reply_pass.push(_reply(urls=["https://x/1", "https://x/2"]))
    run_turn("s1", "just show me what you have")

    snapshot, _conversation, _lead = conversation_store.load_session("s1")
    state = from_snapshot("s1", snapshot)
    assert state["shown_urls"] == ["https://x/1", "https://x/2"]
    assert state["results_shown"] is True


def test_backstop_renders_listings_when_the_reply_pass_fails(fake_llm, no_reply_pass, no_search):
    """A model failure is a flatter reply, never a customer shown nothing."""
    from src.graph.build import run_turn

    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="Dump"))
    run_turn("s1", "I need a dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    # no_reply_pass.queue stays empty -> returns None, exactly as a failed call does.
    result = run_turn("s1", "just show me what you have")

    assert no_reply_pass.calls == 1
    assert len(no_search) == 1, "the backstop must still run the search itself"
    assert "Diamond C Dump" in result["assistant_text"]


# ------------------------------------------------------------- no-match honesty (3f)
# These travel WITH the tool result rather than sitting in the static prompt, because how
# well a batch matches is a fact about that call, not a rule the agent must remember.
def _quality(outcome, listings, **state):
    return ToolRunner(_qualified_state(**state), turn_output())._match_quality(outcome, listings)


def test_zero_results_says_nothing_matches_and_offers_to_adjust():
    text = _quality({"search_ran": True, "result_count": 0}, [])
    assert "NO MATCHES" in text
    assert "nothing in our current stock matches" in text
    assert "adjust" in text
    assert "979-532-1486" in text


def test_zero_results_after_paging_says_they_have_seen_them_all():
    """A different message: we HAVE stock, they have just already been shown it."""
    text = _quality({"result_count": 0}, [], shown_urls=["https://x/1"])
    assert "already seen every match" in text


def test_brand_relaxed_opens_by_naming_the_brand_we_lack():
    text = _quality(
        {"result_count": 1, "brand_relaxed": True},
        [{"title": "A", "url": "https://x/1"}],
        brand_preference="Diamond C",
    )
    assert "DIAMOND C" in text.upper()
    assert "other" in text.lower()
    assert "Never imply" in text


def test_filters_relaxed_names_what_was_dropped():
    text = _quality(
        {"result_count": 1, "filters_relaxed": True, "relaxed_filters_dropped": ["length", "hitch type"]},
        [{"title": "A", "url": "https://x/1"}],
    )
    assert "ALTERNATIVES, NOT EXACT MATCHES" in text
    assert "length, hitch type" in text


def test_a_clean_match_adds_no_caveat():
    assert _quality({"result_count": 1}, [{"title": "A", "url": "https://x/1"}]) == ""


# ------------------------------------------------------------------- pitch material
# The pitch used to be written from the bullet fields alone, so all it could do was repeat
# them. The tool line now carries material, floor and features for it - separately labelled so
# they never become bullets of their own.
def test_the_tool_line_carries_pitch_material_the_card_does_not_show():
    from src.llm.tools import listing_line

    line = listing_line(1, {
        "title": "2026 Galyean Cattle Trailer - 15079",
        "url": "https://x/15079",
        "length": "32 ft 0 in",
        "trailer_material": "STEEL",
        "floor": "CLEATED RUBBER FLOOR",
        "features": ["Torsion suspension", "10 LED dome light with toggle switch"],
    })
    assert "Length: 32 ft 0 in" in line
    pitch = line.split("FOR THE PITCH ONLY (never a bullet): ", 1)[1]
    assert "cleated rubber floor" in pitch
    assert "Torsion suspension" in pitch
    assert "material steel" in pitch


def test_pitch_material_is_capped_and_skips_placeholders():
    from src.llm.tools import listing_line
    from src.search.listing_search import PITCH_FEATURE_CHARS as _PITCH_FEATURE_CHARS
    from src.search.listing_search import PITCH_FEATURES as _PITCH_FEATURES

    features = [f"Feature number {i} " + "x" * 200 for i in range(20)] + ["None", ""]
    line = listing_line(1, {
        "title": "T", "url": "https://x/1",
        "trailer_material": "Unspecified", "floor": None, "features": features,
    })
    pitch = line.split("FOR THE PITCH ONLY (never a bullet): ", 1)[1]
    kept = pitch.removeprefix("features: ").split("; ")
    assert len(kept) == _PITCH_FEATURES
    assert all(len(item) <= _PITCH_FEATURE_CHARS for item in kept)
    assert "material" not in pitch and "None" not in pitch


def test_a_listing_with_nothing_extra_has_no_pitch_part():
    from src.llm.tools import listing_line

    line = listing_line(1, {"title": "T", "url": "https://x/1", "features": []})
    assert "FOR THE PITCH ONLY" not in line


def test_the_prompt_forbids_repeating_card_values_in_the_pitch():
    from src.llm.respond import _CARD_FORMAT

    assert "THE PITCH NEVER REPEATS THE CARD" in _CARD_FORMAT
    # The example pitch must itself obey the rule - the old one restated the axle bullet.
    example = [l for l in _CARD_FORMAT.splitlines() if l.startswith("   - ")][-1]
    assert not any(ch.isdigit() for ch in example), example


def test_search_results_keep_a_capped_feature_list_for_the_pitch(monkeypatch):
    """search_listings strips the full features list - which silently emptied the pitch
    material on every real search. A capped slice now survives the strip."""
    from types import SimpleNamespace

    from src.search import listing_search

    raw = {"title": "T", "url": "https://x/1", "match_evidence_text": "long evidence",
           "features": ["Long Arm Tarp System", "Long Arm Tarp System", "None", "LED lights"]}
    monkeypatch.setattr(
        listing_search, "search_listing_result",
        lambda **kw: SimpleNamespace(listings=[raw]),
    )
    [item] = listing_search.search_listings(category="Dump", slots={}, metadata_filters={})
    assert "features" not in item and "match_evidence_text" not in item
    assert item["pitch_features"] == ["Long Arm Tarp System", "LED lights"]


def test_search_names_the_material_field_material():
    """The key search uses. Reading only trailer_material found nothing."""
    from src.llm.tools import listing_line

    line = listing_line(1, {"title": "T", "url": "https://x/1", "material": "Aluminum",
                            "pitch_features": []})
    assert "material aluminum" in line


@pytest.mark.parametrize("empty", [None, float("nan"), 3, ""])
def test_pitch_features_treats_anything_but_a_list_as_none(empty):
    """A DataFrame cell with no features is NaN, and iterating a float raises."""
    import numpy as np

    from src.search.listing_search import pitch_features

    assert pitch_features(empty) == []
    assert pitch_features(np.float64("nan")) == []
    assert pitch_features(np.array(["Tarp kit", "Ramps"], dtype=object)) == ["Tarp kit", "Ramps"]
