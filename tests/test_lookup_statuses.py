"""What the model is told a lookup result MEANS, not just what it is called.

The matcher returns one of four statuses. The agent used to receive the bare label -
"MATCH STATUS: no_exact" above five listings - and nothing saying those five were substitutes
rather than the trailer the customer named. The rules are New Prompt's, carried in the tool
result so they sit beside the listings they apply to.
"""
from __future__ import annotations

import pytest
from factories import turn_output

from src.graph.state import new_state
from src.llm.tools import ToolRunner, lookup_guidance
from src.tools.category import set_trailer_category

_LISTING = {
    "title": "2025 Diamond C FMAX212 Stock #81382",
    "url": "https://www.trailerplace.com/inventory/81382",
    "category": "Equipment",
    "make": "Diamond C",
    "price": "$9,999",
}


@pytest.fixture
def matcher(monkeypatch):
    """Returns whatever the test sets, through the REAL lookup node."""
    from src.graph.nodes import inventory_lookup as lookup_module

    result = {"match_status": "none", "matches": [], "requested_label": "stock 81382"}

    def _fake(*, year, make, model_text, stock_number, listing_url=None, limit):
        return dict(result)

    monkeypatch.setattr(lookup_module, "lookup_inventory", _fake)
    return result


def _lookup(matcher, status, matches):
    matcher.update(match_status=status, matches=matches)
    state = new_state("s1")
    set_trailer_category(state, "Equipment")
    state["turn_outcome"] = {}
    return ToolRunner(state, turn_output())._lookup_inventory(
        year=None, make=None, model_text=None, stock_number="81382"
    )


# ------------------------------------------------------------------------ each status
def test_an_exact_match_is_presented_as_the_trailer_they_asked_about(matcher):
    text = _lookup(matcher, "exact", [_LISTING])
    assert "these are the trailer(s) they asked about" in text
    assert _LISTING["url"] in text


def test_no_exact_says_the_listings_are_substitutes_and_names_what_is_missing(matcher):
    """The failure this exists for: alternatives presented as the trailer they named."""
    text = _lookup(matcher, "no_exact", [_LISTING])
    assert "we do NOT currently show stock 81382" in text
    assert "not the trailer they asked for" in text
    assert "never invent a spec" in text


def test_ambiguous_replaces_the_usual_closing_with_which_one(matcher):
    text = _lookup(matcher, "ambiguous", [_LISTING, _LISTING])
    assert "SEVERAL models" in text
    assert "which of these they mean" in text
    assert "REPLACES the usual closing" in text


def test_none_lets_the_model_say_we_cannot_find_it(matcher):
    """The search wording forbade any comment on stock, which left "do you have stock 81382?"
    with no answer at all. A lookup has to be allowed to say it found nothing."""
    text = _lookup(matcher, "none", [])
    assert "can't find stock 81382" in text
    assert "979-532-1486" in text
    assert "do not say anything about our stock levels" not in text
    assert "URL:" not in text, "no listings to show"


def test_no_exact_with_nothing_to_offer_is_answered_as_not_found(matcher):
    """What the real matcher returns for a stock number that does not exist - checked live
    against trailer_listings with stock 55555. The no_exact rule refers to listings that are
    not there, so the not-found rule applies instead."""
    text = _lookup(matcher, "no_exact", [])
    assert "can't find stock 81382" in text
    assert "listings below" not in text


def test_a_stock_number_lookup_names_the_stock_number(matcher):
    """The real matcher labels a stock-number lookup "that exact trailer"."""
    matcher["requested_label"] = "that exact trailer"
    text = _lookup(matcher, "no_exact", [])
    assert "can't find stock #81382" in text


def test_an_implausible_stock_number_is_never_quoted_back(matcher):
    """A load weight that slipped into the field must not be read out as a stock number."""
    matcher["requested_label"] = "that exact trailer"
    matcher.update(match_status="no_exact", matches=[])
    state = new_state("s1")
    set_trailer_category(state, "Equipment")
    state["turn_outcome"] = {}
    turn = turn_output()
    # The same number as a measurement: the gate's cross-field conflict check.
    turn.extracted.payload_capacity = 7000.0
    text = ToolRunner(state, turn)._lookup_inventory(
        year=2025, make="Diamond C", model_text=None, stock_number="7000"
    )
    assert "#7000" not in text


def test_an_unknown_status_is_treated_as_no_match():
    assert lookup_guidance("something_new", "the FMAX") == lookup_guidance("none", "the FMAX")


def test_a_missing_label_still_reads_as_a_sentence():
    assert "{label}" not in lookup_guidance("no_exact", "")
    assert "that trailer" in lookup_guidance("no_exact", "")


# ------------------------------------------------------------------- the contact invite
def test_a_lookup_tells_the_agent_not_to_invite_contact_details(matcher):
    text = _lookup(matcher, "exact", [_LISTING])
    assert "Do NOT invite them to share their name" in text


def test_the_deterministic_path_holds_to_the_same_rule():
    """Turn one would normally carry the contact opener. Not on a lookup."""
    from src.graph.nodes.compose import _contact_ask_is_due

    state = new_state("s1")
    state["turn_index"] = 1
    state["turn_outcome"] = {}
    assert _contact_ask_is_due(state), "sanity: the opener is due on turn one"

    state["turn_outcome"] = {"contact_invite_suppressed": True}
    assert not _contact_ask_is_due(state)
