"""Links a customer pastes: our listing pages are looked up, and interest goes to the team."""
from __future__ import annotations

import pandas as pd
import pytest

from src.conversation_store import load_session
from src.domain import links
from src.graph.build import run_turn
from src.graph.state import from_snapshot, new_state
from src.search import inventory_matcher

from tests.factories import complete_welcome, turn_output

GALYEAN = "https://www.trailerplace.com/inventory/2026-galyean-32-cattle-trailer-w-butterfly-gates-015087/"
FACEBOOK = "https://www.facebook.com/share/p/1AbCdEf/"
INSTAGRAM = "https://www.instagram.com/p/C9xYz/"


@pytest.fixture(autouse=True)
def mail(monkeypatch):
    from src.tools import email_sender

    sent = []
    monkeypatch.setattr(email_sender, "send_email", lambda subject, body: sent.append((subject, body)) or True)
    return sent


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


# ------------------------------------------------------------------------- the URL itself
def test_links_are_named_by_where_they_point():
    text = f"saw this {FACEBOOK} and this one {INSTAGRAM}, also {GALYEAN}."
    assert links.find_trailer_links(text) == [
        ("Facebook post link", FACEBOOK),
        ("Instagram post link", INSTAGRAM),
        ("TrailerPlace website listing link", GALYEAN),
    ]
    assert links.find_trailer_links("see https://example.com/x") == []


@pytest.mark.parametrize("variant", [
    GALYEAN,
    GALYEAN.rstrip("/"),
    GALYEAN.replace("https://www.", "http://"),
    GALYEAN.replace("https://", "") + "?utm_source=fb",
    GALYEAN.upper(),
])
def test_a_listing_url_is_recognised_however_it_was_copied(variant):
    assert links.normalize_listing_url(variant) == GALYEAN


@pytest.mark.parametrize("not_a_listing", [
    "https://www.trailerplace.com/", "https://www.trailerplace.com/inventory/",
    FACEBOOK, "https://trailerplace.com.evil.io/inventory/x/", None, "",
])
def test_anything_else_is_not_a_listing_url(not_a_listing):
    assert links.normalize_listing_url(not_a_listing) is None


# ------------------------------------------------------------------------- the matcher
@pytest.fixture
def catalogue(monkeypatch):
    frame = inventory_matcher.prepare_inventory(pd.DataFrame([
        {"title": "2026 Galyean 32' Cattle Trailer - 015087", "url": GALYEAN, "stock_number": "15087",
         "year": "2026", "make": "Galyean", "model": "32' Cattle Trailer", "category": "Livestock"},
        {"title": "2026 Galyean 32' Cattle Trailer - 15086", "url": GALYEAN.replace("015087", "15086"),
         "stock_number": "15086", "year": "2026", "make": "Galyean", "model": "32' Cattle Trailer",
         "category": "Livestock"},
    ]))
    monkeypatch.setattr(inventory_matcher, "prepared_inventory", lambda: frame)
    return frame


def test_a_listing_link_finds_exactly_that_trailer(catalogue):
    result = inventory_matcher.lookup_inventory(
        year=None, make=None, model_text=None, stock_number=None,
        listing_url=GALYEAN.rstrip("/"),
    )
    assert result["match_status"] == "exact"
    assert [m["stock_number"] for m in result["matches"]] == ["15087"], "not its twin, 15086"


def test_a_link_to_a_trailer_no_longer_listed_finds_nothing(catalogue):
    result = inventory_matcher.lookup_inventory(
        year=None, make=None, model_text=None, stock_number=None,
        listing_url="https://www.trailerplace.com/inventory/2024-sold-unit-99999/",
    )
    assert result["match_status"] == "none" and result["matches"] == []


def test_a_stock_number_already_shown_is_not_shown_again(catalogue):
    """Live: "I like the 81419" printed the same card a second time.

    The already-shown check was written for links and tested only with one, so a stock
    number - which is how people actually pick a trailer off a list - walked straight past
    it. Nothing about the customer's screen depends on which identifier they used.
    """
    from src.graph.nodes.inventory_lookup import inventory_lookup_node

    state = new_state("s1")
    state["shown_urls"] = [GALYEAN]
    output = turn_output(intent="listing_interest", listing_reference=1)
    output.inventory_lookup.is_lookup = True
    output.inventory_lookup.confidence = "high"
    output.inventory_lookup.stock_number = "15087"
    state["turn"] = output

    inventory_lookup_node(state)

    outcome = state["turn_outcome"]
    assert outcome["listings"] == [], "no second card"
    assert outcome["inventory_already_shown"][0]["stock_number"] == "15087", "details still there"
    assert outcome["inventory_match_status"] == "already_shown"


def test_a_link_already_shown_is_not_shown_again(catalogue):
    from src.graph.nodes.inventory_lookup import inventory_lookup_node

    state = new_state("s1")
    state["shown_urls"] = [GALYEAN]
    output = turn_output(intent="inventory_lookup")
    output.inventory_lookup.is_lookup = True
    output.inventory_lookup.confidence = "high"
    output.inventory_lookup.listing_url = GALYEAN
    state["turn"] = output

    inventory_lookup_node(state)

    outcome = state["turn_outcome"]
    assert outcome["listings"] == [], "no second card"
    assert outcome["inventory_already_shown"][0]["stock_number"] == "15087", "details still there"
    assert outcome["inventory_match_status"] == "already_shown"


# ------------------------------------------------------------- interest goes to the team
def shares(link, wants="availability", **extra):
    output = turn_output(intent="listing_interest", shared_link_interest=True, **extra)
    output.inventory_lookup.wants = wants
    return output


def test_a_facebook_link_they_want_is_emailed_naming_the_facebook_post(fake_llm, mail, no_reply_pass):
    complete_welcome(fake_llm)
    fake_llm.push(shares(FACEBOOK))
    run_turn("s1", f"is this one still available? {FACEBOOK}")

    assert len(mail) == 1
    subject, body = mail[0]
    assert "Listing Interest" in subject + body
    assert "Shared a Facebook post link and asked if it is still available" in body
    # The URL itself is NOT in the email. A post URL is opaque, it says nothing about which
    # trailer and it expires; the platform is what tells the team where they saw us.
    assert FACEBOOK not in body
    assert "| Customer shared a Facebook link" in body
    assert no_reply_pass.calls == 1, "the reply pass tells them it went to the team"


def test_an_instagram_link_on_the_first_message_waits_for_their_details(fake_llm, mail):
    fake_llm.push(shares(INSTAGRAM, wants="price"))
    run_turn("s1", f"how much is this? {INSTAGRAM}")

    assert mail == [], "held until we can reach them"
    stashed = state_after()["pending_email_actions"]
    assert "Shared an Instagram post link and asked about the price" in stashed[0]["description"]


def test_a_link_with_no_interest_sends_nothing(fake_llm, mail):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(intent="smalltalk_other", shared_link_interest=False))
    run_turn("s1", f"lol my buddy sent me {FACEBOOK}")

    assert mail == []


def test_the_agent_escalating_the_same_interest_sends_no_second_email(fake_llm, mail):
    from src.llm.tools import ToolRunner

    state = new_state("s1")
    state["contact"].update({"name": "Dana", "phone": "979-555-0100"})
    state["turn_outcome"] = {"link_interest": {"status": "sent", "label": "Facebook post link"}}
    runner = ToolRunner(state, turn_output())

    reply = runner.call("escalate", '{"reason": "listing_interest", "summary": "wants the FB trailer"}')

    assert mail == [] and not state["turn_outcome"].get("outbox_events")
    assert "PASSED TO THE TEAM" in reply


# ------------------------------------------------- a lookup in the middle of the questions
def _dump_state_after_lookup(reply):
    from src.tools.category import set_trailer_category

    state = new_state("s1")
    set_trailer_category(state, "Dump")
    state["turn_outcome"] = {"reply_text": reply, "inventory_lookup_ran": True}
    return state


def test_a_lookup_mid_questions_goes_back_to_the_next_question():
    """Live: the Galyeans were shown and the dump trailer they came for was never mentioned again."""
    from src.graph.nodes.compose import compose_node

    state = _dump_state_after_lookup(
        "Yes, we have these 2026 Galyeans: 1. [card] Do any of these look like a fit, or would you like to see more options?"
    )
    compose_node(state, turn_output())

    text = state["turn_outcome"]["assistant_text"]
    assert "Do any of these look like a fit" not in text, "their generic closing is replaced"
    assert "Back to your Dump trailer - " in text
    assert state["pending_slot"] == state["turn_outcome"]["asked_slot"] is not None


def test_nothing_is_tacked_on_when_the_model_already_asked_it():
    from src.graph.nodes.compose import compose_node
    from src.tools.questions import carry_on_question

    probe = _dump_state_after_lookup("")
    _slot, question = carry_on_question(probe)
    state = _dump_state_after_lookup(f"Here it is: [card]\n\n{question}")
    compose_node(state, turn_output())

    assert state["turn_outcome"]["assistant_text"].count(question) == 1
