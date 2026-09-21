"""What the team's email says, beyond the reason and the description.

Three things, all ported from New Prompt:

* a required question asked twice and never answered is itself worth reporting - it is the
  bot failing to get somewhere, and only the team can change the wording;
* a link back to the conversation, so the email is a starting point rather than a summary;
* a Facebook or Instagram URL taken OUT of the description and replaced by the platform's
  name. A post URL is opaque, says nothing about which trailer and expires. Our own listing
  links stay: they name exactly one trailer.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot
from src.tools import email_sender

from tests.factories import complete_welcome, turn_output


@pytest.fixture
def mail(monkeypatch):
    sent = []
    monkeypatch.setattr(
        email_sender, "send_email", lambda subject, body: sent.append((subject, body)) or True
    )
    return sent


def bodies(mail, reason):
    return [body for subject, body in mail if reason in subject]


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


# ------------------------------------------------- a question asked twice and never answered
def dodges_the_question(fake_llm, times=2):
    """A customer who is asked the same thing twice and answers neither."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "I need a dump trailer")
    asked = state_after()["pending_slot"]
    for index in range(times):
        fake_llm.push(turn_output(intent="general_question",
                                  user_question_to_answer="do you open on Sundays?"))
        run_turn("s1", f"but do you open on Sundays? ({index})")
    return asked


def test_giving_up_on_a_question_tells_the_team(fake_llm, no_search, mail):
    slot = dodges_the_question(fake_llm)
    assert slot, "the fixture has to actually get a question asked"

    sent = bodies(mail, "Unanswered Question")
    assert len(sent) == 1
    assert "No answer after two asks" in sent[0]
    assert slot.replace("_", " ") in sent[0], "the team is told WHICH question"


def test_one_dodge_is_not_giving_up(fake_llm, no_search, mail):
    dodges_the_question(fake_llm, times=1)
    assert bodies(mail, "Unanswered Question") == [], "it still has an ask left"


def test_a_stalled_conversation_is_reported_once(fake_llm, no_search, mail):
    """Four dodges loses two questions. A customer who ignores one usually ignores the
    next, and six emails about one stalled chat is noise the team learns to filter out.
    The first one is the signal; the chat link carries the rest."""
    dodges_the_question(fake_llm, times=4)
    assert len(bodies(mail, "Unanswered Question")) == 1


def test_no_preference_is_an_answer_not_a_failure(fake_llm, no_search, mail):
    """"Whatever works" closes the question on the spot. Nothing went wrong."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "I need a dump trailer")
    slot = state_after()["pending_slot"]

    output = turn_output(intent="qualification_answer")
    output.extracted.numeric_no_preference = [slot]
    fake_llm.push(output)
    run_turn("s1", "whatever works")

    assert bodies(mail, "Unanswered Question") == []


def test_an_explicit_skip_is_not_a_failure_either(fake_llm, no_search, mail):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "I need a dump trailer")

    fake_llm.push(turn_output(intent="skip_current"))
    run_turn("s1", "skip that one")

    assert bodies(mail, "Unanswered Question") == [], "they answered; the answer was no"


# ----------------------------------------------------------------- the link to the chat
def test_the_email_links_back_to_the_conversation(fake_llm, no_search, mail, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(email_sender, "settings",
                        replace(settings, chat_ui_url="https://chat.example.com/"))
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(intent="general_question", faq_key="financing"))
    run_turn("s1", "do you do financing?")

    body = bodies(mail, "FAQ")[0]
    assert "| Chat: https://chat.example.com/?chat_session=" in body


def test_with_no_ui_configured_there_is_no_link(fake_llm, no_search, mail, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(email_sender, "settings", replace(settings, chat_ui_url=""))
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(intent="general_question", faq_key="financing"))
    run_turn("s1", "do you do financing?")

    assert "| Chat:" not in bodies(mail, "FAQ")[0], "a wrong link is worse than none"


# --------------------------------------------------------------- social links in the body
FACEBOOK = "https://www.facebook.com/trailerplace/posts/pfbid02abc"
OURS = "https://www.trailerplace.com/inventory/2026-iron-bull-dtb-15081/"


def test_a_post_url_is_replaced_by_the_platform():
    body = email_sender.render_email_body(
        name="Dave", email="d@x.ai", phone=None, reason="Listing Interest",
        description=f"Shared a Facebook post link and wants this trailer: {FACEBOOK}",
        shared_platforms=["Facebook"],
    )
    assert FACEBOOK not in body
    assert "[Listing Interest] Shared a Facebook post link and wants this trailer" in body
    assert "| Customer shared a Facebook link" in body


def test_our_own_listing_link_stays():
    """It names exactly one trailer, which is the most useful thing the line can carry."""
    body = email_sender.render_email_body(
        name="Dave", email="d@x.ai", phone=None, reason="Listing Interest",
        description=f"Interested in 2026 Iron Bull DTB - 15081 ({OURS})",
    )
    assert OURS in body


def test_the_platform_is_remembered_across_turns(fake_llm, no_search, mail):
    """The email may not go out for several turns. Where they found us stays true."""
    fake_llm.push(turn_output(intent="general_question"))
    run_turn("s2", f"saw this on your page {FACEBOOK}")
    assert state_after("s2")["shared_platforms"] == ["Facebook"]

    fake_llm.push(turn_output(intent="general_question", faq_key="financing"))
    run_turn("s2", "do you do financing?")
    fake_llm.push(turn_output(intent="contact_info_provided", name="Ibrahim", email="i@x.ai"))
    run_turn("s2", "Ibrahim, i@x.ai")

    assert "| Customer shared a Facebook link" in bodies(mail, "FAQ")[0]
