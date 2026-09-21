"""Every set of trailers the customer is shown is told to the team.

A browsing session that goes quiet is still a lead. The team can see the transcript, but
not at a glance, and "six livestock trailers went in front of somebody this afternoon" is
the one line that says a visit was worth something.

Raised from compose, from what the reply actually SHOWED - the search and the lookup both
know what they found, which stops being the same thing the moment the reply chooses what to
present. It goes through the same contact gate as every other notification, so with no way
to reach the customer it waits, and the request for their details stays alive until it can
go out.
"""
from __future__ import annotations

import pytest

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.nodes import greeting
from src.graph.state import from_snapshot, new_state
from src.tools import team_notify

from tests.factories import complete_welcome, turn_output

REASON = team_notify.RESULTS_SHOWN_REASON


@pytest.fixture
def mail(monkeypatch):
    """Nothing leaves the machine; record what would have."""
    from src.tools import email_sender

    sent = []
    monkeypatch.setattr(
        email_sender, "send_email", lambda subject, body: sent.append((subject, body)) or True
    )
    return sent


def alerts(mail):
    return [body for subject, body in mail if REASON in subject]


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def shown_two_trailers(fake_llm, session_id="s1"):
    """A qualified customer who has just been shown the fixture's two listings."""
    complete_welcome(fake_llm, session_id)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn(session_id, "dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    return run_turn(session_id, "just show me what you have")


# ------------------------------------------------------------------ the alert itself
def test_showing_trailers_tells_the_team(fake_llm, no_search, mail):
    result = shown_two_trailers(fake_llm)
    assert result["listings"], "the fixture has to actually show trailers"

    sent = alerts(mail)
    assert len(sent) == 1
    assert "[Results Shown to User] Showed 2 Dump trailers" in sent[0]


def test_the_line_is_the_count_and_the_category(fake_llm, no_search, mail):
    """The stock numbers used to be listed here and made this the longest line the team
    ever read. They are in the conversation the chat link opens."""
    state = new_state("s2")
    state["category"] = "Livestock"
    state["contact"] = {"name": "Dave", "phone": "979-555-0100", "declined": False}
    from src.graph.nodes.compose import _tell_the_team_what_they_saw

    _tell_the_team_what_they_saw(state, [{"stock_number": "15079"}, {"stock_number": "00564"}])

    body = state["turn_outcome"]["outbox_events"][0]["payload"]["body"]
    assert "[Results Shown to User] Showed 2 Livestock trailers" in body
    assert "15079" not in body


def test_a_turn_that_shows_nothing_tells_them_nothing(fake_llm, no_search, mail):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    assert alerts(mail) == [], "a question is not a set of results"


def test_each_batch_is_its_own_alert(fake_llm, no_search, mail):
    shown_two_trailers(fake_llm)
    fake_llm.push(turn_output(intent="show_more_results"))
    run_turn("s1", "show me more")

    assert len(alerts(mail)) == 2, "a second batch is a second thing the team should know"


# ------------------------------------------------- with nobody to attribute it to, it waits
def browsing_anonymously(fake_llm):
    """Listings shown to someone who has not said who they are.

    The gate asks twice and then gets out of the way, which is what lets a customer who will
    not say see trailers at all. These are the four turns that takes.
    """
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s3", "I need a dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s3", "gravel")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    run_turn("s3", "just show me what you have")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    return run_turn("s3", "seriously, just show me")


def test_with_no_contact_the_alert_waits(fake_llm, no_search, mail):
    result = browsing_anonymously(fake_llm)
    assert result["listings"], "the listings still go out; only the email waits"
    assert alerts(mail) == []
    assert state_after("s3")["pending_email_actions"], "held, not dropped"


def test_the_next_turn_asks_who_they_are_and_says_why(fake_llm, no_search, mail):
    browsing_anonymously(fake_llm)
    fake_llm.push(turn_output(intent="general_question"))
    reply = run_turn("s3", "do you do financing?")["assistant_text"]

    assert "so our team can follow up with you" in reply


def test_and_goes_out_when_they_answer(fake_llm, no_search, mail):
    browsing_anonymously(fake_llm)
    fake_llm.push(turn_output(intent="general_question"))
    run_turn("s3", "do you do financing?")

    fake_llm.push(turn_output(intent="contact_info_provided", name="Ibrahim", email="i@x.ai"))
    run_turn("s3", "Ibrahim, i@x.ai")

    sent = alerts(mail)
    assert len(sent) == 1
    assert "Full Name: Ibrahim" in sent[0]


def test_someone_who_refuses_is_not_asked_again(fake_llm, no_search, mail):
    browsing_anonymously(fake_llm)
    fake_llm.push(turn_output(intent="contact_declined", declined=True))
    run_turn("s3", "no thanks")

    after = state_after("s3")
    assert after["pending_email_actions"] == [], "dropped with the question"
    assert alerts(mail) == []


# --------------------------------------------------------------- the two budgets differ
def spent_the_ordinary_budget(**extra):
    state = new_state("s1")
    state["turn_index"] = 9
    state["contact"] = {
        "name": None, "email": None, "phone": None, "declined": False,
        "asks_without_progress": 2, "last_asked_turn": 4,
    }
    state.update(extra)
    return state


def test_a_waiting_alert_buys_two_more_asks():
    state = spent_the_ordinary_budget(pending_email_actions=[{"reason": REASON}])
    assert greeting.contact_ask_is_due(state)


def test_without_one_the_asking_stops():
    assert not greeting.contact_ask_is_due(spent_the_ordinary_budget())


def test_but_the_listings_are_never_held_back_for_it():
    """contact_gate_applies also holds the results gate shut. Left open for a stranded
    notification, the customer would never see another trailer."""
    state = spent_the_ordinary_budget(pending_email_actions=[{"reason": REASON}])
    assert not greeting.contact_gate_applies(state)
