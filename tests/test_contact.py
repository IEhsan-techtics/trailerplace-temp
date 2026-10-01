"""Contact capture and the lead row. Asked once, then never again."""
from __future__ import annotations

import pytest

from src.conversation_store import load_lead, load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


@pytest.fixture
def capture(fake_llm):
    """Run one turn carrying contact details; return the resulting lead row."""

    def _capture(**contact):
        fake_llm.push(turn_output(intent="contact_info_provided", **contact))
        run_turn("s1", "here are my details")
        return load_lead("s1")

    return _capture


# ------------------------------------------------------------------------ storing the lead
def test_details_are_stored_on_the_lead(capture):
    lead = capture(name="Dave", email="dave@example.com", phone="979-555-0100")
    assert lead["name"] == "Dave"
    assert lead["email"] == "dave@example.com"
    assert lead["phone_number"] == "979-555-0100"
    assert lead["contact_status"] == "complete"


def test_details_alone_are_not_a_hard_lead_until_the_team_is_emailed(capture):
    """A way to reach them is not enough: the team has to have heard about them."""
    lead = capture(name="Dave", email="dave@example.com")
    assert lead["lead_type"] == "soft"
    assert lead["contact_status"] == "complete"


def test_a_name_alone_is_partial_contact(capture):
    lead = capture(name="Dave")
    assert lead["lead_type"] == "soft"
    assert lead["contact_status"] == "partial"


def test_a_session_that_gives_nothing_stays_a_soft_lead(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hi")
    lead = load_lead("s1")
    assert lead["lead_type"] == "soft"
    assert lead["contact_status"] == "missing_contact"


# ---------------------------------------------------------------------------- hard leads
#
# Hard = a team email went out about someone we can reach by email or phone. A name is not
# required. Tested on the store directly, both the in-memory path and the Postgres one.
EMAIL = [{"event_type": "results_shown_to_user", "payload": {}}]


def _save(session_id="s1", *, contact=None, emailed=False):
    from src.conversation_store import save_turn

    save_turn(
        session_id, conversation=[], state_snapshot={}, request_message="x", response={},
        contact=contact, outbox_events=EMAIL if emailed else None,
    )


def test_an_email_to_the_team_about_someone_reachable_makes_a_hard_lead():
    _save(contact={"email": "dave@example.com"}, emailed=True)
    assert load_lead("s1")["lead_type"] == "hard", "a name is not required"


def test_a_phone_number_counts_as_reachable():
    _save(contact={"phone": "979-555-0100"}, emailed=True)
    assert load_lead("s1")["lead_type"] == "hard"


def test_details_without_a_team_email_stay_soft():
    _save(contact={"name": "Dave"})
    _save(contact={"name": "Dave", "phone": "979-555-0100"})
    assert load_lead("s1")["lead_type"] == "soft"


def test_once_hard_it_stays_hard():
    _save(contact={"email": "dave@example.com"}, emailed=True)
    _save(contact={"email": "dave@example.com"})
    assert load_lead("s1")["lead_type"] == "hard"


@pytest.mark.parametrize(
    "lead_fields, contact, emailed, expected",
    [
        ({}, {"email": "d@x.com"}, True, "hard"),
        ({}, {"phone": "979-555-0100"}, True, "hard"),
        ({}, {"name": "Dave", "email": "d@x.com"}, False, "soft"),
        ({}, {"name": "Dave"}, True, "soft"),
        ({"email": "d@x.com"}, {}, True, "hard"),
        ({"lead_type": "hard", "email": "d@x.com"}, {}, False, "hard"),
    ],
)
def test_the_database_lead_follows_the_same_rule(lead_fields, contact, emailed, expected):
    """_update_lead is the Postgres path, which the suite otherwise never reaches."""
    from types import SimpleNamespace

    from src.conversation_store import _update_lead

    lead = SimpleNamespace(
        psid=None, name=None, email=None, phone_number=None, lead_type="soft",
        contact_status="missing_contact",
    )
    vars(lead).update(lead_fields)
    sql = SimpleNamespace(get=lambda _model, _id: lead)

    _update_lead(sql, "lead-1", contact, emailed=emailed)
    assert lead.lead_type == expected


# ------------------------------------------------------------------- asked once, then never
def test_the_opener_is_marked_asked_after_the_first_turn(fake_llm):
    """Whatever they replied, they have now been asked - that is the whole policy."""
    assert state_after()["contact"]["asked"] is False

    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hi")

    assert state_after()["contact"]["asked"] is True


def test_ignoring_the_opener_does_not_get_a_second_ask(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hi")
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "I need a dump trailer")

    state = state_after()
    assert state["contact"]["asked"] is True
    assert state["contact"]["name"] is None
    # The prompt is what carries the instruction; assert it says so for this state.
    from src.llm.prompt import state_block

    assert "Never ask again" in state_block(state)


def test_declining_is_remembered_and_never_pursued(fake_llm):
    fake_llm.push(turn_output(intent="contact_declined", declined=True))
    run_turn("s1", "I'd rather not share that")

    assert state_after()["contact"]["declined"] is True

    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    assert state_after()["contact"]["declined"] is True


def test_a_decline_on_a_message_that_does_something_else_still_counts(fake_llm):
    """ContactInfo.declined is set even when the dominant intent is something bigger."""
    fake_llm.push(
        turn_output(category_mentioned="dump", intent="category_selection", declined=True)
    )
    run_turn("s1", "dump trailer, and no I won't give my number")

    state = state_after()
    assert state["contact"]["declined"] is True
    assert state["category"] == "Dump", "the rest of the message still applied"


def test_details_are_never_overwritten_by_a_later_turn(fake_llm):
    fake_llm.push(turn_output(intent="contact_info_provided", name="Dave", email="dave@x.com"))
    run_turn("s1", "I'm Dave, dave@x.com")

    fake_llm.push(turn_output(intent="contact_info_provided", name="Sam", email="sam@x.com"))
    run_turn("s1", "actually I'm Sam")

    state = state_after()
    assert state["contact"]["name"] == "Dave", "the lead keeps the details it qualified on"
    assert state["contact"]["email"] == "dave@x.com"


# ------------------------------------------- a refusal stops the asking, not always the send
#
# The flag means "this message refuses contact details", which covers two different things:
# "don't contact me" and "I'd rather not give my name". With an email already on file the
# second is the one that happened, and throwing that lead away was the old rule taken past
# its own reason for existing - there was no way to reach them, so there was nothing to send.
def _state(**contact):
    return {
        "session_id": "s1",
        "contact": {"name": None, "email": None, "phone": None, "declined": False, **contact},
        "turn_outcome": {},
    }


def test_refusing_the_name_does_not_throw_away_a_reachable_lead():
    from src.tools import team_notify

    state = _state(email="d@x.ai", declined=True)
    assert team_notify.record(state, reason="Listing Interest", description="wants it") == "sent"
    assert len(state["turn_outcome"]["outbox_events"]) == 1


def test_refusing_with_no_way_to_reach_them_still_drops_it():
    from src.tools import team_notify

    state = _state(declined=True)
    assert team_notify.record(state, reason="Listing Interest", description="wants it") == "dropped"
    assert not state.get("pending_email_actions"), "nothing to wait for - they said no"


def test_a_stash_from_an_earlier_turn_survives_the_refusal():
    """They asked about a trailer on turn one, gave an email on turn two and would rather
    not give their name on turn three. The turn-one request is still a real lead."""
    from src.tools import team_notify

    state = _state(email="d@x.ai", declined=True)
    state["pending_email_actions"] = [
        {"reason": "Listing Interest", "description": "wants it", "event_type": "listing_interest"}
    ]
    assert team_notify.flush(state) == 1
    assert state["pending_email_actions"] == []


def test_but_we_never_ask_them_again():
    """The half of the old rule that does not change."""
    from src.graph.nodes import greeting

    state = _state(email="d@x.ai", declined=True)
    assert not greeting.contact_gate_applies(state)
    assert not greeting.contact_ask_is_due(state)


def test_the_status_the_customer_is_told_matches():
    from src.tools import team_notify

    assert team_notify.status_now(_state(email="d@x.ai", declined=True)) == "sent"
    assert team_notify.status_now(_state(declined=True)) == "dropped"
    assert team_notify.status_now(_state()) == "stashed"
