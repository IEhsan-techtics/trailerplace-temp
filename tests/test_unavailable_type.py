"""A trailer type we do not carry: say so, list what we have, and hand them to the team."""
from __future__ import annotations

import pytest

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output


@pytest.fixture(autouse=True)
def mail(monkeypatch):
    from src.tools import email_sender

    sent = []
    monkeypatch.setattr(email_sender, "send_email", lambda subject, body: sent.append((subject, body)) or True)
    return sent


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def ask_for(type_text, **extra):
    extra.setdefault("intent", "category_exploration")
    return turn_output(unavailable_type_requested=type_text, **extra)


def test_a_type_we_never_sell_is_answered_with_what_we_do_have(fake_llm, mail):
    fake_llm.push(ask_for("boat trailer"))
    text = run_turn("s1", "do you have boat trailers?")["assistant_text"]

    assert "we don't currently have boat trailers available" in text
    for stocked in ("- Dump", "- Utility", "- Equipment", "- Enclosed"):
        assert stocked in text
    assert "- Diesel Tank" not in text, "only what is in stock is listed"
    assert "Our team will contact you shortly" in text
    assert "your name and an email or phone number" in text
    assert text.startswith("Thank you for contacting TrailerPlace."), "it owns the first turn"
    assert mail == [], "held until we can reach them"
    assert state_after()["pending_email_actions"], "stashed, not dropped"


def test_the_team_hears_about_it_as_soon_as_they_give_their_details(fake_llm, mail):
    fake_llm.push(ask_for("camper"))
    run_turn("s1", "I'm after a camper")
    fake_llm.push(turn_output(intent="contact_info_provided", name="Dana", phone="979-555-0100"))
    text = run_turn("s1", "Dana, 979-555-0100")["assistant_text"]

    assert len(mail) == 1
    subject, body = mail[0]
    assert "camper" in (subject + body).lower()
    assert "team" in text.lower()


def test_with_details_on_file_the_team_is_told_at_once(fake_llm, mail):
    complete_welcome(fake_llm)
    fake_llm.push(ask_for("horse trailer with living quarters"))
    text = run_turn("s1", "any horse trailers with living quarters?")["assistant_text"]

    assert len(mail) == 1
    assert "I've passed this on to our team" in text
    assert "Could I take" not in text
    assert not text.startswith("Thank you for contacting")


def test_a_category_we_know_but_hold_no_stock_in_is_caught_without_the_model(fake_llm, mail):
    """Diesel Tank is canonical but not in the fixture catalogue. The model only named it."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="diesel tank trailer", intent="category_selection"))
    text = run_turn("s1", "I need a diesel tank trailer")["assistant_text"]

    assert "we don't currently have Diesel Tank trailers available" in text
    assert len(mail) == 1
    assert state_after()["category"] is None


def test_a_type_we_do_stock_is_never_treated_as_unavailable(fake_llm, mail):
    complete_welcome(fake_llm)
    fake_llm.push(ask_for("dump trailer", category_mentioned="dump trailer"))
    text = run_turn("s1", "a dump trailer")["assistant_text"]

    assert "don't currently have" not in text
    assert mail == []
    assert state_after()["category"] == "Dump"


def test_asking_again_does_not_email_the_team_twice(fake_llm, mail):
    complete_welcome(fake_llm)
    fake_llm.push(ask_for("boat trailer"))
    run_turn("s1", "boat trailers?")
    fake_llm.push(ask_for("boat trailers"))
    text = run_turn("s1", "so no boat trailers at all?")["assistant_text"]

    assert len(mail) == 1
    assert "don't currently have boat trailers" in text


def test_someone_who_declined_contact_gets_the_phone_number_and_no_ask(fake_llm, mail):
    fake_llm.push(turn_output(intent="contact_declined", declined=True))
    run_turn("s1", "I'd rather not share my details")
    fake_llm.push(ask_for("boat trailer"))
    text = run_turn("s1", "do you do boat trailers?")["assistant_text"]

    assert "979-532-1486" in text
    assert "Could I take" not in text
    assert mail == []


def test_the_rest_of_the_conversation_carries_on_normally(fake_llm, mail):
    complete_welcome(fake_llm)
    fake_llm.push(ask_for("boat trailer"))
    run_turn("s1", "boat trailer?")
    fake_llm.push(turn_output(category_mentioned="utility", intent="category_selection"))
    text = run_turn("s1", "ok, a utility trailer then")["assistant_text"]

    assert state_after()["category"] == "Utility"
    assert "?" in text and "don't currently have" not in text


def test_exactly_one_model_call_and_no_reply_pass(fake_llm, no_reply_pass):
    complete_welcome(fake_llm)
    calls_before = fake_llm.calls
    fake_llm.push(ask_for("boat trailer", intent="team_request_escalation"))
    run_turn("s1", "can someone get me a boat trailer?")

    assert fake_llm.calls == calls_before + 1
    assert no_reply_pass.calls == 0, "no second escalation through the reply pass"


@pytest.mark.parametrize("requested, label", [
    ("Diesel Tank", "Diesel Tank trailers"),
    ("boat trailer", "boat trailers"),
    ("boat trailers", "boat trailers"),
    ("campers", "campers"),
    ("camper", "campers"),
    ("horse trailer with living quarters", "horse trailer with living quarters"),
])
def test_the_type_is_named_naturally(requested, label):
    """Live run: the model reported "campers", and the reply said "campers trailers"."""
    from src.tools.unavailable import _label

    assert _label(requested) == label


def test_the_reply_invites_them_to_look_at_what_we_do_have(fake_llm, mail):
    """A list is not an answer on its own. Without something to reply to, a customer told
    "we don't have that, here's what we do have" has been closed down rather than helped."""
    fake_llm.push(turn_output(intent="contact_info_provided", name="Dave", phone="979-555-0100"))
    run_turn("s1", "hi, I'm Dave on 979-555-0100")
    fake_llm.push(turn_output(unavailable_type_requested="boat trailer"))
    text = run_turn("s1", "do you have boat trailers?")["assistant_text"]

    assert "Would any of those work for what you need?" in text
    assert text.count("?") == 1, "one question per reply"


def test_but_not_while_we_are_still_asking_who_they_are(fake_llm, mail):
    """Then the contact request is the question, and two question marks in one reply is how
    a customer ends up answering neither."""
    fake_llm.push(turn_output(unavailable_type_requested="boat trailer"))
    text = run_turn("s2", "do you have boat trailers?")["assistant_text"]

    assert "Would any of those work" not in text
    assert text.count("?") == 1
