"""LLM_WRITES_REPLY: the model's reply goes out as written, or it is written again.

Live, compose's name check read "You're welcome! What material will you be hauling?" as a
greeting to someone called What, cut the word out, and sent "You're welcome material will
you be hauling". With the flag on, the model's reply is never edited: it goes out as written,
or it is turned down and the model writes it again, told why. The checks read the model's own
report on its reply (asked_slots, question_count, asked_for_contact, reply_covers), never the
text.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.graph.nodes import compose
from src.llm import client
from src.llm.client import empty_output
from src.llm.schemas import ChatbotTurnReplyOutput, ReplyRewrite


@pytest.fixture(autouse=True)
def writes_reply(monkeypatch):
    from src import config

    on = replace(config.settings, llm_writes_reply=True)
    monkeypatch.setattr(config, "settings", on)
    monkeypatch.setattr(compose, "settings", on)


class Rewrites:
    """Stands in for the rewrite call: hands back what a test queued, else fails."""

    def __init__(self):
        self.queue: list[ReplyRewrite | None] = []
        self.calls: list[dict] = []

    def __call__(self, state, user_message, first_reply, problem, needs):
        self.calls.append({"first": first_reply, "problem": problem, "needs": needs})
        return self.queue.pop(0) if self.queue else None


@pytest.fixture(autouse=True)
def rewrites(monkeypatch):
    fake = Rewrites()
    monkeypatch.setattr(client, "rewrite_reply", fake)
    return fake


def _fields(reply, asked=None, questions=None, contact=False, covers=(), offered=()):
    asked = list(asked or [])
    return {
        "reply": reply,
        "asked_slots": asked,
        "asked_for_contact": contact,
        "question_count": (1 if asked else 0) if questions is None else questions,
        "reply_covers": list(covers),
        "offered_categories": list(offered),
    }


def _output(reply, asked=None, questions=None, contact=False, covers=(), offered=(), **pieces):
    fields = {**empty_output().model_dump(), "about_trailers": True, "only_acknowledges": False}
    if "contact_info" in pieces:  # the output's own contact field; `contact` here is asked_for_contact
        fields["contact"] = pieces.pop("contact_info").model_dump()
    fields.update(pieces)
    return ChatbotTurnReplyOutput.model_validate(
        {**fields, **_fields(reply, asked, questions, contact, covers, offered)}
    )


def _rewrite(reply, asked=None, questions=None, contact=False, covers=(), offered=()):
    return ReplyRewrite.model_validate(_fields(reply, asked, questions, contact, covers, offered))


def _state(**overrides):
    """Mid-qualification on a Dump trailer, contact settled, so no contact ask is due."""
    state = {
        "session_id": "s1",
        "turn_index": 3,
        "category": "Dump",
        "required_slots": ["haul_item", "payload_capacity"],
        "slots": {"length": 10.0, "width": 6.0},
        "asked_counts": {},
        "declined_slots": [],
        "contact": {"name": "Dave", "phone": "979-555-0100", "greeted": True, "asked": True},
        "turn_outcome": {"user_message": "ok thanks"},
    }
    state.update(overrides)
    return state


def _send(state, output):
    compose.compose_node(state, output)
    return state["turn_outcome"]


def _sent_as_written(state, output):
    return _send(state, output)["assistant_text"] == output.reply


def _turned_down(state, output, rewrites):
    """Turned down: a rewrite was asked for (none queued, so it fails) and compose answered."""
    outcome = _send(state, output)
    return bool(rewrites.calls) and outcome["assistant_text"] != output.reply


# ---- the ordinary flow ----


def test_the_live_what_reply_goes_out_word_for_word():
    reply = "You're welcome! What material will you be hauling (dirt, gravel, debris, etc.)?"
    outcome = _send(_state(), _output(reply, ["haul_item"], covers=["thanked_them"]))

    assert outcome["assistant_text"] == reply
    assert outcome["asked_slot"] == "haul_item"


def test_the_question_it_asks_is_counted():
    state = _state()
    _send(state, _output("Got it. What will you be hauling?", ["haul_item"]))

    assert state["asked_counts"] == {"haul_item": 1}
    assert state["pending_slot"] == "haul_item"


def test_it_may_pick_any_open_question_not_only_the_first():
    state = _state()
    assert _sent_as_written(state, _output("What's the approximate weight of the load?", ["payload_capacity"]))
    assert state["pending_slot"] == "payload_capacity"


def test_two_questions_are_turned_down(rewrites):
    output = _output("What will you be hauling? And how heavy is it?", ["haul_item"], questions=2)
    assert _turned_down(_state(), output, rewrites)
    assert "2 questions" in rewrites.calls[0]["problem"]


def test_two_slots_claimed_are_turned_down(rewrites):
    output = _output("What will you haul, and how heavy is it?", ["haul_item", "payload_capacity"])
    assert _turned_down(_state(), output, rewrites)


def test_a_question_already_answered_is_turned_down(rewrites):
    state = _state(slots={"length": 10.0, "width": 6.0, "haul_item": "gravel"})
    assert _turned_down(state, _output("What will you be hauling?", ["haul_item"]), rewrites)


def test_a_question_asked_twice_already_is_turned_down(rewrites):
    state = _state(asked_counts={"haul_item": 2})
    assert _turned_down(state, _output("What will you be hauling?", ["haul_item"]), rewrites)


def test_a_slot_question_it_did_not_name_is_turned_down(rewrites):
    assert _turned_down(_state(), _output("What will you be hauling?", [], questions=1), rewrites)


def test_asking_nothing_with_questions_left_is_turned_down(rewrites):
    assert _turned_down(_state(), _output("Great, thanks for that.", []), rewrites)


def test_no_category_the_type_question_needs_no_slot():
    state = _state(category=None, required_slots=[], slots={})
    reply = "What type of trailer are you looking for? We have Utility, Dump and more - which fits?"
    assert _sent_as_written(state, _output(reply, [], questions=1))


# ---- the rewrite ----


def test_a_turned_down_reply_is_written_again_and_the_rewrite_goes_out(rewrites):
    rewrites.queue.append(_rewrite("Got it. What will you be hauling?", ["haul_item"]))
    state = _state()
    outcome = _send(state, _output("Great, thanks for that.", []))

    assert outcome["assistant_text"] == "Got it. What will you be hauling?"
    assert state["pending_slot"] == "haul_item"


def test_the_rewrite_is_told_what_was_wrong_and_what_is_needed(rewrites):
    _send(_state(), _output("Great, thanks for that.", []))

    call = rewrites.calls[0]
    assert call["first"] == "Great, thanks for that."
    assert "asks nothing" in call["problem"]
    assert "haul_item" in call["needs"] and "payload_capacity" in call["needs"]


def test_a_rewrite_that_still_breaks_a_rule_goes_to_compose(rewrites):
    rewrites.queue.append(_rewrite("Still nothing to ask.", []))
    output = _output("Great, thanks for that.", [], acknowledgement="Thanks for that.",
                     next_question_slot="haul_item", next_question_text="What will you be hauling?")
    outcome = _send(_state(), output)

    assert outcome["assistant_text"] == "Thanks for that. What will you be hauling?"


def test_a_turn_python_owns_is_not_rewritten(rewrites):
    state = _state(pending_category_switch={"suggested": "Equipment", "from_haul_item": "a skid steer"})
    _send(state, _output("What will you be hauling?", ["haul_item"]))

    assert rewrites.calls == []


# ---- a wrong value ----


def _retry(**overrides):
    return _state(**{"invalid_retry_slot": "payload_capacity", "invalid_retry_reason": "negative", **overrides})


def test_a_rejected_value_moving_on_to_another_question_is_turned_down(rewrites):
    assert _turned_down(_retry(), _output("Thanks! What will you be hauling?", ["haul_item"]), rewrites)


def test_a_rejected_value_re_asked_with_the_reason_goes_out():
    state = _retry()
    reply = "That came through as a negative number. What's the rough haul weight per load?"
    assert _sent_as_written(state, _output(reply, ["payload_capacity"], covers=["flagged_wrong_value"]))
    assert state["asked_counts"] == {"payload_capacity": 1}


def test_thanks_for_a_rejected_value_is_turned_down(rewrites):
    """Live: "Thanks, I've noted that. That came through as a negative number..." """
    reply = "Thanks, I've noted that. That came through as a negative number. What's the weight?"
    output = _output(reply, ["payload_capacity"], covers=["flagged_wrong_value", "thanked_them"])
    assert _turned_down(_retry(), output, rewrites)


def test_a_re_ask_that_does_not_say_what_was_wrong_is_turned_down(rewrites):
    assert _turned_down(_retry(), _output("What's the rough haul weight per load?", ["payload_capacity"]), rewrites)


def test_the_axle_count_is_still_re_asked_by_python(rewrites):
    state = _state(invalid_retry_slot="axle_count", invalid_retry_reason="axle_range",
                   required_slots=["haul_item", "axle_count"])
    output = _output("That seems off. How many axles?", ["axle_count"], covers=["flagged_wrong_value"])
    assert _send(state, output)["assistant_text"] != output.reply
    assert rewrites.calls == []


def test_a_rejected_value_already_asked_twice_is_not_asked_again(rewrites):
    state = _retry(asked_counts={"payload_capacity": 2})
    reply = "That came through as a negative number. What's the rough haul weight per load?"
    assert _turned_down(state, _output(reply, ["payload_capacity"], covers=["flagged_wrong_value"]), rewrites)


# ---- the contact request ----


def test_asking_for_contact_when_it_is_not_due_is_turned_down(rewrites):
    output = _output("What will you be hauling? And your name and number?", ["haul_item"], contact=True)
    assert _turned_down(_state(), output, rewrites)


def test_a_due_contact_ask_rides_along():
    """An answered standard question is one of the moments to ask."""
    state = _state(contact={})
    reply = "We offer financing - call 979-532-1486. What will you be hauling? And your name and number?"
    assert _sent_as_written(state, _output(reply, ["haul_item"], contact=True, faq_key="financing"))
    assert state["contact"]["asks_without_progress"] == 1


def test_a_due_contact_ask_it_left_out_is_written_again_not_bolted_on(rewrites):
    rewrites.queue.append(_rewrite("What will you be hauling? And may I have your name and number?",
                                   ["haul_item"], contact=True))
    state = _state(contact={})
    outcome = _send(state, _output("We offer financing. What will you be hauling?", ["haul_item"],
                                   faq_key="financing"))

    assert outcome["assistant_text"] == "What will you be hauling? And may I have your name and number?"
    assert "contact details are due" in rewrites.calls[0]["problem"]


def test_a_due_contact_ask_may_be_the_one_question():
    state = _state(contact={})
    reply = "We're open 8 to 6. Could you share your name and an email or phone so our team can follow up?"
    outcome = _send(state, _output(reply, [], contact=True, faq_key="store_info"))

    assert outcome["assistant_text"] == reply
    assert outcome["asked_slot"] is None


def test_a_request_passed_to_the_team_must_say_so(rewrites):
    state = _state(turn_outcome={"user_message": "Sam, sam@x.com", "emails_flushed": 1})
    assert _turned_down(state, _output("Thanks, Sam. What will you be hauling?", ["haul_item"]), rewrites)

    state = _state(turn_outcome={"user_message": "Sam, sam@x.com", "emails_flushed": 1})
    reply = "Thanks, Sam - I've passed that on to our team. What will you be hauling?"
    assert _sent_as_written(state, _output(reply, ["haul_item"], covers=["passed_to_team"]))


# ---- the flag and the prompt ----


def test_with_the_flag_off_the_reply_is_ignored(monkeypatch):
    monkeypatch.setattr(compose, "settings", replace(compose.settings, llm_writes_reply=False))
    output = _output("You're welcome! What material will you be hauling?", ["haul_item"],
                     acknowledgement="Thanks for that.", next_question_slot="haul_item",
                     next_question_text="What will you be hauling?")

    assert _send(_state(), output)["assistant_text"] == "Thanks for that. What will you be hauling?"


def test_the_prompt_only_asks_for_a_reply_with_the_flag_on():
    from src.llm.prompt import _system_prompt_for

    assert "REPORT ON YOUR REPLY" in _system_prompt_for(0, True)
    assert "REPORT ON YOUR REPLY" not in _system_prompt_for(0, False)


# ---- off topic ----


def _off_topic(state=None):
    state = state or _state()
    state["turn_outcome"]["off_topic"] = True
    return state


def test_an_off_topic_decline_goes_out_as_written():
    reply = "Sorry, I can only help with trailers here. What will you be hauling?"
    assert _sent_as_written(_off_topic(), _output(reply, ["haul_item"], covers=["declined_off_topic"]))


def test_an_off_topic_reply_that_does_what_they_asked_is_turned_down(rewrites):
    """Live, the model apologised and then gave the recipe anyway."""
    reply = "I mainly help with trailers, but here you go: " + "toast the bread, add ham and cheese. " * 12
    output = _output(reply + "What will you be hauling?", ["haul_item"], covers=["declined_off_topic"])
    assert _turned_down(_off_topic(), output, rewrites)


def test_an_off_topic_reply_must_decline(rewrites):
    assert _turned_down(_off_topic(), _output("What will you be hauling?", ["haul_item"]), rewrites)


def test_off_topic_with_nothing_left_to_ask_may_just_stop():
    state = _off_topic(_state(slots={"haul_item": "gravel", "payload_capacity": 5000}))
    reply = "Sorry, I can only help with trailers and TrailerPlace here."
    assert _sent_as_written(state, _output(reply, [], covers=["declined_off_topic"]))


# ---- a type we do not stock ----


@pytest.fixture
def stocked(monkeypatch):
    from src.tools import unavailable

    monkeypatch.setattr(unavailable, "_stocked", lambda: ("Utility", "Enclosed", "Car Hauler", "Dump", "Livestock"))


def _unavailable(status, **overrides):
    state = _state(**overrides)
    state["turn_outcome"]["unavailable_type"] = {"type": "boat trailer", "status": status}
    return state


def test_a_boat_trailer_reply_with_the_team_told_goes_out(stocked):
    reply = "Sorry - we don't carry boat trailers. A Utility trailer may work; our team has your request."
    output = _output(reply, [], questions=0, covers=["said_not_stocked", "passed_to_team"], offered=["Utility"])
    assert _sent_as_written(_unavailable("sent"), output)


def test_a_reply_that_does_not_say_we_do_not_stock_it_is_turned_down(stocked, rewrites):
    output = _output("We can order one for you!", [], covers=["passed_to_team"], offered=["Utility"])
    assert _turned_down(_unavailable("sent"), output, rewrites)


def test_a_reply_that_offers_nothing_we_carry_is_turned_down(stocked, rewrites):
    output = _output("Sorry, we don't carry them.", [], covers=["said_not_stocked", "passed_to_team"])
    assert _turned_down(_unavailable("sent"), output, rewrites)


def test_offering_a_type_we_do_not_stock_is_turned_down(stocked, rewrites):
    output = _output("Sorry, none. Try a Concession trailer.", [],
                     covers=["said_not_stocked", "passed_to_team"], offered=["Concession"])
    assert _turned_down(_unavailable("sent"), output, rewrites)


def test_needing_their_details_the_reply_must_ask_for_them_and_nothing_else(stocked, rewrites):
    covers, offered = ["said_not_stocked"], ["Utility"]
    assert _turned_down(_unavailable("stashed", contact={}),
                        _output("Sorry, we don't carry them. A Utility may work.", [], covers=covers, offered=offered),
                        rewrites)
    assert _turned_down(_unavailable("stashed", contact={}),
                        _output("Sorry. Would a Utility work? Your name and number?", [], questions=1,
                                contact=True, covers=covers, offered=offered),
                        rewrites)
    assert _sent_as_written(_unavailable("stashed", contact={}),
                            _output("Sorry. A Utility may work. Your name and number?", [], contact=True,
                                    covers=covers, offered=offered))


def test_declined_contact_gets_the_phone_number(stocked, rewrites):
    covers, offered = ["said_not_stocked"], ["Livestock"]
    assert _turned_down(_unavailable("dropped", contact={"declined": True}),
                        _output("Sorry, no horse trailers. Livestock may work.", [], covers=covers, offered=offered),
                        rewrites)
    assert _sent_as_written(_unavailable("dropped", contact={"declined": True}),
                            _output("Sorry, no horse trailers. Livestock may work; call 979-532-1486.", [],
                                    covers=covers + ["gave_phone"], offered=offered))


def test_no_qualification_question_on_an_unavailable_turn(stocked, rewrites):
    output = _output("Sorry, none. Utility may work. What will you haul?", ["haul_item"],
                     covers=["said_not_stocked", "passed_to_team"], offered=["Utility"])
    assert _turned_down(_unavailable("sent"), output, rewrites)


# ---- the first message ----


def _first_turn_with_contact():
    return _state(turn_index=1, category=None, required_slots=[], slots={},
                  contact={"name": "Tony Stephens", "phone": "806-555-0199"})


def test_a_first_reply_with_the_welcome_and_their_name_goes_out():
    state = _first_turn_with_contact()
    reply = "Hi Tony, thank you for contacting TrailerPlace! Which type of trailer fits what you need?"
    assert _sent_as_written(state, _output(reply, [], questions=1,
                                           covers=["welcome", "thanked_for_contacting", "greeted_by_name"]))
    assert state["contact"]["greeted"] is True


def test_the_first_reply_is_worded_by_the_model_with_no_fixed_line():
    """The welcome is up to the model: no script is required to open the first reply."""
    reply = "Hi Tony! Thank you for contacting TrailerPlace - glad you found us. Which type fits you?"
    assert _sent_as_written(_first_turn_with_contact(), _output(
        reply, [], questions=1, covers=["thanked_for_contacting", "greeted_by_name"]))


def test_the_state_block_no_longer_scripts_the_first_reply():
    from src import config
    from src.graph.nodes import greeting
    from src.llm.prompt import state_block

    assert config.settings.llm_writes_reply
    block = state_block({"turn_index": 1, "contact": {}, "slots": {}})
    assert greeting.OPENING not in block
    assert "own words" in block


def test_their_own_first_name_is_not_taken_for_a_guess():
    state = {"contact": {"name": "Tony Stephens"}, "session_id": "s1"}
    assert compose._no_invented_name(state, "Thanks, Tony - great to hear from you.") == (
        "Thanks, Tony - great to hear from you."
    )


def test_a_name_they_never_gave_is_still_taken_out():
    state = {"contact": {"name": "Tony Stephens"}, "session_id": "s1"}
    assert compose._no_invented_name(state, "Thanks, Ibrahim - great to hear from you.") == (
        "Thanks - great to hear from you."
    )


# ---- the contact policy: four moments, no others ----


def test_mid_flow_with_no_moment_contact_is_never_asked(rewrites):
    """No moment, no ask - even with nothing on file and nothing asked before."""
    output = _output("What will you be hauling? And your name and number?", ["haul_item"], contact=True)
    assert _turned_down(_state(contact={}), output, rewrites)
    assert _sent_as_written(_state(contact={}), _output("What will you be hauling?", ["haul_item"]))


def test_a_first_message_about_trailers_is_not_asked_for_contact(rewrites):
    state = _state(turn_index=1, contact={})
    reply = "Thank you for contacting TrailerPlace! A dump trailer it is. What material will you be hauling?"
    output = _output(reply, ["haul_item"], covers=["welcome", "thanked_for_contacting"], about_trailers=True)
    assert _sent_as_written(state, output)


def test_a_first_message_with_nothing_about_trailers_is_asked(rewrites):
    state = _state(turn_index=1, category=None, required_slots=[], slots={}, contact={})
    bare = "Hi there, thank you for contacting TrailerPlace! What kind of trailer can I help you find?"
    opening = ["welcome", "thanked_for_contacting"]
    assert _turned_down(state, _output(bare, [], questions=1, covers=opening, about_trailers=False), rewrites)

    state = _state(turn_index=1, category=None, required_slots=[], slots={}, contact={})
    asked = bare + " And could I get your name and an email or phone?"
    assert _sent_as_written(state, _output(asked, [], questions=1, contact=True, covers=opening,
                                           about_trailers=False))


def test_a_customer_who_declined_is_still_asked_at_a_moment(rewrites):
    state = _state(contact={"declined": True})
    reply = "We offer financing - call 979-532-1486. What will you be hauling? Your name and number?"
    assert _sent_as_written(state, _output(reply, ["haul_item"], contact=True, faq_key="financing"))


def test_the_rewrite_is_told_which_details_are_missing(rewrites):
    state = _state(contact={"name": "Dave"})
    _send(state, _output("We offer financing. What will you be hauling?", ["haul_item"], faq_key="financing"))

    assert "an email or phone number" in rewrites.calls[0]["needs"]
    assert "their name" not in rewrites.calls[0]["needs"]


# ---- the opening: "Thank you for contacting TrailerPlace", and "Hi <name>" when given ----


def test_a_first_reply_without_thank_you_for_contacting_is_turned_down(rewrites):
    reply = "Hi Tony, welcome! Which type of trailer fits what you need?"
    assert _turned_down(_first_turn_with_contact(),
                        _output(reply, [], questions=1, covers=["welcome", "greeted_by_name"]), rewrites)
    assert "Thank you for contacting TrailerPlace" in rewrites.calls[0]["problem"]


def test_a_first_reply_to_someone_who_gave_their_name_must_open_with_it(rewrites):
    reply = "Thank you for contacting TrailerPlace! Which type of trailer fits what you need?"
    assert _turned_down(_first_turn_with_contact(),
                        _output(reply, [], questions=1, covers=["thanked_for_contacting"]), rewrites)
    assert 'open with "Hi Tony,"' in rewrites.calls[0]["needs"]


def test_no_name_given_no_name_needed():
    state = _state(turn_index=1, category=None, required_slots=[], slots={}, contact={})
    reply = "Thank you for contacting TrailerPlace! Which type of trailer fits what you need?"
    assert _sent_as_written(state, _output(reply, [], questions=1, covers=["thanked_for_contacting"],
                                           about_trailers=True))


def test_later_turns_need_no_opening():
    assert _sent_as_written(_state(), _output("Got it. What will you be hauling?", ["haul_item"]))


def test_the_reply_to_a_refusal_does_not_ask_again(rewrites):
    """Asked again at the next moment, never in the answer to "I'd rather not"."""
    from src.llm.schemas import ContactInfo

    state = _state(turn_index=1, category=None, required_slots=[], slots={}, contact={"declined": True})
    refusal = ContactInfo(name=None, email=None, phone=None, declined=True)
    reply = "Thank you for contacting TrailerPlace. No problem at all. Which type of trailer do you need?"
    assert _sent_as_written(state, _output(reply, [], questions=1, covers=["thanked_for_contacting"],
                                           about_trailers=False, contact_info=refusal))


# ---- a question waiting on them: nothing more is asked until they answer or skip ----


def _holding(**overrides):
    state = _state(pending_slot="haul_item", asked_counts={"haul_item": 1}, **overrides)
    state["turn_outcome"]["holding"] = "haul_item"
    return state


def test_ok_thanks_gets_a_short_reply_and_no_question():
    reply = "You're welcome! Let me know if you have any other questions."
    output = _output(reply, [], covers=["invited_questions"], only_acknowledges=True)
    state = _holding()
    assert _sent_as_written(state, output)
    assert state["asked_counts"] == {"haul_item": 1}  # not asked again, not counted again


def test_asking_anything_while_a_question_waits_is_turned_down(rewrites):
    output = _output("You're welcome! What will you be hauling?", ["haul_item"],
                     covers=["invited_questions"], only_acknowledges=True)
    assert _turned_down(_holding(), output, rewrites)
    assert "still waiting" in rewrites.calls[0]["problem"]


def test_ok_thanks_without_inviting_questions_is_turned_down(rewrites):
    assert _turned_down(_holding(), _output("You're welcome.", [], only_acknowledges=True), rewrites)


def test_a_question_of_theirs_is_answered_with_nothing_asked_after_it():
    reply = "Yes, we offer financing - call 979-532-1486 to talk to our finance team."
    assert _sent_as_written(_holding(), _output(reply, [], faq_key="financing"))


def test_holding_and_a_moment_still_asks_for_contact():
    reply = "We offer financing - call 979-532-1486. Could I get your name and a phone or email?"
    assert _sent_as_written(_holding(contact={}), _output(reply, [], contact=True, faq_key="financing"))


def test_apply_keeps_the_question_open_on_ok_thanks(monkeypatch):
    """The ask is not spent and the slot is not given up on, even at its second ask."""
    from src import config
    from src.graph.nodes import apply as apply_module
    from dataclasses import replace as _replace

    monkeypatch.setattr(config, "settings", _replace(config.settings, llm_writes_reply=True))
    state = _state(pending_slot="haul_item", asked_counts={"haul_item": 2})
    output = _output("You're welcome!", [], only_acknowledges=True, intent="smalltalk_other")
    apply_module._apply_attempts(state, output, type("R", (), {"stored": {}, "no_preference": []})())

    assert state["pending_slot"] == "haul_item"
    assert "haul_item" not in state["declined_slots"]
    assert state["turn_outcome"]["holding"] == "haul_item"
