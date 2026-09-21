"""Saying the same thing twice in one reply.

Every case here was produced by the live bot, not invented. The acknowledgement field is
meant to react to what the customer said; once the prompt insisted a reply must respond
before it asks for anything, the model started writing one even when the answer beside it
already did the job, and both went out.
"""
from __future__ import annotations

import pytest

from src.graph.nodes.compose import _restates


@pytest.mark.parametrize(
    "candidate, established",
    [
        # Live, turn 3: "do you offer financing?"
        (
            "Yes, we do offer financing.",
            "We offer financing. Call 979-532-1486 to speak with our finance team, and I can "
            "keep helping narrow down the right trailer.",
        ),
        # Live, turn 5: "what are your opening hours?"
        (
            "We're open from 8:00 AM to 6:00 PM.",
            "We're located in Wharton, TX and open 8:00 AM to 6:00 PM. Call 979-532-1486 or "
            "visit https://trailerplace.com.",
        ),
        # Live, turn 4: the welcome, where "info" and "information" are the same word.
        (
            "Thanks, Ibrahim - I've got your contact information.",
            "Great to have your contact info, Ibrahim!",
        ),
    ],
)
def test_a_restatement_is_recognised(candidate, established):
    assert _restates(candidate, established)


@pytest.mark.parametrize(
    "candidate, established",
    [
        # Shares the topic and nothing else. Dropping this would lose the apology.
        (
            "I'm sorry your trailer arrived damaged.",
            "Our team will reach out about the trailer as soon as I know how to contact you.",
        ),
        # A real second point: the answer says nothing about delivery.
        (
            "We can deliver to most of Texas.",
            "We offer financing. Call 979-532-1486 to speak with our finance team.",
        ),
        # Their name is not a restatement of anything.
        ("Thanks, Ibrahim!", "We're open 8:00 AM to 6:00 PM."),
    ],
)
def test_a_second_point_is_kept(candidate, established):
    assert not _restates(candidate, established)


def test_an_empty_line_restates_nothing():
    assert not _restates("", "We offer financing.")
    assert not _restates("We offer financing.", "")
    assert not _restates("Thanks!", "Sure, okay.")


# ------------------------------------------------------------ the handoff confirmation
# The pattern that spots "the agent already said it" was written with a backspace character
# where its word boundary belonged, so it matched nothing and the confirmation was appended to
# every reply that already carried one.
@pytest.mark.parametrize(
    "reply",
    [
        "Thanks, Dave - I've passed it on to our team.",
        "I've passed your request to the team.",
        "Our team will reach out shortly.",
        "I've notified our team about the damaged trailer.",
        "I've logged your interest in that trailer.",
    ],
)
def test_a_reply_that_already_confirms_the_handoff_is_left_alone(reply):
    from src.graph.nodes.compose import _with_handoff

    assert _with_handoff({"emails_flushed": 1}, reply) == reply


def test_a_reply_that_does_not_mention_it_gets_the_confirmation():
    from src.graph.nodes.compose import _with_handoff

    text = _with_handoff({"emails_flushed": 2}, "What type of trailer are you after?")
    assert "I've passed your requests on to our team" in text


def test_nothing_is_confirmed_when_nothing_went_out():
    from src.graph.nodes.compose import _with_handoff

    assert _with_handoff({}, "Hello.") == "Hello."


# ------------------------------------------------------------ a question asked twice
from src.graph.nodes.compose import _repeats  # noqa: E402

# Live, Tilt: the answer re-asked with a curly apostrophe, the closing had a straight one.
_TILT_ANSWER = (
    "It helps us match the trailer’s capacity to your load so we don’t recommend one that’s "
    "under-rated. If you’re not sure, that’s okay—what’s the approximate weight of the load?"
)


def test_the_same_question_with_different_apostrophes_is_one_question():
    assert _repeats([_TILT_ANSWER], "What's the approximate weight of the load?", "payload_capacity")


def test_typography_alone_is_enough_even_without_a_slot():
    assert _repeats([_TILT_ANSWER], "What's the approximate weight of the load?")


@pytest.mark.parametrize("reworded, slot, closing", [
    ("No problem - what's the rough weight?", "payload_capacity", "What's the approximate weight of the load?"),
    ("Roughly how heavy is a full load?", "payload_capacity", "What's the rough haul weight per load?"),
    ("And how long is the longest piece?", "length", "About how long is the load (or what deck length do you need)?"),
    ("Would a gooseneck or bumper pull suit you?", "hitch_type", "Do you prefer a bumper pull or gooseneck hitch?"),
    ("So what are you planning to haul?", "haul_item", "What will you be hauling on the flatbed?"),
])
def test_a_reworded_re_ask_is_caught(reworded, slot, closing):
    assert _repeats([f"Good question. {reworded}"], closing, slot)


@pytest.mark.parametrize("parts, slot, closing", [
    # A statement about weight is not a question about it.
    (["Gravel is heavy, so payload matters."], "payload_capacity", "What's the rough haul weight per load?"),
    # A question about something else leaves ours standing.
    (["Would you like our financing number?"], "payload_capacity", "What's the rough haul weight per load?"),
    (["What will you be hauling?"], "length", "What length trailer are you looking for?"),
])
def test_a_different_question_is_kept(parts, slot, closing):
    assert not _repeats(parts, closing, slot)


# ----------------------------------------------- the model's own questions, kept in line
from src.graph.nodes.compose import _up_to_the_question_mark, _without_other_questions  # noqa: E402


@pytest.mark.parametrize("proposed, sent", [
    # Both live.
    ("What vehicle will you be hauling? צור", "What vehicle will you be hauling?"),
    ("What's the approximate weight of the vehicle? (This question is already included in the answer.)",
     "What's the approximate weight of the vehicle?"),
    ("What length do you need?", "What length do you need?"),
    ("", ""),
    (None, ""),
])
def test_a_proposed_question_ends_at_its_question_mark(proposed, sent):
    assert _up_to_the_question_mark(proposed) == sent


def test_a_question_about_another_slot_is_dropped_from_the_answer():
    """Live, Car Hauler: the answer asked the length while the closing asked the weight."""
    answer = ("The vehicle’s weight helps us narrow down a car hauler with enough payload "
              "capacity. About how long is the vehicle?")
    assert _without_other_questions(answer, "payload_capacity") == (
        "The vehicle’s weight helps us narrow down a car hauler with enough payload capacity."
    )


@pytest.mark.parametrize("text, slot", [
    ("What's the rough weight of what you'll haul?", "payload_capacity"),  # about the asked slot too
    ("Which one fits what you need?", "payload_capacity"),                   # about no slot
    ("About how long is the vehicle?", None),                                # nothing being asked
    ("About how long is the vehicle?", "dump_mechanism"),                    # a rules-panel slot
])
def test_questions_that_are_not_out_of_turn_are_kept(text, slot):
    assert _without_other_questions(text, slot) == text


# ------------------------------------------- the category question, which has no slot
# Live, turn 2 of the long conversation, "what sort of trailers do you carry?":
#   "We carry Utility, Enclosed, Equipment, Dump, Flatbed and many more. ...
#    What type of trailer fits what you need? What type of trailer are you looking for?
#    We have Utility, Enclosed, Equipment, Tilt, Livestock, Flatbed and many more -
#    which one fits what you need?"
# The closing comes back with no slot, so _repeats compared a fourteen-word canned menu
# against a six-word question by word overlap and called them different questions.
_MENU = (
    "What type of trailer are you looking for? We have Utility, Enclosed, Equipment, Tilt, "
    "Livestock, Flatbed and many more - which one fits what you need?"
)
_CARRIED = (
    "We carry Utility, Enclosed, Equipment, Dump, Flatbed and many more. Utility trailers are "
    "open general-purpose haulers, Enclosed trailers are lockable and weatherproof. "
    "What type of trailer fits what you need?"
)


@pytest.mark.parametrize("already_asked", [
    "What type of trailer fits what you need?",
    "What kind of trailer are you after?",
    "So what type of trailer would suit you best?",
    # Live, after the first fix: "Yes - we carry Aluminum, Car Hauler, ... and Roll Off
    # trailers. Which type fits what you need? What type of trailer are you looking for?
    # We have..." The phrase never says "trailer", so the narrow topic list missed it.
    "Which type fits what you need?",
    "Which kind suits you best?",
    "What type would work best for you?",
])
def test_the_category_menu_is_dropped_when_the_reply_already_asked(already_asked):
    assert _repeats([already_asked], _MENU, "base_category")


def test_the_whole_live_reply_is_caught():
    assert _repeats([_CARRIED], _MENU, "base_category")


@pytest.mark.parametrize("other", [
    "Would you like our financing number?",
    "What will you be hauling?",
    "Can I take your name and number?",
])
def test_a_question_about_anything_else_leaves_the_menu_standing(other):
    assert not _repeats([other], _MENU, "base_category")


def test_a_statement_naming_the_types_is_not_a_question_about_them():
    """Listing what we carry is an answer. The question still has to be asked."""
    listed = "We carry Utility, Enclosed, Equipment, Dump and Flatbed trailers, among many others."
    assert not _repeats([listed], _MENU, "base_category")


# ---------------------------------------- a question the model wrote twice by itself
# _repeats compares the CLOSING against the rest of the reply, so a question repeated inside
# the model's own fields walked straight past it. Live, after the category-menu fix:
#   "We carry Utility, Enclosed, ... What will you be hauling? What will you be hauling? Why?"
from src.graph.nodes.compose import _without_a_repeated_question  # noqa: E402


def test_the_same_question_twice_becomes_one():
    text = "We carry Utility and Dump trailers. What will you be hauling? What will you be hauling?"
    assert _without_a_repeated_question(text).count("What will you be hauling?") == 1


def test_typography_does_not_make_it_a_different_question():
    text = "What’s the rough weight? What's the rough weight?"
    assert _without_a_repeated_question(text).count("?") == 1


def test_two_different_questions_both_stay():
    text = "What will you be hauling? How long does it need to be?"
    assert _without_a_repeated_question(text) == text


def test_the_category_menu_is_left_alone():
    """It is deliberately one question written as two sentences."""
    menu = (
        "What type of trailer are you looking for? We have Utility, Enclosed, Equipment and "
        "many more - which one fits what you need?"
    )
    assert _without_a_repeated_question(menu) == menu


def test_statements_are_never_touched():
    text = "We carry Dump trailers. We carry Dump trailers."
    assert _without_a_repeated_question(text) == text, "only questions are de-duplicated"
