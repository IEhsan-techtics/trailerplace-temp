"""The list of what we carry, said once per reply.

Live, a first reply came out as:

    Thank you for contacting TrailerPlace... Thanks, Ibrahim. What type of trailer are you
    looking for? We have Car Hauler, Equipment, Enclosed, Utility, and Dump and many more.
    We have Car Hauler, Equipment, Enclosed, Utility, and Dump and many more - which one
    fits what you need?

The model had written the menu into the acknowledgement AND into next_question_text. The
existing duplicate check compares wording, and the two phrasings were different enough to
slip past it - so this one compares the categories NAMED, which is what makes them the same
sentence.
"""
from __future__ import annotations

from src.graph.nodes.compose import _both_recite_the_menu, _category_names_in


def test_it_reads_the_categories_out_of_a_sentence():
    named = _category_names_in("We have Car Hauler, Equipment, Enclosed, Utility and Dump")

    assert named == {"Car Hauler", "Equipment", "Enclosed", "Utility", "Dump"}


def test_the_live_duplicate_is_caught():
    first = "What type of trailer are you looking for? We have Car Hauler, Equipment, Enclosed, Utility, and Dump and many more."
    second = "We have Car Hauler, Equipment, Enclosed, Utility, and Dump and many more - which one fits what you need?"

    assert _both_recite_the_menu(first, second) is True


def test_two_categories_in_common_is_not_the_menu():
    """"Would a Dump or an Equipment trailer suit you better?" beside a question naming the
    same two is a comparison, not the catalogue read out twice."""
    first = "A Dump trailer or an Equipment trailer would both work for that."
    second = "Would you like to look at Dump or Equipment trailers?"

    assert _both_recite_the_menu(first, second) is False


def test_an_acknowledgement_about_something_else_is_kept():
    first = "Thanks, Ibrahim - I've got your number."
    second = "We have Utility, Enclosed, Equipment, Dump, Flatbed and many more - which fits?"

    assert _both_recite_the_menu(first, second) is False


def test_an_empty_part_is_never_a_duplicate():
    assert _both_recite_the_menu("", "We have Utility, Enclosed, Equipment, Dump") is False


def test_dropping_the_repeated_closing_leaves_a_question_behind():
    """"We sell Aluminum, Car Hauler, Equipment, Enclosed and Utility trailers, and many
    more." is true, complete, and a dead end - the conversation stops on our full stop."""
    from src.graph.nodes.compose import WHICH_ONE, _asks_something

    answered = ["We sell Aluminum, Car Hauler, Equipment, Enclosed and Utility trailers."]
    assert _asks_something(answered) is False
    assert WHICH_ONE.endswith("?")

    already = ["We carry Dump and Utility. What are you hauling?"]
    assert _asks_something(already) is True


def test_the_acknowledgement_never_carries_a_question():
    """Its contract is "one short sentence, no question", and Python picks the one question
    a reply asks. Live: "Thanks, Ibrahim - what type of trailer are you looking for?" beside
    a closing asking exactly that."""
    from src.graph.nodes.compose import _without_questions

    kept = _without_questions("Thanks, Ibrahim. What type of trailer are you looking for?")
    assert kept == "Thanks, Ibrahim."

    assert _without_questions("Thanks, Ibrahim.") == "Thanks, Ibrahim.", "left alone"


def test_an_acknowledgement_that_is_only_a_question_is_left_alone():
    """Emptying it would lose the whole part, and on a turn with no closing that question
    may be all the reply has."""
    from src.graph.nodes.compose import _without_questions

    assert _without_questions("What will you be hauling?") == "What will you be hauling?"
