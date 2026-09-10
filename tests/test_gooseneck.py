"""Gooseneck is both a hitch type and a make we stock. Which did they mean?"""
from __future__ import annotations

import pytest

from src.domain.gooseneck import (
    AMBIGUOUS,
    BRAND,
    HITCH,
    apply_clarification_answer,
    clarification_question,
    mentions_gooseneck,
    resolve_gooseneck_mention,
)

# The conftest stub has no Gooseneck make, so tests that need the ambiguity say so.
MAKES_WITH_GOOSENECK = ("Gooseneck", "Diamond C", "Iron Bull Trailers", "Aluma", "PJ Trailers")


def resolve(text, **kwargs):
    kwargs.setdefault("known_makes", MAKES_WITH_GOOSENECK)
    return resolve_gooseneck_mention(text, **kwargs)


@pytest.mark.parametrize(
    "text", ["gooseneck", "goose neck", "goose-neck", "GOOSENECK", "a Gooseneck trailer"]
)
def test_the_word_is_recognised_however_it_is_spelled(text):
    assert mentions_gooseneck(text) is True


def test_text_without_it_is_not_flagged():
    assert mentions_gooseneck("bumper pull please") is False


# ---------------------------------------------------- 1. answering the hitch question
def test_gooseneck_answering_the_hitch_question_is_the_hitch():
    reading = resolve("gooseneck", pending_slot="hitch_type")
    assert reading.meaning == HITCH
    assert reading.reason == "answering_hitch_question"
    assert not reading.needs_clarification


def test_even_a_bare_gooseneck_is_unambiguous_when_the_hitch_was_asked():
    """Without the pending question this same text would be ambiguous."""
    assert resolve("gooseneck").meaning == AMBIGUOUS
    assert resolve("gooseneck", pending_slot="hitch_type").meaning == HITCH


# ------------------------------------------------------------- 2. the word "hitch" nearby
@pytest.mark.parametrize(
    "text",
    [
        "I want a gooseneck hitch",
        "gooseneck coupler please",
        "does it come with a goose neck hitch?",
        "a hitch gooseneck setup",
        "gooseneck style",
    ],
)
def test_a_coupling_word_settles_it_as_the_hitch(text):
    reading = resolve(text)
    assert reading.meaning == HITCH


# ------------------------------------------------------------- 3. named as a manufacturer
@pytest.mark.parametrize(
    "text",
    [
        "I want the Gooseneck brand",
        "do you carry the gooseneck make?",
        "a trailer made by Gooseneck",
        "is it manufactured by gooseneck?",
    ],
)
def test_a_manufacturer_word_settles_it_as_the_brand(text):
    reading = resolve(text)
    assert reading.meaning == BRAND
    assert reading.reason == "brand_word_nearby"


# ------------------------------------------------------- 4. another make in the message
def test_another_make_named_means_gooseneck_is_the_hitch():
    """"a gooseneck Diamond C" - Diamond C is who built it, gooseneck is how it attaches.

    The make comes from the MODEL, which has the full list in its prompt and copes with the
    misspellings customers actually type. Reading it off the text here required fuzzy
    matching, which read the word "trailer" as the make Load Trail.
    """
    reading = resolve("looking for a gooseneck Diamond C", extracted_brand="Diamond C")
    assert reading.meaning == HITCH
    assert reading.reason == "another_make_named"
    assert reading.other_brand == "Diamond C"


def test_the_other_brand_is_reported_so_it_can_be_stored():
    reading = resolve("a gooseneck iron bull dump", extracted_brand="Iron Bull Trailers")
    assert reading.other_brand == "Iron Bull Trailers"


def test_the_model_naming_gooseneck_itself_is_not_another_make():
    """It must not read its own name as "some other brand was mentioned"."""
    assert resolve("the gooseneck trailer", extracted_brand="Gooseneck").meaning == AMBIGUOUS


# ----------------------------------------------------------------- 5. genuinely ambiguous
@pytest.mark.parametrize(
    "text", ["I want the gooseneck trailer", "show me gooseneck", "gooseneck ones"]
)
def test_a_bare_mention_is_ambiguous_and_asks(text):
    reading = resolve(text)
    assert reading.meaning == AMBIGUOUS
    assert reading.needs_clarification is True


def test_the_clarification_question_offers_both_readings_plainly():
    question = clarification_question()
    assert "hitch" in question.lower()
    assert "brand" in question.lower()
    assert len(question) < 160


def test_there_is_no_ambiguity_when_we_do_not_stock_the_make():
    """The ambiguity is a property of the inventory. Sell the last one and it disappears."""
    reading = resolve_gooseneck_mention(
        "I want the gooseneck trailer", known_makes=("Diamond C", "Aluma")
    )
    assert reading.meaning == HITCH
    assert reading.reason == "make_not_stocked"


def test_no_mention_at_all_resolves_without_asking():
    assert resolve("bumper pull is fine").reason == "not_mentioned"


# ------------------------------------------------------------ reading their clarification
@pytest.mark.parametrize(
    "text", ["the hitch", "hitch please", "I meant the coupler", "how it tows"]
)
def test_a_clarification_naming_the_hitch_is_read_as_hitch(text):
    assert apply_clarification_answer(text) == HITCH


@pytest.mark.parametrize(
    "text", ["the brand", "the make", "Gooseneck the manufacturer", "the company"]
)
def test_a_clarification_naming_the_brand_is_read_as_brand(text):
    assert apply_clarification_answer(text) == BRAND


def test_brand_wins_when_they_name_both_to_contrast_them():
    """"the brand, not the hitch" mentions both words but says only one thing."""
    assert apply_clarification_answer("the brand, not the hitch") == BRAND


@pytest.mark.parametrize("text", ["", "   ", "yes", "sure", "whatever"])
def test_an_unclear_clarification_returns_none(text):
    assert apply_clarification_answer(text) is None


# --------------------------------------------------- the fuzzy-brand-match landmine
@pytest.mark.parametrize(
    "text", ["I want the gooseneck trailer", "a gooseneck trailer please", "gooseneck units"]
)
def test_the_word_trailer_is_not_read_as_a_brand(text):
    """No brand is invented from the text itself. Only the model names a make, and here it
    named none - so the mention stays ambiguous and gets asked about."""
    reading = resolve(text)
    assert reading.other_brand is None
    assert reading.meaning == AMBIGUOUS
