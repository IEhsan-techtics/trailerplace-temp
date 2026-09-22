"""Never open a reply on a listing card.

On the web a missing lead-in is a cosmetic wart. On Messenger it is worse: the reply is
split on the cards and each one is sent as its own bubble, so a reply that starts at "1."
reaches the customer as a stack of trailers with no sentence in front of any of them.

The model writes the line (BEFORE THE FIRST CARD in respond._CARD_FORMAT) because it can
tie it to what the customer actually asked for. These tests cover the backstop for when it
does not.
"""
from __future__ import annotations

import pytest

from src.domain.reply_chunks import first_line_is_a_card, split_reply_into_chunks
from src.llm.respond import LEAD_IN, _with_a_lead_in

NL = chr(10)
CARD = "1. [2026 Iron Bull DTB - 15081](https://www.trailerplace.com/inventory/x/)"
BODY = CARD + NL + "   - Category: Dump" + NL + "   - Price: $9,995"


@pytest.mark.parametrize(
    "text, expected",
    [
        (CARD, True),
        (NL * 2 + CARD, True),                      # leading blank lines do not count as prose
        ("   " + NL + CARD, True),                  # nor does whitespace
        ("Here's what we have:" + NL * 2 + CARD, False),
        ("We don't have that right now.", False),
        ("", False),
        ("- Category: Dump", False),                # a bullet is not a card start
    ],
)
def test_spotting_a_reply_that_opens_on_a_card(text, expected):
    assert first_line_is_a_card(text) is expected


def test_a_reply_that_opens_on_a_card_gets_the_standard_lead_in():
    out = _with_a_lead_in(BODY)

    assert out.startswith(LEAD_IN)
    assert BODY in out


def test_the_lead_in_becomes_its_own_message():
    """A blank line, not just a newline: the chunker splits on paragraphs, so without the
    gap the lead-in would ride along inside the first trailer's bubble."""
    chunks = split_reply_into_chunks(_with_a_lead_in(BODY))

    assert chunks[0] == LEAD_IN
    assert chunks[1].startswith("1. [")


def test_the_model_s_own_line_is_left_alone():
    """It is better than ours every time - it knows what they asked for."""
    written = "I found a few that should work for the mulch:" + NL * 2 + BODY

    assert _with_a_lead_in(written) == written
    assert LEAD_IN not in _with_a_lead_in(written)


def test_a_reply_with_no_cards_is_untouched():
    plain = "We're open 8 to 6, and the team is on 979-532-1486. What will you be hauling?"

    assert _with_a_lead_in(plain) == plain
