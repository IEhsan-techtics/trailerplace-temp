"""One reply, two channels: the website renders the markdown, Messenger cannot.

The reply Luna writes is the same either way. What differs is what the channel can draw -
the web chat shows the markdown card and puts the trailer's own picture card beside it,
while Messenger has to be handed an actual generic template, because a markdown link
arrives there as literal square brackets.
"""
from __future__ import annotations

from src.domain import cards

DUMP = {
    "title": "2026 Diamond C 14' Dump Trailer - 15081",
    "url": "https://www.trailerplace.com/inventory/2026-diamond-c-dump-15081/",
    "price_display": "$12,500",
    "length": "14 ft 0 in",
    "category": "Dump",
    "image_url": "https://images.trailerplace.com/15081.jpg",
}
TILT = {
    "title": "2025 PJ 20' Tilt",
    "url": "https://www.trailerplace.com/inventory/2025-pj-tilt-15082/",
    "price_display": "$9,995",
    "category": "Tilt",
}

REPLY = "\n\n".join([
    "Here are two that fit what you described:",
    f"1. [{DUMP['title']}]({DUMP['url']})\n   - Price: $12,500\n   - Plenty of capacity for gravel.",
    f"2. [{TILT['title']}]({TILT['url']})\n   - Price: $9,995",
    "Do either of these look right, or shall I pull a few more?",
])


# ---------------------------------------------------------------------------- the website
def test_each_trailer_sits_under_its_own_bubble():
    bubbles = cards.website_bubbles(REPLY, [DUMP, TILT])

    assert [len(bubble["listings"]) for bubble in bubbles] == [0, 1, 1, 0]
    assert bubbles[1]["listings"][0]["listing"] is DUMP
    assert bubbles[1]["listings"][0]["rank"] == 1, "the number the reply calls it by"
    assert bubbles[2]["listings"][0]["rank"] == 2
    assert bubbles[0]["text"].startswith("Here are two")
    assert "[" not in bubbles[3]["text"], "the closing question carries no card"


def test_a_trailer_named_twice_still_gets_exactly_one_card():
    reply = f"{REPLY}\n\nThat first one - [{DUMP['title']}]({DUMP['url']}) - is the popular pick."
    bubbles = cards.website_bubbles(reply, [DUMP, TILT])

    assert sum(len(bubble["listings"]) for bubble in bubbles) == 2, "two trailers, two cards"


def test_a_trailer_the_reply_never_mentioned_is_still_shown():
    """The search found it; it is never silently dropped just because the prose skipped it."""
    bubbles = cards.website_bubbles(REPLY, [DUMP, TILT, {"title": "Third", "url": "https://x/3"}])

    leftover = bubbles[-1].get("leftover")
    assert [entry["rank"] for entry in leftover] == [3]


def test_a_reply_with_no_cards_at_all_shows_the_listings_after_it():
    bubbles = cards.website_bubbles("We have a few of those.", [DUMP])

    assert bubbles[0]["listings"] == []
    assert bubbles[0]["leftover"][0]["listing"] is DUMP


def test_the_markdown_itself_is_left_alone():
    """The web chat renders it; taking it apart would only have to be put back together."""
    bubbles = cards.website_bubbles(REPLY, [DUMP, TILT])
    assert f"[{DUMP['title']}]({DUMP['url']})" in bubbles[1]["text"]


# -------------------------------------------------------------------------- Messenger
def test_a_trailer_becomes_a_card_and_the_prose_stays_text():
    sends = cards.messenger_sends(REPLY, [DUMP, TILT])
    kinds = [kind for kind, _ in sends]

    assert kinds == ["text", "card", "text", "card", "text", "text"], (
        "intro, card + its bullets, card + its bullets, closing question"
    )
    _, element = sends[1]
    assert element["title"] == DUMP["title"]
    assert element["image_url"] == DUMP["image_url"]
    assert element["buttons"][0] == {
        "type": "web_url", "url": DUMP["url"], "title": "View Trailer"
    }
    assert element["default_action"]["url"] == DUMP["url"]


def test_no_markdown_reaches_the_customer():
    """Sent as written, "[title](url)" arrives as those literal characters."""
    for kind, payload in cards.messenger_sends(REPLY, [DUMP, TILT]):
        if kind == "text":
            assert "](" not in payload


def test_the_card_subtitle_leads_with_the_price():
    assert cards.card_subtitle(DUMP) == "$12,500 · 14 ft 0 in · Dump"


def test_a_listing_with_no_photo_still_sends():
    """Meta fetches image_url server-side and rejects the element if it cannot."""
    element = cards.card_element(TILT)

    assert "image_url" not in element, "omitted, never sent empty"
    assert element["title"] and element["buttons"], "still a card"


def test_an_overlong_title_is_clipped_where_we_choose():
    element = cards.card_element({"title": "x" * 200, "url": "https://x/1"})

    assert len(element["title"]) == cards.CARD_TITLE_MAX
    assert element["title"].endswith("…")


def test_with_cards_switched_off_the_trailer_still_reaches_them():
    sends = cards.messenger_sends(REPLY, [DUMP, TILT], cards_enabled=False)
    kinds = [kind for kind, _ in sends]
    texts = [payload for kind, payload in sends if kind == "text"]

    assert "card" not in kinds
    assert DUMP["url"] in texts, "a bare, tappable link of its own"


def test_a_url_the_turn_did_not_present_falls_back_to_plain_bubbles():
    """Cards print database values, so a card is only built from a row we actually have."""
    sends = cards.messenger_sends(REPLY, [TILT])
    kinds = [kind for kind, _ in sends]

    assert kinds.count("card") == 1, "only the trailer we hold a row for"
    assert DUMP["url"] in [payload for kind, payload in sends if kind == "text"]


def test_a_long_bubble_is_split_to_fit_the_send_api():
    long_reply = "\n".join(["A sentence about trailers."] * 200)
    sends = cards.messenger_sends(long_reply, [])

    assert len(sends) > 1
    assert all(len(payload) <= cards.MESSENGER_TEXT_LIMIT for _, payload in sends)
    assert "".join(payload for _, payload in sends).replace("\n", "") == long_reply.replace("\n", "")


def test_an_empty_reply_sends_nothing():
    assert cards.messenger_sends("", [DUMP]) == []
