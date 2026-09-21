"""One reply, rendered as the channel it is going to actually draws it.

Luna writes a reply once, in markdown, with a listing card per trailer:

    1. [2026 Iron Bull DTB - 15081](https://www.trailerplace.com/inventory/...)
       - Category: Utility
       - Price: $9,995
       - One sales sentence.

The **website** can render that as written. src/domain/reply_chunks.py splits it into
bubbles, and the UI puts the trailer's own picture card under the bubble that links it -
so the markdown stays, and the card is an extra beside it.

**Messenger renders no markdown at all.** Sent as written, that first line arrives as the
literal characters ``[2026 Iron Bull DTB - 15081](https://...)``, brackets and all. Nor
does the Send API preview a link into a card: that preview is the Messenger app's courtesy
to a link a PERSON pastes, and a link we deliver stays plain text. A trailer therefore has
to be taken apart and re-sent as a *generic template* - Meta's own card, with the photo,
the title, a subtitle and a View Trailer button - followed by the bullets as a text bubble.

Both renderings read the SAME listing rows the search returned, so every value on a card is
the database's own. The reply's text is used only to work out which trailer a chunk is
about, and in what order the trailers were presented.
"""
from __future__ import annotations

from typing import Any, Iterable

from src.domain.reply_chunks import parse_listing_card, split_reply_into_chunks, urls_in_chunk

# Meta truncates a longer title or subtitle with an ellipsis of its own, mid-word. Clipping
# on this side keeps the cut where we choose it.
CARD_TITLE_MAX = 80
CARD_SUBTITLE_MAX = 80
# The Send API's hard limit on one text bubble.
MESSENGER_TEXT_LIMIT = 2000
CARD_BUTTON_TITLE = "View Trailer"


def url_key(url: Any) -> str:
    """How a URL is compared. The same normalisation the reply path uses throughout."""
    return str(url or "").strip().rstrip("/").lower()


def _get(listing: Any, key: str) -> Any:
    """Listings arrive as dicts from the search and as objects from the UI's own model."""
    if isinstance(listing, dict):
        return listing.get(key)
    return getattr(listing, key, None)


def listings_by_url(listings: Iterable[Any] | None) -> dict[str, Any]:
    """The turn's listings, keyed by URL; the first row wins a repeated URL."""
    index: dict[str, Any] = {}
    for listing in listings or []:
        key = url_key(_get(listing, "url"))
        if key:
            index.setdefault(key, listing)
    return index


def _clip(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- the website
def website_bubbles(assistant_text: str, listings: Iterable[Any] | None) -> list[dict[str, Any]]:
    """The reply as the web chat shows it: each bubble's markdown, with its trailers.

    A listing belongs to the FIRST bubble that links it, so a trailer named twice still gets
    exactly one card. Anything no bubble linked - or every listing, when the reply has no
    cards in it at all - is returned under ``leftover`` for the UI to put after the last
    bubble, because a trailer the search found is never silently dropped.

    ``rank`` is the listing's 1-based position in the turn's own results, which is the number
    the reply refers to it by and the number the card shows.
    """
    rows = list(listings or [])
    chunks = split_reply_into_chunks(assistant_text) or [str(assistant_text or "")]
    claimed: set[int] = set()
    bubbles: list[dict[str, Any]] = []

    for chunk in chunks:
        linked = {url_key(url) for url in urls_in_chunk(chunk)}
        matched: list[dict[str, Any]] = []
        if linked:
            for index, listing in enumerate(rows):
                if index in claimed:
                    continue
                url = url_key(_get(listing, "url"))
                if url and url in linked:
                    claimed.add(index)
                    matched.append({"rank": index + 1, "listing": listing})
        bubbles.append({"text": chunk, "listings": matched})

    leftover = [
        {"rank": index + 1, "listing": listing}
        for index, listing in enumerate(rows) if index not in claimed
    ]
    if leftover and bubbles:
        bubbles[-1]["leftover"] = leftover
    return bubbles


# ------------------------------------------------------------------------- Messenger
def card_subtitle(listing: Any) -> str:
    """The line under the title. Price first: it is the thing customers ask for."""
    parts: list[str] = []
    price = _get(listing, "price_display") or _get(listing, "price")
    if str(price or "").strip():
        parts.append(str(price).strip())
    for key in ("length", "category"):
        value = str(_get(listing, key) or "").strip()
        if value:
            parts.append(value)
    return _clip(" · ".join(parts), CARD_SUBTITLE_MAX)


def card_element(listing: Any, fallback_title: str = "", fallback_url: str = "") -> dict[str, Any]:
    """One generic-template element: the card Messenger draws for a trailer.

    ``image_url`` is left out rather than sent empty when a listing has no photo. Meta
    fetches that URL server-side and rejects the whole element if it cannot, and a card with
    a title, a subtitle and a button is still a card - whereas a rejected element is a
    trailer the customer never sees.
    """
    url = str(_get(listing, "url") or fallback_url or "").strip()
    element: dict[str, Any] = {
        "title": _clip(_get(listing, "title") or fallback_title, CARD_TITLE_MAX),
        "default_action": {"type": "web_url", "url": url, "webview_height_ratio": "full"},
        "buttons": [{"type": "web_url", "url": url, "title": CARD_BUTTON_TITLE}],
    }
    subtitle = card_subtitle(listing)
    if subtitle:
        element["subtitle"] = subtitle
    image = str(_get(listing, "image_url") or "").strip()
    if image:
        element["image_url"] = image
    return element


def split_for_messenger(text: str) -> list[str]:
    """Keep every bubble inside the Send API's limit, splitting on line breaks."""
    body = str(text or "").strip()
    if len(body) <= MESSENGER_TEXT_LIMIT:
        return [body] if body else []
    parts: list[str] = []
    current = ""
    for line in body.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= MESSENGER_TEXT_LIMIT:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        # One line over the limit is rare - no listing card is - so a hard slice is an
        # acceptable last resort rather than something worth a word wrapper.
        while len(line) > MESSENGER_TEXT_LIMIT:
            parts.append(line[:MESSENGER_TEXT_LIMIT])
            line = line[MESSENGER_TEXT_LIMIT:]
        current = line
    if current:
        parts.append(current)
    return parts


def _sends_for_chunk(chunk: str, index: dict[str, Any], cards_enabled: bool) -> list[tuple[str, Any]]:
    """One bubble -> the sends it becomes: ``("text", str)`` or ``("card", element)``."""
    parsed = parse_listing_card(chunk)
    if not parsed:
        return [("text", part) for part in split_for_messenger(chunk)]

    marker, title, url, body = parsed
    listing = index.get(url_key(url)) if cards_enabled else None
    if listing is None:
        # No row to build a card from - cards switched off, or a URL this turn did not
        # present. Three plain bubbles: worse than a card, but the trailer still reaches
        # the customer with a tappable link, which is what matters.
        bubbles: list[tuple[str, Any]] = [
            ("text", f"{marker} {title}".strip() if marker else title),
            ("text", url),
        ]
        if body:
            bubbles.extend(("text", part) for part in split_for_messenger(body))
        return bubbles

    sends: list[tuple[str, Any]] = [("card", card_element(listing, title, url))]
    if body:
        # The bullets and the sales sentence - the part of the card Luna actually wrote.
        # The title and the price it copied are on the card above.
        sends.extend(("text", part) for part in split_for_messenger(body))
    return sends


def messenger_sends(
    assistant_text: str, listings: Iterable[Any] | None, *, cards_enabled: bool = True
) -> list[tuple[str, Any]]:
    """The reply as the ordered sends Messenger needs, in the order to send them.

    ``("text", body)`` goes out as a message bubble; ``("card", element)`` goes out as a
    one-element generic template. A webhook loops over this and sends each in turn - there
    is no further decision for it to make.
    """
    index = listings_by_url(listings)
    sends: list[tuple[str, Any]] = []
    for chunk in split_reply_into_chunks(assistant_text):
        sends.extend(_sends_for_chunk(chunk, index, cards_enabled))
    return sends
