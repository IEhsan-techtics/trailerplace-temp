"""Split one finished assistant reply into the bubbles the chat UI sends it in.

A reply that shows six trailers is one wall of text in a single bubble: the customer has to
scroll past five trailers they did not want to reach the closing question, and every listing
title - the hyperlink that opens the trailer - is buried in the middle of it. Sent as chunks,
each trailer arrives as its own message with its own tappable title, exactly the way a
salesperson would send them one after another.

The card shape this parses is the one src/llm/respond.py's prompt dictates, and the same one
_strip_listing_blocks() edits:

    1. [2026 Iron Bull DTB - 15081](https://...)
       - Category: Utility
       - Price: $9,995
       - One sales-pitch sentence.

So: everything before the first card is the intro (split on blank lines), each card is its own
chunk, and whatever follows the last card (the closing question) is the final chunk. A reply
with no cards is split on blank lines alone - which is what makes the mandated opening line
("Thank you for contacting TrailerPlace.") arrive as its own short message.
"""
from __future__ import annotations

import re

# A card's first line: an optional "1." / "1)" marker, optional bold/italic wrappers, then a
# markdown link. A bullet ("- [foo](url)") is deliberately NOT a card start - bullets are card
# BODY, and treating one as a start would cut a card in half.
_CARD_START_RE = re.compile(r"^[*_]{0,2}(?:\d+[.)]\s*)?[*_]{0,2}\[[^\]]+\]\(\s*https?://[^)\s]+", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[-*+\u2022]\s")


def _is_card_start(line: str) -> bool:
    return bool(_CARD_START_RE.match(line.strip())) and not _BULLET_RE.match(line.strip())


def _is_card_body(line: str) -> bool:
    """A continuation line of the card above: a bullet, or any indented wrap of one."""
    if not line.strip():
        return True
    if line[:1].isspace():
        return True
    return bool(_BULLET_RE.match(line.strip()))


def _paragraphs(text: str) -> list[str]:
    """Blank-line-separated blocks, empties dropped."""
    return [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]


def split_reply_into_chunks(text: str) -> list[str]:
    """One reply -> the ordered list of messages to send it as. Never returns an empty list
    for non-empty input, and joining the result with blank lines round-trips the reply."""
    source = str(text or "")
    if not source.strip():
        return []

    lines = source.split("\n")
    chunks: list[str] = []
    buffer: list[str] = []  # prose not yet flushed (intro, or the tail after the last card)

    def flush_prose() -> None:
        chunks.extend(_paragraphs("\n".join(buffer)))
        buffer.clear()

    index = 0
    while index < len(lines):
        if not _is_card_start(lines[index]):
            buffer.append(lines[index])
            index += 1
            continue
        # A card starts here: everything before it is prose that ships first.
        flush_prose()
        card = [lines[index]]
        index += 1
        while index < len(lines) and not _is_card_start(lines[index]) and _is_card_body(lines[index]):
            card.append(lines[index])
            index += 1
        # Blank lines collected as body are just the gap before whatever comes next.
        while card and not card[-1].strip():
            card.pop()
        chunks.append("\n".join(card).strip())
    flush_prose()
    return chunks or [source.strip()]


_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def urls_in_chunk(chunk: str) -> list[str]:
    """Every listing URL a chunk links to, so the UI can put that trailer's card under it."""
    return _URL_RE.findall(str(chunk or ""))


# A card's title line, taken apart: the "1." marker, the linked title, and the URL it opens.
# Same shape _CARD_START_RE matches, anchored at both ends so a line carrying anything else
# is left alone rather than half-parsed.
_CARD_TITLE_RE = re.compile(
    r"^[*_]{0,2}(?P<marker>\d+[.)])?\s*[*_]{0,2}\[(?P<title>[^\]]+)\]\(\s*(?P<url>https?://[^)\s]+)\s*\)[*_]{0,2}\s*$",
    re.IGNORECASE,
)


def parse_listing_card(chunk: str):
    """Split one card chunk into (marker, title, url, body) - or None if it is not a card.

    Channels that cannot render a markdown hyperlink need the three pieces separately;
    everything that can render one keeps using the chunk as it stands.
    """
    lines = str(chunk or "").strip().split("\n")
    if not lines:
        return None
    match = _CARD_TITLE_RE.match(lines[0].strip())
    if not match:
        return None
    body = "\n".join(line.strip() for line in lines[1:]).strip()
    return match.group("marker") or "", match.group("title").strip(), match.group("url"), body
