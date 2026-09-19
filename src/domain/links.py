"""Links a customer pastes: which site they point at, and our own listing URLs. Pure, no I/O.

Whether the customer WANTS the trailer behind a link is the model's reading
(``shared_link_interest``). This module only does what a URL settles by itself: where it
points, and whether it is one of our listing pages.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.I)

# Host -> how the team email names it.
_SOURCES = (
    (("trailerplace.com",), "TrailerPlace website listing link"),
    (("facebook.com", "fb.com", "fb.watch", "fb.me"), "Facebook post link"),
    (("instagram.com", "instagr.am"), "Instagram post link"),
)


def _host(url: str) -> str:
    raw = url if "://" in url else f"https://{url}"
    host = (urlsplit(raw).hostname or "").lower()
    return host.removeprefix("www.").removeprefix("m.")


def _source(url: str) -> str | None:
    host = _host(url)
    for domains, label in _SOURCES:
        if any(host == domain or host.endswith("." + domain) for domain in domains):
            return label
    return None


def find_trailer_links(text: str) -> list[tuple[str, str]]:
    """Every link in the message to our site, Facebook or Instagram, as (label, url)."""
    found: list[tuple[str, str]] = []
    for match in _URL_RE.finditer(str(text or "")):
        url = match.group(0).rstrip(".,;:!?")
        label = _source(url)
        if label:
            found.append((label, url))
    return found


def normalize_listing_url(value) -> str | None:
    """Our listing URL in one canonical form, or None when it is not one of our listing pages.

    Scheme, "www.", query string, fragment, case and the trailing slash vary with how it was
    copied; the path is what identifies the listing.
    """
    url = str(value or "").strip()
    if not url:
        return None
    raw = url if "://" in url else f"https://{url}"
    parts = urlsplit(raw)
    if _host(url) != "trailerplace.com":
        return None
    path = parts.path.rstrip("/").lower()
    if not path.startswith("/inventory/") or path == "/inventory":
        return None
    return f"https://www.trailerplace.com{path}/"
