"""Links a customer pastes: which site they point at, and our own listing URLs. Pure, no I/O.

Whether the customer WANTS the trailer behind a link is the model's reading
(``shared_link_interest``). This module only does what a URL settles by itself: where it
points, and whether it is one of our listing pages.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.I)

# Host -> how the team email names it.
_SOURCES = (
    (("trailerplace.com",), "TrailerPlace website listing link"),
    (("facebook.com", "fb.com", "fb.watch", "fb.me"), "Facebook post link"),
    (("instagram.com", "instagr.am"), "Instagram post link"),
)


# What the platform is CALLED, for a message written out to the model and the team. The
# labels above name a link in an email; these name the platform in a sentence.
_PLATFORMS = (
    ("Facebook", ("facebook.com", "fb.com", "fb.me", "fb.watch", "messenger.com")),
    ("Instagram", ("instagram.com", "instagr.am")),
)

# Facebook wraps an outbound link in a redirect (l.facebook.com/l.php?u=<the real url>),
# so a customer sharing one of our listings hands us the wrapper, not the listing.
_REDIRECT_HOSTS = ("l.facebook.com", "lm.facebook.com", "l.instagram.com")


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


def unwrap_redirect(url: str) -> str:
    """The real destination behind a Facebook or Instagram redirect link, else the url itself.

    Without this, a listing of ours shared from a Facebook post arrives as an l.facebook.com
    wrapper, and neither the lookup nor the team email can see which trailer it is.
    """
    text = str(url or "").strip()
    parts = urlsplit(text if "://" in text else f"https://{text}")
    if (parts.hostname or "").lower() in _REDIRECT_HOSTS:
        target = (parse_qs(parts.query).get("u") or [""])[0]
        if target.startswith(("http://", "https://")):
            return target
    return text


def platform_of(url: str) -> str | None:
    """'Facebook', 'Instagram', or None for anything else.

    By host alone: the path of a post URL is opaque and Meta changes its shape often.
    """
    host = _host(str(url or ""))
    for platform, domains in _PLATFORMS:
        if any(host == domain or host.endswith("." + domain) for domain in domains):
            return platform
    return None


def strip_social_links(text: str) -> str:
    """The text with every Facebook or Instagram URL removed, and the gap tidied up.

    For the team's email. A post URL is opaque - it says nothing about which trailer, it
    expires, and pasted into a one-line summary it swamps the sentence that matters. The
    platform is named instead (``shared_platforms``), which is the part that tells the team
    where the customer saw us.

    OUR OWN listing links are deliberately left alone: trailerplace.com/inventory/... names
    exactly one trailer, and that is the most useful thing the line can carry.
    """
    def _drop(match: re.Match) -> str:
        url = match.group(0).rstrip(".,;:!?")
        if not platform_of(url) and not platform_of(unwrap_redirect(url)):
            return match.group(0)
        # A Facebook wrapper around one of OUR listings is not a post - it is a trailer,
        # reached the long way round. Unwrapped in place rather than dropped, so the line
        # still says which one.
        ours = normalize_listing_url(unwrap_redirect(url))
        return ours or ""

    stripped = _URL_RE.sub(_drop, str(text or ""))
    stripped = re.sub(r"\(\s*\)", "", stripped)           # "(  )" left by a dropped url
    stripped = re.sub(r"\s+([.,;:!?])", r"\1", stripped)  # " ." left in front of punctuation
    return re.sub(r"\s{2,}", " ", stripped).strip(" :-")


def shared_platforms(texts) -> list[str]:
    """Which platforms the customer has sent us links from, e.g. ['Facebook']."""
    platforms: list[str] = []
    for text in texts:
        for match in _URL_RE.finditer(str(text or "")):
            url = match.group(0).rstrip(".,;:!?")
            # The WRAPPER first: a customer who reached us through l.facebook.com came from
            # Facebook, whatever our listing page at the other end of it says.
            platform = platform_of(url) or platform_of(unwrap_redirect(url))
            if platform and platform not in platforms:
                platforms.append(platform)
    return platforms
