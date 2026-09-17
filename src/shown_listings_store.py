from __future__ import annotations

from typing import Any

_shown: dict[str, set[str]] = {}


def add_shown_urls(sid: str, urls: list[str] | tuple[str, ...] | set[str]) -> None:
    bucket = _shown.setdefault(sid, set())
    bucket.update(str(url) for url in urls if url)


def add_shown_keys_and_urls(
    sid: str,
    keys: list[str] | tuple[str, ...] | set[str],
    urls: list[str] | tuple[str, ...] | set[str],
) -> None:
    add_shown_urls(sid, urls)


def _listing_url(listing: Any) -> str | None:
    url = getattr(listing, "url", None)
    if url:
        return str(url)
    if isinstance(listing, dict):
        url = listing.get("url")
        if url:
            return str(url)
    return None


def accumulate_shown_urls_from_chat_messages(messages: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for message in messages or []:
        for listing in (message or {}).get("listings") or []:
            url = _listing_url(listing)
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls
