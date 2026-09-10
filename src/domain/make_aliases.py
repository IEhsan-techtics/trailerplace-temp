"""Single source of truth for trailer make/brand aliases.

Previously three near-duplicate maps existed:
- ``normalizer.MAKE_MAP`` (used at ingest to canonicalize the make written into
  the listings table),
- ``make_resolver._ALIASES`` (query-time user-text -> canonical make),
- ``listing_search.MAKE_ALIAS_MAP`` (rerank display normalization).

They diverged in both membership and target form (e.g. one produced
"Cargo Craft Trailers", the others "Cargo Craft"), which was only reconciled by
``$in`` filters. This module is the one map all three consume.

Canonical (value) forms are the display forms already returned by
``make_inventory.known_makes()``, so switching each consumer over does not change
what the resolver returns or what is displayed. The only intended data change is
that ingest now writes the short forms "Cargo Craft" / "Diamond C" (matching the
display form) instead of the long "... Trailers" forms — which requires a
re-ingest to take effect in the index.

Keys are given in several normalized variants (spaces, hyphens, ``&``/"and"/"n")
because each consumer re-normalizes keys differently:
- ``normalizer.normalize_make`` looks up ``make.strip().lower()`` (hyphens/``&``
  preserved),
- ``listing_search._canonical_make`` looks up ``_norm_key`` (``&`` preserved,
  hyphens -> space),
- ``make_resolver`` re-normalizes every key via its own ``_norm`` (``&`` -> "and").
Redundant variants keep every consumer correct.
"""

from __future__ import annotations

# alias (lowercased) -> canonical make (the make_inventory.known_makes() form)
MAKE_ALIASES: dict[str, str] = {
    "alcom": "Alcom",
    "aluma": "Aluma",
    "alum": "Aluma",
    "ameritrail": "AmeriTrail",
    "ameri trail": "AmeriTrail",
    "ameraitrail": "AmeriTrail",
    "baseline": "Baseline",
    "cm trailers": "Cm Trailers",
    "cm trailer": "Cm Trailers",
    "calico trailers": "Calico Trailers",
    "calico": "Calico Trailers",
    "cargo craft": "Cargo Craft",
    "cargo craft trailers": "Cargo Craft",
    "continental cargo": "Continental Cargo",
    "diamond c": "Diamond C",
    "diamond c trailers": "Diamond C",
    "dimond c": "Diamond C",
    "east texas": "East Texas Trailers",
    "east texas trailers": "East Texas Trailers",
    "fairwest": "Fairwest",
    "galyean": "Galyean",
    "gooseneck": "Gooseneck",
    "haulmark": "Haulmark",
    "iron bull": "Iron Bull Trailers",
    "iron bull trailers": "Iron Bull Trailers",
    "j&j": "J&J Trailer",
    "j and j": "J&J Trailer",
    "j j": "J&J Trailer",
    "kaufman": "Kaufman Trailers",
    "kaufman trailers": "Kaufman Trailers",
    "lark united manufacturing": "Lark United Manufacturing",
    "liberty": "Liberty",
    "norstar": "Norstar",
    "p&c": "P&C",
    "p and c": "P&C",
    "p n c": "P&C",
    "p c": "P&C",
    "pace": "Pace",
    "pace american": "Pace American",
    "ranch king": "Ranch King",
    "rd trailers": "Rd Trailers",
    "rd trailer": "Rd Trailers",
    "stallion": "Stallion",
    "star": "Star",
    "texas pride": "Texas Pride",
    "w w": "W-W",
    "w-w": "W-W",
}
