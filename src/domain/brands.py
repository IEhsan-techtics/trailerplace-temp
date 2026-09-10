from __future__ import annotations

import difflib
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.domain.categories import CANONICAL_CATEGORIES
from src.domain.normalizer import normalize_category, normalize_make

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
# Fallback only. The brands/categories the prompts advertise must be the inventory
# the search can actually return, so the primary source is the trailer_listings
# table search itself reads. This module used to read listings_final_v5.xlsx while
# ingest read listings.xlsx, and the prompts promised makes the index had never
# seen; reading the same rows as search removes that class of drift entirely.
_LISTINGS_FILE = Path(os.getenv("LISTINGS_DATA_FILE", str(_ROOT / "listings.xlsx")))

# Non-canonical labels the catalogue uses, folded onto the canonical category they mean.
# "Concession" was here too, back when concession units were filed under Enclosed in the
# workbook. They now carry their own category, so folding them lost the two we actually
# stock: the bot reported no concession trailers while holding two, and would have told a
# customer to look elsewhere. Only ever map a NON-canonical label - mapping one canonical
# category onto another hides real inventory.
_CATEGORY_ALIASES = {
    "Atv Trailer": "Utility",
    "Landscape": "Utility",
    "Tank": "Diesel Tank",
}


@dataclass(frozen=True)
class MakeInventory:
    canonical_makes: tuple[str, ...]
    categories_by_make: dict[str, tuple[str, ...]]
    filter_values_by_make: dict[str, tuple[str, ...]]


def _display_make(value: Any) -> str:
    # normalize_make now yields the canonical short display form directly
    # (make_aliases.py), so no post-hoc display override is needed.
    return normalize_make(str(value or "").strip())


def _display_category(value: Any) -> str:
    normalized = normalize_category(str(value or "").strip())
    return _CATEGORY_ALIASES.get(normalized, normalized)


def _make_category_pairs_from_db() -> list[tuple[str, str]]:
    """(make, category) for every listing, straight from the table.

    Returns an empty list when the database is off or the table is empty, so the
    caller can fall back to the workbook rather than advertise no brands at all.
    """
    from sqlalchemy import select

    from src import db
    from src.db_models import TrailerListingRow

    if not db.database_enabled():
        return []
    try:
        with db.get_session_factory()() as session:
            rows = session.execute(
                select(TrailerListingRow.make, TrailerListingRow.category).distinct()
            ).all()
    except Exception:
        # Brand advertising must never take the chatbot down; the workbook
        # fallback below still produces a usable prompt block.
        logger.exception("make_inventory_db_read_failed | falling back to workbook")
        return []
    return [(str(make or ""), str(category or "")) for make, category in rows]


def _make_category_pairs_from_workbook() -> list[tuple[str, str]]:
    if not _LISTINGS_FILE.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_excel(_LISTINGS_FILE)
    except ImportError:
        return []
    if "make" not in df.columns or "category" not in df.columns:
        return []
    return [
        (str(row.get("make") or "").strip(), str(row.get("category") or "").strip())
        for _, row in df.iterrows()
    ]


@lru_cache(maxsize=1)
def load_make_inventory() -> MakeInventory:
    """The brands the prompts may advertise, derived from what we actually stock.

    Sourced from trailer_listings so an advertised brand is by construction a
    brand search can return. The workbook is the fallback for a database that is
    off or not yet ingested.
    """
    pairs = _make_category_pairs_from_db()
    source = "trailer_listings"
    if not pairs:
        pairs = _make_category_pairs_from_workbook()
        source = "workbook"
    if not pairs:
        return MakeInventory((), {}, {})

    categories: dict[str, set[str]] = {}
    filter_values: dict[str, set[str]] = {}
    for raw_make, raw_category in pairs:
        if not raw_make or not raw_category:
            continue

        make = _display_make(raw_make)
        category = _display_category(raw_category)
        if category == "Unknown" or category not in CANONICAL_CATEGORIES:
            # Only categories the rest of the system can actually qualify and search. The
            # catalogue carries a handful of rows outside the canonical 13 (e.g. "Welding")
            # — advertised in the makes block, they contradicted the same prompt's "we
            # carry exactly 13 categories" line and could be offered in the brand-category
            # question as a choice nothing downstream could handle.
            continue

        categories.setdefault(make, set()).add(category)
        variants = filter_values.setdefault(make, set())
        variants.add(raw_make)
        variants.add(normalize_make(raw_make))

    canonical_makes = tuple(sorted(categories))
    logger.info(
        "make_inventory_loaded | source=%s | makes=%s", source, len(canonical_makes)
    )
    return MakeInventory(
        canonical_makes=canonical_makes,
        categories_by_make={make: tuple(sorted(values)) for make, values in categories.items()},
        filter_values_by_make={
            make: tuple(sorted(v for v in values if v and v != "Unknown"))
            for make, values in filter_values.items()
        },
    )


def stocked_categories() -> tuple[str, ...]:
    """Canonical categories we currently hold stock in, in canonical order.

    The vocabulary stays hand-written in categories.py — the type/cargo terms
    encode judgment no column can supply — but which of those categories the
    prompts advertise is decided by the catalogue. Falls back to all canonical
    categories when nothing is loaded, so an unreachable database can never make
    the bot claim we sell nothing.
    """
    inventory = load_make_inventory()
    available = {
        category
        for categories in inventory.categories_by_make.values()
        for category in categories
    }
    if not available:
        return tuple(CANONICAL_CATEGORIES)
    return tuple(c for c in CANONICAL_CATEGORIES if c in available)


def known_makes() -> tuple[str, ...]:
    return load_make_inventory().canonical_makes


def make_prompt_block() -> str:
    inventory = load_make_inventory()
    prompt_makes = tuple(
        make for make in inventory.canonical_makes
        if make not in {"Gooseneck", "Bumper Pull"}
    )
    lines = [
        "Canonical trailer makes/brands available in the current inventory, with their available trailer categories:",
        "Gooseneck and Bumper Pull are strictly hitch types, never trailer makes/brands. "
        "Do not infer, recommend, or return either one as a make.",
    ]
    if not prompt_makes:
        lines.append("- No makes are currently available from the inventory workbook.")
        return "\n".join(lines)

    for make in prompt_makes:
        categories = tuple(
            category
            for category in inventory.categories_by_make.get(make, ())
            if category not in {"Gooseneck", "Bumper Pull"}
        )
        category_text = ", ".join(categories) if categories else "category unavailable"
        lines.append(f"- {make}: {category_text}")
    return "\n".join(lines)


# Words that appear in a make's name but carry no identifying weight — "trailer" would
# otherwise match the word "trailer" in almost any message and hand us a brand filter the
# customer never asked for.
_GENERIC_MAKE_WORDS = frozenset({"trailer", "trailers", "inc", "llc", "co", "company", "industries", "mfg"})


def brand_mentioned_in_text(brand: str, text: str) -> bool:
    """Did the customer actually NAME this brand in this message?

    The extractor will happily report a make it read off a listing already on screen — so a
    turn that only hands over an email address comes back with brand_preference="Iron Bull
    Trailers", which then silently narrows every later search to one manufacturer. A brand is
    a brand only when they said it, typos included ("dimond c" -> Diamond C).
    """
    words = re.findall(r"[a-z0-9&]+", (text or "").lower())
    brand_words = [
        word
        for word in re.findall(r"[a-z0-9&]+", (brand or "").lower())
        if len(word) >= 3 and word not in _GENERIC_MAKE_WORDS
    ]
    if not words or not brand_words:
        return False
    return any(
        word == brand_word or difflib.SequenceMatcher(None, word, brand_word).ratio() >= 0.8
        for brand_word in brand_words
        for word in words
    )


def categories_for_make(make: str) -> tuple[str, ...]:
    return load_make_inventory().categories_by_make.get(make, ())


def make_filter_values(make: str) -> tuple[str, ...]:
    inventory = load_make_inventory()
    return inventory.filter_values_by_make.get(make, (make,))
