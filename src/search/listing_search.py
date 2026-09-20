"""SQL search over the ``trailer_listings`` table.

Replaces the former Pinecone vector search. The five hard gates that used to be
metadata filters are now a SQL WHERE, and there is no embedding: every row that
passes the filter is handed to the rerankers, which do all the ordering.

Ranking responsibilities, in full:
  * stated columns (category, make, hitch type, min length) -> the SQL filter
  * dimensional fit (length/width/height/payload)           -> _rerank_listings_by_fit
  * non-metadata features ("sliding gates", "insulated")    -> feature_ranker (gpt-5-nano)
Nothing else is ranked, by design, so no similarity score is needed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from rapidfuzz import fuzz
from sqlalchemy import select

from src import config, db
from src.db_models import TrailerListingRow
from src.domain.brands import make_filter_values
from src.domain.make_aliases import MAKE_ALIASES as MAKE_ALIAS_MAP
from src.domain.normalizer import normalize_category, normalize_hitch, normalize_subcategory
# Single-sourced parsers (units.py) so the rerank path parses raw catalog
# strings the same way ingest did when it wrote the table. _parse_number and
# _parse_length_ft are kept as local aliases to avoid churning the many call sites.
from src.domain.units import (
    parse_length_ft as _parse_length_ft,
    parse_weight_lbs as _parse_number,
)
from src.search.feature_ranker import ValidatedFeatureRerank, rank_non_metadata_features

logger = logging.getLogger(__name__)

RERANK_ENABLED = (os.getenv("RERANK_ENABLED") or "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
RERANK_WARN_RATIO = float((os.getenv("RERANK_WARN_RATIO") or "1.35").strip())
RERANK_EXTREME_RATIO = float((os.getenv("RERANK_EXTREME_RATIO") or "1.9").strip())
RERANK_LENGTH_WEIGHT = float((os.getenv("RERANK_LENGTH_WEIGHT") or "8.0").strip())
RERANK_MISSING_DIM_PENALTY = float((os.getenv("RERANK_MISSING_DIM_PENALTY") or "0.35").strip())
RERANK_VERBOSE_LOGS = (os.getenv("RERANK_VERBOSE_LOGS") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAKE_RERANK_VERBOSE_LOGS = (os.getenv("MAKE_RERANK_VERBOSE_LOGS") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FEATURE_RERANK_WEIGHT = config.settings.feature_rerank_weight
FIT_RERANK_WEIGHT = config.settings.feature_fit_weight
FEATURE_MATCH_THRESHOLD = 0.70
FEATURE_RERANK_VERBOSE_LOGS = (
    os.getenv("FEATURE_RERANK_VERBOSE_LOGS") or "1"
).strip().lower() in {"1", "true", "yes", "on"}

# Stage precedence:
# 1) fit rerank (length-first, strict no under-length)
# 2) category make-priority reorder
# Strict under-length behavior remains unchanged by make-priority.
CATEGORY_MAKE_PREFERENCES: dict[str, list[str]] = {
    "Aluminum": ["Aluma"],
    "Car Hauler": ["Diamond C", "Iron Bull Trailers", "P&C"],
    "Diesel Tank": ["East Texas Trailers"],
    "Dump": ["Iron Bull Trailers", "Diamond C", "Texas Pride"],
    "Enclosed": ["Cargo Craft", "Haulmark", "Stallion"],
    "Equipment": ["Diamond C", "Iron Bull Trailers", "P&C"],
    "Fiber": ["Cargo Craft", "Haulmark", "Stallion"],
    "Flatbed": ["Diamond C", "Iron Bull Trailers", "P&C"],
    "Livestock": ["Galyean", "Gooseneck", "Calico Trailers"],
    "Race Trailer": ["Haulmark", "Cargo Craft"],
    "Roll Off": ["Iron Bull Trailers", "East Texas Trailers"],
    "Tilt": ["Diamond C", "Iron Bull Trailers", "Aluma"],
    "Utility": ["Diamond C", "Iron Bull Trailers", "East Texas Trailers", "P&C"],
}


@dataclass
class ListingSearchResult:
    listings: list[dict[str, Any]]
    query_text: str
    metadata_filter: dict[str, Any] | None
    rerank_debug: dict[str, Any] = field(default_factory=dict)
    make_debug: dict[str, Any] = field(default_factory=dict)
    match_analysis: dict[str, Any] = field(default_factory=dict)


_ALLOWED_HITCH_TYPES = {"Gooseneck", "Bumper Pull"}


def _listing_filters(
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
    category_only: bool = False,
) -> list[tuple[str, str, Any]]:
    """The hard gates, as ``(column, op, value)`` descriptors.

    Descriptors rather than SQLAlchemy clauses so they stay comparable (see
    narrowing_filters_present) and loggable. _to_sql_clauses turns them into
    the actual WHERE.

    To gate on another column the customer stated — colour, material, floor —
    add a descriptor here and a label in search_node's _HARD_FILTER_LABELS so
    the relaxation message names it. Two things to remember: compare through
    the matching normalizer so the value lines up with what ingest wrote, and
    note that ``col == value`` excludes NULL rows, so each new gate narrows the
    result set and pushes more searches onto the relaxation ladder.
    """
    metadata_filters = metadata_filters or {}
    filters: list[tuple[str, str, Any]] = []
    normalized_category = normalize_category(category) if category else None
    if category:
        filters.append(("category", "eq", normalized_category))

    if category_only:
        # The last-resort pass: every hard filter but the category is dropped so we can show the
        # customer the closest alternatives instead of an empty screen. Their requirements are not
        # thrown away — the fit rerank and the feature rerank still order what comes back by how
        # near it gets. Only the all-or-nothing gates are gone.
        return filters[:1]

    make_value = metadata_filters.get("make")
    if make_value:
        values = [value for value in make_filter_values(str(make_value)) if value]
        if values:
            # make_filter_values returns every raw workbook spelling AND the
            # normalized form of each, so this matches the normalized column.
            filters.append(("make", "in", tuple(values)))

    hitch_value = metadata_filters.get("hitch_type") or slots.get("hitch_type")
    if hitch_value:
        hitch = normalize_hitch(str(hitch_value))
        if hitch in _ALLOWED_HITCH_TYPES:
            filters.append(("hitch_type", "eq", hitch))
        else:
            logger.info("listing_hitch_filter_rejected | value=%r | normalized=%r", hitch_value, hitch)

    subcategory_value = metadata_filters.get("subcategory")
    if normalized_category == "Aluminum" and subcategory_value:
        subcategory = normalize_subcategory(str(subcategory_value))
        if subcategory:
            filters.append(("subcategory", "eq", subcategory))

    # The SQL min-length gate. Same slot sources as the fit reranker's
    # _required_length_ft_from_filters, kept in step with it deliberately: this decides
    # which rows come back at all, that one decides how they are ordered.
    min_length = (
        _parse_length_ft(metadata_filters.get("length_ft"))
        or _parse_length_ft(slots.get("length"))
        or _parse_length_ft(slots.get("trailer_size"))
        or _parse_length_ft(slots.get("cargo_size"))
    )
    if min_length:
        filters.append(("length_ft_num", "gte", min_length))

    return filters


def _to_sql_clauses(filters: list[tuple[str, str, Any]]) -> list[Any]:
    clauses = []
    for column_name, op, value in filters:
        column = getattr(TrailerListingRow, column_name)
        if op == "eq":
            clauses.append(column == value)
        elif op == "in":
            clauses.append(column.in_(list(value)))
        elif op == "gte":
            clauses.append(column >= value)
        else:  # pragma: no cover - guards a typo in _listing_filters
            raise ValueError(f"unsupported filter op: {op!r}")
    return clauses


def _filters_as_dict(filters: list[tuple[str, str, Any]]) -> dict[str, Any] | None:
    """Readable form for logs, state and the API contract."""
    if not filters:
        return None
    rendered: dict[str, Any] = {}
    for column_name, op, value in filters:
        rendered[column_name] = list(value) if op == "in" else value
    return rendered


def narrowing_filters_present(
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
) -> bool:
    """Does the hard filter constrain anything BEYOND the category?

    If it does not, a category-only retry would run the identical query for a second time and come
    back just as empty — the category really has nothing left to show.
    """
    full = _listing_filters(category, slots, metadata_filters)
    return full != _listing_filters(category, slots, metadata_filters, category_only=True)


_FILTER_LABELS = {
    "category": "Category",
    "make": "Make",
    "hitch_type": "Hitch Type",
    "subcategory": "Subcategory",
    "length_ft_num": "Length",
}


def _filter_description(filters: list[tuple[str, str, Any]]) -> str:
    """Human-readable rendering of the active gates, for logs and debug output.

    Replaces the old embedding query text. Nothing consumes it as an input any
    more — it exists so a search's behaviour is legible in the logs.
    """
    parts: list[str] = []
    for column_name, op, value in filters:
        label = _FILTER_LABELS.get(column_name, column_name.replace("_", " ").title())
        if op == "in":
            rendered = ", ".join(str(item) for item in value)
        elif op == "gte":
            rendered = f">= {value:g} ft" if column_name.endswith("_ft_num") else f">= {value}"
        else:
            rendered = str(value)
        parts.append(f"{label}: {rendered}")
    return " | ".join(parts)


def _row_to_listing(row: TrailerListingRow) -> dict[str, Any]:
    """Flatten a table row into the card dict the rest of the app expects.

    Key set and coalescing behaviour are unchanged from the Pinecone version:
    every column is nullable, exactly as metadata fields were optional before.
    """
    price = float(row.price) if row.price is not None else None
    raw_features = row.features or []
    if isinstance(raw_features, str):
        raw_features = [raw_features]
    return {
        "title": row.title or "",
        "image_url": row.image_url,
        "condition": row.condition or "New",
        "price": row.price_display or price,
        "price_display": row.price_display or (f"${price:,.0f}" if price else None),
        "category": row.category or "",
        "subcategory": row.subcategory or "",
        "make": row.make or "",
        "model": row.model or "",
        "trim": row.trim or "",
        "stock_number": row.stock_number or "",
        "color": row.color or "",
        "hitch_type": row.hitch_type,
        "year": row.year,
        "length": row.length,
        "width": row.width,
        "height": row.height,
        "axles": row.axles,
        "gvwr": row.gvwr,
        "axle_capacity": row.axle_capacity,
        # The two below are numbers already, not display strings, and are the only
        # ranking inputs read straight off the column rather than re-parsed from text:
        # `axle_count` has no display twin by design (see db_models) and the total is
        # derived, so there is no raw string to parse in the first place.
        "axle_count": row.axle_count,
        "total_axle_capacity": row.total_axle_capacity_lbs_num,
        "payload_capacity": row.payload_capacity,
        "material": row.trailer_material,
        "floor": row.floor,
        "url": row.url or "",
        # No similarity score exists any more. Kept at 0.0 because it is the
        # last term in both rerank sort keys, where a constant is a no-op.
        "relevance_score": 0.0,
        # Kept only while ranking. search_listings strips this internal
        # evidence before results enter conversation state or the API response.
        "features": [
            str(value).strip()
            for value in raw_features
            if str(value or "").strip()
        ],
        "match_evidence_text": row.match_evidence_text or "",
    }


def _required_length_ft_from_filters(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> Optional[float]:
    # `length` is the one length slot. trailer_size/cargo_size keep their own names because
    # they ask for several numbers at once, but their length still counts as this one - they
    # normally arrive already split out into metadata_filters, and are read here as well so a
    # direct search_listings() call with raw slots behaves the same.
    return (
        _parse_length_ft(metadata_filters.get("length_ft"))
        or _parse_length_ft(slots.get("length"))
        or _parse_length_ft(slots.get("trailer_size"))
        or _parse_length_ft(slots.get("cargo_size"))
    )


def _required_payload_lbs_from_filters(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> Optional[float]:
    return _parse_number(
        metadata_filters.get("payload_lbs")
        or slots.get("payload_capacity")
    )


def _required_axle_capacity_lbs_from_filters(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> Optional[float]:
    return _parse_number(
        metadata_filters.get("axle_capacity_lbs") or slots.get("axle_capacity")
    )


def _required_total_axle_capacity_lbs_from_filters(
    slots: dict[str, Any], metadata_filters: dict[str, Any]
) -> Optional[float]:
    """What the axles must carry BETWEEN them.

    Falls back to the arithmetic when they gave a per-axle rating and a count but never
    said a total - the same multiplication ingest does. Never the reverse: a total alone
    says nothing about how many axles carry it.
    """
    stated = _parse_number(
        metadata_filters.get("total_axle_capacity_lbs") or slots.get("total_axle_capacity_lbs")
    )
    if stated is not None:
        return stated
    per_axle = _required_axle_capacity_lbs_from_filters(slots, metadata_filters)
    count = _required_axle_count_from_filters(slots, metadata_filters)
    if per_axle is not None and count is not None:
        return per_axle * count
    return None


def _required_axle_count_from_filters(
    slots: dict[str, Any], metadata_filters: dict[str, Any]
) -> Optional[int]:
    """How many axles, as a whole number. A fractional axle is not a thing."""
    value = _parse_number(metadata_filters.get("axle_count") or slots.get("axle_count"))
    return None if value is None else int(round(value))


def _required_width_ft_from_filters(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> Optional[float]:
    # cargo_size is deliberately NOT read here. It answers width only as part of a full
    # "8x25x6.5", which parse_dimensions splits correctly on the way into metadata_filters -
    # read raw, a plain "about 20 ft" cargo_size would be taken as a 20 ft WIDTH.
    return _parse_length_ft(
        metadata_filters.get("width_ft")
        or slots.get("width")
    )


def _required_height_ft_from_filters(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> Optional[float]:
    return _parse_length_ft(
        metadata_filters.get("height_ft")
        or slots.get("height")
    )


def _norm_key(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"[^\w&]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canonical_make(value: Any) -> str:
    key = _norm_key(value)
    if not key:
        return ""
    return MAKE_ALIAS_MAP.get(key, str(value or "").strip())


def _brand_order_for_category(
    listings: list[dict[str, Any]],
    *,
    category: str | None,
) -> tuple[list[str], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    cat = normalize_category(category) if category else None
    preferred = CATEGORY_MAKE_PREFERENCES.get(cat) or []
    preferred_norm: list[str] = []
    seen_preferred: set[str] = set()
    for item in preferred:
        canonical = _canonical_make(item)
        if canonical and canonical not in seen_preferred:
            preferred_norm.append(canonical)
            seen_preferred.add(canonical)

    buckets: dict[str, list[dict[str, Any]]] = {}
    brand_order: list[str] = []
    remainder: list[dict[str, Any]] = []

    for item in listings:
        brand = _canonical_make(item.get("make"))
        if not brand:
            remainder.append(item)
            continue
        if brand not in buckets:
            buckets[brand] = []
            brand_order.append(brand)
        buckets[brand].append(item)

    ordered_brands: list[str] = []
    seen_brands: set[str] = set()

    for brand in preferred_norm:
        if brand in buckets and brand not in seen_brands:
            ordered_brands.append(brand)
            seen_brands.add(brand)

    for brand in brand_order:
        if brand not in seen_brands:
            ordered_brands.append(brand)
            seen_brands.add(brand)

    return ordered_brands, buckets, remainder


def _brand_quota_template(brand_count: int, max_recommendations: int) -> list[int]:
    if brand_count <= 1:
        return [max_recommendations]
    if brand_count == 2:
        return [3, 3]
    return [3, 2, 1]


def _apply_category_make_preference(
    listings: list[dict[str, Any]],
    *,
    category: str | None,
    max_recommendations: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not listings or not category:
        return listings, {"applied": False, "reason": "missing_listings_or_category"}

    cat = normalize_category(category)
    preferred = CATEGORY_MAKE_PREFERENCES.get(cat)
    if not preferred:
        return listings, {"applied": False, "reason": "no_category_preference", "category": cat}

    ordered_brands, buckets, remainder = _brand_order_for_category(listings, category=category)
    brand_count = len(ordered_brands)
    quota_template = _brand_quota_template(brand_count, max_recommendations)
    quota_brand_count = 1 if brand_count <= 1 else 2 if brand_count == 2 else 3
    quota_brands = ordered_brands[:quota_brand_count]

    ranked: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    selected_urls: set[str] = set()

    def _take_from_brand(brand: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        taken: list[dict[str, Any]] = []
        for item in buckets.get(brand, []):
            if len(taken) >= limit:
                break
            url = str(item.get("url") or "").strip()
            if url and url in selected_urls:
                continue
            taken.append(item)
            if url:
                selected_urls.add(url)
        return taken

    # First pass: enforce the 3/2/1 or 3/3 style quota for the highest-priority brands.
    for idx, brand in enumerate(quota_brands):
        taken = _take_from_brand(brand, quota_template[idx])
        ranked.extend(taken)
        selected_counts[brand] = len(taken)

    def _backfill_from_brand(brand: str) -> None:
        remaining = []
        already_selected = selected_counts.get(brand, 0)
        for item in buckets.get(brand, [])[already_selected:]:
            url = str(item.get("url") or "").strip()
            if url and url in selected_urls:
                continue
            remaining.append(item)
        if not remaining:
            return
        ranked.extend(remaining)
        selected_counts[brand] = selected_counts.get(brand, 0) + len(remaining)
        for item in remaining:
            url = str(item.get("url") or "").strip()
            if url:
                selected_urls.add(url)

    # Backfill from lower-priority brands first, then from any leftovers of the quota brands.
    for brand in ordered_brands[quota_brand_count:]:
        _backfill_from_brand(brand)

    if len(ranked) < max_recommendations:
        for brand in quota_brands:
            if len(ranked) >= max_recommendations:
                break
            already_selected = selected_counts.get(brand, 0)
            extras = []
            for item in buckets.get(brand, [])[already_selected:]:
                url = str(item.get("url") or "").strip()
                if url and url in selected_urls:
                    continue
                extras.append(item)
                if len(ranked) + len(extras) >= max_recommendations:
                    break
            if not extras:
                continue
            ranked.extend(extras)
            selected_counts[brand] = selected_counts.get(brand, 0) + len(extras)
            for item in extras:
                url = str(item.get("url") or "").strip()
                if url:
                    selected_urls.add(url)

    if len(ranked) < max_recommendations and remainder:
        for item in remainder:
            if len(ranked) >= max_recommendations:
                break
            url = str(item.get("url") or "").strip()
            if url and url in selected_urls:
                continue
            ranked.append(item)
            if url:
                selected_urls.add(url)

    ranked = ranked[:max_recommendations]
    selected_by_brand = {
        brand: count
        for brand, count in selected_counts.items()
        if count > 0
    }
    before_top = [str(x.get("make") or "") for x in listings[:8]]
    after_top = [str(x.get("make") or "") for x in ranked[:8]]
    logger.info(
        "make_rerank_summary | category=%r | brand_order=%s | quota_template=%s | selected_by_brand=%s | total=%s | before_top=%s | after_top=%s",
        cat,
        ordered_brands,
        quota_template,
        selected_by_brand,
        len(listings),
        before_top,
        after_top,
    )
    if MAKE_RERANK_VERBOSE_LOGS:
        for i, item in enumerate(ranked, 1):
            mk = _canonical_make(item.get("make"))
            logger.info(
                "make_rerank_item | rank=%s | title=%r | make=%r | canonical_make=%r | brand_rank=%s",
                i,
                item.get("title"),
                item.get("make"),
                mk,
                ordered_brands.index(mk) + 1 if mk in ordered_brands else None,
            )

    return ranked, {
        "applied": True,
        "category": cat,
        "brand_order": ordered_brands,
        "quota_template": quota_template,
        "selected_by_brand": selected_by_brand,
        "brand_count": brand_count,
    }


def _rerank_listings_by_fit(
    listings: list[dict[str, Any]],
    *,
    required_length_ft: Optional[float],
    required_payload_lbs: Optional[float],
    required_width_ft: Optional[float],
    required_height_ft: Optional[float],
    warn_ratio: float,
    extreme_ratio: float,
    length_weight: float,
    missing_dim_penalty: float,
    required_axle_capacity_lbs: Optional[float] = None,
    required_total_axle_capacity_lbs: Optional[float] = None,
    required_axle_count: Optional[int] = None,
    retain_all: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    needs_present = any(
        x is not None
        for x in (
            required_length_ft, required_payload_lbs, required_width_ft, required_height_ft,
            required_axle_capacity_lbs, required_total_axle_capacity_lbs, required_axle_count,
        )
    )
    if not listings or not needs_present:
        return listings, {"applied": False, "reason": "missing_clear_requirements_or_no_listings"}

    entries: list[dict[str, Any]] = []
    for idx, listing in enumerate(listings, 1):
        length_ft = _parse_length_ft(listing.get("length"))
        gvwr_lbs = _parse_number(listing.get("gvwr"))
        payload_lbs = _parse_number(listing.get("payload_capacity"))
        width_ft = _parse_length_ft(listing.get("width"))
        height_ft = _parse_length_ft(listing.get("height"))
        axle_capacity_lbs = _parse_number(listing.get("axle_capacity"))
        # Already numeric on the card - see _row_to_listing.
        total_axle_capacity_lbs = _parse_number(listing.get("total_axle_capacity"))
        axle_count = _parse_number(listing.get("axle_count"))

        # Weight requirement can be satisfied by payload or gvwr if payload missing.
        weight_from = "payload_capacity"
        weight_cap = payload_lbs
        if weight_cap is None:
            weight_cap = gvwr_lbs
            if weight_cap is not None:
                weight_from = "gvwr"
            else:
                weight_from = "unknown"

        length_ratio = (
            (length_ft / required_length_ft)
            if required_length_ft is not None and length_ft is not None and required_length_ft > 0
            else None
        )
        weight_ratio = (
            (weight_cap / required_payload_lbs)
            if required_payload_lbs is not None and weight_cap is not None and required_payload_lbs > 0
            else None
        )
        width_ratio = (
            (width_ft / required_width_ft)
            if required_width_ft is not None and width_ft is not None and required_width_ft > 0
            else None
        )
        height_ratio = (
            (height_ft / required_height_ft)
            if required_height_ft is not None and height_ft is not None and required_height_ft > 0
            else None
        )
        # Both sides are TOTALS, so this compares like with like - and it is a different
        # question from the per-axle one below: four 3,500 lb axles and two 7,000 lb axles
        # both carry 14,000 lb, but only one of them satisfies "7,000 lb axles".
        total_axle_ratio = (
            (total_axle_capacity_lbs / required_total_axle_capacity_lbs)
            if required_total_axle_capacity_lbs is not None
            and total_axle_capacity_lbs is not None
            and required_total_axle_capacity_lbs > 0
            else None
        )
        # Both sides are PER-AXLE ratings, so this compares like with like.
        axle_ratio = (
            (axle_capacity_lbs / required_axle_capacity_lbs)
            if required_axle_capacity_lbs is not None and axle_capacity_lbs is not None and required_axle_capacity_lbs > 0
            else None
        )

        fail_count = 0
        missing_count = 0
        penalty = 0.0

        if required_length_ft is not None:
            if length_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif length_ratio < 1.0:
                # Strict under-length disallow.
                fail_count += 1
                penalty += (1.0 - length_ratio) * (length_weight * 2.0)
            else:
                over = length_ratio - 1.0
                penalty += over * length_weight
                if length_ratio > warn_ratio:
                    penalty += (length_ratio - warn_ratio) * (length_weight * 1.8)
                if length_ratio > extreme_ratio:
                    penalty += (length_ratio - extreme_ratio) * (length_weight * 3.2)

        if required_payload_lbs is not None:
            if weight_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif weight_ratio < 1.0:
                fail_count += 1
                penalty += (1.0 - weight_ratio) * 6.0
            else:
                over = weight_ratio - 1.0
                penalty += over * 1.4
                if weight_ratio > warn_ratio:
                    penalty += (weight_ratio - warn_ratio) * 2.0
                if weight_ratio > extreme_ratio:
                    penalty += (weight_ratio - extreme_ratio) * 4.0
                if weight_from == "gvwr":
                    penalty += 0.2
                    logger.info(
                        "payload_rerank_gvwr_fallback | title=%r | gvwr=%r | payload_capacity=%r",
                        listing.get("title"),
                        listing.get("gvwr"),
                        listing.get("payload_capacity"),
                    )

        if required_width_ft is not None:
            if width_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif width_ratio < 1.0:
                fail_count += 1
                penalty += (1.0 - width_ratio) * 4.0
            else:
                over = width_ratio - 1.0
                penalty += over * 1.2
                if width_ratio > warn_ratio:
                    penalty += (width_ratio - warn_ratio) * 1.8
                if width_ratio > extreme_ratio:
                    penalty += (width_ratio - extreme_ratio) * 3.2

        if required_height_ft is not None:
            if height_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif height_ratio < 1.0:
                fail_count += 1
                penalty += (1.0 - height_ratio) * 4.0
            else:
                over = height_ratio - 1.0
                penalty += over * 1.0
                if height_ratio > warn_ratio:
                    penalty += (height_ratio - warn_ratio) * 1.5
                if height_ratio > extreme_ratio:
                    penalty += (height_ratio - extreme_ratio) * 2.6

        if required_axle_capacity_lbs is not None:
            # Softer than the payload block and deliberately never fail_count: only ~70% of the
            # catalogue carries an axle rating, and in the legacy path fail_count>0 culls the
            # listing outright whenever any non-failing row exists. An axle preference must
            # ORDER results, not empty the screen — that is why it is not a SQL gate either.
            if axle_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif axle_ratio < 1.0:
                # As steep as the payload block's under-capacity term: an axle rated below what
                # they asked for cannot do the job, so it must rank below even a wildly
                # over-specified trailer, which can.
                penalty += (1.0 - axle_ratio) * 6.0
            else:
                # Over-specification is only a mild demerit here, unlike length or payload: a
                # heavier-rated axle still does the job, it is just more trailer than they asked
                # for. The escalation stays gentle so an over-rated trailer never sinks below an
                # under-rated one that cannot carry the load at all.
                over = axle_ratio - 1.0
                penalty += over * 0.8
                if axle_ratio > warn_ratio:
                    penalty += (axle_ratio - warn_ratio) * 0.6
                if axle_ratio > extreme_ratio:
                    penalty += (axle_ratio - extreme_ratio) * 0.8

        if required_total_axle_capacity_lbs is not None:
            # Same shape and the same reasoning as the per-axle block above, and never
            # fail_count for the same reason: a quarter of the catalogue carries no total,
            # and culling those would empty the screen rather than order it.
            if total_axle_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif total_axle_ratio < 1.0:
                # Axles that together carry less than the load cannot do the job.
                penalty += (1.0 - total_axle_ratio) * 6.0
            else:
                over = total_axle_ratio - 1.0
                penalty += over * 0.8
                if total_axle_ratio > warn_ratio:
                    penalty += (total_axle_ratio - warn_ratio) * 0.6
                if total_axle_ratio > extreme_ratio:
                    penalty += (total_axle_ratio - extreme_ratio) * 0.8

        if required_axle_count is not None:
            # Deliberately NOT a ratio. A count is categorical, not a magnitude: 4 axles is
            # not "twice as much trailer" as 2 the way 20 ft is twice 10 ft, and a ratio
            # would rank a quad as a mild over-spec of a tandem when it is a different
            # trailer entirely. A flat step per axle away from the ask says that plainly.
            if axle_count is None:
                missing_count += 1
                penalty += missing_dim_penalty
            else:
                off_by = abs(int(round(axle_count)) - required_axle_count)
                # Under-supplying axles is worse than over-supplying: three axles will still
                # carry what two would, two will not do the work of three.
                penalty += off_by * (2.5 if axle_count < required_axle_count else 1.5)

        length_overage = (
            max(0.0, float(length_ratio) - 1.0)
            if required_length_ft is not None and length_ratio is not None
            else 999.0
        )
        base_score = float(listing.get("relevance_score") or 0.0)
        entries.append(
            {
                "listing": listing,
                "fetch_pos": idx,
                "base_score": base_score,
                "penalty": round(penalty, 6),
                "fail_count": fail_count,
                "missing_count": missing_count,
                "length_ratio": None if length_ratio is None else round(length_ratio, 6),
                "weight_ratio": None if weight_ratio is None else round(weight_ratio, 6),
                "width_ratio": None if width_ratio is None else round(width_ratio, 6),
                "height_ratio": None if height_ratio is None else round(height_ratio, 6),
                "axle_ratio": None if axle_ratio is None else round(axle_ratio, 6),
                "total_axle_ratio": None if total_axle_ratio is None else round(total_axle_ratio, 6),
                "axle_count": None if axle_count is None else int(round(axle_count)),
                "length_overage": round(length_overage, 6),
                "fit_score": round(base_score - penalty, 6),
            }
        )

    # The legacy/no-feature path drops failing entries whenever a non-failing
    # option exists. Feature-aware search sets retain_all=True so every hard-
    # filtered candidate reaches the combined feature/fit scorer.
    nonfailing = [e for e in entries if e["fail_count"] == 0]
    fallback_pool = entries if retain_all or not nonfailing else nonfailing
    ranked_entries = sorted(
        fallback_pool,
        key=lambda e: (
            e["fail_count"] if retain_all else 0,
            e["missing_count"],
            e["length_overage"],
            e["penalty"],
            -e["base_score"],
        ),
    )
    fit_rank_by_fetch_pos = {
        entry["fetch_pos"]: rank for rank, entry in enumerate(ranked_entries, 1)
    }
    for entry in entries:
        entry["fit_rank"] = fit_rank_by_fetch_pos.get(entry["fetch_pos"])

    if RERANK_VERBOSE_LOGS:
        decision_rank = {id(e): i for i, e in enumerate(ranked_entries, 1)}
        for e in entries:
            logger.info(
                "rerank_score | fetched_pos=%s | decision_rank=%s | title=%r | base_score=%.6f | penalty=%.6f | fit_score=%.6f | length_ratio=%s | weight_ratio=%s | width_ratio=%s | height_ratio=%s | axle_ratio=%s | total_axle_ratio=%s | axle_count=%s | fail_count=%s | missing_count=%s",
                e["fetch_pos"],
                decision_rank.get(id(e)),
                e["listing"].get("title"),
                e["base_score"],
                e["penalty"],
                e["fit_score"],
                e["length_ratio"],
                e["weight_ratio"],
                e["width_ratio"],
                e["height_ratio"],
                e["axle_ratio"],
                e["total_axle_ratio"],
                e["axle_count"],
                e["fail_count"],
                e["missing_count"],
            )

    logger.info(
        "rerank_summary | applied=true | required_length_ft=%s | required_payload_lbs=%s | required_width_ft=%s | required_height_ft=%s | required_axle_capacity_lbs=%s | required_total_axle_capacity_lbs=%s | required_axle_count=%s | candidates=%s | kept_pool=%s",
        required_length_ft,
        required_payload_lbs,
        required_width_ft,
        required_height_ft,
        required_axle_capacity_lbs,
        required_total_axle_capacity_lbs,
        required_axle_count,
        len(entries),
        len(fallback_pool),
    )

    return [e["listing"] for e in ranked_entries], {
        "applied": True,
        "required_length_ft": required_length_ft,
        "required_payload_lbs": required_payload_lbs,
        "required_width_ft": required_width_ft,
        "required_height_ft": required_height_ft,
        "required_axle_capacity_lbs": required_axle_capacity_lbs,
        "required_total_axle_capacity_lbs": required_total_axle_capacity_lbs,
        "required_axle_count": required_axle_count,
        "candidate_count": len(entries),
        "retained_candidate_count": len(ranked_entries),
        "fit_entries": entries if retain_all else [],
    }


def _normalize_feature_phrase(value: Any) -> str:
    text = str(value or "").casefold()
    # Treat common feature abbreviations such as A/C as one token ("ac").
    text = re.sub(r"(?<=\w)/(?=\w)", "", text)
    text = re.sub(r"[^\w]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _feature_token_similarity(requested_token: str, candidate_token: str) -> float:
    """Compare word forms without a hand-maintained feature vocabulary.

    Edit similarity naturally connects productive forms such as insulation/insulated,
    electrical/electric, rails/rail, and lights/light. Very short tokens (AC, LED, TV)
    must match exactly because fuzzy matching them creates too many coincidences.
    """
    if min(len(requested_token), len(candidate_token)) <= 3:
        return 1.0 if requested_token == candidate_token else 0.0
    return fuzz.ratio(requested_token, candidate_token) / 100.0


def _feature_match_details(
    requested_features: list[str], match_sources: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    stored = []
    all_candidate_tokens: list[tuple[str, str]] = []
    for source in match_sources:
        normalized = _normalize_feature_phrase(source)
        if not normalized:
            continue
        tokens = normalized.split()
        stored.append((source, tokens))
        all_candidate_tokens.extend((source, token) for token in tokens)
    details: list[dict[str, Any]] = []
    scores: list[float] = []

    for requested in requested_features:
        requested_norm = _normalize_feature_phrase(requested)
        requested_tokens = requested_norm.split()
        token_matches: list[dict[str, Any]] = []
        for requested_token in requested_tokens:
            best_source = ""
            best_candidate_token = ""
            best_similarity = 0.0
            for source, candidate_token in all_candidate_tokens:
                similarity = _feature_token_similarity(requested_token, candidate_token)
                if similarity > best_similarity:
                    best_source = source
                    best_candidate_token = candidate_token
                    best_similarity = similarity
            token_matches.append(
                {
                    "requested_token": requested_token,
                    "matched_token": best_candidate_token or None,
                    "source": best_source or None,
                    "similarity": round(best_similarity, 6),
                    "accepted": best_similarity >= FEATURE_MATCH_THRESHOLD,
                }
            )

        best_raw = (
            sum(match["similarity"] for match in token_matches) / len(token_matches)
            if token_matches
            else 0.0
        )
        # Every word in a clean feature phrase matters. This prevents "enclosed" alone
        # from satisfying "insulated enclosed" while retaining inflectional matches.
        accepted = bool(token_matches) and all(match["accepted"] for match in token_matches)
        score = best_raw if accepted else 0.0
        scores.append(score)
        best_feature = max(
            stored,
            key=lambda item: (
                sum(
                    max(
                        (_feature_token_similarity(token, candidate) for candidate in item[1]),
                        default=0.0,
                    )
                    for token in requested_tokens
                )
                / len(requested_tokens)
                if requested_tokens
                else 0.0
            ),
            default=(None, []),
        )[0]
        details.append(
            {
                "requested": requested,
                "best_stored_feature": best_feature,
                "raw_similarity": round(best_raw, 6),
                "accepted": accepted,
                "score": round(score, 6),
                "token_matches": token_matches,
            }
        )

    coverage = sum(scores) / len(scores) if scores else 0.0
    return round(coverage, 6), details


def _candidate_feature_match_sources(listing: dict[str, Any]) -> list[str]:
    """Return deduplicated searchable phrases from all candidate evidence.

    match_evidence_text is the exact readable text embedded at ingest time, so
    splitting it into fields lets feature reranking inspect core fields and
    arbitrary info/specification values in addition to the explicit features.
    """
    candidates: list[str] = []
    candidates.extend(str(value) for value in (listing.get("features") or []))
    candidates.extend(
        str(listing.get(key) or "") for key in ("title", "model", "trim")
    )
    evidence = str(listing.get("match_evidence_text") or "")
    candidates.extend(re.split(r"[|\n;]+", evidence))

    unique: dict[str, str] = {}
    for candidate in candidates:
        text = re.sub(r"\s+", " ", str(candidate or "")).strip(" ,")
        normalized = _normalize_feature_phrase(text)
        if normalized:
            unique.setdefault(normalized, text)
    return list(unique.values())


def _combined_feature_fit_rerank(
    listings: list[dict[str, Any]],
    *,
    requested_features: list[str],
    required_length_ft: Optional[float],
    required_payload_lbs: Optional[float],
    required_width_ft: Optional[float],
    required_height_ft: Optional[float],
    metadata_filter: dict[str, Any] | None,
    query_text: str,
    required_axle_capacity_lbs: Optional[float] = None,
    required_total_axle_capacity_lbs: Optional[float] = None,
    required_axle_count: Optional[int] = None,
    semantic_rerank: ValidatedFeatureRerank | None = None,
    semantic_fallback_reason: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank every hard-filtered candidate with an 85/15 blend.

    Unlike the legacy fit reranker, this never removes dimensionally failing
    candidates. Fit only supplies the secondary 15% ordering signal. When a
    validated GPT assessment is supplied, Python derives feature coverage from
    its evidence-grounded booleans; otherwise the deterministic matcher is the
    availability fallback.
    """
    if not listings:
        return listings, {
            "applied": False,
            "reason": "no_listings",
            "requested_features": requested_features,
        }

    _, fit_debug = _rerank_listings_by_fit(
        listings,
        required_length_ft=required_length_ft,
        required_payload_lbs=required_payload_lbs,
        required_width_ft=required_width_ft,
        required_height_ft=required_height_ft,
        required_axle_capacity_lbs=required_axle_capacity_lbs,
        required_total_axle_capacity_lbs=required_total_axle_capacity_lbs,
        required_axle_count=required_axle_count,
        warn_ratio=RERANK_WARN_RATIO,
        extreme_ratio=RERANK_EXTREME_RATIO,
        length_weight=RERANK_LENGTH_WEIGHT,
        missing_dim_penalty=RERANK_MISSING_DIM_PENALTY,
        retain_all=True,
    )
    fit_entries = fit_debug.get("fit_entries") or []
    if not fit_entries:
        # With no dimension/weight requirement, the existing order is the
        # SQL fetch order, so it becomes the secondary fit-order signal.
        fit_entries = [
            {
                "listing": listing,
                "fetch_pos": pos,
                "fit_rank": pos,
                "base_score": float(listing.get("relevance_score") or 0.0),
                "penalty": 0.0,
                "fail_count": 0,
                "missing_count": 0,
                "length_ratio": None,
                "weight_ratio": None,
                "width_ratio": None,
                "height_ratio": None,
                "axle_ratio": None,
                "total_axle_ratio": None,
                "axle_count": None,
                "length_overage": 999.0,
                "fit_score": float(listing.get("relevance_score") or 0.0),
            }
            for pos, listing in enumerate(listings, 1)
        ]

    candidate_count = len(fit_entries)
    scored: list[dict[str, Any]] = []
    for entry in fit_entries:
        listing = entry["listing"]
        fit_rank = int(entry.get("fit_rank") or entry["fetch_pos"])
        fit_order_score = (
            1.0
            if candidate_count == 1
            else 1.0 - ((fit_rank - 1) / (candidate_count - 1))
        )
        stored_features = list(listing.get("features") or [])
        match_sources = _candidate_feature_match_sources(listing)
        candidate_id = f"C{int(entry['fetch_pos']):03d}"
        semantic_source = "deterministic_fallback"
        if semantic_rerank is not None and candidate_id in semantic_rerank.assessments_by_id:
            assessment = semantic_rerank.assessments_by_id[candidate_id]
            matched_count = sum(1 for match in assessment.feature_matches if match.matched)
            feature_coverage = round(matched_count / len(requested_features), 6)
            feature_matches = [match.model_dump() for match in assessment.feature_matches]
            semantic_source = "gpt_semantic"
        else:
            feature_coverage, feature_matches = _feature_match_details(
                requested_features, match_sources
            )
        final_score = (
            FEATURE_RERANK_WEIGHT * feature_coverage
            + FIT_RERANK_WEIGHT * fit_order_score
        )
        scored.append(
            {
                "listing": listing,
                "candidate_id": candidate_id,
                "fetch_pos": entry["fetch_pos"],
                "fit_rank": fit_rank,
                "base_score": float(listing.get("relevance_score") or 0.0),
                "feature_coverage": round(feature_coverage, 6),
                "fit_order_score": round(fit_order_score, 6),
                "final_score": round(final_score, 6),
                "feature_matches": feature_matches,
                "semantic_source": semantic_source,
                "stored_features": stored_features,
                "match_source_count": len(match_sources),
                "penalty": entry.get("penalty", 0.0),
                "fail_count": entry.get("fail_count", 0),
                "missing_count": entry.get("missing_count", 0),
                "length_ratio": entry.get("length_ratio"),
                "weight_ratio": entry.get("weight_ratio"),
                "width_ratio": entry.get("width_ratio"),
                "height_ratio": entry.get("height_ratio"),
            }
        )

    scored.sort(
        key=lambda item: (
            -item["final_score"],
            -item["feature_coverage"],
            item["fit_rank"],
            -item["base_score"],
            item["fetch_pos"],
        )
    )

    analysis_candidates: list[dict[str, Any]] = []
    for final_rank, item in enumerate(scored, 1):
        listing = item["listing"]
        diagnostic = {
            "fetch_position": item["fetch_pos"],
            "candidate_id": item["candidate_id"],
            "fit_position": item["fit_rank"],
            "final_position": final_rank,
            "title": listing.get("title"),
            "url": listing.get("url"),
            "make": listing.get("make"),
            "requested_features": requested_features,
            "stored_features": item["stored_features"],
            "match_source_count": item["match_source_count"],
            "feature_matches": item["feature_matches"],
            "semantic_source": item["semantic_source"],
            "feature_coverage": item["feature_coverage"],
            "fit_order_score": item["fit_order_score"],
            "feature_weight": FEATURE_RERANK_WEIGHT,
            "fit_weight": FIT_RERANK_WEIGHT,
            "final_score": item["final_score"],
            "length": listing.get("length"),
            "width": listing.get("width"),
            "height": listing.get("height"),
            "payload_capacity": listing.get("payload_capacity"),
            "gvwr": listing.get("gvwr"),
            "length_ratio": item["length_ratio"],
            "weight_ratio": item["weight_ratio"],
            "width_ratio": item["width_ratio"],
            "height_ratio": item["height_ratio"],
            "fit_penalty": item["penalty"],
            "fit_fail_count": item["fail_count"],
            "fit_missing_count": item["missing_count"],
            "rank_movement": {
                "fetch_to_fit": item["fetch_pos"] - item["fit_rank"],
                "fit_to_final": item["fit_rank"] - final_rank,
            },
        }
        analysis_candidates.append(diagnostic)
        if FEATURE_RERANK_VERBOSE_LOGS:
            logger.info(
                "feature_fit_candidate | %s",
                json.dumps(diagnostic, ensure_ascii=False, default=str),
            )

    matched_candidates = sum(
        1 for item in scored if item["feature_coverage"] > 0.0
    )
    logger.info(
        "feature_fit_summary | candidates=%s matched_candidates=%s feature_weight=%.2f fit_weight=%.2f threshold=%.2f metadata_filter=%s query_text=%r final_urls=%s",
        candidate_count,
        matched_candidates,
        FEATURE_RERANK_WEIGHT,
        FIT_RERANK_WEIGHT,
        FEATURE_MATCH_THRESHOLD,
        json.dumps(metadata_filter or {}, default=str),
        query_text,
        [item["listing"].get("url") for item in scored],
    )
    return [item["listing"] for item in scored], {
        "applied": True,
        "feature_match_mode": (
            "gpt_semantic"
            if semantic_rerank is not None and not semantic_rerank.fallback_candidate_ids
            else "hybrid_semantic_deterministic"
            if semantic_rerank is not None
            else "deterministic_fallback"
        ),
        "semantic_fallback_reason": semantic_fallback_reason,
        "semantic_model": semantic_rerank.model if semantic_rerank is not None else None,
        "semantic_reasoning_effort": semantic_rerank.reasoning_effort if semantic_rerank is not None else None,
        "semantic_latency_ms": semantic_rerank.latency_ms if semantic_rerank is not None else None,
        "semantic_request_id": semantic_rerank.request_id if semantic_rerank is not None else None,
        "semantic_request_ids": list(semantic_rerank.request_ids) if semantic_rerank is not None else [],
        "semantic_batch_count": semantic_rerank.batch_count if semantic_rerank is not None else 0,
        "semantic_fallback_candidate_ids": sorted(semantic_rerank.fallback_candidate_ids) if semantic_rerank is not None else [],
        "semantic_validation_errors": semantic_rerank.validation_errors_by_id if semantic_rerank is not None else {},
        "candidate_count": candidate_count,
        "matched_candidate_count": matched_candidates,
        "requested_features": requested_features,
        "feature_weight": FEATURE_RERANK_WEIGHT,
        "fit_weight": FIT_RERANK_WEIGHT,
        "feature_match_threshold": FEATURE_MATCH_THRESHOLD,
        "hard_metadata_filter": metadata_filter or {},
        "filter_description": query_text,
        "candidates": analysis_candidates,
    }


def fetch_listings(filters: list[tuple[str, str, Any]]) -> list[TrailerListingRow]:
    """Every row matching the hard gates, in a stable order.

    No LIMIT: the catalogue is small enough that a category filter returns far
    fewer rows than the old top_k=50, and the rerankers want the full set.

    The ORDER BY is not cosmetic. Fetch position feeds the final rerank
    tiebreaker and, when the customer stated no dimension at all, becomes the
    fallback fit rank — so an unordered SELECT would let results reshuffle
    between identical runs. Price ascending makes that fallback "cheapest
    first"; stock_number breaks any remaining tie.
    """
    if not db.database_enabled():
        # Same guard every public function in conversation_store uses: with the
        # database off, return the neutral value rather than raising. A turn
        # then reports "nothing found" instead of 500ing, and readiness.py is
        # the thing that fails loudly about a misconfigured database.
        logger.warning("listing_search_skipped | reason=database_disabled")
        return []

    statement = (
        select(TrailerListingRow)
        .where(*_to_sql_clauses(filters))
        .order_by(
            TrailerListingRow.price.asc().nullslast(),
            TrailerListingRow.stock_number.asc().nullslast(),
            TrailerListingRow.listing_id.asc(),
        )
    )
    with db.get_session_factory()() as session:
        return list(session.execute(statement).scalars())


def search_listing_result(
    *,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
    requested_features: list[str] | None = None,
    already_shown_urls: list[str] | None = None,
    top_k: int | None = None,
    max_recommendations: int | None = None,
    category_only_filters: bool = False,
) -> ListingSearchResult:
    metadata_filters = metadata_filters or {}
    requested_features = [
        str(feature).strip()
        for feature in (requested_features or [])
        if str(feature or "").strip()
    ]
    max_recommendations = max_recommendations or int(os.getenv("SEARCH_MAX_RECOMMENDATIONS", "5"))
    filters = _listing_filters(category, slots, metadata_filters, category_only_filters)
    metadata_filter = _filters_as_dict(filters)
    filter_description = _filter_description(filters)
    shown_urls = {str(u).strip() for u in (already_shown_urls or []) if str(u or "").strip()}

    rows = fetch_listings(filters)

    logger.info(
        "listing_search | category=%r | matched_rows=%s | max_recommendations=%s | "
        "already_shown_url_count=%s | filters=%s | slots=%s | metadata_filters_collected=%s",
        category,
        len(rows),
        max_recommendations,
        len(shown_urls),
        json.dumps(metadata_filter, default=str) if metadata_filter else "{}",
        json.dumps(slots or {}, default=str),
        json.dumps(metadata_filters or {}, default=str),
    )

    shown = shown_urls
    listings: list[dict[str, Any]] = []
    for row in rows:
        item = _row_to_listing(row)
        if item["url"] and item["url"] in shown:
            continue
        listings.append(item)

    rerank_debug: dict[str, Any] = {"applied": False, "reason": "disabled"}
    make_debug: dict[str, Any]
    match_analysis: dict[str, Any] = {}
    if requested_features:
        required_length_ft = _required_length_ft_from_filters(slots, metadata_filters)
        required_payload_lbs = _required_payload_lbs_from_filters(slots, metadata_filters)
        required_width_ft = _required_width_ft_from_filters(slots, metadata_filters)
        required_height_ft = _required_height_ft_from_filters(slots, metadata_filters)
        required_axle_capacity_lbs = _required_axle_capacity_lbs_from_filters(slots, metadata_filters)
        required_total_axle_capacity_lbs = _required_total_axle_capacity_lbs_from_filters(slots, metadata_filters)
        required_axle_count = _required_axle_count_from_filters(slots, metadata_filters)
        semantic_rerank: ValidatedFeatureRerank | None = None
        semantic_fallback_reason: str | None = None
        if config.settings.feature_llm_rerank_enabled and listings:
            try:
                semantic_rerank = rank_non_metadata_features(listings, requested_features)
            except Exception as exc:  # noqa: BLE001 - deterministic availability fallback
                semantic_fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "feature_rank_fallback | reason=%r candidates=%s requested_features=%s",
                    semantic_fallback_reason,
                    len(listings),
                    json.dumps(requested_features, ensure_ascii=False),
                )
        else:
            semantic_fallback_reason = (
                "no_candidates" if not listings else "feature_llm_rerank_disabled"
            )
            logger.info(
                "feature_rank_not_invoked | reason=%s candidates=%s requested_features=%s",
                semantic_fallback_reason,
                len(listings),
                json.dumps(requested_features, ensure_ascii=False),
            )

        listings, match_analysis = _combined_feature_fit_rerank(
            listings,
            requested_features=requested_features,
            required_length_ft=required_length_ft,
            required_payload_lbs=required_payload_lbs,
            required_width_ft=required_width_ft,
            required_height_ft=required_height_ft,
            required_axle_capacity_lbs=required_axle_capacity_lbs,
        required_total_axle_capacity_lbs=required_total_axle_capacity_lbs,
        required_axle_count=required_axle_count,
            metadata_filter=metadata_filter,
            query_text=filter_description,
            semantic_rerank=semantic_rerank,
            semantic_fallback_reason=semantic_fallback_reason,
        )
        rerank_debug = {
            "applied": True,
            "mode": "combined_feature_fit",
            "candidate_count": len(listings),
        }
        make_debug = {
            "applied": False,
            "reason": "feature_order_preserved",
            "category": normalize_category(category) if category else None,
        }
    elif RERANK_ENABLED:
        required_length_ft = _required_length_ft_from_filters(slots, metadata_filters)
        required_payload_lbs = _required_payload_lbs_from_filters(slots, metadata_filters)
        required_width_ft = _required_width_ft_from_filters(slots, metadata_filters)
        required_height_ft = _required_height_ft_from_filters(slots, metadata_filters)
        required_axle_capacity_lbs = _required_axle_capacity_lbs_from_filters(slots, metadata_filters)
        required_total_axle_capacity_lbs = _required_total_axle_capacity_lbs_from_filters(slots, metadata_filters)
        required_axle_count = _required_axle_count_from_filters(slots, metadata_filters)
        listings, rerank_debug = _rerank_listings_by_fit(
            listings,
            required_length_ft=required_length_ft,
            required_payload_lbs=required_payload_lbs,
            required_width_ft=required_width_ft,
            required_height_ft=required_height_ft,
            required_axle_capacity_lbs=required_axle_capacity_lbs,
        required_total_axle_capacity_lbs=required_total_axle_capacity_lbs,
        required_axle_count=required_axle_count,
            warn_ratio=RERANK_WARN_RATIO,
            extreme_ratio=RERANK_EXTREME_RATIO,
            length_weight=RERANK_LENGTH_WEIGHT,
            missing_dim_penalty=RERANK_MISSING_DIM_PENALTY,
            # This is the category-only relaxed retry: the hard gates were dropped from the SQL
            # filter precisely because nothing met them. If the fit rerank then
            # hard-culls the same under-size candidates (retain_all=False drops fail_count>0
            # whenever any non-failing row exists), the relaxation is undone one stage later and
            # the screen collapses to the lone row that happened to meet the gate — seen live as
            # a 50 ft livestock search returning 1 of 17 candidates. Keep every candidate and let
            # the fit score ORDER them closest-first, which is what "closest alternatives" means.
            retain_all=category_only_filters,
        )
        logger.info("rerank_debug=%s", json.dumps(rerank_debug, default=str))
        listings, make_debug = _apply_category_make_preference(
            listings,
            category=category,
            max_recommendations=max_recommendations,
        )
    else:
        listings, make_debug = _apply_category_make_preference(
            listings,
            category=category,
            max_recommendations=max_recommendations,
        )
    logger.info("make_rerank_debug=%s", json.dumps(make_debug, default=str))

    return ListingSearchResult(
        listings=listings[:max_recommendations],
        query_text=filter_description,
        metadata_filter=metadata_filter,
        rerank_debug=rerank_debug,
        make_debug=make_debug,
        match_analysis=match_analysis,
    )


def search_listings(
    *,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
    requested_features: list[str] | None = None,
    already_shown_urls: list[str] | None = None,
    top_k: int | None = None,
    max_recommendations: int | None = None,
    category_only_filters: bool = False,
) -> list[dict[str, Any]]:
    result = search_listing_result(
        category=category,
        slots=slots,
        metadata_filters=metadata_filters,
        requested_features=requested_features,
        already_shown_urls=already_shown_urls,
        top_k=top_k,
        max_recommendations=max_recommendations,
        category_only_filters=category_only_filters,
    )
    return [
        {
            **{
                key: value
                for key, value in item.items()
                if key not in {"match_evidence_text", "features"}
            },
            # The full list stays behind, but a short slice of it is what the reply's pitch
            # sentence is written from - without it the pitch had nothing to say that the
            # card's own bullets had not already said.
            "pitch_features": pitch_features(item.get("features")),
        }
        for item in result.listings
    ]


# Sized from the catalogue: a listing carries 15 features at the median and 19 at p75, about
# 28 characters each, stored alphabetically. At 8 the cut fell mid-alphabet, so the Diamond C
# LPT208's "Long Arm Tarp System" - feature 17 of 21 - never reached the pitch. At 20 a typical
# search turn adds roughly 700 input tokens across six trailers.
PITCH_FEATURES = 20
PITCH_FEATURE_CHARS = 80


def pitch_features(features: Any) -> list[str]:
    """At most PITCH_FEATURES distinct features, each clipped, placeholders dropped.

    Anything that is not a list is treated as no features - the lookup path reads them out of
    a DataFrame, where an empty cell is NaN, and iterating a float raises.
    """
    if isinstance(features, str):
        features = [features]
    elif hasattr(features, "tolist") and isinstance(features.tolist(), list):
        features = features.tolist()  # a numpy array; a numpy NaN's tolist() is a float
    elif not isinstance(features, (list, tuple)):
        features = []
    kept: list[str] = []
    seen: set[str] = set()
    for feature in features or []:
        text = " ".join(str(feature or "").split())
        if not text or text.lower() in {"none", "null", "n/a", "unspecified"}:
            continue
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        kept.append(text[:PITCH_FEATURE_CHARS].rstrip())
        if len(kept) >= PITCH_FEATURES:
            break
    return kept
