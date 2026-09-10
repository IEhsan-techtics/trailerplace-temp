"""Deterministic inventory matcher for direct lookups.

The reference file's three embedded mini-LLMs (query extraction, reply-intro,
feature-framing) and its regex-fallback extraction stack are NOT ported here —
identifier extraction is now the Analyze LLM's job (``TurnAnalysis.inventory_lookup``)
and reply wording is the Respond LLM's job (both already built in M3). This module
is a pure function of identifiers: given year/make/model_text/stock_number it
fuzzy-matches the catalogue and returns card dicts shaped identically to
inventory search results.

Source of truth is the ``trailer_listings`` table — the same rows semantic
search returns, so a stock-number lookup can never quote a trailer the search
cannot find. The workbook remains a fallback for when the database is
unavailable (and is what the unit tests drive). Matching itself is unchanged:
rows are loaded into the same DataFrame shape and scored by the same fuzzy
pipeline, whichever source they came from.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from src.domain.normalizer import clean_dealer_notes

logger = logging.getLogger(__name__)

# Same depth as src/domain/brands.py (src/search/inventory_matcher.py -> search -> src -> root).
_ROOT = Path(__file__).resolve().parents[2]
# The SAME workbook (and env override) the ingest and the prompts read — direct
# lookups must quote the same inventory the search returns and the prompts advertise.
# prepare_inventory backfills any column this workbook lacks, so the schema difference
# from the old listings_final_v5.xlsx source is harmless.
_LISTINGS_FILE = Path(os.getenv("LISTINGS_DATA_FILE", str(_ROOT / "listings.xlsx")))

_MAKE_ONLY = "MAKE_SEARCH"
_YEAR_MAKE = "YEAR_MAKE_SEARCH"
_MODEL_SEARCH = "MODEL_SEARCH"
_YEAR_MAKE_MODEL = "YEAR_MAKE_MODEL_SEARCH"
_STOCK = "STOCK_SEARCH"
_POSSIBLE_MODEL = "POSSIBLE_MODEL_SEARCH"
_UNKNOWN = "UNKNOWN_SEARCH"

_COMMON_MODEL_WORDS = {
    "a", "an", "and", "available", "cost", "do", "does", "for", "have", "how",
    "is", "looking", "much", "price", "that", "the", "this", "trailer",
    "trailers", "want", "what", "aluminum", "car", "cargo", "dump", "enclosed",
    "equipment", "fiber", "flatbed", "hauler", "livestock", "race", "roll",
    "tilt", "utility",
}

_INVENTORY_EVIDENCE_MAX_CHARS = 3000


def normalize_text(text: Any) -> str:
    value = "" if text is None else str(text)
    if value.lower() == "nan":
        return ""
    value = value.lower()
    value = (
        value.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("�", " ")
    )
    value = value.replace("&", " and ")
    value = re.sub(r"\btrailers\b", "trailer", value)
    value = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " x ", value)
    value = re.sub(r"[^a-z0-9#]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _stock_text(value: Any) -> str:
    text = _clean_scalar(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    digits = re.sub(r"\D+", "", text)
    # Excel stores stock numbers as numbers (02570 -> 2570) while listing titles — and
    # therefore customers — keep the leading zero. Compare without it on both sides.
    return digits.lstrip("0") or digits


def extract_model_code(model: Any) -> str:
    text = normalize_text(model)
    for token in text.split():
        if token in _COMMON_MODEL_WORDS or token in {"base", "w", "with"}:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        return token
    return ""


def load_inventory(excel_path: str | Path = _LISTINGS_FILE) -> pd.DataFrame:
    return pd.read_excel(excel_path)


def prepare_inventory(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in (
        "title", "url", "year", "make", "model", "trim", "category",
        "subcategory", "stock_number", "price", "condition", "length", "width",
        "gvwr", "payload_capacity", "hitch_type", "color", "axles", "axle_capacity",
        "trailer_material", "floor", "dealer_notes", "info_specs_json",
        "info_spec_json",
    ):
        if column not in prepared.columns:
            prepared[column] = ""

    prepared["make_norm"] = prepared["make"].map(normalize_text)
    prepared["model_norm"] = prepared["model"].map(normalize_text)
    prepared["title_norm"] = prepared["title"].map(normalize_text)
    prepared["category_norm"] = prepared["category"].map(normalize_text)
    prepared["subcategory_norm"] = prepared["subcategory"].map(normalize_text)
    prepared["stock_number_norm"] = prepared["stock_number"].map(_stock_text)
    prepared["year_norm"] = prepared["year"].map(_stock_text)
    prepared["model_code"] = prepared["model"].map(extract_model_code)
    prepared["search_text"] = (
        prepared["year_norm"].astype(str)
        + " "
        + prepared["make_norm"].astype(str)
        + " "
        + prepared["model_norm"].astype(str)
        + " "
        + prepared["model_code"].astype(str)
        + " "
        + prepared["title_norm"].astype(str)
        + " "
        + prepared["stock_number_norm"].astype(str)
        + " "
        + prepared["category_norm"].astype(str)
        + " "
        + prepared["subcategory_norm"].astype(str)
    ).map(lambda value: re.sub(r"\s+", " ", value).strip())
    return prepared


# Table column -> the column name prepare_inventory expects. Everything else
# lines up by name already.
_DB_COLUMN_ALIASES = {"price_display": "price"}


def load_inventory_from_db() -> pd.DataFrame:
    """Every listing row as a DataFrame in the workbook's column shape.

    Reads through listing_search so there is one definition of "a listing row",
    and one place that knows the database might be switched off.
    """
    from src.search.listing_search import fetch_listings

    rows = fetch_listings([])
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {
            column.name: getattr(row, column.name)
            for column in row.__table__.columns
        }
        for source, target in _DB_COLUMN_ALIASES.items():
            record[target] = record.pop(source, None)
        # Ingest already flattened the spec blob into match_evidence_text, so the
        # JSON columns the workbook path re-parses per lookup are not needed.
        record.pop("info_json_source", None)
        records.append(record)
    return pd.DataFrame(records)


@lru_cache(maxsize=1)
def prepared_inventory() -> pd.DataFrame:
    """The catalogue to match against: the table when available, else the workbook.

    Cached for the process, like the workbook path always was — a re-ingest
    needs a restart to be picked up here.
    """
    from src import db

    if db.database_enabled():
        frame = load_inventory_from_db()
        if not frame.empty:
            logger.info("inventory_matcher_source | source=trailer_listings | rows=%s", len(frame))
            return prepare_inventory(frame)
        # An empty table is far more likely to be "not ingested yet" than "we
        # sold everything", so fall through rather than answer every lookup with
        # nothing.
        logger.warning("inventory_matcher_empty_table | falling back to workbook")

    if not _LISTINGS_FILE.exists():
        logger.warning("inventory_matcher_file_missing | path=%s", _LISTINGS_FILE)
        return prepare_inventory(pd.DataFrame())
    logger.info("inventory_matcher_source | source=workbook | path=%s", _LISTINGS_FILE)
    return prepare_inventory(load_inventory(_LISTINGS_FILE))


def _spec_blob_text(value: Any) -> str:
    raw = _clean_scalar(value)
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw[:1200]

    values: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if val in (None, "", [], {}):
                    continue
                if isinstance(val, (dict, list)):
                    walk(val)
                else:
                    values.append(f"{key}: {val}")
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif obj not in (None, ""):
            values.append(str(obj))

    walk(parsed)
    return " | ".join(values)[:1200]


def _inventory_match_evidence_text(row: pd.Series | dict[str, Any]) -> str:
    get = row.get
    # DB rows arrive with the evidence already flattened by ingest — the same
    # text the feature reranker reads, so both paths quote identical evidence.
    prebuilt = _clean_scalar(get("match_evidence_text"))
    if prebuilt:
        return prebuilt[:_INVENTORY_EVIDENCE_MAX_CHARS]
    parts = [
        _clean_scalar(get("title")),
        " ".join(
            part
            for part in (
                _clean_scalar(get("year")),
                _clean_scalar(get("make")),
                _clean_scalar(get("model")),
                _clean_scalar(get("trim")),
                _clean_scalar(get("category")),
                _clean_scalar(get("subcategory")),
            )
            if part
        ),
        _clean_scalar(get("search_text")),
        clean_dealer_notes(_clean_scalar(get("dealer_notes"))),
        _spec_blob_text(get("info_specs_json") or get("info_spec_json")),
    ]
    evidence = " | ".join(part for part in parts if part)
    return evidence[:_INVENTORY_EVIDENCE_MAX_CHARS]


def _row_to_listing(row: pd.Series | dict[str, Any], score: float | None = None) -> dict[str, Any]:
    get = row.get
    price = _clean_scalar(get("price"))
    listing = {
        "title": _clean_scalar(get("title")),
        "url": _clean_scalar(get("url")),
        "year": _clean_scalar(get("year")),
        "make": _clean_scalar(get("make")),
        "model": _clean_scalar(get("model")),
        "trim": _clean_scalar(get("trim")),
        "category": _clean_scalar(get("category")),
        "subcategory": _clean_scalar(get("subcategory")),
        "stock_number": _stock_text(get("stock_number")),
        "condition": _clean_scalar(get("condition")) or "New",
        "price": price,
        "price_display": price,
        "length": _clean_scalar(get("length")),
        "width": _clean_scalar(get("width")),
        "gvwr": _clean_scalar(get("gvwr")),
        "axle_capacity": _clean_scalar(get("axle_capacity")),
        "payload_capacity": _clean_scalar(get("payload_capacity")),
        "hitch_type": _clean_scalar(get("hitch_type")),
        # Card-shape parity with search results (milestone.md M6 step 6).
        "color": _clean_scalar(get("color")),
        "axles": _clean_scalar(get("axles")),
        "material": _clean_scalar(get("trailer_material")),
        "floor": _clean_scalar(get("floor")),
        "match_evidence_text": _inventory_match_evidence_text(row),
    }
    if score is not None:
        listing["relevance_score"] = round(float(score), 4)
    return listing


def _score_candidate(
    row: pd.Series,
    *,
    query_norm: str,
    make_norm: str,
    model_code_norm: str,
    model_text_norm: str,
) -> dict[str, float]:
    make_score = fuzz.partial_ratio(make_norm, row["make_norm"]) if make_norm else 0.0
    model_code_score = (
        max(
            fuzz.ratio(model_code_norm, row["model_code"]),
            fuzz.partial_ratio(model_code_norm, row["model_norm"]),
            fuzz.partial_ratio(model_code_norm, row["title_norm"]),
        )
        if model_code_norm
        else 0.0
    )
    model_text_score = (
        fuzz.token_set_ratio(model_text_norm, row["model_norm"])
        if model_text_norm
        else 0.0
    )
    title_score = fuzz.token_set_ratio(query_norm, row["title_norm"]) if query_norm else 0.0
    search_score = fuzz.token_set_ratio(query_norm, row["search_text"]) if query_norm else 0.0
    overall = (
        0.30 * make_score
        + 0.35 * model_code_score
        + 0.15 * model_text_score
        + 0.10 * title_score
        + 0.10 * search_score
    )
    return {
        "make_score": float(make_score),
        "model_code_score": float(model_code_score),
        "model_text_score": float(model_text_score),
        "title_score": float(title_score),
        "search_score": float(search_score),
        "overall": float(overall),
    }


@dataclass
class _Identifiers:
    """Direct-lookup identifiers already extracted by the Analyze LLM.

    Replaces the reference file's ``TrailerQueryExtraction`` (deleted along with
    its mini-LLM extractor) — this module never parses user text itself.
    """

    year: int | None = None
    possible_make: str | None = None
    possible_model_code: str | None = None
    possible_model_text: str | None = None
    stock_number: str | None = None


def _filter_same_make_rows(df: pd.DataFrame, make_norm: str) -> pd.DataFrame:
    if not make_norm or df.empty:
        return df.iloc[0:0]
    return df[df["make_norm"].map(lambda value: fuzz.ratio(make_norm, str(value or "")) >= 85)]


def _uses_year_metadata_filter(identifiers: _Identifiers) -> bool:
    # Every call into match_inventory is already a validated direct lookup
    # (the Analyze gate decided that upstream), so a year identifier alone is
    # enough to narrow the candidate frame — no wants-based gating needed here.
    return bool(identifiers.year)


def _candidate_frame(df: pd.DataFrame, identifiers: _Identifiers) -> pd.DataFrame:
    candidates = df
    if _uses_year_metadata_filter(identifiers):
        candidates = candidates[candidates["year_norm"] == str(identifiers.year)]
    return candidates


def _no_exact_alternative_rows(
    *,
    df: pd.DataFrame,
    identifiers: _Identifiers,
    query_norm: str,
    make_norm: str,
    top_scored: list[tuple[float, pd.Series, dict[str, float]]],
    limit: int,
) -> list[tuple[float, pd.Series, dict[str, float]]]:
    selected: list[tuple[float, pd.Series, dict[str, float]]] = []
    seen: set[str] = set()

    def add_rows(rows: list[tuple[float, pd.Series, dict[str, float]]]) -> None:
        for item in rows:
            row = item[1]
            key = _stock_text(row.get("stock_number")) or _clean_scalar(row.get("url")) or _clean_scalar(row.get("title"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= limit:
                return

    same_make_df = _filter_same_make_rows(df, make_norm)
    if identifiers.year and not same_make_df.empty:
        same_make_other_years = same_make_df[same_make_df["year_norm"] != str(identifiers.year)]
        if not same_make_other_years.empty:
            rows: list[tuple[float, pd.Series, dict[str, float]]] = []
            for _, row in same_make_other_years.iterrows():
                scores = _score_candidate(
                    row,
                    query_norm=query_norm,
                    make_norm=make_norm,
                    model_code_norm=normalize_text(identifiers.possible_model_code),
                    model_text_norm=normalize_text(identifiers.possible_model_text or identifiers.possible_model_code),
                )
                year_distance = abs(int(row.get("year_norm") or 0) - identifiers.year) if str(row.get("year_norm") or "").isdigit() else 99
                rows.append((max(0.0, 100.0 - year_distance), row, scores))
            rows.sort(key=lambda item: item[0], reverse=True)
            add_rows(rows)

    if len(selected) < limit:
        safe_same_year = [
            item
            for item in top_scored
            if identifiers.year
            and str(item[1].get("year_norm") or "") == str(identifiers.year)
            and item[2]["overall"] >= 25
        ]
        add_rows(safe_same_year)

    return selected


def match_inventory(
    identifiers: _Identifiers,
    df: pd.DataFrame | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    df = df if df is not None else prepared_inventory()
    query_norm = normalize_text(
        " ".join(
            part
            for part in (
                str(identifiers.year) if identifiers.year else "",
                identifiers.possible_make or "",
                identifiers.possible_model_text or "",
            )
            if part
        )
    )
    stock = _stock_text(identifiers.stock_number)
    requested_identifiers = {
        "stock_number": stock or None,
        "year": identifiers.year,
        "make": identifiers.possible_make or None,
        "model_code": identifiers.possible_model_code or None,
        "model_text": identifiers.possible_model_text or None,
    }
    metadata_filter = {"year": str(identifiers.year)} if _uses_year_metadata_filter(identifiers) else {}

    if stock:
        stock_matches = df[df["stock_number_norm"] == stock]
        if not stock_matches.empty:
            best = _row_to_listing(stock_matches.iloc[0], 100.0)
            logger.info("inventory_lookup_result | entity_type=%s | exact_match_count=1", _STOCK)
            return {
                "entity_type": _STOCK,
                "confidence": 1.0,
                "best_match": best,
                "top_matches": [best],
                "exact_match_count": 1,
                "requested_identifiers": requested_identifiers,
                "metadata_filter": metadata_filter,
                "no_exact_reason": "",
            }

    make_norm = normalize_text(identifiers.possible_make)
    model_code_norm = normalize_text(identifiers.possible_model_code)
    model_text_norm = normalize_text(identifiers.possible_model_text or identifiers.possible_model_code)

    candidates = _candidate_frame(df, identifiers)
    scored: list[tuple[float, pd.Series, dict[str, float]]] = []
    for _, row in candidates.iterrows():
        scores = _score_candidate(
            row,
            query_norm=query_norm,
            make_norm=make_norm,
            model_code_norm=model_code_norm,
            model_text_norm=model_text_norm,
        )
        scored.append((scores["overall"], row, scores))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_scored = scored[: max(limit, 8)]

    strong_make_rows = [item for item in top_scored if make_norm and item[2]["make_score"] >= 85]

    entity_type = _UNKNOWN
    selected = top_scored
    confidence = 0.0
    exact_match_count = 0
    no_exact_reason = ""

    if identifiers.year and make_norm and strong_make_rows and not (model_code_norm or model_text_norm):
        entity_type = _YEAR_MAKE
        selected = strong_make_rows
        confidence = max(item[2]["make_score"] for item in strong_make_rows) / 100
        exact_match_count = len(strong_make_rows)
    elif model_code_norm or model_text_norm:
        model_make_rows = [
            item
            for item in top_scored
            if (not make_norm or item[2]["make_score"] >= 75)
            and (item[2]["model_code_score"] >= 75 or item[2]["model_text_score"] >= 78)
            and item[2]["overall"] >= 58
        ]
        if model_make_rows:
            selected = model_make_rows
            confidence = min(0.99, max(item[2]["overall"] for item in model_make_rows) / 100)
            if identifiers.year and make_norm:
                entity_type = _YEAR_MAKE_MODEL
            elif model_make_rows[0][2]["model_code_score"] >= 85 or model_make_rows[0][2]["model_text_score"] >= 88:
                entity_type = _MODEL_SEARCH
            else:
                entity_type = _POSSIBLE_MODEL
            exact_match_count = len(model_make_rows)
    elif make_norm and strong_make_rows:
        entity_type = _MAKE_ONLY
        selected = strong_make_rows
        confidence = max(item[2]["make_score"] for item in strong_make_rows) / 100
    elif top_scored and top_scored[0][2]["overall"] >= 70:
        entity_type = _POSSIBLE_MODEL
        selected = top_scored
        confidence = top_scored[0][2]["overall"] / 100

    if not exact_match_count:
        no_exact_reason = (
            "no_exact_year_filtered_match_fallback_discarded_year"
            if metadata_filter
            else "no_exact_inventory_match_for_requested_identifiers"
        )
        alternative_rows = _no_exact_alternative_rows(
            df=df,
            identifiers=identifiers,
            query_norm=query_norm,
            make_norm=make_norm,
            top_scored=top_scored,
            limit=limit,
        )
        selected = alternative_rows
        confidence = 0.0
        if entity_type == _MAKE_ONLY:
            entity_type = _YEAR_MAKE if identifiers.year and make_norm else _POSSIBLE_MODEL

    top_matches = [_row_to_listing(row, score) for score, row, _scores in selected[:limit]]
    unique_exact = len(top_matches) == 1 and entity_type in {_MODEL_SEARCH, _YEAR_MAKE_MODEL}
    best_match = top_matches[0] if top_matches and (entity_type == _STOCK or unique_exact) else None
    if top_matches and entity_type == _POSSIBLE_MODEL and confidence >= 0.9:
        best_match = top_matches[0]

    logger.info(
        "inventory_lookup_result | entity_type=%s | confidence=%.3f | exact_match_count=%s | no_exact_reason=%s | matches=%s",
        entity_type,
        confidence,
        exact_match_count,
        no_exact_reason,
        len(top_matches),
    )
    return {
        "entity_type": entity_type,
        "confidence": round(confidence, 4),
        "best_match": best_match,
        "top_matches": top_matches,
        "exact_match_count": exact_match_count,
        "requested_identifiers": requested_identifiers,
        "metadata_filter": metadata_filter,
        "no_exact_reason": no_exact_reason,
    }


def lookup_inventory(
    *,
    year: int | None,
    make: str | None,
    model_text: str | None,
    stock_number: str | None,
    limit: int = 5,
) -> dict[str, Any]:
    """Pure function of identifiers — no user-text parsing.

    Returns ``{"match_status": "exact"|"no_exact"|"ambiguous"|"none", "matches": [...], "requested_label": str}``.
    """
    identifiers = _Identifiers(
        year=year,
        possible_make=make,
        possible_model_code=extract_model_code(model_text) if model_text else None,
        possible_model_text=model_text,
        stock_number=stock_number,
    )
    result = match_inventory(identifiers, limit=limit)
    top_matches = result["top_matches"]
    best_match = result.get("best_match")
    no_exact_reason = result.get("no_exact_reason") or ""
    entity_type = result["entity_type"]

    if no_exact_reason:
        match_status = "no_exact"
    elif not top_matches:
        match_status = "none"
    elif len(top_matches) == 1:
        match_status = "exact"
    elif entity_type in (_STOCK, _YEAR_MAKE):
        # Multiple genuinely valid rows for a stock/year+make lookup is a normal
        # result set, not an ambiguity about which single item the user meant.
        match_status = "exact"
    elif best_match is not None:
        match_status = "exact"
    else:
        # Several distinct model candidates cleared threshold with no clear
        # winner -> ask the user which one they mean (milestone.md M6 step 6).
        match_status = "ambiguous"

    label_parts = [str(year) if year else "", make or "", model_text or ""]
    requested_label = " ".join(part for part in label_parts if part).strip() or "that exact trailer"

    return {
        "match_status": match_status,
        "matches": top_matches,
        "requested_label": requested_label,
    }
