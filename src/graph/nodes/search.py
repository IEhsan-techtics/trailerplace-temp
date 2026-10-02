from __future__ import annotations

import logging
import random
from typing import Any

from src import turn_status
from src.config import settings
from src.tools.category import ALUMINUM
from src.domain.slot_map import normalize_slot_targets, sanitize_non_metadata_features
from src.search.listing_search import narrowing_filters_present, search_listings

logger = logging.getLogger(__name__)

# The hard SQL gates, in the words a customer would recognise them by. Everything else they
# told us (width, payload, features) is already a ranking signal rather than a gate.
# Add a label here whenever a new column becomes a filter in listing_search._listing_filters,
# so a relaxed retry can still tell the customer what it stopped filtering on.
_HARD_FILTER_LABELS = {
    "make": "brand",
    "hitch_type": "hitch type",
    "subcategory": "trailer type",
    "length_ft": "length",
}


# Said the moment the inventory search actually fires, so the customer knows we have gone to
# look rather than staring at a silent pause. Hardcoded on purpose: it costs no tokens and no
# latency, and it can never claim we checked stock on a turn where no search ran - this list is
# only ever read from inside search_node.
SEARCH_STATUS_LINES: tuple[str, ...] = (
    "Let me pull up what we have that fits.",
    "Give me a moment - I'll check what matches your requirements.",
    "Let me see what we have on the lot for you.",
    "I'll take a look through our current inventory.",
    "Let me find the ones that suit what you need.",
    "One moment while I check what we have in stock.",
    "Let me see what we've got that would work for you.",
)


def pick_search_status_line() -> str:
    return random.choice(SEARCH_STATUS_LINES)


def _relaxed_filter_labels(state: dict, metadata_filters: dict[str, Any]) -> list[str]:
    """What we stopped filtering on, for the reply to own up to."""
    labels = [label for key, label in _HARD_FILTER_LABELS.items() if metadata_filters.get(key)]
    if (state.get("slots", {}) or {}).get("hitch_type") and "hitch type" not in labels:
        labels.append("hitch type")
    return labels


def _build_metadata_filters(state: dict, category: str | None = None) -> dict[str, Any]:
    """Translate answered slots into SQL filter targets.

    Only category/make/hitch_type/subcategory/min-length become hard filters
    here; width/payload/height stay in ``slots`` for listing_search's fit
    rerank to weigh but are never gated on in the query itself
    (milestone.md M6 step 2).
    """
    category = category or state.get("category") or ""
    slots = state.get("slots", {}) or {}
    filters: dict[str, Any] = {}
    for slot_name, value in slots.items():
        if value is None:
            continue
        filters.update(normalize_slot_targets(category, slot_name, value))

    hitch_value = slots.get("hitch_type")
    if isinstance(hitch_value, list) and len(hitch_value) == 1:
        filters["hitch_type"] = hitch_value[0]

    if state.get("brand_preference"):
        filters["make"] = state["brand_preference"]

    return filters


# Several types at once (state["candidate_categories"]): each is searched and ranked on its
# own, and this many of each are shown - so one type's best never crowds out another's.
# More types than this table covers are searched together instead, as every type.
PER_TYPE_RESULTS = {2: 3, 3: 2}


def types_to_search(state: dict) -> list[str | None]:
    """The categories this search covers: the chosen one, each wanted one, or None for all."""
    if state.get("category"):
        return [state["category"]]
    wanted = list(state.get("candidate_categories") or [])
    if len(wanted) in PER_TYPE_RESULTS:
        return wanted
    return [None]


def _search_one(
    state: dict, category: str | None, max_recommendations: int,
    requested_features: list[str], shown_urls: list[str], *, relax_to_category: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool, bool]:
    """One category's search, with the usual fallbacks. Returns the results, the filters, and
    whether the brand or the other hard filters had to be let go."""
    slots = state.get("slots", {}) or {}
    metadata_filters = _build_metadata_filters(state, category)

    results = search_listings(
        category=category,
        slots=slots,
        metadata_filters=metadata_filters,
        requested_features=requested_features,
        already_shown_urls=shown_urls,
        max_recommendations=max_recommendations,
    )

    brand_relaxed = False
    # A requested make remains a hard constraint in feature-aware search. The
    # legacy/no-feature path keeps its existing zero-result make relaxation.
    if not results and metadata_filters.get("make") and not requested_features:
        relaxed_filters = {key: value for key, value in metadata_filters.items() if key != "make"}
        logger.info(
            "TOOL search: session=%s zero results with make filter, relaxing and retrying filters=%s",
            state.get("session_id"), relaxed_filters,
        )
        results = search_listings(
            category=category,
            slots=slots,
            metadata_filters=relaxed_filters,
            requested_features=requested_features,
            already_shown_urls=shown_urls,
            max_recommendations=max_recommendations,
        )
        brand_relaxed = True

    # Still nothing. Every hard filter is all-or-nothing — one 24 ft minimum, one hitch type, one
    # brand — so a single unmet requirement empties the screen even when the category is full of
    # trailers the customer would happily look at. Drop the gates, keep the CATEGORY, and search
    # again: their size requirements survive in the fit rerank and their feature requests in the
    # feature rerank, so what comes back is ordered by how close it gets. The reply presents it
    # as alternatives rather than as matches.
    #
    # The non-metadata features are NOT dropped here. They are the customer's standing preferences
    # (sliding gates, a ramp), they only ever ranked rather than filtered, and they are cleared in
    # one place only: a category change.
    filters_relaxed = False
    if not results and relax_to_category and narrowing_filters_present(category, slots, metadata_filters):
        logger.info(
            "TOOL search: session=%s zero results, relaxing every filter except category=%r and retrying",
            state.get("session_id"), category,
        )
        results = search_listings(
            category=category,
            slots=slots,
            metadata_filters=metadata_filters,
            requested_features=requested_features,
            already_shown_urls=shown_urls,
            max_recommendations=max_recommendations,
            category_only_filters=True,
        )
        filters_relaxed = bool(results)
    return results, metadata_filters, brand_relaxed, filters_relaxed


def _off_aluminum(state: dict, base: str) -> str:
    """No aluminum trailers of the type they want: drop Aluminum and make that type the category.

    Recorded on the outcome, so the reply tells them these are not aluminum.
    """
    from src.tools.category import normalize_category, set_trailer_category

    canonical = normalize_category(base) or base
    (state.get("slots") or {}).pop("base_category", None)
    (state.get("slot_sources") or {}).pop("base_category", None)
    set_trailer_category(state, canonical)
    state.setdefault("turn_outcome", {})["aluminum_dropped"] = canonical
    logger.info(
        "TOOL search: session=%s no aluminum %s - searching %s instead", state.get("session_id"), canonical, canonical,
    )
    return canonical


def search_node(state: dict) -> dict:
    assert state.get("qualification_complete"), "search_node requires qualification_complete"
    outcome = state.setdefault("turn_outcome", {})
    types = types_to_search(state)
    if types == [None]:
        # No type, or more of them than are searched one by one: every type, ranked on the specs.
        outcome["all_types"] = True
    if len(types) > 1:
        outcome["several_types"] = list(types)
    per_type = PER_TYPE_RESULTS.get(len(types), settings.search_max_recommendations)
    # Every non-searchable preference they have voiced so far (sliding gates, tandem axle, ramp),
    # not just this turn's — the feature reranker scores the customer's full spec, not their
    # last line.
    requested_features, _ = sanitize_non_metadata_features(
        state.get("non_metadata_features", []) or []
    )
    # Repair feature phrases persisted by an older Analyze prompt as well:
    # ["insulated", "insulated enclosed"] becomes ["insulated"].
    state["non_metadata_features"] = requested_features
    shown_urls = state.get("shown_urls", []) or []

    # Set here, beside the tool-call log line, so it exists if and only if a search really ran.
    # Published to turn_status as well as the outcome: the outcome reaches the client when the
    # turn ends, but the UI is waiting NOW and polls turn_status to show this during the search.
    status_line = pick_search_status_line()
    outcome["search_status_message"] = status_line
    turn_status.publish(state.get("session_id"), status_line)

    results: list[dict[str, Any]] = []
    brand_relaxed = filters_relaxed = False
    relaxed_labels: list[str] = []
    metadata_filters: dict[str, Any] = {}
    searched: list[str] = []
    for category in types:
        logger.info(
            "TOOL search: session=%s category=%s filters=%s features=%s already_shown=%d",
            state.get("session_id"), category, _build_metadata_filters(state, category),
            requested_features, len(shown_urls),
        )
        # Aluminum built as another type ("an aluminum utility trailer"): letting the type go
        # would show aluminum trailers of every kind, so it is never relaxed away. If there are
        # none, the type itself is searched instead, and the reply says so.
        base = (state.get("slots") or {}).get("base_category") if category == ALUMINUM else None
        base = None if base == ALUMINUM else base
        found, metadata_filters, brand_off, filters_off = _search_one(
            state, category, per_type, requested_features, shown_urls, relax_to_category=not base,
        )
        if not found and base:
            category = _off_aluminum(state, base)
            found, metadata_filters, brand_off, filters_off = _search_one(
                state, category, per_type, requested_features, shown_urls,
            )
        if category:
            searched.append(category)
        results.extend(found)
        brand_relaxed = brand_relaxed or brand_off
        if filters_off:
            filters_relaxed = True
            relaxed_labels += [
                label for label in _relaxed_filter_labels(state, metadata_filters)
                if label not in relaxed_labels
            ]

    logger.info(
        "TOOL search: session=%s results=%d brand_relaxed=%s filters_relaxed=%s urls=%s",
        state.get("session_id"), len(results), brand_relaxed, filters_relaxed,
        [r.get("url") for r in results],
    )

    # NOT recorded as shown here: respond decides what actually reaches the customer, and it
    # is respond that records it. Marking them shown from this side told the team we had
    # presented six trailers the customer never saw, and locked those six out of every later
    # "show me more".
    state["last_search_filters"] = metadata_filters
    state["search_pending"] = False

    outcome["listings"] = results
    outcome["search_ran"] = True
    outcome["result_count"] = len(results)
    if brand_relaxed:
        outcome["brand_relaxed"] = True
    if filters_relaxed:
        outcome["filters_relaxed"] = True
        outcome["relaxed_filters_dropped"] = relaxed_labels

    if results:
        filter_desc = ", ".join(f"{key}={value}" for key, value in metadata_filters.items())
        description = f"Inventory search — {len(results)} results — {', '.join(searched) or 'All types'}"
        if filter_desc:
            description += f" ({filter_desc})"
        outcome.setdefault("system_email_triggers", []).append(
            {"kind": "results_shown", "description": description}
        )
    return state
