from __future__ import annotations

import logging

from src.config import settings
from src.domain.links import normalize_listing_url
from src.search.inventory_matcher import lookup_inventory
from src.tools.lookup_gate import lookup_requested, usable_stock_number

logger = logging.getLogger(__name__)


def inventory_lookup_node(state: dict) -> dict:
    outcome = state.setdefault("turn_outcome", {})
    turn = state.get("turn")
    lookup = turn.inventory_lookup if turn else None
    # No intent check: a lookup rides along with any intent (see build.py::_lookup_requested).
    # One source of truth with _route and the tool handler, so the node can never run on a
    # turn the gate would have refused.
    assert turn is not None and lookup is not None and lookup_requested(turn), (
        "inventory_lookup_node requires an approved inventory_lookup gate"
    )

    # Side-query invariant: a lookup never touches qualification state.
    guarded_keys = ("category", "slots", "brand_preference", "skipped_slots", "declined_slots", "qualification_complete")
    before = {key: state.get(key) for key in guarded_keys}

    logger.info(
        "TOOL inventory_lookup: session=%s year=%s make=%s model=%s stock=%s url=%s confidence=%s",
        state.get("session_id"), lookup.year, lookup.make, lookup.model_text, lookup.stock_number,
        getattr(lookup, "listing_url", None), lookup.confidence,
    )
    listing_url = normalize_listing_url(getattr(lookup, "listing_url", None))

    # A stock number that failed the plausibility check ("7000 lbs", or a number this turn
    # also extracted as the payload) is dropped rather than searched on: left in, it joins
    # the fuzzy query text and nudges make/model scoring toward nothing in particular. The
    # rest of the identifiers still run - the gate already established there are some.
    result = lookup_inventory(
        year=lookup.year,
        make=lookup.make,
        model_text=lookup.model_text,
        stock_number=usable_stock_number(turn),
        listing_url=listing_url,
        limit=settings.inventory_lookup_limit,
    )
    matches = result["matches"]

    shown = {normalize_listing_url(url) for url in state.get("shown_urls") or []}
    if matches and normalize_listing_url(matches[0].get("url")) in shown:
        # A trailer we already showed them: the card is on their screen, so it is not shown
        # again. The reply pass still gets its details, to answer their question.
        #
        # Deliberately NOT restricted to the link case it was written for. "I like the
        # 81419" is how people pick one off a list, and that arrives as a stock number: it
        # went straight past this check and printed the same card a second time.
        outcome["inventory_already_shown"] = matches[:1]
        result = {**result, "match_status": "already_shown", "matches": []}
        matches = []

    logger.info(
        "TOOL inventory_lookup: session=%s status=%s matches=%d requested=%r",
        state.get("session_id"), result["match_status"], len(matches), result.get("requested_label"),
    )

    # Recorded as shown by respond, from what it actually cited — see nodes/respond.py.
    # Ambiguous matches are DELIBERATELY passed as listings too: they render as full,
    # structured cards (real fields from the block, never invented), and the closing
    # question asks which one they mean. An earlier titles-only presentation was rolled
    # back — bare names with no link, price, or specs read worse than the cards.
    outcome["listings"] = matches
    outcome["inventory_result"] = result
    outcome["inventory_match_status"] = result["match_status"]
    # Durable signal the respond node/prompt key off — preserved from the M4 stub
    # contract exactly (respond.py:14 checks inventory_lookup_ran, not this stub).
    outcome["inventory_lookup_ran"] = True
    outcome["contact_invite_suppressed"] = True

    if matches:
        description = f"Inventory lookup — {len(matches)} results — {result['requested_label']}"
        outcome.setdefault("system_email_triggers", []).append(
            {"kind": "results_shown", "description": description}
        )

    for key in guarded_keys:
        assert state.get(key) == before[key], f"inventory_lookup_node must not mutate {key}"

    return state
