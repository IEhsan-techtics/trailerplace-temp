from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.domain.links import normalize_listing_url
from src.search.inventory_matcher import lookup_inventory
from src.tools.lookup_gate import lookup_requested, usable_stock_number

logger = logging.getLogger(__name__)


def _about_one_trailer(turn: Any, total_matched: int) -> bool:
    """Is this about a single trailer they have already seen?

    Two ways it can be. They are PICKING one - "I like the 81419", or a link to a card still
    on their screen - in which case the answer is to log the interest, not to print the card
    twice. Or their question resolves to exactly one unit, and the reply pass can answer it
    from that unit's details in a sentence; re-pasting the card they are looking at reads
    like a bot that was not listening.

    A question that matches SEVERAL units is neither. "Do you have any 2026 Galyean
    trailers?" collapsed to one here and was answered "the available unit is 32 ft long".
    There were six.
    """
    if bool(getattr(turn, "shared_link_interest", False)):
        return True
    if getattr(turn, "intent", "") == "listing_interest":
        return True
    return total_matched <= 1


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
    total_matched = int(result.get("total_matched") or len(matches))
    if (
        matches
        and normalize_listing_url(matches[0].get("url")) in shown
        and _about_one_trailer(turn, total_matched)
    ):
        # One trailer they have already seen: the card is on their screen, so it is not
        # printed again and the reply pass answers from its details instead.
        #
        # The condition used to be the already-shown check ALONE, testing only matches[0] and
        # then throwing the whole result set away. Asked "do you have any 2026 Galyean
        # trailers?" with six on the lot, five of them came back, the first had been shown
        # earlier in the chat, and all five were discarded - so the model answered from
        # conversation memory and said "the available unit is 32 ft long".
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
    outcome["inventory_total_matched"] = total_matched
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
