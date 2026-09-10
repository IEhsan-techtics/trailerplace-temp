"""The tools the model may call, and what it is shown when one returns.

Only the reply-writing pass gets these (see src/llm/respond.py). The analysis pass runs
before ``apply_node``, where ``state["slots"]`` still holds LAST turn's answers - a search
fired from there would filter on stale values and miss the "make it 24 ft" the customer
just said. By the time this pass runs, apply has folded the turn in and the filters are
current.

The tool SCHEMAS live on the ``@tool`` functions in src/graph/agent.py, which is the single
place they are declared. This module is the implementation behind them, and two rules shape
it:

* ``search_inventory`` takes NO arguments. The filters are built from state by
  ``search_node``'s own ``_build_metadata_filters``; letting the model pass them would let
  it search on a size or a brand the customer never gave.
* ``lookup_inventory`` takes only identifiers, and they go through the same
  ``src/tools/lookup_gate`` the deterministic path uses, so a bare make or a category word
  cannot become a side query here either.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _listing_get(listing: Any, key: str, default: Any = "") -> Any:
    return listing.get(key, default) if isinstance(listing, dict) else getattr(listing, key, default)


# The fields a card can carry, in the order they are shown. Every one is missing on some
# trailer in the catalogue, so none is guaranteed.
_LISTING_FIELDS: tuple[tuple[str, str], ...] = (
    ("Category", "category"),
    ("Make", "make"),
    ("Price", "price_display"),
    ("Length", "length"),
    ("Width", "width"),
    ("Payload", "payload_capacity"),
    ("Axle capacity", "axle_capacity"),
    ("Hitch type", "hitch_type"),
)


def listing_line(index: int, listing: Any) -> str:
    """One listing, with the fields it does NOT have left out entirely.

    A missing field rendered as the literal "None" gets copied onto the card, and a trailer
    whose length we simply do not hold is advertised as having a length of None. The model
    cannot omit what it is never shown, so the omission happens here.
    """
    title = str(_listing_get(listing, "title") or "").strip()
    price = _listing_get(listing, "price_display") or _listing_get(listing, "price")
    parts = [f"{index}. TITLE: {title}"]
    for label, key in _LISTING_FIELDS:
        value = price if key == "price_display" else _listing_get(listing, key)
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value if str(item or "").strip())
        text = str(value or "").strip()
        if text and text.lower() not in {"none", "null", "n/a"}:
            parts.append(f"{label}: {text}")
    parts.append(f"URL: {str(_listing_get(listing, 'url') or '').strip()}")
    return " | ".join(parts)


def listing_block(listings: list[Any]) -> str:
    """The tool result the model reads. Plain lines, not JSON - it has to copy values out of
    this into prose, and a flat labelled line is far harder to misread than nested JSON."""
    if not listings:
        return "NO MATCHES. Do not show any listings and do not say anything about our stock levels."
    lines = [listing_line(index, listing) for index, listing in enumerate(listings, start=1)]
    return "\n".join(lines)


class ToolRunner:
    """Executes one turn's tool calls against the session state.

    Holds the state so the tools can take no (or few) arguments, and records what ran so the
    caller can tell whether listings reached the model.
    """

    def __init__(self, state: dict, turn: Any):
        self.state = state
        self.turn = turn
        self.ran: list[str] = []

    # -- preconditions ---------------------------------------------------------------
    def _search_refusal(self) -> str | None:
        """Why a search must not run, in words the model can act on. None means go ahead.

        These are the HARD preconditions, and they are Python's call, not the model's: the
        model decides whether the customer wants to see trailers, never whether we are in a
        position to show them any.
        """
        from src.graph.nodes import greeting

        if not self.state.get("category"):
            return (
                "NO SEARCH RAN: no trailer category has been chosen yet, so there is nothing to "
                "search. This says NOTHING about our stock. Ask which type of trailer they want."
            )
        for key, what in (
            ("pending_keep_filters", "whether to keep their earlier answers"),
            ("pending_category_switch", "whether to switch category"),
            ("pending_gooseneck_clarification", "whether they meant the gooseneck hitch or the brand"),
        ):
            if self.state.get(key):
                return (
                    f"NO SEARCH RAN: we are waiting on their answer about {what}. "
                    "Ask that question and show no listings."
                )
        if greeting.contact_gate_applies(self.state):
            return (
                "NO SEARCH RAN: we have not asked who we are speaking with yet. "
                "Ask for their name and a way to reach them, and show no listings."
            )
        return None

    # -- the tools -------------------------------------------------------------------
    def _search_inventory(self) -> str:
        refusal = self._search_refusal()
        if refusal:
            logger.info("TOOL search_inventory refused: session=%s %s", self.state.get("session_id"), refusal)
            return refusal

        from src.graph.nodes.search import search_node

        # search_node asserts on this. The gate above has established the search is warranted;
        # apply_node recomputes the flag properly on the next turn either way.
        self.state["qualification_complete"] = True
        search_node(self.state)
        self.ran.append("search_inventory")
        listings = (self.state.get("turn_outcome") or {}).get("listings") or []
        return listing_block(listings)

    def _lookup_inventory(self, **identifiers: Any) -> str:
        from src.graph.nodes.inventory_lookup import inventory_lookup_node
        from src.tools.lookup_gate import lookup_requested

        # The model's identifiers, not the analysis pass's - but judged by the same gate, so a
        # bare make or a category word is refused here exactly as it is on the deterministic path.
        turn = self.turn.model_copy(deep=True)
        for field, value in identifiers.items():
            setattr(turn.inventory_lookup, field, value)
        turn.inventory_lookup.is_lookup = True
        turn.inventory_lookup.confidence = "high"

        if not lookup_requested(turn):
            logger.info("TOOL lookup_inventory refused: session=%s %r", self.state.get("session_id"), identifiers)
            return (
                "NO LOOKUP RAN: that is not a specific trailer we can look up. A make on its own "
                "is a brand preference and a trailer type is not a model. Ask which type of "
                "trailer they want, or which model."
            )

        self.state["turn"] = turn
        try:
            inventory_lookup_node(self.state)
        finally:
            self.state.pop("turn", None)
        self.ran.append("lookup_inventory")

        outcome = self.state.get("turn_outcome") or {}
        status = outcome.get("inventory_match_status") or "none"
        listings = outcome.get("listings") or []
        return f"MATCH STATUS: {status}\n{listing_block(listings)}"

    # -- dispatch --------------------------------------------------------------------
    def call(self, name: str, arguments: str) -> str:
        """Run one tool call. Never raises - a broken tool degrades to a line the model can
        still write a sensible reply around."""
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            logger.warning("TOOL %s: unparseable arguments %r", name, arguments)
            args = {}
        try:
            if name == "search_inventory":
                return self._search_inventory()
            if name == "lookup_inventory":
                return self._lookup_inventory(
                    year=args.get("year"),
                    make=args.get("make"),
                    model_text=args.get("model_text"),
                    stock_number=args.get("stock_number"),
                )
        except Exception:
            logger.exception("TOOL %s failed: session=%s", name, self.state.get("session_id"))
            return "That lookup failed on our side. Do not mention it; answer them without listings."
        logger.warning("TOOL unknown tool requested: %r", name)
        return f"No such tool: {name}."
