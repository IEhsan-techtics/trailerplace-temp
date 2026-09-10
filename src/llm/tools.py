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
        # Everything the tools have handed over THIS turn, accumulated across calls.
        #
        # ``state["shown_urls"]`` only grows in compose, which runs after the agent loop has
        # finished - so without this a second search_inventory inside one loop would see the
        # same exclusion list and return the same trailers, and the agent would ping-pong on
        # identical results. Tracked here instead, so a second call is a genuine next page.
        self.served_listings: list[Any] = []
        self._served_urls: list[str] = []

    def _remember(self, listings: list[Any]) -> None:
        """Record what a tool just returned, de-duplicated, and publish the running total.

        ``turn_outcome["listings"]`` becomes everything served this turn rather than only the
        last call's batch: it is what the API response carries and what the cited-URL check
        reads, and a first batch the reply cited would otherwise be forgotten.
        """
        for listing in listings:
            url = str(_listing_get(listing, "url") or "").strip()
            if url and url in self._served_urls:
                continue
            if url:
                self._served_urls.append(url)
            self.served_listings.append(listing)
        self.state.setdefault("turn_outcome", {})["listings"] = list(self.served_listings)

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

        # Hide what we have already handed over this turn, so a second call is the NEXT page
        # rather than the same one again. Restored afterwards because what the customer was
        # actually shown is decided by what the reply cites, and compose records that.
        persisted = list(self.state.get("shown_urls") or [])
        if self._served_urls:
            self.state["shown_urls"] = persisted + [
                url for url in self._served_urls if url not in persisted
            ]
        try:
            search_node(self.state)
        finally:
            self.state["shown_urls"] = persisted

        self.ran.append("search_inventory")
        outcome = self.state.get("turn_outcome") or {}
        fresh = outcome.get("listings") or []
        if not fresh and self._served_urls:
            return (
                "NO FURTHER MATCHES: you have already been given every trailer we have for "
                "these requirements. Present the ones you already have and do not search again."
            )
        self._remember(fresh)
        return "\n".join(
            part for part in (self._match_quality(outcome, fresh), listing_block(fresh)) if part
        )

    def _match_quality(self, outcome: dict, listings: list[Any]) -> str:
        """How well these actually match - stated with the results, not left to be inferred.

        The search relaxes its filters rather than showing an empty screen, so "here are five
        trailers" can mean three different things. Which one is a fact about THIS call, so it
        travels with the results instead of sitting in the static prompt as a rule the agent
        has to remember to apply.
        """
        if not listings:
            seen_before = bool(self.state.get("shown_urls"))
            reason = (
                "they have already seen every match we have for these requirements"
                if seen_before
                else "nothing in our current stock matches their requirements"
            )
            return (
                f"NO MATCHES: {reason}. Say exactly that in one honest, friendly sentence. Show "
                "no cards and do not repeat or re-link any trailer already shown. Offer to "
                "adjust a requirement - a different size, hitch or feature - to open up more "
                "options, and give them 979-532-1486 and our website."
            )

        if outcome.get("brand_relaxed"):
            brand = self.state.get("brand_preference") or "the brand they asked for"
            return (
                f"NOT {brand.upper()}: nothing from {brand} matched their requirements, so these "
                f"are the closest we have from OTHER makes. OPEN your reply with one honest "
                f"sentence saying we do not currently have a {brand} matching what they asked "
                "for, and that these are close alternatives from other brands. Then show every "
                f"card as normal. Never imply any of these IS a {brand}."
            )

        if outcome.get("filters_relaxed"):
            dropped = ", ".join(outcome.get("relaxed_filters_dropped") or [])
            on = f" (particularly {dropped})" if dropped else ""
            return (
                "ALTERNATIVES, NOT EXACT MATCHES: nothing in stock met every requirement they "
                f"gave, so these are the closest we have{on}. OPEN your reply with one honest, "
                "matter-of-fact sentence saying BOTH that no trailer matches all their "
                "requirements and that these are close alternatives that could still suit them. "
                "Then show every card as normal. Never present these as exact matches, and "
                "never imply one meets the requirement it misses."
            )

        return ""

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
        listings = list(outcome.get("listings") or [])
        self._remember(listings)
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
