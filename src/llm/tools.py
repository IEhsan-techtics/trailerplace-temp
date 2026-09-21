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
    pitch_material = _pitch_material(listing)
    if pitch_material:
        parts.append(f"FOR THE PITCH ONLY (never a bullet): {pitch_material}")
    return " | ".join(parts)


# What the pitch sentence is written FROM. The model used to see only the fields that become
# bullets, so the one thing it could say about a trailer was a bullet it had just printed -
# "Its 32-foot length and 16,345-pound payload provide substantial capacity" under Length:
# 32 ft and Payload: 16345 lbs. 182 of the 258 rows carry a features list that says what the
# trailer is actually like.
def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "none", "null", "n/a", "unspecified"} else text


def _pitch_material(listing: Any) -> str:
    """A capped slice of the features, then material and floor - never a value on the card.

    Features come FIRST. The model takes the first detail it is given, and material and floor
    are the same on most of the lot: with them in front, five steel dump trailers were each
    pitched on their "steel floor" while one of them had a long-arm tarp system nobody
    mentioned.

    Search results carry ``pitch_features`` - it strips the full ``features`` list before
    results leave it, so reading ``features`` alone found nothing on the path that matters
    most. ``features`` is still read for any caller that hands over a raw row.
    """
    from src.search.listing_search import pitch_features

    bits: list[str] = []
    kept = _listing_get(listing, "pitch_features", None)
    if kept is None:
        kept = pitch_features(_listing_get(listing, "features"))
    if kept:
        bits.append("features: " + "; ".join(kept))

    # Search names it "material"; the table column is trailer_material.
    material = _clean(_listing_get(listing, "material") or _listing_get(listing, "trailer_material"))
    if material:
        bits.append(f"material {material.lower()}")
    floor = _clean(_listing_get(listing, "floor"))
    if floor and floor.lower() != material.lower():
        bits.append(f"floor {floor.lower()}")
    return ", ".join(bits)


def listing_block(listings: list[Any]) -> str:
    """The tool result the model reads. Plain lines, not JSON - it has to copy values out of
    this into prose, and a flat labelled line is far harder to misread than nested JSON."""
    if not listings:
        return "NO MATCHES. Do not show any listings and do not say anything about our stock levels."
    lines = [listing_line(index, listing) for index, listing in enumerate(listings, start=1)]
    return "\n".join(lines)


# What each lookup status means and what the reply has to do with it. Ported from New Prompt's
# reply prompt (src/llm/respond.py), where the same four statuses come out of the same matcher.
#
# Sent WITH the result rather than sitting in the system prompt: the model reads the meaning
# right beside the listings it applies to, and the turns with no lookup - most of them - pay
# nothing for it. Before this the model was handed "MATCH STATUS: no_exact" and left to work
# out from the word alone that the trailers under it were substitutes.
_LOOKUP_GUIDANCE = {
    "exact": (
        "WHAT THIS MEANS: these are the trailer(s) they asked about. Present them warmly as "
        "full cards, then ask whether they're interested in any of them."
    ),
    "no_exact": (
        "WHAT THIS MEANS: we do NOT currently show {label}. The listings below are the closest "
        "alternatives, not the trailer they asked for. OPEN with one honest sentence saying we "
        "don't currently show {label}, then present these as close alternatives. Never present "
        "any of them as the one they asked about, and never invent a spec to make one fit."
    ),
    "ambiguous": (
        "WHAT THIS MEANS: we carry SEVERAL models matching what they asked, and nothing says "
        "which one they mean. Open with one short line saying so, show EVERY listing below as a "
        "full card, and END with ONE question asking which of these they mean or are most "
        "interested in. That question REPLACES the usual closing - never end this reply with "
        '"Do any of these look like a fit, or would you like to see more options?".'
    ),
    # New Prompt has no rule for this one. The search wording ("do not say anything about our
    # stock levels") is wrong for a lookup: the customer asked whether we have a specific
    # trailer, and being forbidden to say we can't find it leaves the question unanswered.
    "none": (
        "WHAT THIS MEANS: we can't find {label} in our current inventory, and there is nothing "
        "close enough to offer instead. Say that plainly in one friendly sentence - show no "
        "cards and invent nothing. Give them 979-532-1486 and our website so the team can check "
        "for them, and offer to help them find something similar if they tell you what they "
        "need it for."
    ),
}


_GENERIC_LOOKUP_LABEL = "that exact trailer"


def _lookup_label(turn: Any, outcome: dict) -> str:
    """How the rule names what they asked for.

    The matcher builds its label from year, make and model only, so a stock-number lookup
    came back as "that exact trailer" and the rule read "we can't find that exact trailer".
    The stock number is the thing they typed, so it is the thing to name. Taken through the
    plausibility gate, so a weight that slipped into the field is never quoted back.
    """
    from src.tools.lookup_gate import usable_stock_number

    label = (outcome.get("inventory_result") or {}).get("requested_label") or ""
    if label and label != _GENERIC_LOOKUP_LABEL:
        return label
    stock = usable_stock_number(turn)
    return f"stock #{stock}" if stock else _GENERIC_LOOKUP_LABEL


def lookup_guidance(status: str, label: str) -> str:
    """The reply rule for one lookup result. An unknown status is treated as no match."""
    template = _LOOKUP_GUIDANCE.get(status, _LOOKUP_GUIDANCE["none"])
    return template.format(label=label or "that trailer")


# Also from New Prompt. They came with a specific trailer in mind, and the general invitation
# to leave their details gets in the way of the answer. A request they made us ACT on is
# different - if an escalation this turn asked for their details, that ask still stands.
_LOOKUP_CONTACT_RULE = (
    "Do NOT invite them to share their name, email or phone in this reply - they asked about a "
    "specific trailer, so answer that. (If an escalation this turn told you to ask for their "
    "details, that request still stands.)"
)


def _reply_instruction(state: dict, answer: str, status: str) -> str:
    """What to tell the customer, given whether the notification went out or is waiting.

    Two sentences in the stashed case, and the order matters: the canned line FIRST, so they
    know their request landed, then the ask, so it reads as the reason we need the detail
    rather than a toll gate in front of the answer.
    """
    from src.tools import team_notify

    if status == "stashed":
        return (
            f'RECORDED, but we cannot send it to the team until we can reach them. Say TWO '
            f'things, in this order, in your own words but keeping the meaning and the phone '
            f'number: (1) "{answer}" (2) "{team_notify.ask_for_missing(state)}" '
            "Ask for nothing else and ask no qualification question."
        )
    if status == "dropped":
        return (
            f'They declined to share contact details, so nothing was sent and we do not ask '
            f'again. Answer them helpfully and give them the number: "{answer}" Do NOT ask '
            "for their details and do not imply anyone will call them back."
        )
    return (
        f'PASSED TO THE TEAM. Tell them this, in your own words but keeping the meaning and '
        f'the phone number: "{answer}" Ask no qualification question.'
    )


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
            ("pending_axle_basis", "whether their axle capacity is per axle or the total"),
            ("pending_axle_count", "how many axles they want"),
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
        already = outcome.get("inventory_already_shown")
        if already:
            return (
                "ALREADY SHOWN: that is a trailer you already showed them, so do NOT show "
                "its card again. Answer what they asked about it in a sentence or two, from "
                "these details only:\n" + listing_block(already)
            )
        listings = list(outcome.get("listings") or [])
        label = _lookup_label(turn, outcome)
        self._remember(listings)

        # The matcher reports a stock number that does not exist as no_exact with NO rows - it
        # found nothing to offer as an alternative either. The no_exact rule talks about "the
        # listings below", so the rule is chosen by what actually came back, not the label.
        rule = status if listings else "none"
        parts = [f"MATCH STATUS: {status}", lookup_guidance(rule, label)]
        if listings:
            parts.append(listing_block(listings))
        if outcome.get("contact_invite_suppressed"):
            parts.append(_LOOKUP_CONTACT_RULE)
        return "\n".join(parts)

    # -- escalation ------------------------------------------------------------------
    # Reason -> the fixed Reason vocabulary for the email subject and body, and which canned
    # line the customer hears back. company_and_email_scenarios.md fixes both.
    _ESCALATION_REASONS = {
        "complaint": ("Escalation", "complaint"),
        "callback": ("Team Request", "team_request"),
        "meeting": ("Team Request", "team_request"),
        "quote": ("Team Request", "team_request"),
        "pricing": ("Team Request", "team_request"),
        "delivery": ("Team Request", "team_request"),
        "paperwork": ("Team Request", "team_request"),
        "viewing": ("Team Request", "team_request"),
        "stock_question": ("Team Request", "team_request"),
        "unstocked_type": ("Team Request", "team_request"),
        "listing_interest": ("Listing Interest", "listing_interest"),
        "other": ("Team Request", "team_request"),
    }

    def _contact(self) -> dict[str, Any]:
        return self.state.get("contact") or {}

    def _escalate(self, reason: str, summary: str) -> str:
        """Record something for the team, under the contact gate.

        Nothing is sent until we hold a name AND an email or phone. Short of that the request
        is STASHED - never dropped - and goes out the moment they tell us how to reach them.
        The customer still gets their answer this turn either way; only the email waits.
        """
        from src.domain import canned_responses
        from src.tools import team_notify

        key = str(reason or "other").strip().lower()
        reason_line, canned_key = self._ESCALATION_REASONS.get(key, self._ESCALATION_REASONS["other"])

        outcome_now = self.state.get("turn_outcome") or {}
        already = outcome_now.get("link_interest") or outcome_now.get("listing_interest")
        if already and canned_key == "listing_interest":
            # Already sent (or stashed) by apply: from the link they shared, or from the
            # trailer they pointed at. A second record would be a second email about the same
            # trailer - and the customer is told the same thing either way.
            self.ran.append("escalate")
            answer = canned_responses.escalation_answer(canned_key, already["status"])
            return _reply_instruction(self.state, answer, already["status"])

        status = team_notify.record(self.state, reason=reason_line, description=summary)
        outcome = self.state.setdefault("turn_outcome", {})
        outcome["escalated"] = True
        if canned_key == "complaint":
            # Nothing else happens this turn. Read by compose's fallback and by the gate.
            outcome["escalation_owns_turn"] = True
        self.ran.append("escalate")

        logger.info(
            "TOOL escalate: session=%s reason=%r status=%s", self.state.get("session_id"),
            reason_line, status,
        )

        # Worded for what actually happened to the email: "I've passed it to our team" is
        # only true when it went out, and it is the wrong thing to say either to someone we
        # are still waiting on details from or to someone who refused to give them.
        answer = canned_responses.escalation_answer(canned_key, status)
        return _reply_instruction(self.state, answer, status)

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
            if name == "escalate":
                return self._escalate(args.get("reason") or "other", args.get("summary") or "")
            if name == "lookup_inventory":
                return self._lookup_inventory(
                    year=args.get("year"),
                    make=args.get("make"),
                    model_text=args.get("model_text"),
                    stock_number=args.get("stock_number"),
                    listing_url=args.get("listing_url"),
                )
        except Exception:
            logger.exception("TOOL %s failed: session=%s", name, self.state.get("session_id"))
            return "That lookup failed on our side. Do not mention it; answer them without listings."
        logger.warning("TOOL unknown tool requested: %r", name)
        return f"No such tool: {name}."
