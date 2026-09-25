"""The conversation state carried between turns.

Key names are not free. ``category``, ``slots``, ``brand_preference``,
``non_metadata_features``, ``shown_urls``, ``qualification_complete`` and ``turn_outcome``
are the contract ``src/graph/nodes/search.py`` already reads - renaming any of them breaks
the finished retrieval path, so they stay exactly as they are.

Everything except ``turn_outcome`` is persisted to
``chatbot_conversations.state_snapshot``; ``turn_outcome`` is per-turn scratch and is
rebuilt from empty each turn.
"""
from __future__ import annotations

from typing import Any, TypedDict

# Bumped when the persisted shape changes in a way an old snapshot cannot satisfy.
STATE_SCHEMA_VERSION = 3

# A required question is asked at most twice, then dropped (brief S23).
MAX_ASKS_PER_SLOT = 2


class SessionState(TypedDict, total=False):
    session_id: str
    lead_id: str
    messages: list[dict[str, Any]]

    # ---- the search_node contract: do not rename ----
    category: str | None
    slots: dict[str, Any]
    brand_preference: str | None
    non_metadata_features: list[str]
    shown_urls: list[str]
    # The LAST batch put in front of them, in the order it was presented, trimmed to the
    # few fields a later turn needs. This is what "the 5th one" counts into, so it has to
    # outlive the turn that showed it - and it is replaced, never appended to, because the
    # customer counts down the list currently on their screen.
    last_shown_listings: list[dict[str, Any]]
    # Every trailer shown so far, trimmed the same way and capped. The customer can scroll
    # back, so "the 81419" may name something from three batches ago; the index counts into
    # the last batch, but an identifier is matched against all of them.
    shown_listings: list[dict[str, Any]]
    qualification_complete: bool
    turn_outcome: dict[str, Any]

    # ---- qualification bookkeeping ----
    required_slots: list[str]
    optional_slots: list[str]
    slot_questions: dict[str, str]
    asked_counts: dict[str, int]
    declined_slots: list[str]
    volunteered_slots: list[str]
    pending_slot: str | None
    invalid_retry_slot: str | None
    invalid_retry_reason: str | None

    # ---- question rules (src/rules) ----
    # Trait keys the model reported for the cargo named ("lightweight", "large_or_heavy").
    cargo_traits: list[str]
    # slot -> "default" | "user". Tells a value a rule filled in from one the customer gave.
    slot_sources: dict[str, str]
    # Defaults currently applied: slot -> {"value", "reason"}.
    rule_defaults: dict[str, dict[str, Any]]
    # Required questions a rule removed for this customer: slot -> reason.
    rule_skipped: dict[str, str]
    # Where each injected question sits: slot -> the slot it goes in front of (None = last).
    rule_ask_anchors: dict[str, str | None]

    # ---- pending confirmations ----
    pending_keep_filters: dict[str, Any] | None
    pending_category_switch: dict[str, Any] | None
    rejected_switches: list[str]
    # Trailer types they asked for that we do not carry, lower-cased. The team is told about
    # each one once; asking again gets the same answer without a second email.
    unavailable_requests: list[str]
    # "gooseneck" is both a hitch type and a make we stock, so a bare mention is worth one
    # short question rather than a guess. Holds the text that was ambiguous.
    pending_gooseneck_clarification: str | None
    # An axle capacity whose wording says neither per-axle nor total ("14,000 lbs of axle
    # capacity"): {"value": 14000.0, "asks": 1}. Held here, stored nowhere, until they say.
    pending_axle_basis: dict | None
    # "How many axles?" is open: {"asks": 1}. Asked after a per-axle rating with no count.
    pending_axle_count: dict | None

    # ---- contact / lead ----
    contact: dict[str, Any]
    # Notifications waiting on a name and a way to reach them. Persisted, so a request made
    # three turns ago still goes out when the customer finally hands over their number.
    pending_email_actions: list[dict[str, Any]]
    contact_followup_pending: str | None
    # Platforms the customer has sent us links from ("Facebook"). Kept on the session, not
    # read off the transcript: a notification stashed on the turn they shared the post may
    # not go out for another three turns, and it should still say where they saw us.
    shared_platforms: list[str]
    # Whether the team has already been told this conversation stalled on a question. Once
    # per conversation: the first one we lose is the signal, and the email links to the chat.
    gave_up_reported: bool
    # Trailers whose interest we have already told the team about, keyed by listing URL ("" for
    # an interest we could not pin to one row). A customer says "I like that one" more than
    # once; the team should hear about it once.
    listing_interest_keys: list[str]
    # Whether any listing interest has been recorded in this session. Read by compose: once a
    # customer has picked a trailer and we have logged it, "what type of trailer are you
    # looking for?" is the wrong thing to say next.
    listing_interest_logged: bool
    # The trailer they said they want, by title. It is what the lead row should be called:
    # a customer who picked one off the list is not shopping for "Livestock | length=32".
    interest_listing: str | None

    turn_index: int
    # Whether listings have been put in front of them at least once. It changes what a
    # later requirement change means: the first search is a search, the second is a redo.
    results_shown: bool
    # The categories trailers have been shown for. The 5-minute rule (src/idle_timer.py) runs
    # once per category, so a category in here never arms it again.
    results_categories: list[str]


# Scratch that must never be written to state_snapshot: it is rebuilt every turn and
# holds listing dicts that would bloat the row for no benefit.
_TRANSIENT_KEYS = ("turn_outcome",)


def new_state(session_id: str) -> SessionState:
    """A fresh session. Every collection is materialised so no node has to None-guard."""
    return SessionState(
        session_id=session_id,
        lead_id="",
        messages=[],
        category=None,
        slots={},
        brand_preference=None,
        non_metadata_features=[],
        shown_urls=[],
        last_shown_listings=[],
        shown_listings=[],
        qualification_complete=False,
        turn_outcome={},
        required_slots=[],
        optional_slots=[],
        slot_questions={},
        asked_counts={},
        declined_slots=[],
        volunteered_slots=[],
        pending_slot=None,
        invalid_retry_slot=None,
        invalid_retry_reason=None,
        cargo_traits=[],
        slot_sources={},
        rule_defaults={},
        rule_skipped={},
        rule_ask_anchors={},
        pending_keep_filters=None,
        pending_category_switch=None,
        rejected_switches=[],
        unavailable_requests=[],
        pending_gooseneck_clarification=None,
        pending_axle_basis=None,
        pending_axle_count=None,
        contact={
            "name": None, "email": None, "phone": None,
            "asked": False, "declined": False,
            # Asks that produced nothing new. Progress resets it; the gate stops at 2.
            "asks_without_progress": 0,
            # The turn the last ask went out on, so the second one is not asked the turn
            # after the first. 0 means we have never asked.
            "last_asked_turn": 0,
            # Whether the "great to have your details" line has already been said.
            "greeted": False,
        },
        pending_email_actions=[],
        contact_followup_pending=None,
        shared_platforms=[],
        gave_up_reported=False,
        listing_interest_keys=[],
        listing_interest_logged=False,
        interest_listing=None,
        turn_index=0,
        results_shown=False,
        results_categories=[],
    )


def to_snapshot(state: SessionState) -> dict[str, Any]:
    """The JSONB payload for chatbot_conversations.state_snapshot."""
    return {key: value for key, value in state.items() if key not in _TRANSIENT_KEYS}


def from_snapshot(session_id: str, snapshot: dict[str, Any] | None) -> SessionState:
    """Rebuild a session from a persisted snapshot.

    Defaults are laid down first and the snapshot merged over them, so a snapshot written
    by an earlier version simply lacks the newer keys rather than crashing a node that
    reads one. Unknown keys are dropped for the same reason in reverse.
    """
    state = new_state(session_id)
    if not snapshot:
        return state
    known = set(SessionState.__annotations__)
    for key, value in snapshot.items():
        if key in known and key not in _TRANSIENT_KEYS:
            state[key] = value  # type: ignore[literal-required]
    state["session_id"] = session_id
    state["turn_outcome"] = {}
    return state
