"""The deterministic layer: the model's evidence becomes state. No LLM calls, no I/O.

Every business rule in the brief is enforced here or in ``src/tools/``. The model's output
is read as EVIDENCE about what the customer said, never as an instruction about what to do
- which is why a bad classification degrades a reply rather than corrupting a session.

The order of the steps is load-bearing:

1. contact         - independent of everything, so it cannot be lost to an early return
2. confirmations   - a pending yes/no is about the PREVIOUS turn and must be read before
                     this turn's category or filters can overwrite the thing it answers
3. gooseneck       - decides whether "gooseneck" is a hitch or a make, before either is stored
4. category        - may open a keep-filters question instead of switching immediately
5. filters         - runs with the category settled, so per-category parsing applies
5a. axles          - needs what step 5 stored: files or holds a capacity, opens the count question
5b. question rules - needs this turn's cargo and values; decides the questions steps 7-8 read
6. haul suggestion - needs the haul_item that step 5 just stored
7. attempts        - needs to know what steps 5 and 6 resolved
8. results gate    - needs everything above
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.domain import axles
from src.domain import gooseneck as gooseneck_domain
from src.domain import links
from src.domain import listing_echo
from src.domain import quantities as quantity_math
from src.graph.nodes import greeting
from src.domain.slot_map import (
    axle_count_out_of_range,
    brand_is_actually_a_hitch,
    slot_value_kind,
)
from src.tools.category import (
    FEATURES_KEY,
    meaningful_filters,
    normalize_category,
    set_trailer_category,
    suggest_category_from_haul_item,
)
from src.rules.engine import apply_rules, mark_user_value
from src.rules.store import current_rules
from src.tools.filters import apply_extracted_fields
from src.tools import scope, team_notify, unavailable
from src.tools.lookup_gate import brand_is_lookup_make, referenced_listing
from src.tools.questions import (
    all_required_resolved,
    decline_slot,
    is_answered,
    record_no_preference,
    resolve_pending_slot,
    sweep_exhausted_slots,
)

logger = logging.getLogger(__name__)

# Intents that mean "stop asking and show me what you have".
_SHOW_RESULTS_INTENTS = {"skip_all_show_results", "show_more_results", "recommendation_request"}


def apply_node(state: dict, output: Any, user_message: str = "") -> dict:
    """Fold one turn's structured output into the session state."""
    outcome = state.setdefault("turn_outcome", {})
    state["invalid_retry_slot"] = None
    state["invalid_retry_reason"] = None

    _apply_contact(state, output)
    _note_shared_platforms(state, user_message)
    # Straight after the contact merge, so a turn that hands over the missing piece sends
    # everything that was waiting on it - and before the FAQ below, so a question asked on
    # that same turn joins the same batch.
    team_notify.flush(state)
    _apply_faq_notification(state, output)
    _apply_shared_link(state, output, user_message)
    _apply_listing_interest(state, output)
    _apply_unavailable_type(state, output)
    _apply_off_topic(state, output)
    handled = _apply_pending_confirmations(state, output, user_message)
    _apply_gooseneck(state, output, user_message)
    if not handled:
        _apply_category(state, output)
    _apply_brand(state, output, user_message)

    result = apply_extracted_fields(state, output, user_message)
    if result.invalid_slot:
        state["invalid_retry_slot"] = result.invalid_slot
        state["invalid_retry_reason"] = result.invalid_reason
    record_no_preference(state, result.no_preference)
    _apply_axles(state, output, user_message, result)

    _apply_question_rules(state, output)
    _apply_haul_item_suggestion(state, result)
    _apply_attempts(state, output, result)
    _report_questions_we_gave_up_on(state)
    _apply_refined_search(state, output, result)
    _apply_results_gate(state, output)

    outcome["applied_slots"] = dict(result.stored)
    outcome["declined_this_turn"] = list(result.no_preference)
    return state


# ------------------------------------------------------------------------- 1. contact
def _apply_contact(state: dict, output: Any) -> None:
    """Merge contact details. Asked once, then never again whatever they do.

    Only blanks are filled: a later turn must not overwrite the details the lead was
    qualified on.
    """
    contact = state.setdefault(
        "contact", {"name": None, "email": None, "phone": None, "asked": False, "declined": False}
    )
    incoming = getattr(output, "contact", None)
    if incoming is None:
        return

    progressed = False
    for field in ("name", "email", "phone"):
        value = getattr(incoming, field, None)
        if value and not contact.get(field):
            contact[field] = str(value).strip()
            progressed = True

    # A customer who gives their name has engaged, so asking once more for the number is
    # persistence rather than nagging. Only a turn that adds nothing counts against the
    # budget, which is what lets the gate ask twice per missing half instead of twice ever.
    if progressed:
        contact["asks_without_progress"] = 0
        # And the turn gap with it. The gap is there to stop us asking the same thing twice
        # in a row with nothing in between; a customer who has just given their name HAS put
        # something in between, and following up for the number now is responsive rather
        # than pestering.
        contact["last_asked_turn"] = 0

    if getattr(incoming, "declined", False) or getattr(output, "intent", "") == "contact_declined":
        # Respected immediately and permanently, whether they refused everything or only the
        # half we were still missing. A customer who says no is not asked again and is not
        # held back from results either - the gate comes down with the question.
        contact["declined"] = True
        logger.info("CONTACT declined: session=%s", state.get("session_id"))

    # NOT marked as asked here. Whether the opener actually went out is something only
    # compose knows, and setting it from this side meant a first message that named a
    # category ("looking for a 20ft livestock trailer") skipped the question entirely while
    # recording that it had been put - so the lead was lost with no way to notice.


def _note_shared_platforms(state: dict, user_message: str) -> None:
    """Remember that they sent us a Facebook or Instagram link, for the team's email.

    Kept on the session rather than re-read off the transcript when the email is built: the
    notification may be stashed for several turns waiting on a phone number, and where the
    customer found us does not stop being true in the meantime.
    """
    known = state.setdefault("shared_platforms", [])
    for platform in links.shared_platforms([user_message]):
        if platform not in known:
            known.append(platform)
            logger.info("LINK platform noted: session=%s %s", state.get("session_id"), platform)


_STOCK_IN_URL_RE = re.compile(r"-(\d{4,6})/?$")


def _stock_in(url: Any) -> str:
    """The stock number on the end of one of our listing URLs, if it is one of ours.

    Our slugs end in the stock number ("...-5-bale-hay-81419/"), which is the one part of a
    listing URL the team can actually act on.
    """
    cleaned = links.normalize_listing_url(url)
    match = _STOCK_IN_URL_RE.search(str(cleaned or ""))
    return match.group(1) if match else ""


def _apply_faq_notification(state: dict, output: Any) -> None:
    """Every one of the five standard questions is worth telling the team about.

    Handled here rather than through the escalate tool because an FAQ must NOT cost the
    customer their place in the qualification flow: the analysis pass writes the answer, and
    compose still asks the pending question afterwards. Routing these to the agent threw that
    away. The notification is the only part that needs adding, and it goes through the same
    contact gate as everything else - so it waits, and it batches, exactly like an escalation.
    """
    faq_key = getattr(output, "faq_key", None)
    if not faq_key:
        return
    team_notify.record(
        state,
        reason=f"FAQ - {faq_key}",
        description=_FAQ_DESCRIPTIONS.get(
            str(faq_key), f"Asked about {str(faq_key).replace('_', ' ')}"
        ),
    )


# What each standard question is, in a handful of words. The customer's verbatim question
# used to go here and ran to a paragraph; the reason line already names the FAQ, and the
# chat link goes to the question itself.
_FAQ_DESCRIPTIONS = {
    "financing": "Asked about financing",
    "trade_in": "Asked about trade-ins",
    "service_parts": "Asked about service and parts",
    "store_info": "Asked about hours or location",
    "contact_human": "Asked to speak to someone",
}


# What they asked about the linked trailer, as the team email says it. Each one has to read
# under seven words with the longest thing that can fill {what} ("an Instagram listing").
_LINK_ASKS = {
    "price": "Asked the price of {what}",
    "availability": "Asked if {what} is available",
    "details": "Asked for details of {what}",
}
_WANTS_IT = "Wants {what}"


def _apply_shared_link(state: dict, output: Any, user_message: str) -> None:
    """A link to a trailer - ours, a Facebook or an Instagram post - that they want.

    The model reads whether they want it (``shared_link_interest``); the URL itself says
    where it points. The team hears about it like any listing interest, under the same
    contact gate, and the email names the kind of link so they know where the customer saw it.
    """
    if not getattr(output, "shared_link_interest", False):
        return
    found = links.find_trailer_links(user_message)
    if not found:
        return
    label, url = found[0]
    wants = getattr(getattr(output, "inventory_lookup", None), "wants", None)
    # OUR listing URL goes in the line - it is one click to the exact trailer. A social one
    # does not: it says nothing about which trailer, it expires, and it is stripped out of
    # the body in any case. The kind of link is appended to the body separately.
    ours = links.normalize_listing_url(url) or links.normalize_listing_url(links.unwrap_redirect(url))
    site = label.split()[0]
    what = ours or f"{'an' if site[:1] in 'AEIOU' else 'a'} {site} listing"
    description = _LINK_ASKS.get(wants, _WANTS_IT).format(what=what)
    status = team_notify.record(state, reason="Listing Interest", description=description)
    state.setdefault("turn_outcome", {})["link_interest"] = {"status": status, "label": label}
    # Sharing a link to a trailer IS picking one. Live, the reply that finally captured the
    # lead still went on to read the catalogue out: "...noted your interest in the trailer
    # you shared. What type of trailer are you looking for? We have Utility, Enclosed..."
    # The guard existed but keyed on the other path into the same place.
    state["listing_interest_logged"] = True
    logger.info("LINK interest: session=%s %s status=%s", state.get("session_id"), label, status)


def interest_status(state: dict) -> str:
    """What would happen to a listing-interest email raised right now.

    Recomputed rather than remembered: it is about the contact we hold, and the customer
    should be told the true thing each time.
    """
    return team_notify.status_now(state)


def _apply_listing_interest(state: dict, output: Any) -> None:
    """They said they want one of the trailers we showed. Tell the team - here, not there.

    This used to depend entirely on the reply pass choosing to call ``escalate``. Live, on a
    turn where every detail was already in front of it, it simply did not:

        > I like the 6th one
        "The 2026 Calico Trailers HOGPEN/LIVESTOCK TRAILER 16' - 00564 sounds like the one
         that fits your livestock-hauling needs. Our sales team can help..."

    No email, no lead - on the one turn in the conversation where the customer said yes. The
    model is the right judge of WHETHER they expressed interest (that is reading a customer,
    and it is what ``intent`` says); it is not the thing that should decide whether the
    dealership hears about it. So Python records it, and the reply pass is told it is done -
    the same shape as a shared link (``_apply_shared_link``).

    Recorded once per trailer. People say "I like that one" more than once about the same
    trailer, and the team does not need to hear it twice.
    """
    if getattr(output, "intent", "") != "listing_interest":
        return
    outcome = state.setdefault("turn_outcome", {})
    if outcome.get("link_interest"):
        # The link they shared already told the team, with the kind of link named.
        return

    listing = referenced_listing(state, output) or {}
    title = str(listing.get("title") or "").strip()
    url = str(listing.get("url") or "").strip()
    state["listing_interest_logged"] = True
    if title:
        # What the lead row is FOR. Without it a customer who asked about one trailer by
        # stock number and said yes to it was filed under "no details yet", because nothing
        # they said was a category or a slot.
        state["interest_listing"] = title

    recorded = state.setdefault("listing_interest_keys", [])
    if url in recorded:
        status = interest_status(state)
        outcome["listing_interest"] = {"status": status, "title": title, "url": url, "repeat": True}
        logger.info(
            "LISTING interest already recorded: session=%s url=%r status=%s",
            state.get("session_id"), url, status,
        )
        return
    recorded.append(url)

    # The listing URL, not the title: it is one click to the exact trailer, and one "word"
    # rather than eleven. The stock number and the title only stand in when there is no URL
    # to give - a trailer picked out of a batch we somehow hold no link for.
    stock = str(listing.get("stock_number") or "").strip() or _stock_in(url)
    if url:
        description = f"Customer is interested in {url}"
    elif stock:
        description = f"Customer is interested in stock {stock}"
    elif title:
        description = f"Customer is interested in {title}"
    else:
        description = "Interested in a trailer we showed"

    status = team_notify.record(state, reason="Listing Interest", description=description)
    outcome["listing_interest"] = {"status": status, "title": title, "url": url, "repeat": False}
    logger.info(
        "LISTING interest: session=%s title=%r status=%s", state.get("session_id"), title, status,
    )


def _apply_off_topic(state: dict, output: Any) -> None:
    """They asked about something that has nothing to do with us. Compose says so.

    Nothing is stored and nobody is emailed: there is no lead in "how do I make a sandwich",
    and a team told about every stray message stops reading the ones that matter.
    """
    if not scope.is_off_topic(output):
        return
    state.setdefault("turn_outcome", {})["off_topic"] = True
    logger.info("OFF-TOPIC turn: session=%s", state.get("session_id"))


def _apply_unavailable_type(state: dict, output: Any) -> None:
    """They asked for a trailer type we do not carry: tell the team, and let compose say so.

    Here, with the other notifications, so the contact details merged above are already in
    place - a customer who asks for a boat trailer AND gives their number in one message gets
    the email sent straight away rather than stashed.
    """
    requested = unavailable.requested_type(output)
    if not requested:
        return
    status = unavailable.record(state, output, requested)
    state.setdefault("turn_outcome", {})["unavailable_type"] = {"type": requested, "status": status}
    logger.info(
        "UNAVAILABLE type requested: session=%s type=%r status=%s",
        state.get("session_id"), requested, status,
    )


# --------------------------------------------------------------- 2. pending confirmations
def _apply_pending_confirmations(state: dict, output: Any, user_message: str) -> bool:
    """Settle a yes/no we asked last turn. Returns True when it consumed this message."""
    if _apply_gooseneck_answer(state, user_message):
        return True
    if _apply_category_switch_answer(state, output):
        return True
    return _apply_keep_filters_answer(state, output)


def _apply_gooseneck_answer(state: dict, user_message: str) -> bool:
    pending = state.get("pending_gooseneck_clarification")
    if not pending:
        return False

    meaning = gooseneck_domain.apply_clarification_answer(user_message)
    if meaning is None:
        # Still unclear. Left pending so the question can be repeated once, but not
        # treated as consuming the turn.
        return False

    state["pending_gooseneck_clarification"] = None
    if meaning == gooseneck_domain.BRAND:
        state["brand_preference"] = gooseneck_domain.GOOSENECK_MAKE
        logger.info("GOOSENECK resolved: session=%s -> brand", state.get("session_id"))
    else:
        state.setdefault("slots", {})["hitch_type"] = ["Gooseneck"]
        logger.info("GOOSENECK resolved: session=%s -> hitch", state.get("session_id"))
    return True


def _apply_category_switch_answer(state: dict, output: Any) -> bool:
    """They are answering "Equipment suits that better - want to switch?"."""
    pending = state.get("pending_category_switch")
    answer = getattr(output, "category_confirm_answer", None)
    if not pending or answer is None:
        return False

    state["pending_category_switch"] = None
    if answer == "yes":
        # Deliberately NO keep-filters question here: they were offered a better-suited
        # category for the load they already described, so their answers still apply.
        set_trailer_category(state, pending["suggested"])
        logger.info(
            "CATEGORY switched on suggestion: session=%s -> %s",
            state.get("session_id"), pending["suggested"],
        )
    else:
        # Remembered so the same suggestion is never raised twice.
        rejected = state.setdefault("rejected_switches", [])
        if pending.get("pair") and pending["pair"] not in rejected:
            rejected.append(pending["pair"])
    return True


def _haul_item_restated(output: Any) -> bool:
    """Did this message name what they are hauling, as well as answering keep-or-drop?"""
    if getattr(getattr(output, "extracted", None), "haul_item", None):
        return True
    return any(
        getattr(answer, "slot_name", "") == "haul_item"
        for answer in (getattr(output, "slot_answers", None) or [])
    )


def _apply_keep_filters_answer(state: dict, output: Any) -> bool:
    """They are answering "keep what you have already told me?" after a category change."""
    pending = state.get("pending_keep_filters")
    answer = getattr(output, "keep_fields_answer", None)
    if not pending or answer is None:
        return False

    new_category = pending.get("new_category")
    offered = dict(pending.get("filters") or {})
    state["pending_keep_filters"] = None

    if answer == "none":
        keep_slots: set[str] = set()
        state["non_metadata_features"] = []
    elif answer == "some":
        # Whatever they named, restricted to what was actually offered - a customer saying
        # "keep the length" cannot resurrect a slot the question never mentioned.
        keep_slots = {slot for slot in (getattr(output, "kept_fields", None) or []) if slot in offered}
    else:  # "all"
        keep_slots = set(offered)

    slots = state.setdefault("slots", {})
    for slot in offered:
        if slot not in keep_slots:
            slots.pop(slot, None)
    if FEATURES_KEY in offered and FEATURES_KEY not in keep_slots:
        state[FEATURES_KEY] = []

    # What they are hauling belongs to the OLD category. It is a required question for the
    # new one, so it is dropped and asked again rather than silently inherited - unless this
    # same message already supplied a new one, which is the usual way a category change
    # arrives ("actually I need to move a tractor").
    if not _haul_item_restated(output):
        slots.pop("haul_item", None)

    if new_category:
        set_trailer_category(state, new_category)
    logger.info(
        "CATEGORY change confirmed: session=%s -> %s keep=%s",
        state.get("session_id"), new_category, answer,
    )
    return True


# ----------------------------------------------------------------------- 3. gooseneck
def _apply_gooseneck(state: dict, output: Any, user_message: str) -> None:
    """Decide what "gooseneck" meant, or open a clarification question."""
    if state.get("pending_gooseneck_clarification"):
        return
    if not gooseneck_domain.mentions_gooseneck(user_message):
        return

    reading = gooseneck_domain.resolve_gooseneck_mention(
        user_message,
        pending_slot=state.get("pending_slot"),
        # Brand recognition is the model's job - it has the make list in its prompt.
        extracted_brand=getattr(getattr(output, "extracted", None), "brand_preference", None),
    )
    if reading.meaning == gooseneck_domain.AMBIGUOUS:
        state["pending_gooseneck_clarification"] = user_message
        logger.info("GOOSENECK ambiguous: session=%s asking", state.get("session_id"))
        return
    if reading.meaning == gooseneck_domain.BRAND:
        state["brand_preference"] = gooseneck_domain.GOOSENECK_MAKE
        return
    if reading.other_brand:
        # "a gooseneck Diamond C": the other name is the make, gooseneck is the hitch.
        state["brand_preference"] = reading.other_brand
        state.setdefault("slots", {})["hitch_type"] = ["Gooseneck"]


# ------------------------------------------------------------------------ 4. category
def _apply_category(state: dict, output: Any) -> None:
    """Select or change the category.

    A CHANGE with meaningful filters already collected does not mutate anything: it opens
    the keep-or-drop question and waits (brief S12).
    """
    mentioned = getattr(output, "category_mentioned", None)
    if not mentioned or getattr(output, "is_category_info_only", False):
        return

    canonical = normalize_category(mentioned)
    if not canonical:
        # Unrecognized, or "not sure" / "any" - base_category stays None (brief S7).
        return

    current = state.get("category")
    if canonical == current:
        return

    if current:
        existing = meaningful_filters(state)
        if existing:
            state["pending_keep_filters"] = {"new_category": canonical, "filters": existing}
            logger.info(
                "CATEGORY change pending: session=%s %s -> %s with %d filters",
                state.get("session_id"), current, canonical, len(existing),
            )
            return

    set_trailer_category(state, canonical)


def _apply_brand(state: dict, output: Any, user_message: str) -> None:
    """Store a brand preference, unless the "brand" is really a hitch type."""
    brand = getattr(getattr(output, "extracted", None), "brand_preference", None)
    if not brand:
        return
    if brand_is_actually_a_hitch(brand, user_message):
        state.setdefault("slots", {}).setdefault("hitch_type", None)
        return
    if brand_is_lookup_make(output, brand):
        # It is the make half of this turn's lookup identifier ("I'm looking for an Iron
        # Bull DTB"), not a standing instruction to filter every later search to that make.
        return
    if listing_echo.is_pointing_turn(output) and brand.strip().casefold() not in (user_message or "").casefold():
        # "I like the 81419" came back with brand=Gooseneck, read off that listing's card.
        # Naming a trailer they like is a finger, not a filter: recorded as a preference it
        # would narrow every later search to one make on the strength of a single trailer.
        logger.info(
            "BRAND ignored: session=%s brand=%r came off a listing they pointed at",
            state.get("session_id"), brand,
        )
        return
    state["brand_preference"] = str(brand).strip()


# ---------------------------------------------------------------- 5b. question rules
def _apply_question_rules(state: dict, output: Any) -> None:
    """Record what the cargo is like, then let the rules settle this turn's questions.

    The model's traits describe the cargo it matched THIS turn, so they replace the old
    ones. A turn that names no cargo leaves them alone - unless what they are hauling has
    been dropped (a category change, keep-filters "none"), which takes its traits with it.

    Run every turn rather than only when something changed: the rules themselves may have
    changed since the last message, and the conversation should follow them.
    """
    haul = getattr(output, "haul_classification", None)
    matched = str(getattr(haul, "haul_item_matched", None) or "").strip()
    if matched:
        state["cargo_traits"] = list(dict.fromkeys(getattr(haul, "cargo_traits", None) or []))
    elif not is_answered(state, "haul_item"):
        state["cargo_traits"] = []
    apply_rules(state, current_rules())
    _skip_weight_after_axle_rating(state)


# ------------------------------------------------------------------------- 5a. axles
_CAPACITY_SLOTS = ("axle_capacity", "total_axle_capacity_lbs")
_SLOT_FOR_BASIS = {"per_axle": "axle_capacity", "total": "total_axle_capacity_lbs"}


def _apply_axles(state: dict, output: Any, user_message: str, result: Any) -> None:
    """Per-axle or total, and how many axles. After the filters, so it sees what they stored.

    Three things, in this order:

    1. A capacity held last turn is filed once they say which it is - using the number we
       held, not the model's re-read of it (New Prompt saw the model halve 14,000 to 7,000 on
       the follow-up, having assumed two axles).
    2. A new capacity whose wording says neither ("14,000 lbs of axle capacity") is held
       rather than stored, and asked about.
    3. After a per-axle rating, "how many axles?" - the search needs it to work out the total.
    """
    extracted = getattr(output, "extracted", None)
    model_basis = getattr(extracted, "axle_capacity_basis", None)

    if getattr(output, "intent", "") in _SHOW_RESULTS_INTENTS:
        # "Just show me what you have" skips our open axle question like any other: the
        # count becomes no preference, and a capacity we could not place is let go. Live, the
        # count question was asked again with "Sorry, I didn't catch that".
        if state.get("pending_axle_count"):
            record_no_preference(state, ["axle_count"])
            state["pending_axle_count"] = None
            if state.get("invalid_retry_slot") == "axle_count":
                state["invalid_retry_slot"] = state["invalid_retry_reason"] = None
        if state.get("pending_axle_basis"):
            logger.info("AXLE capacity dropped: they asked to see results: session=%s", state.get("session_id"))
            state["pending_axle_basis"] = None
        return

    held = state.get("pending_axle_basis")
    if held:
        _resolve_held_capacity(state, held, user_message, model_basis, result)
    else:
        _hold_unclear_capacity(state, output, user_message, model_basis, result)

    _apply_axle_count(state, output, user_message, result)


def _unstore(state: dict, slot: str, result: Any) -> None:
    (state.get("slots") or {}).pop(slot, None)
    (state.get("slot_sources") or {}).pop(slot, None)
    result.stored.pop(slot, None)


def _resolve_held_capacity(state: dict, held: dict, user_message: str,
                           model_basis: str | None, result: Any) -> None:
    # Whatever the model filed under either capacity this turn is its re-read of the number
    # we are holding. The held one is what they said.
    for slot in _CAPACITY_SLOTS:
        if slot in result.stored:
            _unstore(state, slot, result)

    basis = axles.basis_from_reply(user_message, model_basis)
    if basis is None:
        if int(held.get("asks") or 0) >= axles.MAX_CLARIFY_ASKS:
            state["pending_axle_basis"] = None
            logger.info(
                "AXLE capacity dropped after %d asks: session=%s value=%s",
                held.get("asks"), state.get("session_id"), held.get("value"),
            )
        return  # still open: compose asks again

    slot = _SLOT_FOR_BASIS[basis]
    value = float(held["value"])
    state.setdefault("slots", {})[slot] = value
    mark_user_value(state, slot)
    result.stored[slot] = value
    state["pending_axle_basis"] = None
    logger.info("AXLE capacity %s resolved as %s: session=%s", value, basis, state.get("session_id"))


def _hold_unclear_capacity(state: dict, output: Any, user_message: str,
                           model_basis: str | None, result: Any) -> None:
    stored = [slot for slot in _CAPACITY_SLOTS if slot in result.stored]
    if not stored or axles.infer_basis(user_message, model_basis) != "unclear":
        return
    # The number as the model read it from their words - the quantity, converted here -
    # falling back to what the filters stored. Given both fields, the model's per-axle one is
    # a halved guess (it assumes two axles), so the larger is the number they actually said.
    value = None
    for quantity in getattr(getattr(output, "extracted", None), "quantities", None) or []:
        if getattr(quantity, "slot_name", None) in _CAPACITY_SLOTS:
            value = quantity_math.to_canonical(quantity.slot_name, quantity)
            if value:
                break
    value = value or max(float(result.stored[slot]) for slot in stored)
    for slot in stored:
        _unstore(state, slot, result)
    state["pending_axle_basis"] = {"value": float(value), "asks": 0}
    logger.info("AXLE capacity %s held pending per-axle/total: session=%s", value, state.get("session_id"))


def _apply_axle_count(state: dict, output: Any, user_message: str, result: Any) -> None:
    """Close last turn's "how many axles?", or open it after a per-axle rating."""
    pending = state.get("pending_axle_count")
    if pending:
        _close_axle_count(state, pending, output, user_message, result)
        return

    if state.get("invalid_retry_slot") == "axle_count":
        # They volunteered a count we do not stock ("five axles"). Asked with our own
        # question rather than a bare "axle count?", with the one-to-four reason in front.
        state["pending_axle_count"] = {"asks": 0}
        return

    if (
        state.get("category")
        and is_answered(state, "axle_capacity")
        and (state.get("slot_sources") or {}).get("axle_capacity") != "default"
        and not state.get("pending_axle_basis")
        and not is_answered(state, "axle_count")
        and "axle_count" not in (state.get("declined_slots") or [])
    ):
        state["pending_axle_count"] = {"asks": 0}
        logger.info("AXLE count question opened: session=%s", state.get("session_id"))


def _close_axle_count(state: dict, pending: dict, output: Any, user_message: str, result: Any) -> None:
    asks = int(pending.get("asks") or 0)

    if is_answered(state, "axle_count"):
        state["pending_axle_count"] = None  # they answered
        return
    if "axle_count" in result.no_preference or "axle_count" in (state.get("declined_slots") or []):
        state["pending_axle_count"] = None
        return

    reason = state.get("invalid_retry_reason") if state.get("invalid_retry_slot") == "axle_count" else None
    if reason is None:
        # The model does not always turn a bare "Tandem." into a number, so their words are
        # read with the same vocabulary the question offered them.
        spoken = axles.count_from_reply(user_message)
        if spoken is None and any(slot != "axle_count" for slot in result.stored):
            # They answered something else ("about 5,000 lbs" - the load). Not a wrong
            # count, so no correction: the question simply stands, within its two asks.
            if asks >= axles.MAX_CLARIFY_ASKS:
                record_no_preference(state, ["axle_count"])
                state["pending_axle_count"] = None
            return
        if spoken is not None and not axle_count_out_of_range(spoken):
            state.setdefault("slots", {})["axle_count"] = spoken
            mark_user_value(state, "axle_count")
            result.stored["axle_count"] = spoken
            state["pending_axle_count"] = None
            logger.info("AXLE count %s read from their reply: session=%s", spoken, state.get("session_id"))
            return
        if spoken is not None:
            reason = "axle_range"
        elif axles.is_no_preference(user_message):
            record_no_preference(state, ["axle_count"])
            state["pending_axle_count"] = None
            return
        else:
            reason = "unclear"

    if asks >= axles.MAX_CLARIFY_ASKS:
        # Two asks spent. Whatever they said, it is no preference now - never a third ask.
        record_no_preference(state, ["axle_count"])
        state["pending_axle_count"] = None
        state["invalid_retry_slot"] = None
        state["invalid_retry_reason"] = None
        return
    state["invalid_retry_slot"] = "axle_count"
    state["invalid_retry_reason"] = reason


def _skip_weight_after_axle_rating(state: dict) -> None:
    """An axle rating sizes the trailer from the capacity end - don't ask the load weight.

    "How heavy is your load?" exists to size the trailer. Someone who said "7,000 lb axles"
    has done that themselves, and asking anyway reads as not having listened. The rating
    already ranks the search in every category.
    """
    sources = state.get("slot_sources") or {}
    rated = any(
        is_answered(state, slot) and sources.get(slot) != "default"
        and isinstance(state["slots"][slot], (int, float)) and state["slots"][slot] > 0
        for slot in _CAPACITY_SLOTS
    )
    if not rated:
        return
    required = state.get("required_slots") or []
    skipped = state.setdefault("rule_skipped", {})
    for slot in list(required):
        if slot_value_kind(slot) == "payload_lbs" and not is_answered(state, slot):
            required.remove(slot)
            skipped[slot] = "they already gave an axle rating"


# ------------------------------------------------------------- 6. haul-item suggestion
def _apply_haul_item_suggestion(state: dict, result: Any) -> None:
    """Their cargo may imply a better-suited category than the one they are on."""
    if state.get("pending_keep_filters") or state.get("pending_category_switch"):
        return
    haul_item = result.stored.get("haul_item")
    if not haul_item:
        return
    suggestion = suggest_category_from_haul_item(state, haul_item)
    if suggestion:
        state["pending_category_switch"] = suggestion


# ------------------------------------------------------------------------ 7. attempts
def _apply_attempts(state: dict, output: Any, result: Any) -> None:
    """Settle last turn's question in light of what they just sent.

    An explicit skip resolves the slot immediately - they answered, the answer was "no".
    Anything else defers to the attempt counter, which was incremented when the question
    was asked, so a counter-question costs an attempt exactly like a non-sequitur does.
    """
    pending = state.get("pending_slot")
    intent = getattr(output, "intent", "")

    if pending and intent == "skip_current":
        decline_slot(state, pending, reason="skip_current")
        state["pending_slot"] = None
        sweep_exhausted_slots(state)
        return

    if intent == "drop_requirements":
        for slot in getattr(output, "dropped_fields", None) or []:
            (state.get("slots") or {}).pop(slot, None)

    answered = bool(getattr(output, "answered_current_question", False))
    # A slot resolved by THIS message counts as answered even if the model said otherwise -
    # the state is the evidence, not the classification.
    if pending and (pending in result.stored or pending in result.no_preference):
        answered = True
    resolve_pending_slot(state, answered=answered)
    # Catches the slot whose two asks were spent on invalid values: those report as
    # "answered", so resolve_pending_slot clears them without ever declining them.
    sweep_exhausted_slots(state)


def _report_questions_we_gave_up_on(state: dict) -> None:
    """A required question asked twice and never answered is worth telling the team about.

    Not because the customer did anything wrong - they can decline anything they like, and
    a "whatever works" is an answer that closes the question on the spot. This is the other
    case: two asks spent and nothing came back, which usually means the question did not
    land. The team is the only one who can see the pattern and change the wording.

    ONCE per conversation, on the first question we lose. A customer who ignores one
    question usually ignores the next as well, and six emails about one stalled chat is
    noise the team will learn to filter out - which costs them the first one too. The email
    carries a link to the conversation, so the rest of the story is one click away.

    A system alert, like the results one: it goes through the gate and waits, and it never
    chases the customer for anything.
    """
    outcome = state.setdefault("turn_outcome", {})
    given_up = outcome.pop("gave_up_on", [])
    if not given_up or state.get("gave_up_reported"):
        return
    slot = given_up[0]
    state["gave_up_reported"] = True
    label = str(slot).replace("_", " ")
    team_notify.record(
        state,
        reason="Unanswered Question",
        description=f"No answer after two asks: {label}",
    )
    logger.info(
        "QUESTION gave up: session=%s slot=%s - telling the team", state.get("session_id"), slot,
    )


# ------------------------------------------------- 7b. refining an already-shown result set
def _apply_refined_search(state: dict, output: Any, result: Any) -> None:
    """A requirement changed after they had already seen listings: search again.

    Two different things a customer can mean, and they need opposite handling:

    * "show me more"       - same criteria, different trailers. The already-shown list stays,
                             so the next page excludes what they have seen.
    * "make it 24 ft"      - different criteria. The already-shown list is CLEARED, because
                             it was built under requirements that no longer apply and the
                             best match for the new ones may well be a trailer they were
                             shown before.

    A category change is neither: it goes through the keep-or-drop question first, and the
    search waits for that answer.
    """
    if not state.get("results_shown"):
        return
    if state.get("pending_keep_filters") or state.get("pending_category_switch"):
        return

    intent = getattr(output, "intent", "")
    if intent == "show_more_results":
        return

    changed = bool(result.stored) or bool(getattr(output, "dropped_fields", None))
    if not changed and intent not in {"requirement_change", "drop_requirements"}:
        return

    state["shown_urls"] = []
    state["turn_outcome"]["search_refined"] = True
    logger.info(
        "SEARCH refine: session=%s changed=%s dropped=%s - clearing shown history",
        state.get("session_id"), sorted(result.stored), getattr(output, "dropped_fields", None),
    )


# --------------------------------------------------------------------- 8. results gate
def in_question_stage(state: dict) -> bool:
    """A category is chosen and at least one of its questions has been asked.

    ``asked_counts`` is written by compose, after this node, so on the turn a category is
    first chosen it is still empty - that turn is never inside the question stage.
    """
    if not state.get("category"):
        return False
    return any(count > 0 for count in (state.get("asked_counts") or {}).values())


def _apply_results_gate(state: dict, output: Any) -> None:
    """Results appear only on an explicit request, or when every required question is done.

    Nothing else opens the gate - not a category selection, not a good answer, not a long
    conversation.
    """
    intent = getattr(output, "intent", "")
    asked_for_results = intent in _SHOW_RESULTS_INTENTS
    # What they asked for, kept apart from whether it is honoured: compose still owes a
    # customer with no category the website line (S25).
    requested_results = asked_for_results
    complete = all_required_resolved(state)
    # Once they have seen listings, a changed requirement is itself a request to look
    # again - they are refining what is in front of them, not starting a new qualification.
    refined = bool(state["turn_outcome"].get("search_refined"))

    # A question of ours is on the table - which category, which hitch, keep the filters.
    # Searching now would answer a question they have not answered yet, and would do it
    # with filters that are about to change. The gate stays shut until they reply.
    #
    # The opening turn is held for the same reason: a visitor who gets their listings
    # immediately and leaves is a lost lead, and there is no second first message in which
    # to ask who they are.
    awaiting_answer = bool(
        state.get("pending_keep_filters")
        or state.get("pending_category_switch")
        or state.get("pending_gooseneck_clarification")
        or state.get("pending_axle_basis")
        or state.get("pending_axle_count")
        or greeting.contact_gate_applies(state)
    )

    if state.get("results_shown"):
        # They have seen listings. Completing qualification is no longer news, so a search
        # needs a REASON: they asked for more, or something they told us changed. Without
        # this the gate stays open and every later turn - a question about opening hours
        # included - runs the same query again and re-prints the same trailers.
        wants_search = asked_for_results or refined
    else:
        # Before any listings, "just show me" / "what do you recommend" only cuts the
        # questions short once they have STARTED: a category is set and at least one of its
        # questions has been put to them. "Recommend a trailer for cattle" picks the
        # category - it does not skip a question nobody has asked yet.
        if asked_for_results and not in_question_stage(state):
            logger.info(
                "GATE held: session=%s intent=%s before any question was asked",
                state.get("session_id"), intent,
            )
            asked_for_results = False
        wants_search = asked_for_results or complete

    state["qualification_complete"] = (
        bool(state.get("category")) and not awaiting_answer and wants_search
    )
    state["turn_outcome"]["wants_results"] = requested_results
    state["turn_outcome"]["qualification_just_completed"] = complete and not asked_for_results
