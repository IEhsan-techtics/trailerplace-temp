"""The deterministic layer: the model's evidence becomes state. No LLM calls, no I/O.

Every business rule in the brief is enforced here or in ``src/tools/``. The model's output
is read as EVIDENCE about what the customer said, never as an instruction about what to do
- which is why a bad classification degrades a reply rather than corrupting a session.

The order of the eight steps is load-bearing:

1. contact         - independent of everything, so it cannot be lost to an early return
2. confirmations   - a pending yes/no is about the PREVIOUS turn and must be read before
                     this turn's category or filters can overwrite the thing it answers
3. gooseneck       - decides whether "gooseneck" is a hitch or a make, before either is stored
4. category        - may open a keep-filters question instead of switching immediately
5. filters         - runs with the category settled, so per-category parsing applies
6. haul suggestion - needs the haul_item that step 5 just stored
7. attempts        - needs to know what steps 5 and 6 resolved
8. results gate    - needs everything above
"""
from __future__ import annotations

import logging
from typing import Any

from src.domain import gooseneck as gooseneck_domain
from src.graph.nodes import greeting
from src.domain.slot_map import brand_is_actually_a_hitch
from src.tools.category import (
    meaningful_filters,
    normalize_category,
    set_trailer_category,
    suggest_category_from_haul_item,
)
from src.tools.filters import apply_extracted_fields
from src.tools import team_notify
from src.tools.lookup_gate import brand_is_lookup_make
from src.tools.questions import (
    all_required_resolved,
    decline_slot,
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
    # Straight after the contact merge, so a turn that hands over the missing piece sends
    # everything that was waiting on it - and before the FAQ below, so a question asked on
    # that same turn joins the same batch.
    team_notify.flush(state)
    _apply_faq_notification(state, output)
    handled = _apply_pending_confirmations(state, output, user_message)
    _apply_gooseneck(state, output, user_message)
    if not handled:
        _apply_category(state, output)
    _apply_brand(state, output, user_message)

    result = apply_extracted_fields(state, output)
    if result.invalid_slot:
        state["invalid_retry_slot"] = result.invalid_slot
        state["invalid_retry_reason"] = result.invalid_reason
    record_no_preference(state, result.no_preference)

    _apply_haul_item_suggestion(state, result)
    _apply_attempts(state, output, result)
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
    question = (getattr(output, "user_question_to_answer", None) or "").strip()
    team_notify.record(
        state,
        reason=f"FAQ - {faq_key}",
        description=question or f"asked about {str(faq_key).replace('_', ' ')}",
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
    state["brand_preference"] = str(brand).strip()


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
def _apply_results_gate(state: dict, output: Any) -> None:
    """Results appear only on an explicit request, or when every required question is done.

    Nothing else opens the gate - not a category selection, not a good answer, not a long
    conversation.
    """
    intent = getattr(output, "intent", "")
    asked_for_results = intent in _SHOW_RESULTS_INTENTS
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
        or greeting.contact_gate_applies(state)
    )

    if state.get("results_shown"):
        # They have seen listings. Completing qualification is no longer news, so a search
        # needs a REASON: they asked for more, or something they told us changed. Without
        # this the gate stays open and every later turn - a question about opening hours
        # included - runs the same query again and re-prints the same trailers.
        wants_search = asked_for_results or refined
    else:
        wants_search = asked_for_results or complete

    state["qualification_complete"] = (
        bool(state.get("category")) and not awaiting_answer and wants_search
    )
    state["turn_outcome"]["wants_results"] = asked_for_results
    state["turn_outcome"]["qualification_just_completed"] = complete and not asked_for_results
