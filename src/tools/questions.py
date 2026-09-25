"""Which question to ask next, and when to stop asking. No LLM calls, no I/O.

Two rules from the brief drive everything here:

* S29 - a question that has been answered is never asked again;
* S23 - a required question is asked at most twice, then dropped.

``asked_counts[slot]`` increments **when the question is asked**, not when the answer
disappoints. That is what makes the ceiling a hard "asked at most twice" rather than
"failed twice after an initial ask", and it is why a counter-question consumes an attempt:
the customer was asked, whatever they chose to send back.

A declined slot is simply absent from the search filters, so giving up on a question costs
no inventory - it only widens what we are willing to show.
"""
from __future__ import annotations

import logging
from typing import Any

from src.graph.state import MAX_ASKS_PER_SLOT

logger = logging.getLogger(__name__)


def _slots(state: dict) -> dict[str, Any]:
    return state.get("slots") or {}


def is_answered(state: dict, slot: str) -> bool:
    """True when the slot holds a real value.

    ``None`` does not count: a slot is set to None only by a vague answer or a stated zero,
    and those are tracked in ``declined_slots`` instead.
    """
    value = _slots(state).get(slot)
    if value is None:
        return False
    if isinstance(value, (list, str)) and not value:
        return False
    return True


def is_declined(state: dict, slot: str) -> bool:
    return slot in (state.get("declined_slots") or [])


def at_attempt_cap(state: dict, slot: str) -> bool:
    return (state.get("asked_counts") or {}).get(slot, 0) >= MAX_ASKS_PER_SLOT


def is_resolved(state: dict, slot: str) -> bool:
    """Resolved means "must not be asked again", for any of the three reasons."""
    return is_answered(state, slot) or is_declined(state, slot) or at_attempt_cap(state, slot)


# Why a slot was given up on rather than answered: both asks spent, nothing came back. Kept
# apart from every other decline reason, because the others are the customer ANSWERING - "no
# preference", "skip that one" - and this one is the question failing to land.
GAVE_UP_REASON = "attempt_cap"


def decline_slot(state: dict, slot: str, reason: str = "unanswered") -> None:
    """Stop pursuing this question. Idempotent."""
    declined = state.setdefault("declined_slots", [])
    if slot not in declined:
        declined.append(slot)
        logger.info("QUESTION declined: slot=%s reason=%s", slot, reason)
        if reason == GAVE_UP_REASON:
            # Noted here rather than at either call site: two different paths give up on a
            # slot (resolve_pending_slot settles the pending one, sweep_exhausted_slots
            # catches the rest) and a report written in one of them misses the other.
            state.setdefault("turn_outcome", {}).setdefault("gave_up_on", []).append(slot)
    # A declined slot must not keep a half-value around: the search would filter on it.
    if slot in _slots(state) and _slots(state)[slot] is None:
        _slots(state).pop(slot, None)


def record_no_preference(state: dict, slots: list[str]) -> None:
    """Brief S18: a vague answer or a stated zero resolves the question immediately.

    No second ask - the customer did answer, they answered "I don't mind".
    """
    for slot in slots:
        if not is_answered(state, slot):
            decline_slot(state, slot, reason="no_preference")


def mark_asked(state: dict, slot: str | None) -> None:
    """Count one ask against a slot and make it the pending question.

    Called when the question actually goes out in a reply, so the count reflects what the
    customer was shown rather than what a node intended to show them.
    """
    if not slot:
        state["pending_slot"] = None
        return
    counts = state.setdefault("asked_counts", {})
    counts[slot] = counts.get(slot, 0) + 1
    state["pending_slot"] = slot
    logger.info("QUESTION asked: slot=%s attempt=%d/%d", slot, counts[slot], MAX_ASKS_PER_SLOT)
    # Reaching the cap here does NOT decline the slot: this ask is still in flight and the
    # customer may yet answer it. sweep_exhausted_slots settles it next turn.


def resolve_pending_slot(state: dict, answered: bool) -> None:
    """Settle last turn's question now that the customer has replied.

    Answered - clear it. Not answered and already at the cap - give up on it for good.
    Not answered with an attempt left - leave it pending so it is asked once more.
    """
    slot = state.get("pending_slot")
    if not slot:
        return
    if answered or is_answered(state, slot):
        state["pending_slot"] = None
        return
    if at_attempt_cap(state, slot):
        decline_slot(state, slot, reason=GAVE_UP_REASON)
        state["pending_slot"] = None


def sweep_exhausted_slots(state: dict) -> None:
    """Move every required slot that has used up both asks into ``declined_slots``.

    ``at_attempt_cap`` alone already stops a slot being offered again, but only
    ``declined_slots`` is visible to the prompt - so without this the model is never told
    to stop raising a question Python has quietly given up on. Run once per turn, after
    the customer's reply has been applied: a slot asked for the second time this turn has
    not been failed yet, it has merely been asked.

    """
    for slot in state.get("required_slots") or []:
        if at_attempt_cap(state, slot) and not is_answered(state, slot):
            decline_slot(state, slot, reason=GAVE_UP_REASON)


def required_remaining(state: dict) -> list[str]:
    """Required slots still worth asking about, in the category's own order."""
    return [slot for slot in (state.get("required_slots") or []) if not is_resolved(state, slot)]


def next_unanswered_slot(state: dict) -> str | None:
    """The next required question, or None when there is nothing left to ask.

    This is the single authority on what gets asked. The model proposes a slot in its
    output, but the reply is only allowed to use that phrasing when it names the slot this
    function independently returns - which is what makes S29 unbreakable by a bad
    proposal.
    """
    # A rejected value is re-asked before moving on, so the correction lands in context.
    retry = state.get("invalid_retry_slot")
    if retry and not is_resolved(state, retry):
        return retry
    remaining = required_remaining(state)
    return remaining[0] if remaining else None


def all_required_resolved(state: dict) -> bool:
    """Brief S13/S25: every required question answered, declined or exhausted.

    False when no category is chosen: without one there is no required list to satisfy,
    and an empty list must not read as "qualification complete".
    """
    if not state.get("category"):
        return False
    if not state.get("required_slots") and not state.get("rule_skipped"):
        # An empty list is only "done" when the rules emptied it - every question skipped
        # for this customer. Otherwise it means the category's questions were never loaded.
        return False
    return not required_remaining(state)


def question_text(state: dict, slot: str) -> str:
    """The canonical wording for a slot.

    Read from the live rules rather than the copy taken when the category was set, so a
    question reworded in the control panel is worded that way in conversations already
    under way. The category's own wording first, then an ask rule's, then that copy.
    """
    from src.rules.store import current_rules

    rules = current_rules()
    if state.get("category"):
        wording = rules.category_spec(state["category"]).questions.get(slot)
        if wording:
            return wording
    for rule in rules.rules:
        if rule.action == "ask_question" and rule.slot == slot and (rule.question or "").strip():
            return rule.question.strip()
    return (state.get("slot_questions") or {}).get(slot) or f"Could you tell me the {slot.replace('_', ' ')}?"


# Questions of our own that outrank the next slot question while they are open.
_OPEN_CONFIRMATIONS = (
    "pending_gooseneck_clarification", "pending_category_switch", "pending_keep_filters",
    "pending_axle_basis", "pending_axle_count",
)


def carry_on_question(state: dict) -> tuple[str, str] | None:
    """After a side question mid-qualification (a lookup), the question that picks the flow
    back up: (slot, wording). None when there is no flow to go back to, or when one of our
    own confirmations is open - that is asked on its own turn, not tacked on here.
    """
    if not state.get("category") or state.get("qualification_complete"):
        return None
    if (state.get("turn_outcome") or {}).get("holding"):
        return None  # our question is still waiting on them; nothing more is asked meanwhile
    if any(state.get(key) for key in _OPEN_CONFIRMATIONS):
        return None
    slot = next_unanswered_slot(state)
    if not slot:
        return None
    wording = question_text(state, slot)
    return slot, f"Back to your {state['category']} trailer - {wording[:1].lower()}{wording[1:]}"
