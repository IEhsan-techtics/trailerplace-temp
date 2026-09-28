"""Send the model's own reply, when it keeps the rules. LLM_WRITES_REPLY only.

With the flag on, the one call writes the whole message (``reply``) and reports what it did:
which question it asked (``asked_slots``), how many questions there are, whether it asked for
their details, and a list of what else it covers (``reply_covers``). Python checks that report
against the rules - it never reads the text itself. Every pattern tried for that misread a good
reply live: "no contact details needed" read as asking for them, "We don’t carry campers" as
never saying so. Reading people is the model's job.

The model wrote its reply BEFORE ``apply_node`` ran, so it could not know what Python then
decided: a value rejected, a question skipped, a notification sent. A reply that breaks a rule
therefore gets one rewrite - a second call that sees the state as it is now and is told what
was wrong. Only when that also fails, or a model call does, does compose build the reply from
the pieces as it does with the flag off.

Never edited. The regex surgery in compose is what sent "You're welcome material will you be
hauling?" live; a reply that needs changing is written again, by the model.

The rules:

* one question per reply (the contact request is separate);
* never a question that is answered, declined or already asked twice (brief S29, S23);
* the contact request exactly when it is due;
* Python's own questions - confirmations, the axle count - are asked by Python.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.graph.nodes import greeting
from src.tools.questions import is_answered, is_resolved, mark_asked, required_remaining

logger = logging.getLogger(__name__)

# The ReplyCover tag the model sets when its reply asks our own open question (hitch or
# brand, switch category, keep answers, the axle questions).
ASKED_OUR_QUESTION = "asked_our_question"

# An off-topic reply is a one-line decline and, at most, one question. Anything this long has
# done what they asked behind an apology - live, "I mainly help with trailers, but here you
# go:" and a recipe.
MAX_OFF_TOPIC_REPLY_CHARS = 400


@dataclass
class _Reply:
    """The model's reply and its own account of it."""

    text: str
    asked: list[str]
    asked_for_contact: bool
    question_count: int
    covers: set[str] = field(default_factory=set)
    offered: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, output: Any) -> "_Reply":
        return cls(
            text=str(getattr(output, "reply", "") or "").strip(),
            asked=[str(s).strip() for s in (getattr(output, "asked_slots", None) or []) if str(s).strip()],
            asked_for_contact=bool(getattr(output, "asked_for_contact", False)),
            question_count=int(getattr(output, "question_count", 0) or 0),
            covers=set(getattr(output, "reply_covers", None) or []),
            offered=[str(c) for c in (getattr(output, "offered_categories", None) or [])],
        )


class _PythonsTurn(Exception):
    """The turn belongs to compose whatever the model wrote - no rewrite can change that."""


def use_model_reply(state: dict, output: Any, situation: str = "flow") -> bool:
    """Send the model's reply, or its one rewrite. True when one went out.

    ``situation`` is the kind of turn compose is on: "flow" for an ordinary one, or one of the
    turns that used to be wholly ours - "off_topic", "unavailable" - with rules of their own.
    """
    reply = _Reply.of(output)
    due = _contact_due(state, output)
    # The model's reading of the message, for the checks and a rewrite to use.
    state.setdefault("turn_outcome", {})["only_acknowledges"] = bool(getattr(output, "only_acknowledges", False))
    try:
        problem = _problem(state, reply, situation, due)
    except _PythonsTurn as turn:
        _log_fallback(state, situation, str(turn), reply)
        return False

    if problem:
        logger.info(
            "COMPOSE asked for a rewrite: session=%s situation=%s reason=%s reply=%r",
            state.get("session_id"), situation, problem, reply.text,
        )
        rewrite = _rewrite(state, reply, problem, situation, due)
        if rewrite is None:
            _log_fallback(state, situation, f"{problem}; the rewrite call failed", reply)
            return False
        second = _problem(state, rewrite, situation, due)
        if second:
            _log_fallback(state, situation, f"{problem}; after the rewrite: {second}", rewrite)
            return False
        reply = rewrite

    _send(state, reply, situation, rewritten=bool(problem))
    return True


def _send(state: dict, reply: _Reply, situation: str, *, rewritten: bool) -> None:
    outcome = state.setdefault("turn_outcome", {})
    contact = state.setdefault("contact", {})
    if reply.asked_for_contact:
        greeting.note_asked(state)
    if greeting.contact_is_complete(contact) and not contact.get("greeted"):
        # The turn their details came in. compose would say its welcome here; the model's
        # reply has already acknowledged them, so it only has to be recorded as said.
        contact["greeted"] = True
        contact["asked"] = True

    slot = reply.asked[0] if reply.asked else None
    outcome["assistant_text"] = reply.text
    outcome["asked_slot"] = slot
    if slot:
        mark_asked(state, slot)
    ours = _our_question(state)
    if ours and ASKED_OUR_QUESTION in reply.covers:
        # Counted where it goes out, as compose counts its own wording of it.
        from src.graph.nodes.compose import note_python_question_asked

        note_python_question_asked(state, ours[0])
    logger.info(
        "COMPOSE used the model's reply: session=%s situation=%s asked_slot=%s rewritten=%s",
        state.get("session_id"), situation, slot, rewritten,
    )


def _log_fallback(state: dict, situation: str, reason: str, reply: _Reply) -> None:
    logger.info(
        "COMPOSE fell back: session=%s situation=%s reason=%s asked_slots=%s reply=%r",
        state.get("session_id"), situation, reason, reply.asked, reply.text,
    )


def _rewrite(state: dict, first: _Reply, problem: str, situation: str, due: bool) -> _Reply | None:
    from src.llm import client

    output = client.rewrite_reply(
        state,
        str((state.get("turn_outcome") or {}).get("user_message") or ""),
        first.text,
        problem,
        _needs(state, situation, due),
    )
    return _Reply.of(output) if output is not None else None


# ------------------------------------------------------------------------------ the checks


def _problem(state: dict, reply: _Reply, situation: str, due: bool) -> str | None:
    """Why this reply cannot go out, or None when it can. Raises _PythonsTurn when no reply
    from the model could have this turn."""
    _pythons_turn(state, situation)
    if not reply.text:
        return "no reply was written"
    problem = _opening_problem(state, reply)
    if problem:
        return problem
    if situation == "unavailable":
        return _unavailable_problem(state, reply)
    if situation == "off_topic":
        problem = _off_topic_problem(reply)
        if problem:
            return problem
    if _our_question(state):
        # Our own question is waiting: the reply answers what they said and asks it, in its
        # own words. The slot-question rules do not apply - it replaces the slot question.
        return (_our_question_problem(state, reply) or _contact_problem(reply, due)
                or _handoff_problem(state, reply))
    return _flow_problem(state, reply, due) or _contact_problem(reply, due) or _handoff_problem(state, reply)


def _our_question(state: dict) -> tuple[str, str] | None:
    """Our own open question - (which, our wording) - or None."""
    from src.graph.nodes.compose import python_question

    return python_question(state)


def _our_question_problem(state: dict, reply: _Reply) -> str | None:
    """The reply must ask our open question, and nothing else.

    It used to be ours outright: the model's reply was dropped and a fixed sentence went out,
    so a customer who said something else heard the identical sentence again with nothing
    they said acknowledged - live, "my daughter lives in Wharton" got the switch question
    word for word, and the template read "For my skid steer" back to them. The model writes
    it now, for the conversation it is in; the fixed sentence is only the fallback.
    """
    which, wording = _our_question(state)
    if ASKED_OUR_QUESTION in reply.covers and reply.question_count == 1 and not reply.asked:
        return None
    return (
        f'our own question is still waiting on them and the reply does not ask it (or asks '
        f'something else too). Respond to what they said, then ask it - once, in your own '
        f'words, keeping this meaning: "{wording}"'
    )


def _opening_problem(state: dict, reply: _Reply) -> str | None:
    """The first reply thanks them for contacting TrailerPlace, and opens "Hi <name>" when
    they gave their name - the dealership's opening. The rest of it is the model's own."""
    if not greeting.is_first_turn(state):
        return None
    # The tags, or the words themselves: live, "Hi Bret, thanks for contacting TrailerPlace!"
    # was rewritten because the model left greeted_by_name out of its tags.
    text = reply.text.lower()
    if "thanked_for_contacting" not in reply.covers and not re.search(
        r"\b(thanks|thank you) for (contacting|reaching out to) trailerplace\b", text
    ):
        return 'it is their first message and it does not say "Thanks for contacting TrailerPlace"'
    first = str((state.get("contact") or {}).get("name") or "").split(" ")[0].lower()
    if greeting.has_name(state.get("contact") or {}) and "greeted_by_name" not in reply.covers and not (
        first and re.match(rf"\s*(hi|hello|hey)\s+{re.escape(first)}\b", text)
    ):
        return 'they gave their name and the reply does not open with "Hi <their name>"'
    return None


def _pythons_turn(state: dict, situation: str) -> None:
    outcome = state.get("turn_outcome") or {}
    if situation == "unavailable":
        return
    if state.get("invalid_retry_slot") == "axle_count":
        raise _PythonsTurn("python asks this turn (axle count)")
    if outcome.get("search_ran"):
        raise _PythonsTurn("a search ran")
    if not state.get("category") and state.get("listing_interest_logged"):
        raise _PythonsTurn("they already picked a trailer")


def _flow_problem(state: dict, reply: _Reply, due: bool) -> str | None:
    outcome = state.get("turn_outcome") or {}
    if outcome.get("holding"):
        # Our question is still waiting on them: nothing else is asked until they answer it
        # or skip it. What they said instead is answered; a plain "ok, thanks" gets a short
        # reply that leaves the door open.
        if reply.question_count or reply.asked:
            return f"our question about {outcome['holding']} is still waiting on them, so it must ask nothing"
        if outcome.get("only_acknowledges") and "invited_questions" not in reply.covers:
            return "they only acknowledged, so it should reply briefly and invite any other questions"
        return None
    if reply.question_count > 1:
        return f"it asks {reply.question_count} questions; one at most"
    if len(reply.asked) > 1:
        return f"it claims {len(reply.asked)} slots; one at most"

    retry = _open_retry(state)
    if retry:
        if reply.asked != [retry]:
            return f"the {retry} they gave was turned down, so {retry} must be asked again"
        if "flagged_wrong_value" not in reply.covers:
            return f"it re-asks {retry} without saying what looked wrong"
        if "thanked_them" in reply.covers:
            return "it thanks them for, or notes, a value that was turned down"

    if outcome.get("type_owed") and not reply.question_count and not (reply.asked_for_contact and due):
        return "they gave specs but no trailer type, and it does not ask which type they are looking for"

    remaining = required_remaining(state) if state.get("category") else []
    if reply.asked:
        slot = reply.asked[0]
        if slot not in remaining:
            if is_answered(state, slot):
                return f"it asks {slot}, which they already answered"
            if is_resolved(state, slot):
                return f"it asks {slot}, which is declined or already asked twice"
            return f"it asks {slot}, which is not a question to ask now"
        if reply.question_count == 0:
            return f"it names {slot} but asks no question"
    elif reply.question_count and remaining:
        return "it asks a question but names no slot, with questions still to ask"
    elif remaining and not (reply.asked_for_contact and due):
        return "it asks nothing, with questions still to ask"
    return None


def _contact_problem(reply: _Reply, due: bool) -> str | None:
    if reply.asked_for_contact and not due:
        return "it asks for their contact details, which is not due this turn"
    if due and not reply.asked_for_contact:
        return "their contact details are due this turn and it does not ask for them"
    return None


def _handoff_problem(state: dict, reply: _Reply) -> str | None:
    """Their details arrived and a request that was waiting on them went to the team."""
    if int((state.get("turn_outcome") or {}).get("emails_flushed") or 0) and "passed_to_team" not in reply.covers:
        return "their request was just passed to our team and the reply does not say so"
    return None


def _unavailable_problem(state: dict, reply: _Reply) -> str | None:
    """A type we do not carry: say so, offer what we do carry, and say what happens next.

    What happens next is the team notification apply already raised, and its status is what
    the reply must match: "sent" (passed on), "stashed" (we need their details first) or
    "dropped" (they declined, so the phone number is how they reach the team).
    """
    from src.tools.unavailable import _stocked

    status = ((state.get("turn_outcome") or {}).get("unavailable_type") or {}).get("status")
    stocked = set(_stocked())

    if "said_not_stocked" not in reply.covers:
        return "it does not say plainly that we do not carry it"
    offered = set(reply.offered)
    if not offered & stocked:
        return "it offers nothing we do carry"
    if offered - stocked:
        return f"it offers {', '.join(sorted(offered - stocked))}, which we do not stock"
    if reply.asked:
        return "it asks a qualification question on this turn"
    if status == "stashed":
        if not reply.asked_for_contact:
            return "the team needs their details and it does not ask for them"
        if reply.question_count:
            return "it asks something besides their details"
    elif reply.question_count > 1:
        return f"it asks {reply.question_count} questions; one at most"
    if status == "sent" and "passed_to_team" not in reply.covers:
        return "the team has their request and it does not say so"
    if status == "dropped" and "gave_phone" not in reply.covers:
        return "they declined contact details and it does not give our phone number"
    if status != "stashed" and reply.asked_for_contact:
        return "it asks for contact details the team does not need"
    return None


def _off_topic_problem(reply: _Reply) -> str | None:
    if "declined_off_topic" not in reply.covers:
        return "it does not decline the off-topic request"
    if len(reply.text) > MAX_OFF_TOPIC_REPLY_CHARS:
        return "it runs long enough to have done what they asked"
    return None


def _open_retry(state: dict) -> str | None:
    """A value Python turned down this turn, still worth asking again."""
    retry = state.get("invalid_retry_slot")
    if not retry or retry == "axle_count" or is_resolved(state, retry):
        return None
    return retry


def _contact_due(state: dict, output: Any) -> bool:
    from src.graph.nodes.compose import _contact_ask_due_now

    return _contact_ask_due_now(state, output)


# -------------------------------------------------------------------- what to tell a rewrite

_PIECES = {"name": "their name", "contact": "an email or phone number"}


def _needs(state: dict, situation: str, due: bool) -> str:
    """What a reply to this turn has to do, as the rewrite is told it."""
    from src.domain.canned_responses import PHONE
    from src.tools import team_notify
    from src.tools.questions import question_text

    outcome = state.get("turn_outcome") or {}
    needs: list[str] = []
    if greeting.is_first_turn(state):
        name = str((state.get("contact") or {}).get("name") or "").split(" ")[0]
        needs.append(
            (f'It is their first message: open with "Hi {name}," and ' if name else
             "It is their first message: ")
            + 'say "Thanks for contacting TrailerPlace!"; the rest in your own words.'
        )

    if situation == "unavailable":
        status = (outcome.get("unavailable_type") or {}).get("status")
        needs.append("Say plainly we do not carry what they asked for, and suggest one to five of "
                     "OUR CATEGORIES that could do the job. No qualification question.")
        needs.append({
            "sent": "Say their request has been passed to our team, who will be in touch.",
            "dropped": f"They declined contact details: give our phone number, {PHONE}.",
        }.get(status, "Say our team can follow up, and ask for "
                      + " and ".join(_PIECES[p] for p in team_notify.missing_pieces(state) or ["contact"])
                      + " - the only question in the reply."))
        return " ".join(needs)

    if situation == "off_topic":
        needs.append("One short line saying you only help with trailers - never do what they asked.")

    ours = _our_question(state)
    if ours:
        needs.append(
            f'Respond to what they said, then ask this question of ours - once, in your own '
            f'words, keeping its meaning: "{ours[1]}" Put {ASKED_OUR_QUESTION} in reply_covers, '
            "leave asked_slots empty, and ask nothing else."
        )
        if due:
            from src.tools import contact_policy

            needs.append(f"End by asking for {contact_policy.describe_missing(state)}, so our team "
                         "can log this.")
        return " ".join(needs)

    if outcome.get("holding"):
        needs.append(
            f"Our question about {outcome['holding']} is still waiting on them: ask NO question at "
            "all. Respond to what they said."
            + (" They only acknowledged: reply briefly and invite them to ask anything else."
               if outcome.get("only_acknowledges") else "")
        )
        if due:
            from src.tools import contact_policy

            needs.append(f"End by asking for {contact_policy.describe_missing(state)}, so our team "
                         "can log this.")
        return " ".join(needs)

    needs.append("At most one question.")
    if outcome.get("type_owed"):
        needs.append("They gave specs but no trailer type: keep every spec they gave, and ask which "
                     "type of trailer they are looking for, naming the few of OUR CATEGORIES their "
                     "specs suit. That is the one question; leave asked_slots empty.")
    retry = _open_retry(state)
    remaining = required_remaining(state) if state.get("category") else []
    if retry:
        needs.append(f"The {retry} they gave looks wrong: say so plainly, do not thank them for it, "
                     f"and ask again: \"{question_text(state, retry)}\" (asked_slots [{retry}]).")
    elif remaining:
        asks = "; ".join(f'{slot}: "{question_text(state, slot)}"' for slot in remaining)
        needs.append(f"Ask ONE of these, word for word, and name it in asked_slots - {asks}.")
    else:
        needs.append("There is no qualification question to ask.")

    if due:
        from src.tools import contact_policy

        needs.append(f"End by asking for {contact_policy.describe_missing(state)}, so our team can "
                     "log this.")
    else:
        needs.append("Do not ask for their contact details.")
    if int(outcome.get("emails_flushed") or 0):
        needs.append("Their request has just been passed to our team: say so.")
    return " ".join(needs)
