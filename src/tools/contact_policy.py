"""When we ask for their name and a way to reach them. LLM_WRITES_REPLY only.

The dealership's rule: never as a toll gate. We ask at the four moments where the details are
worth something to the customer, every time one of them comes round, until we have them:

* their FIRST message has nothing about trailers in it ("hi") - so the welcome asks who they are;
* we have just SHOWN them trailers - the team is told what they saw, and needs to know who;
* we have just answered one of the STANDARD questions (hours, financing, trade-ins...);
* they need our TEAM - a complaint, a callback, a quote, a type we do not stock, a trailer
  they want.

Never otherwise, and never before the trailers: a customer asking for a dump trailer is shown
dump trailers, and asked who they are after. A customer who declined is still asked at these
moments - it is a courtesy each time, and the moment is where it earns its place.

Every judgement here is the model's: whether the message is about trailers is its
``about_trailers``, a standard question its ``faq_key``, an escalation its ``intent``. Python
only puts the moments together and keeps count; the model writes the request.
"""
from __future__ import annotations

from typing import Any

from src import config
from src.tools import team_notify

_ESCALATION_INTENTS = {"team_request_escalation", "listing_interest"}


def active() -> bool:
    return bool(config.settings.llm_writes_reply)


def moment(state: dict, output: Any) -> str | None:
    """Which of the four moments this turn is, or None."""
    outcome = state.get("turn_outcome") or {}
    if outcome.get("search_ran") or outcome.get("cited_listing_urls") or outcome.get("idle_results"):
        return "results"
    if getattr(output, "faq_key", None):
        return "faq"
    if (
        getattr(output, "intent", "") in _ESCALATION_INTENTS
        or outcome.get("needs_a_person")
        or outcome.get("unavailable_type")
        or outcome.get("link_interest")
    ):
        return "escalation"
    if int(state.get("turn_index") or 0) <= 1 and not getattr(output, "about_trailers", True):
        return "first_message"
    return None


def missing(state: dict) -> list[str]:
    """"name" and/or "contact" (an email or phone) - what we do not have yet."""
    return team_notify.missing_pieces(state)


def ask_due(state: dict, output: Any) -> bool:
    """Whether this reply ends by asking for their details.

    Not in the reply to the message that refused them. A refusal only stops the asking for
    that turn - they are asked again at the next moment - but asking in the very answer to
    "I'd rather not share my details" is not listening.
    """
    if getattr(getattr(output, "contact", None), "declined", False):
        return False
    return bool(missing(state)) and moment(state, output) is not None


def describe_missing(state: dict) -> str:
    """"your name and an email or phone number", in the model's words to use."""
    pieces = {"name": "their name", "contact": "an email or phone number"}
    return " and ".join(pieces[piece] for piece in missing(state)) or "nothing"
