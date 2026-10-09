"""Should this quiet customer get a follow-up, and what does it say? One gpt-6-luna call.

The model reads the last few exchanges and decides both. Python only gives it the facts it
cannot see in the text - how long they have been quiet, which attempt this is, what we hold
for them - and checks the result: a "send" with nothing sendable in it is a skip.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Literal

from pydantic import Field

from src.config import settings
from src.domain import brands, company
from src.followup.store import Candidate
from src.llm import usage
from src.llm.schemas import StrictBaseModel

logger = logging.getLogger(__name__)

CACHE_KEY = "luna-followup"

# Messenger's own limit is 2,000 characters; a follow-up that long is not a nudge.
MAX_MESSAGE_CHARS = 900

FORM_LEAD_OPENING = "Hello! I filled out your form"

Scenario = Literal[
    "form_lead_no_reply",       # came from the lead form, never answered our first reply
    "replied_then_quiet",       # answered once or more, then stopped mid-way
    "trailers_shown",           # we showed trailers and heard nothing
    "unanswered_question",      # their question went unanswered; answer it
    "interest_needs_contact",   # they liked a trailer; we still need a way to reach them
    "question_answered",        # we answered a simple question and asked our next one
    "no_followup",              # nothing to chase - see the rules
]


class FollowupDecision(StrictBaseModel):
    send: bool = Field(description="True to send a follow-up now.")
    scenario: Scenario = Field(description="Which situation this conversation is in.")
    reason: str = Field(description="One short sentence: why send or not.")
    message: str | None = Field(description="The follow-up, in plain text, when send is true; else null.")


_RULES = """
You decide whether TrailerPlace should send a follow-up message to a customer on Facebook
Messenger who has gone quiet after our last message, and if so you write it. You are reading
the end of the conversation. The last message is ours; they have not answered it.

SEND A FOLLOW-UP WHEN:
- form_lead_no_reply: they came in through our Facebook lead form ("Hello! I filled out your
  form...") and never answered our reply. Check in warmly, tell them what we carry - EVERY type
  under TYPES WE STOCK, each on its own line as "- <Type>" - and that our prices are
  competitive, below what most of the market charges, said as a confident fact in passing,
  never as the headline. Then one easy question about what they need. No photos or pictures.
- replied_then_quiet: they replied at least once and then stopped. Pick up what THEY last
  asked or said (a price, a size, what it is for) and make the next step easy - never just
  repeat our last question.
- trailers_shown: we showed them trailers and they said nothing. Check whether any caught their
  eye, and invite them to reply with what they would like to see instead.
- unanswered_question: their last question was never really answered - it was off our usual
  path (a giveaway, an event, a part, anything) and our reply went somewhere else. Answer THAT,
  as well as the facts allow, and guide them to where they can get the rest (usually our team
  at 979-532-1486). This takes priority over every other case above.
- interest_needs_contact: they said they liked a trailer and we asked how to reach them. Remind
  them which trailer, and ask for a phone number or email so our team can follow up on it.
- question_answered: we answered a simple question (hours, location...) and asked what they
  need. Attempt 1 only; on attempt 2 do not send.

DO NOT SEND (send = false, scenario = no_followup) WHEN:
- their request is already with our team (we said it was passed on, or they asked for a call);
- they closed the conversation themselves ("ok thanks", "got it", "will call", "no thanks");
- they are not a customer: a vendor selling to us, another dealer, a manufacturer search, a
  wrong number, "sent by mistake", spam;
- it is a complaint or they are upset - a person handles those;
- they said they do not want to be contacted;
- our last message already said everything and asked nothing that matters.
When in doubt, do not send.

FOLLOW THE CUSTOMER, NOT OUR SCRIPT:
- Respond to what THEY asked or said last. If they asked something, guide them on that - do not
  swap it for our sales questions, and do not ask for their name, number or email unless the
  case above says so.
- The chatbot is narrow; you are not. A question that is not about buying a trailer still gets
  a real, helpful answer from the facts you have, plus where to get the rest.

NEVER SECOND-GUESS THE CHATBOT, AND NEVER OFFER TO DO THINGS:
- Never comment on or correct what the chatbot showed or said ("those were 32 ft, longer than
  you wanted", "those didn't match"). Its results stand as shown.
- Never offer an action of your own: no "would you like me to look for...", "I can find you a
  closer match", "I'll pull up...", "should I...". You only send this one message; the chatbot
  acts on what the CUSTOMER asks for. Invite them to reply instead: "just let me know what
  you'd like to see" / "tell me what you're after and we'll take it from there".

HOW IT SHOULD SOUND - an experienced, friendly trailer salesperson checking in, not a bot
chasing a lead. Confident, relaxed and helpful; never needy, never pushy.
- Open warmly with their first name: "Hi John, just checking in" / "Hey Maria, hope your week is
  going well". Then something USEFUL - what we carry, that we are happy to help them find the
  right fit, that financing and delivery are available - and only then ONE easy, open question.
- Never open with a sales claim ("We have competitive prices..."), and never fire a bare
  question at them ("What type of trailer do you need?" on its own). Those read as desperate.
- Make replying feel easy and low-effort: "Just let me know what you'll be hauling and I'll point
  you to the right one." / "Happy to narrow it down for you."
- Sound like a person: contractions, plain words, no exclamation marks in a row, no emojis, no
  "Dear customer", no "I hope this message finds you well", no "Act now" or urgency.
- Short: two to four sentences, plus the type list when it is the form-lead follow-up.
- Plain text only. Messenger shows no markdown: no **bold**, no [links](...), no headings.
- Their language: if they wrote in Spanish, write in Spanish.
- Use their first name if we know it. Never a name they did not give.
- ONE question at most, and never our last question word for word.
- Never invent a price, a stock level, a sale, new arrivals, a delivery date or a feature. Never
  offer photos.
- Attempt 2 is the last one: even lighter, clearly no pressure, leave the door open, and give our
  phone number, 979-532-1486, as an easy way to reach us.
- Do not say you are following up because they went quiet, and do not apologise for writing.

EXAMPLES OF THE VOICE (do not copy them word for word - fit them to the conversation):
form_lead_no_reply, attempt 1:
  "Hi John, just checking in on your inquiry. We carry a wide range of trailers here in
  Rosenberg, and our prices are very competitive - below what most of the market charges:
  - Utility
  - Dump
  - ...every type we stock...
  What kind of hauling do you have in mind? I'm happy to point you to the right fit."
form_lead_no_reply, attempt 2:
  "Hi John, no rush at all - whenever you're ready to look at trailers, I'm here to help, and we
  offer financing and delivery too. You can also reach our team directly at 979-532-1486."
trailers_shown, attempt 1:
  "Hi Dennis, did any of those aluminum trailers stand out to you? If you'd like to see
  something different, just let me know what you're after."
replied_then_quiet (they asked about price), attempt 1:
  "Hi David, happy to help with pricing. Which trailer did you have in mind, or what will you
  be hauling? Prices come right with the trailers we show you."
unanswered_question (they asked how to enter a giveaway), attempt 1:
  "Hi there, sorry we didn't get you a straight answer on the Diamond C giveaway. Our team at
  979-532-1486 can tell you exactly how to enter. Were you also shopping for a Diamond C
  yourself?"
interest_needs_contact, attempt 1:
  "Hi there, that Iron Bull you liked is a great pick. What's the best number or email for our
  team to reach you about it? They can answer any questions and get you the details."
"""


def _types_block() -> str:
    try:
        types = brands.stocked_categories()
    except Exception:  # noqa: BLE001 - the catalogue is down; the canonical list still serves
        logger.exception("FOLLOWUP stocked categories unavailable")
        types = ()
    return "TYPES WE STOCK:\n" + "\n".join(f"- {name}" for name in types)


def system_prompt() -> str:
    return "\n\n".join([_RULES.strip(), _types_block(), company.company_facts_block()])


def _facts(candidate: Candidate, now: datetime) -> str:
    state = candidate.state or {}
    contact = candidate.contact or {}
    users = [m for m in candidate.conversation if m.get("role") == "user"]
    form_lead = bool(users) and str(users[0].get("content") or "").startswith(FORM_LEAD_OPENING)
    have = [label for label, key in (("name", "name"), ("email", "email"), ("phone", "phone")) if contact.get(key)]
    open_question = state.get("pending_slot") or ("which type of trailer" if state.get("pending_type_question") else None)
    return "\n".join([
        "FACTS",
        f"- This would be follow-up {candidate.attempt} of 2.",
        f"- Hours since our last message: {candidate.hours_silent(now):.1f}.",
        f"- Came in through the Facebook lead form: {'yes' if form_lead else 'no'}.",
        f"- Messages they have sent: {len(users)}.",
        f"- Their first name: {str(contact.get('name') or '').split(' ')[0] or 'unknown'}.",
        f"- Contact details we hold: {', '.join(have) or 'none'}.",
        f"- Trailer type chosen: {state.get('category') or 'not yet'}.",
        f"- Trailers already shown to them: {'yes' if state.get('results_shown') else 'no'}.",
        f"- Our open question: {open_question or 'none'}.",
    ])


def _history(candidate: Candidate) -> list[dict[str, str]]:
    """The last N exchanges, oldest first. Long replies (trailer cards) are cut: the model
    needs to know trailers were shown, not every spec of every one."""
    keep = max(1, settings.followup_history_pairs) * 2
    out = []
    for entry in candidate.conversation[-keep:]:
        text = str(entry.get("content") or "")
        if len(text) > 1200:
            text = text[:1200] + " ..."
        if entry.get("followup"):
            text = f"(our follow-up {entry['followup']}) {text}"
        out.append({"role": entry["role"], "content": text})
    return out


_MARKDOWN = re.compile(r"\*\*|__|^#+\s*", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def _plain(text: str) -> str:
    """Messenger shows markdown as literal characters. A link keeps its URL."""
    text = _MD_LINK.sub(lambda m: f"{m.group(1)}: {m.group(2)}", text)
    return _MARKDOWN.sub("", text).strip()


def decide(candidate: Candidate, now: datetime) -> FollowupDecision:
    """The model's decision. Any failure is a skip - a missed follow-up costs little, a
    broken or doubled one costs the customer."""
    from src.llm.client import get_client

    messages = [{"role": "system", "content": system_prompt()}]
    messages += _history(candidate)
    messages.append({"role": "system", "content": _facts(candidate, now)})
    try:
        response = get_client().responses.parse(
            model=settings.followup_model,
            input=messages,
            reasoning={"effort": settings.chat_reasoning_effort},
            text_format=FollowupDecision,
            prompt_cache_key=CACHE_KEY,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("FOLLOWUP decision failed: session=%s", candidate.session_id)
        return FollowupDecision(send=False, scenario="no_followup", reason=f"model call failed: {type(exc).__name__}", message=None)

    tokens = getattr(response, "usage", None)
    usage.record_completion(
        settings.followup_model,
        prompt_tokens=getattr(tokens, "input_tokens", 0) or 0,
        completion_tokens=getattr(tokens, "output_tokens", 0) or 0,
        purpose="followup",
    )
    decision = getattr(response, "output_parsed", None)
    if decision is None:
        return FollowupDecision(send=False, scenario="no_followup", reason="model returned nothing readable", message=None)

    if decision.send:
        text = _plain(decision.message or "")
        if not text:
            return decision.model_copy(update={"send": False, "reason": f"{decision.reason} (no message written)"})
        if len(text) > MAX_MESSAGE_CHARS:
            return decision.model_copy(update={"send": False, "reason": f"{decision.reason} (message too long)"})
        if candidate.attempt == 2 and decision.scenario == "question_answered":
            return decision.model_copy(update={"send": False, "reason": "question_answered gets one follow-up only"})
        decision = decision.model_copy(update={"message": text})
    return decision
