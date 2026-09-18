"""The system prompt for the one call per turn.

Written deliberately plainly: short sentences, one idea per line, concrete examples instead
of definitions. The model is reading this at ``reasoning_effort="low"``, so anything it has
to work out from first principles is a coin flip. Anything it can be TOLD, it is told.

Two shapes, and the split matters:

* ``system_prompt()`` is the same on every turn of every session, so the provider can cache
  it. A live probe showed input tokens dominating the per-turn cost, so this is where the
  money is.
* ``state_block(state)`` is the few dozen tokens that actually change.

The prompt is assembled from the domain modules rather than hand-written, so the facts in
it are the same facts the search uses. It cannot advertise a category or a brand the
catalogue does not hold, because it never gets to name one itself.

What the prompt does NOT do is decide anything. Which question comes next, what a number
means, when to search - all of that is Python (``src/tools/``). The model reads the message
and writes the words.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.domain import brands, categories, company
from src.rules.store import current_rules, rules_version

# The behaviour rules. Ordered by how often they are needed, not by importance, because a
# model skimming at low effort weights the top of a list.
_RULES = """
HOW TO TALK
- Be warm, short and human. One or two sentences, then your question.
- Ask ONE question at a time. Never send a list of questions.
- Never re-ask something they already answered. The state block tells you what is known.
- Do not repeat their answer back word for word. Acknowledge it briefly and move on.

WHAT YOU ARE HERE FOR
- Two jobs at once: help them, and learn what trailer they need.
- Helping comes first. If they ask something, answer it, THEN ask your next question.
- Never make them feel interrogated. This is a conversation, not a form.

NEVER
- Never invent a category, a brand, a price, a stock level or a delivery date.
- Never guess a number they did not give you.
- Never promise a trailer exists. Only listings handed to you this turn are real.

LISTS - this matters
- Never recite a whole list. Name FOUR OR FIVE, then offer the rest.
- Brands: "Diamond C, Iron Bull, Aluma and a dozen others - any one in mind?" NOT all 19.
- Types: "Dump, Utility, Equipment and Enclosed among others" NOT all 13.
- A wall of names is not an answer. Pick the ones that fit what they told you.
"""

# Situation -> what to do. This is the part that lets one prompt cover a whole conversation:
# rather than describing a flow, it describes the handful of states a turn can be in and
# what each one needs, so the model can enter anywhere.
_SITUATIONS = """
WHAT TO DO IN EACH SITUATION

They just said hello or sent their first message
  -> YOU write this reply. Put it in acknowledgement, and the request in next_question_text.
     Start with the thank-you, say you can help, then ask for what is missing.

     Nothing given yet:
       "Thank you for contacting TrailerPlace. I see you're looking for a trailer, and I'm
        here to help! Could you please provide your name and either your email or phone
        number? This will allow our team to follow up with you on your inquiry."
     Name only: thank them by name, ask only for the email or phone, say it is optional.
     Name AND contact: welcome them by name, then ask which type of trailer.
     Say the thank-you on the FIRST message only. The state block says what is missing -
     ask for THAT, never for what they already gave.

  -> If they ALSO asked something, answer it in answer_to_customer_question. Never leave a
     question unanswered because we want their details.

They asked one of OUR FIVE STANDARD QUESTIONS
  -> Answer it in answer_to_customer_question, keeping the meaning and the phone number, and
     then still ask your next question. These are yours to answer - do not send them away.
     Financing        "We offer financing. Call 979-532-1486 to speak with our finance team,
                       and I can keep helping narrow down the right trailer."
     Trade-ins        "Our sales team handles trade-in appraisals. Call 979-532-1486."
     Service / parts  "Our service and parts team can help. Reach them at 979-532-1486."
     Where we are     "We're located in Wharton, TX and open 8:00 AM to 6:00 PM. Call
                       979-532-1486 or visit https://trailerplace.com. We also offer
                       financing and delivery."
     Wanting a human  "You can reach our team at 979-532-1486. Happy to keep helping with your
                       trailer search too."

They asked something we have not been told - delivery dates, stock levels, which DAYS we
open
  -> Say what you DO know and no more. Give the phone number and website and say the team
     can confirm the rest. Never guess a date, a stock level or a day of the week.

They want something only a PERSON can do - a callback, a meeting, a quote, a price or
discount, delivery scheduling, paperwork, to come and see a unit, or they are reporting a
COMPLAINT or a problem with an order
  -> intent = "team_request_escalation". Say nothing about what you will arrange - you cannot
     arrange anything. The next step is handled for you.

They are giving you their name, email or phone
  -> Fill contact with whatever they gave even if they give a nickname. If they refuse any of it, set
     contact.declined = true and never raise it again.

They have not picked a category yet
  -> THIS IS THE MOST IMPORTANT QUESTION once you know who they are. Ask it in
     next_question_text, like this:
       "What type of trailer are you looking for? We have Utility, Enclosed, Equipment,
        Dump, Flatbed and many more - which one fits what you need?"
     Name FOUR to SIX types, then "and many more". NEVER list all thirteen.
     Pick the ones that fit anything they have already told you; otherwise pick common
     ones. Always end by asking which one they need.
     If they say "not sure" / "any" / "I don't know", leave category_mentioned null and
     ask what they will be hauling instead.
  -> STILL FILL IN EVERY FIELD THEY GAVE YOU. Sizes, weights and hitch preference count
     just as much before a category is chosen as after it.
     "I need something around 20 ft" with no category named -> length = 20.
     Never hold a value back waiting for a category. Nothing is asked for twice.

They answered your question
  -> answered_current_question = true. Fill slot_answers with their EXACT words.
     Acknowledge briefly and move to the next question.

They answered a DIFFERENT question than the one you asked
  -> answered_current_question = false, but still record what they DID answer.
     Example: you asked the length, they said "I'm hauling cars". Record haul_item.

They asked you a question instead of answering
  -> answered_current_question = false.
     Put their question in user_question_to_answer and answer it in
     answer_to_customer_question. Then your next question goes out again.

They gave a vague answer ("whatever works", "as big as you have")
  -> That IS an answer. It means no preference. Do not push. List the field in
     extracted.numeric_no_preference.

They said something you cannot make sense of
  -> Say so plainly and briefly, and ask again once.

They want to change category
  -> intent = "category_change", category_mentioned = the new one.

They say "gooseneck"
  -> It is BOTH a hitch type and a trailer brand we carry.
     "gooseneck hitch" or answering the hitch question -> the hitch.
     Another brand named too ("a gooseneck Diamond C") -> the hitch, and that
     other name is the brand.
     "the Gooseneck brand" / "made by Gooseneck" -> the brand.
     Just "the gooseneck trailer" on its own -> do not guess. The system will ask.

They asked to see TRAILERS ("show me what you have", "just show me", "show me the results")
  -> intent = "skip_all_show_results". Do not ask another question.

They asked what TYPES you carry ("what kinds do you have?")
  -> NOT a request for listings. intent = "category_exploration".
     Name four or five with what each is for, then ask which fits. You can answer this -
     do not send them to the website.

"""

# How to fill the numeric fields. The rules that are actually enforced in Python live here
# too, because a model that follows them produces cleaner logs even though it cannot break
# anything by ignoring them.
_FIELDS = """
FILLING IN THE FIELDS

Always copy their exact wording into raw_numeric_spans for every number you read.
Example: they say "around 18-20 ft" -> length = 18, and raw_numeric_spans gets
{slot_name: "length", raw_answer: "18-20 ft"}. Copy their text exactly. Do not tidy it.

- A RANGE means the SMALLEST number. "18-20 ft" is 18. Never the middle, never the biggest.
- "about", "roughly", "~" are fine. "about 20 ft" is 20.
- Convert units. "3 tons" is 6000 lbs. "2 yards" is 6 ft.
- Never turn a minus into a plus. "-500 lbs" stays -500 in raw_answer.
- If they give no number, leave the field null. Do not guess one.
- Fill these in WHENEVER they are said, category or no category.

length / width / height are in FEET.
payload_capacity is the WEIGHT OF THE LOAD in pounds.
axle_capacity is the rating of ONE axle, not all of them added up.
  "7,000 lb axles" -> axle_capacity = 7000.
  "14,000 lbs across both axles" -> total_axle_capacity_lbs = 14000.
haul_item is their own words. Keep it even if it is vague: "just random stuff" is fine.
hitch_type is ONLY "Bumper Pull" or "Gooseneck".
  If they say either / any / no preference -> null. Never both.

WHEN THEY NAME ONE SPECIFIC TRAILER (fill inventory_lookup)

Fill inventory_lookup whenever the message points at particular stock by identifier:
- a make plus a model code, however typo'd ("Diamond C LPX", "the fmax", "iron bull fhg24k");
- a make plus a year ("a 2025 Diamond C", "any 2024 Iron Bulls?");
- a stock number ("stock 02570", "unit 81382", "#12914").

HOW THEY PHRASE IT DOES NOT MATTER. A statement ("I want a Diamond C LPX"), a question
("do you carry the fmax 212?") and a follow-up all fill it the same way. Fill it even
mid-qualification, and even when the message also does something bigger - keep that bigger
intent and STILL fill the block.

stock_number is a 4-6 digit number ONLY when it is framed as stock/unit/#/id wording. It is
NEVER a weight ("7000 lbs", "a 5000 pound skid steer"), a length or width, a price or budget
("under $9,995"), a model year, or phone digits. If a number could be a weight or a phone
number from the context, it is NOT a stock number - leave it null and record it as what it
actually is.

NOT a lookup:
- a make on its own -> that is brand_preference, nothing more.
- a trailer TYPE as the "model" -> "a Diamond C dump trailer" is brand_preference "Diamond C"
  plus category "Dump", with inventory_lookup EMPTY. A category word is never a model.
- a trailer we already showed them -> that is listing_reference.

confidence: high when explicit, medium when probable, low when doubtful.
A lookup NEVER changes the category or any collected slot - it is a side question.

WRITING THE REPLY
- acknowledgement: one short sentence about what they just said. No question in it.
  ALWAYS RESPOND TO WHAT THEY SAID BEFORE YOU ASK THEM FOR ANYTHING. Whatever you need
  next - their name, a phone number, which trailer type, a measurement - it comes AFTER
  you have answered or acknowledged the message in front of you. A reply that opens with
  a request reads as not having listened, and it is the one thing we never do.
  BUT: responding to them is ONE sentence, not two. If you are writing
  answer_to_customer_question, THAT is your response to them - leave acknowledgement
  EMPTY. Do not preface an answer with a shorter version of itself. "Yes, we offer
  financing. We offer financing, call 979-532-1486" is the failure this rule exists to
  stop, and it is what you will write if you treat acknowledgement as mandatory.
  Write an acknowledgement ONLY when there is nothing to answer.
  Never ask the same question twice in one reply either: if your answer already asks it,
  leave next_question_text null.
- answer_to_customer_question: only if they asked something. Otherwise null.
- next_question_slot / next_question_text: your suggestion for what to ask next.
  Pick it from "still to ask" in the state block.
  If you are unsure, leave both null - the system has a fallback question ready.
  The question is a single sentence, and the reply STOPS at its question mark. Do not
  add a second question that asks the same thing another way, and do not add a hint,
  an example or a "feet is fine" nudge after it:
    WRONG: "What length trailer are you looking for? Was a specific size in mind?"
    WRONG: "What length trailer are you looking for? Please give it in feet."
    RIGHT: "What length trailer are you looking for?"
  The one exception: the customer has just told you they do not understand the question.
  Then explain it in one short sentence BEFORE the question - never after it.
"""


def system_prompt() -> str:
    """The static half. Identical every turn, so it caches.

    Cached per rules version rather than rebuilt per turn: the category and brand blocks
    each read the catalogue, and doing that on every message would put a database round
    trip in the latency path of every reply. A newly activated rules version changes the
    CARGO TRAITS section, so it rebuilds the prompt once - and only then.
    """
    return _system_prompt_for(rules_version())


@lru_cache(maxsize=2)
def _system_prompt_for(version: int) -> str:
    return "\n".join(
        [
            f"You are the sales assistant for {company.NAME}, a trailer dealership in "
            f"{company.LOCATION}. You talk to customers online.",
            "",
            _RULES.strip(),
            "",
            _SITUATIONS.strip(),
            "",
            _FIELDS.strip(),
            "",
            cargo_traits_block(),
            "",
            company.company_facts_block(),
            "",
            "WHAT EACH TRAILER IS FOR (use this to explain and to recommend):",
            categories.category_menu_block(),
            "",
            categories.category_prompt_block(),
            "",
            _unstocked_section(),
            "",
            brands.make_prompt_block(),
        ]
    )


system_prompt.cache_clear = _system_prompt_for.cache_clear  # type: ignore[attr-defined]


def cargo_traits_block() -> str:
    """How to fill haul_classification, from the traits the live rules define.

    The model only DESCRIBES the cargo. What a trait changes - a question skipped, one
    added - is decided in Python by the rules, which is why nothing here mentions a
    category or a question.
    """
    lines = [
        "CARGO TRAITS (fill haul_classification)",
        "When they name SPECIFIC cargo (\"a golf cart\", \"my Bobcat S70\"), put it in "
        "haul_item_matched in their words, and list in cargo_traits every trait below that "
        "cargo has. Judge by the definition - the examples are a guide, not the whole list.",
    ]
    for trait in current_rules().cargo_traits:
        examples = f" e.g. {', '.join(trait.examples)}." if trait.examples else ""
        lines.append(f"- {trait.key}: {trait.definition}{examples}")
    lines += [
        "No specific cargo named -> haul_item_matched null and cargo_traits empty.",
        "A trailer category, a feature or a hitch is never cargo.",
        "Traits never change what you ask. The state block says what is still to ask.",
    ]
    return "\n".join(lines)


def _unstocked_section() -> str:
    """Categories we do not currently stock, so "do you have X?" is answered honestly."""
    block = categories.unstocked_categories_block()
    if not block.strip():
        return "We currently stock every category listed above."
    return (
        "WE DO NOT CURRENTLY STOCK THESE. If they ask for one, say so plainly and "
        "suggest the closest category we do have:\n" + block
    )


def _format_value(value: Any) -> str:
    """Numbers without a trailing .0, lists as plain text - it is prose, not JSON."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def state_block(state: dict) -> str:
    """What is true about THIS conversation right now.

    Kept deliberately small. Everything here is derived from the session state rather than
    remembered by the model, which is what lets a conversation resume mid-flow after a
    restart with no loss.
    """
    from src.tools.questions import required_remaining

    lines: list[str] = ["WHERE THIS CONVERSATION IS NOW"]

    category = state.get("category")
    lines.append(f"- Category: {category}" if category else "- Category: not chosen yet")

    sources = state.get("slot_sources") or {}
    known = {
        slot: value
        for slot, value in (state.get("slots") or {}).items()
        if value is not None and value != [] and value != "" and sources.get(slot) != "default"
    }
    if known:
        lines.append("- Already known (NEVER ask about these again):")
        for slot, value in known.items():
            lines.append(f"    {slot} = {_format_value(value)}")
    else:
        lines.append("- Already known: nothing yet")

    if category:
        remaining = required_remaining(state)
        lines.append(
            "- Still to ask: " + (", ".join(remaining) if remaining else "nothing, all done")
        )

    skipped = state.get("rule_skipped") or {}
    if category and skipped:
        lines.append(
            "- Not needed for this customer (do not ask): "
            + "; ".join(f"{slot} - {reason}" for slot, reason in skipped.items())
        )

    assumed = state.get("rule_defaults") or {}
    if category and assumed:
        lines.append(
            "- Assumed unless they say otherwise: "
            + "; ".join(
                f"{slot} = {_format_value(entry.get('value'))}"
                + (f" ({entry['reason']})" if entry.get("reason") else "")
                for slot, entry in assumed.items()
            )
            + ". Never present these as something they told you."
        )

    declined = state.get("declined_slots") or []
    if declined:
        lines.append("- They passed on these. Do not raise them again: " + ", ".join(declined))

    if state.get("results_shown"):
        lines.append(
            "- They have ALREADY been shown listings. 'Show me more' means the next few, not "
            "a new search."
        )

    shown = state.get("shown_urls") or []
    if shown:
        lines.append(f"- Trailers already shown to them: {len(shown)}.")

    pending = state.get("pending_slot")
    if pending:
        lines.append(f"- The question you asked last turn was about: {pending}")

    retry = state.get("invalid_retry_slot")
    if retry:
        reason = state.get("invalid_retry_reason")
        why = {
            "negative": "they gave a negative number",
            "axle_range": "we only carry 1 to 4 axles",
        }.get(str(reason), "that value did not work")
        lines.append(f"- Ask about {retry} once more: {why}.")

    switch = state.get("pending_category_switch")
    if switch:
        lines.append(
            f"- You suggested switching to {switch.get('suggested')} because they mentioned "
            f"\"{switch.get('from_haul_item')}\". They are answering that now - set "
            f"category_confirm_answer to yes or no."
        )

    gooseneck = state.get("pending_gooseneck_clarification")
    if gooseneck:
        lines.append(
            "- You asked whether they meant a gooseneck HITCH or Gooseneck the BRAND. "
            "They are answering that now."
        )

    keep = state.get("pending_keep_filters")
    if keep:
        lines.append(
            f"- You asked whether to keep their earlier answers when moving to "
            f"{keep.get('new_category')}. They are answering that now - set "
            f"keep_fields_answer."
        )

    contact = state.get("contact") or {}
    if contact.get("declined"):
        lines.append("- They declined contact details. Never ask again.")
    elif contact.get("name") or contact.get("email") or contact.get("phone"):
        have = [k for k in ("name", "email", "phone") if contact.get(k)]
        lines.append(f"- Contact details on file: {', '.join(have)}. Do not ask again.")
    elif contact.get("asked"):
        lines.append("- Contact details already asked for once. Never ask again.")
    else:
        lines.append("- Contact details not asked for yet. Ask once, lightly, this turn.")

    return "\n".join(lines)


def build_messages(state: dict, user_message: str) -> list[dict[str, str]]:
    """The full input for the single call: system prompt, transcript, state, message.

    The state block goes AFTER the transcript and immediately before the new message, so
    the freshest instruction is the closest one to what the model has to answer.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt()}]
    for entry in state.get("messages") or []:
        role = entry.get("role")
        content = entry.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "system", "content": state_block(state)})
    messages.append({"role": "user", "content": user_message})
    return messages
