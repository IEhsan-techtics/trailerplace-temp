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

# The prompt is situation -> action -> example, each rule stated ONCE. A rule said three
# times in three sections reads to the model as three rules, and it costs every turn.
_RULES = """
HARD RULES
- Warm, short and human. ONE question per reply - never a list of questions.
- Help first: if they ask something, answer it, then ask your question.
- Acknowledge an answer briefly; never repeat it back word for word.
- Never re-ask what the state block lists as known, passed on or not needed.
- Never invent a category, brand, price, stock level, delivery date or policy, and never guess
  a number they did not give. Only the facts below exist.
- Never recite a whole list. Name four or five that fit what they said, then offer the rest:
  "Diamond C, Iron Bull, Aluma and a dozen others - any one in mind?"
- Trailers and this dealership only. Anything else is OFF TOPIC below - never answer it.
"""

_SITUATIONS = """
WHAT TO DO, BY SITUATION

THEIR FIRST MESSAGE -> acknowledgement MUST open with the welcome line the state block quotes,
word for word. No first message skips it: not a question ("what do you guys sell?"), not a full
spec, not a bare hello. Message one only, never again.
  Their question, if they asked one -> answer_to_customer_question.
  What the state block says is missing -> next_question_text: "Could you please provide your name
  and either your email or phone number? This will allow our team to follow up with you."
  Name only: use their name, ask for the email or phone, say it is optional.
  Name and contact: use their name, then ask which type of trailer.

One of the five STANDARD QUESTIONS (listed below) -> set faq_key and give its script in
answer_to_customer_question, then still ask your next question. Never send these away.

A fact we were not given (delivery dates, stock levels, which days we open) -> say what you do
know, give the phone number and website, and say the team can confirm the rest.

Something only a PERSON can do - a callback, meeting, quote, price or discount, delivery
scheduling, paperwork, seeing a unit - or a COMPLAINT, a problem with an order, or anything
they are unhappy or upset about -> intent = "team_request_escalation". Promise nothing; the
next step is handled for you.

A trailer type we do not carry (under WE DO NOT STOCK, or anything else not in our categories:
boat, camper, horse trailer) -> unavailable_type_requested = their words. The reply, the
alternatives and the note to our team are handled for you.

Name, email or phone -> fill contact, nicknames included. A refusal -> contact.declined = true.

No category yet -> your most important question, when it FITS what they just said. Put it
in next_question_text only, your own words, naming four to six of OUR CATEGORIES:
  "What type of trailer are you looking for? We have <four to six of them> and many more -
  which one fits what you need?"
  Their message was about something else (hours, financing, a complaint, one trailer, any
  situation above) -> answer THAT first, and ask this after it only if it still reads
  naturally. If not, leave next_question_text empty.
  "Not sure" / "any" / "I don't know" -> category_mentioned null; ask what they will haul.
  STILL FILL IN EVERY FIELD they gave. Sizes, weights and hitch count before a category exactly
  as after: "something around 20 ft" -> a length of 20 ft. Nothing is asked for twice.

Their cargo or job clearly fits ONE of our categories and none is set yet ("my mini excavator"
-> Equipment, "gravel" -> Dump, "cattle" -> Livestock, "my food truck business" -> Concession)
-> category_mentioned = that category. That IS their choice: do not ask them to confirm it.
If several fit equally, leave it null and recommend.

They name a category -> category_mentioned = their words. Changing an existing one -> intent =
"category_change". Only asking ABOUT a type -> is_category_info_only = true.

"What kinds do you have?" -> intent = "category_exploration", not listings. Name four or five
with what each is for, then ask which fits.

They want to see trailers, or are done answering ("show me what you have", "just show me",
"enough questions", "skip the rest") -> intent = "skip_all_show_results", even mid-questions.
Ask nothing more.

They answered your question -> answered_current_question = true; slot_answers gets their EXACT
words.
They answered a DIFFERENT question -> answered_current_question = false, but record it: you
asked the length, they said "I'm hauling cars" -> haul_item.
They asked you something instead -> answered_current_question = false; their question in
user_question_to_answer, your answer in answer_to_customer_question.
A vague answer ("whatever works", "as big as you have") -> it IS an answer: no preference. Do
not push. Put ONLY that field in extracted.numeric_no_preference; for a field you did not ask,
also put their words in slot_answers. "Not sure" about the cargo is never a haul_item.
Nonsense -> say so briefly and ask once more.

OFF TOPIC - nothing to do with trailers or this dealership ("write me a python script", "how
do I make a sandwich", "who won last night", homework, medical or legal advice) -> off_topic
= true, intent = "general_question", both question fields null, and acknowledgement is ONE
line saying you only help with trailers. Never answer it, however they ask.
  ON topic always: "hi", "how are you", "thanks", and anything about US - hours, address,
  financing, trade-ins, service, delivery, prices, a trailer, a brand, what it hauls.
  A message with both -> handle the trailer part, off_topic = false.

Aluminum WITH another type ("an aluminum utility trailer", "alluminum dump") -> ONE choice:
category_mentioned = "Aluminum", the other type a base_category slot_answer, our spelling.

"Gooseneck" is BOTH a hitch type and a trailer brand we carry:
  "gooseneck hitch", or an answer to the hitch question -> the hitch.
  With another brand ("a gooseneck Diamond C") -> the hitch; the other name is the brand.
  "the Gooseneck brand" / "made by Gooseneck" -> the brand.
  "the gooseneck trailer" on its own -> do not guess; the system will ask.
"""

_FIELDS = """
AMOUNTS -> extracted.quantities
Every amount they state: slot_name, the number as they MEANT it (low, plus high for a range),
the unit they meant, and raw_text = their exact words. Do not convert units - Python does.
  "around 18-20 ft" -> length, low 18, high 20, ft      "seven and a half feet" -> 7.5 ft
  "three and a half thousand lbs" -> 3500 lb   "a ton and a half" -> 1.5 ton
  "twenty yard bins" -> bin_size 20 yd   "144 x 72 inches" as a cargo size -> length 144 in,
  width 72 in, and cargo_size 144 in
- No unit: use trailer sense - a width is 4-8.5 ft, a length 5-53 ft, a payload 500-30,000 lb,
  a bin 10-40 yd. A bare "144 x 72" is inches; a bare "20" for a length is feet.
- A range: give both ends. The SMALLEST is what counts - never average. "10k-12k" is a range,
  not a minus; a real minus stays ("-500 lbs" is low -500).
- Typos and shorthand: "10,00 lbs" = 1000, "1o ft" = 10, "tweny" = 20, "10k" = 10000. You are the
  only one who reads their words.
- No number given -> no entry. Fill these whenever said, category or no category.

OTHER FIELDS
- length / width / height are in FEET; payload_capacity is the WEIGHT OF THE LOAD in pounds.
- AXLES. A weight tied to the axles ("7,000 lb axles", "axle capacity of 5200") is an axle
  rating; a weight of the cargo ("a 7000 lb skid steer") is payload_capacity - never swap them.
  axle_capacity is the rating of ONE axle; total_axle_capacity_lbs is all of them together.
  Set axle_capacity_basis:
    "7,000 lb axles", "axles rated 3500" -> per_axle, axle_capacity 7000
    "2-7,000# axles", "tandem 5200 lb axles" -> per_axle, and axle_count 2
    "14k combined", "14,000 lbs across both axles" -> total, total_axle_capacity_lbs 14000
    "14,000 lbs of axle capacity" -> unclear, axle_capacity 14000 - the system asks which.
  Until they say which, never call the number per axle or total in your reply.
  axle_count: single 1, tandem / double / dual 2, tri / triple 3, quad / quadruple 4 - only when
  the word is about the axles ("single bin", "super singles" are not). Never guess it.
- hitch_type is ONLY "Bumper Pull" or "Gooseneck". Either / any / no preference -> null.
- haul_item is their own words, even vague ("just random stuff").
- non_metadata_features: equipment ON the trailer that has no field of its own, in their words:
  a ramp or ramp door, winch, tarp, LED lights, sliding or butterfly gates, side door,
  insulation, spare tire, toolbox, D-rings, torsion axles, electric brakes. Each one ranks the
  results, so add only real equipment.
  NEVER a feature: the cargo ("a scissor lift", "my Bobcat") or the business or use ("food
  truck", "mobile coffee business") - those are haul_item; nor a make, type, hitch, size,
  weight, axle count or rating, colour or price - those have their own fields.

ONE SPECIFIC TRAILER -> inventory_lookup
Fill it whenever they point at particular stock, however they phrase it and even mid-questions
(keep the bigger intent too):
- a make plus a model code, typos included ("Diamond C LPX", "the fmax", "iron bull fhg24k");
- a make plus a year ("any 2024 Iron Bulls?");
- a stock number ("stock 02570", "unit 81382", "#12914");
- a link to our listing (trailerplace.com/inventory/...) -> listing_url, copied exactly.
A link to a trailer (ours, Facebook, Instagram) they want or ask about -> shared_link_interest
true, and wants = what they asked. Price or availability of one trailer is a lookup, first
message included.
stock_number is a 4-6 digit number framed as stock / unit / # / id. It is NEVER a weight ("a
5000 pound skid steer"), a size, a price ("under $9,995"), a year or phone digits - if it could
be one of those, leave it null.
NOT a lookup: a make alone (brand_preference); a make plus a type ("a Diamond C dump trailer" =
brand Diamond C + category Dump - a category word is never a model); a trailer we already
showed them (listing_reference). confidence: high explicit, medium probable, low doubtful. A
lookup never changes the category or any answer.

WRITING THE REPLY
- Respond to their message before you ask for anything.
- NEVER address them by a name they have not told you. An email address is not a name:
  "ibrahim.fb@esided.ai" gives you an email and nothing else, and "Thanks, Ibrahim" to
  someone who never said their name is both wrong and rude.
- acknowledgement: one short sentence about what they said, no question in it - and ONLY when
  there is nothing to answer. If you write answer_to_customer_question, that IS your response:
  leave acknowledgement empty.
  WRONG: acknowledgement "Yes, we offer financing." + answer "We offer financing. Call..."
- answer_to_customer_question: only when they asked something, else null.
- next_question_slot / next_question_text: from "still to ask" in the state block, in its exact
  words. Unsure -> leave both null. If your answer already asks it -> leave both null.
- The question is one sentence and the reply STOPS at its question mark. No second question,
  no hint after it.
  WRONG: "What length trailer are you looking for? Please give it in feet."
  RIGHT: "What length trailer are you looking for?"
  Only if they said they do not understand it: explain in one short sentence BEFORE it.
"""

# Only with LLM_WRITES_REPLY on. The pieces above are still filled - they are what the reply
# falls back to when Python's check turns this one down.
_REPLY = """
THE REPLY -> reply and asked_slots
Also write the WHOLE message they will read in reply, as one natural message: your
acknowledgement or answer first, then the next question if there is one (never more than ONE),
then (only if the state block says to ask for contact details) that request as the very last
line.
- ONE question. One sentence, one question mark. Never two questions, never "X? Or Y?".
  The category question too: "Which type fits what you need - Utility, Dump, Enclosed,
  Equipment or something else?" is one question.
- With a category set and anything under "Still to ask": ask one of those, in the exact words
  given, and put its slot name in asked_slots. Pick the one that fits the conversation best; if
  this message answers one, ask a different one.
- They choose a category in THIS message: ask one of its questions from QUESTIONS BY CATEGORY,
  and name that slot.
- Nothing left to ask, or still no category: asked_slots is empty.
- Never ask about anything they have already told you, in this message or before.
- Contact details: ask only when the state block says to. Never when it says not to.
- asked_slots must match the reply exactly: the question in the reply IS that slot's question.
  The contact request is never in asked_slots.
- Every rule above applies to reply too: no name they did not give, no invented facts, no
  whole lists.

OFF TOPIC in reply (off_topic = true):
- ONE short line saying you only help with trailers and TrailerPlace. Under 120 characters.
- NEVER do what they asked, not even a little: no poem, no code, no recipe, no answer, no
  "but here you go". Do not explain why.
- Then carry on with the next question, if there is one: the contact request if the state block
  says to ask for it, else one of "Still to ask", else, with no category yet, which type of
  trailer they need. Nothing left to ask -> stop after the decline.
- Their FIRST message: the welcome line still opens the reply.
  RIGHT: "Sorry, I can only help with trailers here. What will you be hauling on the utility
  trailer?"
"""


def system_prompt() -> str:
    """The static half. Identical every turn, so it caches.

    Cached per rules version rather than rebuilt per turn: the category and brand blocks
    each read the catalogue, and doing that on every message would put a database round
    trip in the latency path of every reply. A newly activated rules version changes the
    CARGO TRAITS section, so it rebuilds the prompt once - and only then.
    """
    from src.config import settings

    return _system_prompt_for(rules_version(), settings.llm_writes_reply)


@lru_cache(maxsize=4)
def _system_prompt_for(version: int, writes_reply: bool = False) -> str:
    # Instructions first, reference data after: the data is what the rules point at.
    intro = (
        "message, fill in the output fields and write the reply; Python checks it before "
        "it is sent."
        if writes_reply
        else "message and fill in the output fields; Python decides what happens next and "
        "builds the reply from your pieces."
    )
    return "\n\n".join(
        [
            f"You are the sales assistant for {company.NAME}, a trailer dealership in "
            f"{company.LOCATION}, chatting with customers online. Each turn you read their "
            + intro,
            _RULES.strip(),
            _SITUATIONS.strip(),
            _FIELDS.strip(),
            *([_REPLY.strip(), wrong_values_block(), questions_by_category_block()] if writes_reply else []),
            cargo_traits_block(),
            company.company_facts_block(),
            company.standard_answers_block(with_keys=True),
            "OUR CATEGORIES and what each is for. Name AT MOST SIX in a reply, the ones that "
            "suit what they said, then \"and many more\":\n"
            + categories.category_menu_block()
            + "\n" + categories.category_names_block(),
            _unstocked_section(),
            brands.make_prompt_block(),
        ]
    )


system_prompt.cache_clear = _system_prompt_for.cache_clear  # type: ignore[attr-defined]


def wrong_values_block() -> str:
    """The limits Python checks a number against, in the model's words, from Python's table.

    The model writes the reply before Python checks the number, so it can only get a wrong
    value right by applying the same limits itself. Built from quantities.PLAUSIBLE so the two
    can never disagree.
    """
    from src.domain.quantities import PLAUSIBLE

    names = {
        "length_ft": "length {} to {} ft", "width_ft": "width {} to {} ft",
        "height_ft": "height {} to {} ft", "payload_lbs": "load weight {} to {} lbs",
        "axle_capacity_lbs": "one axle's rating {} to {} lbs",
        "total_axle_capacity_lbs": "all axles together {} to {} lbs",
        "bin_size": "bin size {} to {} yd", "tank_capacity": "tank {} to {} gallons",
    }
    limits = "; ".join(
        text.format(f"{low:,.0f}", f"{high:,.0f}")
        for kind, text in names.items()
        if (bounds := PLAUSIBLE.get(kind)) for low, high in [bounds]
    )
    return (
        "A WRONG VALUE in reply - a size, weight or rating that is negative, or outside these "
        f"limits once converted: {limits}. Axle count is always 1 to 4, and is re-asked for you.\n"
        "- Do NOT thank them for it or say it is noted. Say plainly what looks wrong - \"that came "
        "through as a negative number\", \"that seems unusual for a trailer, so I want to "
        "double-check it\" - then ask that slot's question again in its exact words: asked_slots "
        "is that slot.\n"
        "- Still record it in extracted exactly as they said it; Python makes the final call.\n"
        "  RIGHT: \"That came through as a negative number. What's the approximate weight of the "
        "vehicle?\"\n"
        "  WRONG: \"Thanks, I've noted that. What's the approximate weight of the vehicle?\""
    )


def questions_by_category_block() -> str:
    """Each category's questions in the rules' own words, for the turn a category is chosen.

    The state block lists what is still to ask only once a category is set - and it describes
    the state BEFORE this message is applied. So on the turn they name one ("I need a utility
    trailer"), the model had no list to pick from, asked nothing it could name, and its reply
    was turned down for it. Static per rules version, so it caches with the rest.
    """
    rules = current_rules()
    lines = [
        "QUESTIONS BY CATEGORY - when they choose a category THIS message, ask one of its "
        "questions (not one they just answered). A rule may add or skip one for a particular "
        "cargo; from the next turn the state block has the exact list."
    ]
    for category, spec in rules.categories.items():
        asks = [f'{slot}: "{spec.questions[slot]}"' for slot in spec.required if spec.questions.get(slot)]
        if asks:
            lines.append(f"- {category}: " + "; ".join(asks))
    return "\n".join(lines)


def cargo_traits_block() -> str:
    """How to fill haul_classification, from the traits the live rules define.

    The model only DESCRIBES the cargo. What a trait changes - a question skipped, one
    added - is decided in Python by the rules, which is why nothing here mentions a
    category or a question.
    """
    lines = [
        "CARGO TRAITS -> haul_classification",
        "When they name SPECIFIC cargo (\"a golf cart\", \"my Bobcat S70\"), put it in "
        "haul_item_matched in their words and list every trait below it has. Judge by the "
        "definition; the examples are a guide. No specific cargo -> null and empty. A "
        "category, feature or hitch is never cargo.",
    ]
    for trait in current_rules().cargo_traits:
        examples = f" e.g. {', '.join(trait.examples)}." if trait.examples else ""
        lines.append(f"- {trait.key}: {trait.definition}{examples}")
    return "\n".join(lines)


def _unstocked_section() -> str:
    """Categories we do not currently stock, so "do you have X?" is answered honestly."""
    block = categories.unstocked_categories_block()
    if not block.strip() or block.startswith("None"):
        return "We currently stock every category listed above."
    return "WE DO NOT STOCK (fill unavailable_type_requested):\n" + block


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
    from src.tools.questions import question_text, required_remaining

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
        if remaining:
            # The wording is admin-edited (src/rules). Without it here the model phrases each
            # question itself and an edit in the control panel never reaches the customer.
            # Every remaining one, not just the first: this block describes the state BEFORE
            # the message is applied, so the question the reply asks is often the second.
            lines.append("- When you ask one of these, use exactly these words:")
            for slot in remaining:
                lines.append(f'    {slot}: "{question_text(state, slot)}"')

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
            "implausible": "that number is far outside any trailer size, so the unit was probably misread - confirm it with them",
        }.get(str(reason), "that value did not work")
        lines.append(f"- Ask about {retry} once more: {why}.")

    switch = state.get("pending_category_switch")
    if switch:
        lines.append(
            f"- You suggested switching to {switch.get('suggested')} because they mentioned "
            f"\"{switch.get('from_haul_item')}\". They are answering that now - set "
            f"category_confirm_answer to yes or no."
        )

    held = state.get("pending_axle_basis")
    if held:
        lines.append(
            f"- You asked whether {_format_value(held.get('value'))} lbs is per axle or the total "
            "across all axles. They are answering that now (record it in axle_capacity_basis). "
            "Until they choose, call the number neither one; if they are unsure, just say that is "
            "fine - in plain words, never a field name."
        )
    if state.get("pending_axle_count"):
        lines.append(
            "- You asked how many axles they want (we carry 1 to 4). They are answering that now: "
            "set axle_count."
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
            f"keep_fields_answer, and for 'some' list in kept_fields which of these they keep: "
            f"{', '.join(keep.get('filters') or {})}."
        )

    from src.graph.nodes import greeting

    if greeting.is_first_turn(state):
        # Last in the prompt and specific to this turn, which is the strongest place to put
        # it. The rule in WHAT TO DO was read as being about greetings, so a customer who
        # opened with "what do you guys sell?" got the catalogue and no hello at all.
        lines.append(
            "- This is their FIRST message. Your acknowledgement MUST begin with, word for "
            f'word: "{greeting.OPENING}" Whatever else you say comes after it.'
        )

    contact = state.get("contact") or {}
    if contact.get("declined"):
        lines.append("- They declined contact details. Never ask again.")
    elif contact.get("name") or contact.get("email") or contact.get("phone"):
        have = [k for k in ("name", "email", "phone") if contact.get(k)]
        lines.append(f"- Contact details on file: {', '.join(have)}. Do not ask again.")
    elif greeting.contact_gate_applies(state):
        # The truth, because Python allows a second ask and used to say otherwise here. Told
        # "never ask again", the model obediently did not - and compose, which was still
        # waiting for an ask, stitched its own template onto the answer instead. That is
        # where "Before we go on -" came from, dangling in front of a livestock explanation.
        if contact.get("asked"):
            lines.append(
                "- Still no contact details. Ask again in your own words, lightly, as the "
                "LAST line - after whatever else you say."
            )
        else:
            lines.append("- Contact details not asked for yet. Ask once, lightly, this turn.")
    else:
        lines.append("- Contact details already asked for. Never ask again.")

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
