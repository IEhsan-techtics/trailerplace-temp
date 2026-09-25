from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _drop_titles(schema: dict, _model: type) -> None:
    """Pydantic titles every model and field ("Length", "Width"...). The provider sends the
    whole schema to the model on every call, and the titles tell it nothing the key does not."""
    schema.pop("title", None)
    for prop in (schema.get("properties") or {}).values():
        prop.pop("title", None)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_drop_titles)


# Field descriptions are one line each. The rules behind them live in the system prompt
# (src/llm/prompt.py), written once - a rule repeated here is paid for twice on every turn.
class ContactInfo(StrictBaseModel):
    name: str | None = Field(description="Their name, else null.")
    email: str | None = Field(description="Their email, else null.")
    phone: str | None = Field(description="Their phone, else null.")
    declined: bool = Field(
        default=False,
        description="True when THIS message refuses contact details, whatever the intent.",
    )


class SlotAnswer(StrictBaseModel):
    slot_name: str = Field(description="A slot name from the state block.")
    raw_answer: str = Field(description="Their exact words for it.")


class EmailTrigger(StrictBaseModel):
    kind: Literal["faq", "escalation", "team_request", "listing_interest"] = Field(description="Email-worthy request kind.")
    faq_key: Literal["contact_human", "financing", "trade_in", "service_parts", "store_info"] | None = Field(description="Required for FAQ triggers, otherwise null.")
    listing_reference: int | None = Field(description="1-based shown-listing index for listing interest, otherwise null.")
    description: str = Field(description="One-line summary for the email body.")


class ExtractedFields(StrictBaseModel):
    length: float | None = Field(description="Feet; smallest end of a range.")
    width: float | None = Field(description="Feet; smallest end of a range.")
    height: float | None = Field(description="Feet; smallest end of a range.")
    payload_capacity: float | None = Field(description="Load weight in lb; smallest end of a range.")
    axle_capacity: float | None = Field(description="Rating of ONE axle in lb.")
    total_axle_capacity_lbs: float | None = Field(description="All axles together in lb.")
    axle_count: int | None = Field(description="Axles wanted: single 1, tandem 2, tri 3.")
    axle_capacity_basis: Literal["per_axle", "total", "unclear"] | None = Field(
        description="What a stated axle capacity meant; null when none was stated."
    )
    hitch_type: list[Literal["Bumper Pull", "Gooseneck"]] | None = Field(description="One clear hitch, else null.")
    haul_item: str | None = Field(description="What they will haul, in their words.")
    brand_preference: str | None = Field(description="A make from MAKES WE STOCK, or an unknown one verbatim.")
    non_metadata_features: list[str] = Field(
        description="Equipment on the trailer with no field of its own (see OTHER FIELDS). Never cargo, a use, a make, category or subcategory, a model word, hitch, size, weight, axle count or rating, colour or price."
    )
    numeric_no_preference: list[str] = Field(description="Slots they answered with no preference.")
    quantities: list["Quantity"] = Field(description="Every amount they stated (see AMOUNTS). Overrides the numbers above.")


# Units the customer can mean. Kept to what trailer questions actually use; Python holds
# the conversion table (src/domain/quantities.py) and rejects a unit that does not fit the slot.
QuantityUnit = Literal["ft", "in", "yd", "m", "cm", "mm", "lb", "kg", "ton", "tonne", "gal", "l", "cu_yd"]


class Quantity(StrictBaseModel):
    """One amount the customer stated. The LLM reads the language; Python does the arithmetic."""

    slot_name: str = Field(
        description=(
            "length, width, height, payload_capacity, axle_capacity, total_axle_capacity_lbs, "
            "cargo_size, trailer_size, bin_size or tank_capacity."
        )
    )
    raw_text: str = Field(description="Their exact words for this amount.")
    low: float = Field(description="The number as meant, in unit; the smaller end of a range.")
    high: float | None = Field(description="The larger end of a range, else null.")
    unit: QuantityUnit = Field(description="The unit they meant, or the one trailer sense implies.")


class HaulClassification(StrictBaseModel):
    # Plain strings, not an enum: the trait list is admin-edited data (src/rules), and the
    # response schema has to stay fixed for the provider to cache it. Python drops any key
    # the live rules do not define.
    cargo_traits: list[str] = Field(description="CARGO TRAITS keys the named cargo has.")
    haul_item_matched: str | None = Field(description="The specific cargo, in their words, else null.")


class InventoryLookup(StrictBaseModel):
    is_lookup: bool = Field(description="True when they name one specific trailer.")
    year: int | None = Field(description="Model year.")
    make: str | None = Field(description="The make.")
    model_text: str | None = Field(description="Model code exactly as typed.")
    stock_number: str | None = Field(description="Stock/unit/# number only - never a weight, size, price, year or phone.")
    listing_url: str | None = Field(description="A trailerplace.com/inventory link they gave, copied exactly.")
    wants: Literal["price", "availability", "details", "general"] | None = Field(description="What they want to know.")
    confidence: Literal["low", "medium", "high"] = Field(description="How sure the lookup is.")


class TurnAnalysis(StrictBaseModel):
    # First on purpose: the model writes this digest before classifying, so the structured
    # fields are filled with the message freshly restated — and the respond prompt re-uses
    # it as a high-priority plain-text brief of what the customer wants this turn.
    turn_summary: str = Field(
        description=(
            "2-3 short plain-text sentences: what the customer just said and what they want "
            "this turn, with the specifics they gave (model names, sizes, weights, stock "
            "numbers, contact details, the question they asked). Facts from THIS message in "
            "context only - no advice, no recommendations, no routing decisions."
        )
    )
    intent: Literal[
        "general_question", "category_exploration", "category_selection",
        "feature_request_no_category", "recommendation_request",
        "qualification_answer", "skip_current", "skip_all_show_results", "show_more_results",
        "requirement_change", "drop_requirements", "category_change",
        "listing_interest", "faq", "team_request_escalation",
        "inventory_lookup", "contact_info_provided", "contact_declined", "smalltalk_other",
    ] = Field(description="Dominant user intent for routing.")
    email_triggers: list[EmailTrigger] = Field(description="Every email-worthy request in message order.")
    haul_classification: HaulClassification = Field(description="Which cargo traits the named cargo has.")
    inventory_lookup: InventoryLookup = Field(description="Specific-inventory lookup identifiers.")
    category_mentioned: str | None = Field(description="Canonical category mentioned or null.")
    is_category_info_only: bool = Field(description="True for category information questions, not selection.")
    extracted: ExtractedFields = Field(description="Structured extracted fields from the latest message.")
    slot_answers: list[SlotAnswer] = Field(description="List of slot answer pairs; never a dict.")
    contact: ContactInfo = Field(description="Contact information extracted from the message.")
    listing_reference: int | None = Field(description="1-based shown-listing reference, if any.")
    dropped_fields: list[str] = Field(description="Fields the user asked to drop.")
    keep_fields_answer: Literal["all", "none", "some"] | None = Field(description="Category-change keep/drop answer.")
    kept_fields: list[str] = Field(description="Specific fields the user wants to keep.")
    category_confirm_answer: Literal["yes", "no"] | None = Field(
        description="Only when a category-switch suggestion is pending: 'yes' to switch to the suggested category, 'no' to stay on the current one. Null otherwise."
    )
    answered_current_question: bool = Field(description="Whether latest message answered the pending question.")
    user_question_to_answer: str | None = Field(description="Interruption question to answer, verbatim.")


class ReplyOutput(StrictBaseModel):
    assistant_text: str = Field(description="Customer-facing assistant reply.")
    cited_listing_urls: list[str] = Field(description="Exact URLs for every listing mentioned in the reply.")

# ---------------------------------------------------------------------------------------
# The single-call turn output (one gpt-6-luna invocation per user message).
#
# Deliberately NOT a superset of TurnAnalysis: email_triggers is absent because email is out
# of scope for this build, and four reply-authoring fields are present because there is no
# second call to write the prose.
# ---------------------------------------------------------------------------------------


class ChatbotTurnOutput(StrictBaseModel):
    turn_summary: str = Field(
        description="2-3 plain sentences: what they said and want this turn, with their specifics. No advice."
    )
    intent: Literal[
        "general_question", "category_exploration", "category_selection",
        "feature_request_no_category", "recommendation_request",
        "qualification_answer", "skip_current", "skip_all_show_results", "show_more_results",
        "requirement_change", "drop_requirements", "category_change",
        "listing_interest", "faq", "team_request_escalation",
        "inventory_lookup", "contact_info_provided", "contact_declined", "smalltalk_other",
    ] = Field(description="Their main intent.")
    category_mentioned: str | None = Field(
        description="The category they named or implied, in their words; null for none or 'not sure'."
    )
    is_category_info_only: bool = Field(description="True when asking ABOUT a category, not choosing it.")
    unavailable_type_requested: str | None = Field(
        description="A trailer type they want that we do not stock, in their words, else null."
    )
    extracted: ExtractedFields = Field(description="Fields from this message.")
    slot_answers: list[SlotAnswer] = Field(description="Each slot this message answers, with their raw words.")
    contact: ContactInfo = Field(description="Contact details in this message.")
    inventory_lookup: InventoryLookup = Field(description="One specific trailer they named.")
    haul_classification: HaulClassification = Field(description="Traits of the cargo they named.")
    keep_fields_answer: Literal["all", "none", "some"] | None = Field(
        description="Only while a keep-filters question is pending."
    )
    kept_fields: list[str] = Field(description="Fields to keep when that answer is 'some'.")
    category_confirm_answer: Literal["yes", "no"] | None = Field(
        description="Only while a category-switch suggestion is pending: yes to switch, no to stay."
    )
    answered_current_question: bool = Field(
        description="True only when they answered the pending question itself."
    )
    user_question_to_answer: str | None = Field(description="Their question, verbatim, else null.")
    faq_key: Literal[
        "contact_human", "financing", "trade_in", "service_parts", "store_info"
    ] | None = Field(description="Which standard question they asked, even alongside something bigger.")
    listing_reference: int | None = Field(description="1-based number of a listing we showed.")
    shared_link_interest: bool = Field(
        description="They shared a link to a trailer (ours, Facebook, Instagram) and want it or ask about it."
    )
    off_topic: bool = Field(
        description="True only when the WHOLE message is nothing to do with us (see OFF TOPIC)."
    )
    dropped_fields: list[str] = Field(description="Fields they asked to drop.")

    # --- reply pieces: there is no second call, so the prose is authored here ---
    acknowledgement: str = Field(description="One short sentence, no question; empty when answering a question.")
    answer_to_customer_question: str | None = Field(
        description="Your answer, from the facts given only; null when they asked nothing."
    )
    next_question_slot: str | None = Field(description="A slot from 'still to ask'; Python has the final say.")
    next_question_text: str | None = Field(description="That question in one sentence.")


# What a reply did, reported by the model that wrote it. Python checks the report against the
# rules instead of reading the text: a pattern cannot read people, and every one tried here
# misread a good reply live ("no contact details needed" read as asking for them).
ReplyCover = Literal[
    "welcome",               # greeted them / thanked them for getting in touch
    "thanked_for_contacting",  # the words "Thank you for contacting TrailerPlace"
    "greeted_by_name",       # opens with "Hi <their name>"
    "declined_off_topic",    # said we only help with trailers, and did not do what they asked
    "flagged_wrong_value",   # told them a number they gave looks wrong
    "thanked_them",          # thanked them, or said a value was noted
    "said_not_stocked",      # said plainly we do not carry the type they asked for
    "passed_to_team",        # said their request has been passed to our team
    "gave_phone",            # gave our phone number
    "invited_questions",     # invited them to ask anything else they want to know
]


class ReplyFields(StrictBaseModel):
    """The reply and the model's own account of it."""

    reply: str = Field(description="The whole message to send them (see THE REPLY).")
    asked_slots: list[str] = Field(
        description="The 'still to ask' slot your reply asks, or empty. Never more than one."
    )
    asked_for_contact: bool = Field(description="True when the reply asks for their name, email or phone.")
    question_count: int = Field(description="Questions in the reply, NOT counting the contact request.")
    reply_covers: list[ReplyCover] = Field(description="Every item on the list that the reply does.")
    offered_categories: list[str] = Field(description="OUR CATEGORIES the reply suggests, else empty.")


class _MessageReading(StrictBaseModel):
    """Read from the message before the reply is written, so the reply can follow it."""

    about_trailers: bool = Field(
        description=(
            "True when this message says something about a trailer, a trailer need, cargo or our "
            "dealership. A bare greeting or small talk ('hi', 'hello', 'how are you') is false."
        )
    )
    only_acknowledges: bool = Field(
        description=(
            "True when the message ONLY acknowledges - 'ok', 'thanks', 'cool', 'got it', in any "
            "words - with no answer, no question, no request and no skip in it."
        )
    )


class ChatbotTurnReplyOutput(ReplyFields, _MessageReading, ChatbotTurnOutput):
    """The same output plus the whole reply, used when LLM_WRITES_REPLY is on.

    A subclass rather than more fields on the base, so with the flag off the model is asked
    for exactly what it was asked for before - no extra output tokens, no new schema. The
    bases are in this order so the fields come out analysis first, reply last.
    """


class ReplyRewrite(ReplyFields):
    """A second go at the reply, once Python has turned the first one down and said why."""
