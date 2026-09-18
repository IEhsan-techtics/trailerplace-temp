from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContactInfo(StrictBaseModel):
    name: str | None = Field(description="Customer name if provided, otherwise null.")
    email: str | None = Field(description="Customer email if provided, otherwise null.")
    phone: str | None = Field(description="Customer phone if provided, otherwise null.")
    declined: bool = Field(
        default=False,
        description=(
            "True when THIS message refuses to share contact details ('I'd rather not', 'no "
            "contact info') - even when the message also does something bigger and the intent "
            "is not contact_declined. False otherwise."
        ),
    )


class SlotAnswer(StrictBaseModel):
    slot_name: str = Field(description="One current-category slot name exactly as listed in the prompt.")
    raw_answer: str = Field(description="The user's raw answer for that slot.")


class EmailTrigger(StrictBaseModel):
    kind: Literal["faq", "escalation", "team_request", "listing_interest"] = Field(description="Email-worthy request kind.")
    faq_key: Literal["contact_human", "financing", "trade_in", "service_parts", "store_info"] | None = Field(description="Required for FAQ triggers, otherwise null.")
    listing_reference: int | None = Field(description="1-based shown-listing index for listing interest, otherwise null.")
    description: str = Field(description="One-line summary for the email body.")


class ExtractedFields(StrictBaseModel):
    length: float | None = Field(description="Trailer length in feet, or null. For a RANGE give the SMALLEST value - never the midpoint, average or largest.")
    width: float | None = Field(description="Trailer width in feet, or null. For a RANGE give the SMALLEST value - never the midpoint, average or largest.")
    height: float | None = Field(description="Trailer height in feet, or null. For a RANGE give the SMALLEST value - never the midpoint, average or largest.")
    payload_capacity: float | None = Field(description="Payload in pounds, or null. For a RANGE give the SMALLEST value - never the midpoint, average or largest.")
    axle_capacity: float | None = Field(description="PER-AXLE capacity rating in pounds ('7000 lb axles'), never the load weight, or null.")
    total_axle_capacity_lbs: float | None = Field(description="Capacity across ALL axles together in pounds ('14,000 lbs total across the axles'), never the load weight, or null.")
    axle_count: int | None = Field(description="How many axles the customer wants ('tandem'=2, 'single'=1, 'tri'=3), or null. Never a capacity.")
    axle_capacity_basis: Literal["per_axle", "total", "unclear"] | None = Field(
        description=(
            "Which of the two capacities the customer meant. 'per_axle' when the wording ties "
            "the number to each axle ('7,000 lb axles'); 'total' when it ties it to all of them "
            "together ('14,000 total', 'combined'); 'unclear' when a bare 'axle capacity' number "
            "could honestly be either. Null when no capacity was stated."
        )
    )
    hitch_type: list[Literal["Bumper Pull", "Gooseneck"]] | None = Field(description="A single clear hitch preference, or null (including when the customer says either/any/no preference).")
    haul_item: str | None = Field(description="Cargo or item to haul, in the user's words.")
    brand_preference: str | None = Field(description="Canonical known make, or unknown brand verbatim.")
    non_metadata_features: list[str] = Field(
        description=(
            "Only newly stated functional/equipment features with no dedicated metadata field; "
            "exclude makes, categories/subcategories, trailer/model words, hitches, dimensions, "
            "weights, axle capacities/ratings, colours, and prices."
        )
    )
    numeric_no_preference: list[str] = Field(description="Slot names where the user gave no numeric preference.")
    raw_numeric_spans: list["SlotAnswer"] = Field(
        description=(
            "For EVERY numeric field you filled above, the customer's verbatim wording for "
            "it: slot_name is the field name ('length', 'payload_capacity', ...) and "
            "raw_answer is the exact substring they typed ('18-20 ft', 'about 3 tons', "
            "'-500 lbs'). Copy their text character for character - do not tidy or convert "
            "it. Python re-parses this and it OVERRIDES your converted number."
        )
    )


class HaulClassification(StrictBaseModel):
    # Plain strings, not an enum: the trait list is admin-edited data (src/rules), and the
    # response schema has to stay fixed for the provider to cache it. Python drops any key
    # the live rules do not define.
    cargo_traits: list[str] = Field(description="Keys from CARGO TRAITS that the named cargo has. Empty when no specific cargo was named.")
    haul_item_matched: str | None = Field(description="Specific cargo grounded in user words, or null.")


class InventoryLookup(StrictBaseModel):
    is_lookup: bool = Field(description="True when the message references specific inventory identifiers.")
    year: int | None = Field(description="Model year mentioned, never a stock number.")
    make: str | None = Field(description="Canonical make if mentioned for lookup.")
    model_text: str | None = Field(description="Model code or phrase exactly as the user typed it.")
    stock_number: str | None = Field(description="Explicit stock/unit/id number only, never weight/length/price/year/phone.")
    wants: Literal["price", "availability", "details", "general"] | None = Field(description="What the user wants about the inventory item.")
    confidence: Literal["low", "medium", "high"] = Field(description="Lookup confidence; low must fail closed.")


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
# The single-call turn output (one gpt-5.6-luna invocation per user message).
#
# Deliberately NOT a superset of TurnAnalysis: email_triggers is absent because email is out
# of scope for this build, and four reply-authoring fields are present because there is no
# second call to write the prose.
# ---------------------------------------------------------------------------------------


class ChatbotTurnOutput(StrictBaseModel):
    turn_summary: str = Field(
        description=(
            "2-3 short plain sentences: what the customer just said and what they want this "
            "turn, with the specifics they gave. Facts from THIS message only - no advice, "
            "no recommendations, no routing decisions."
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
    category_mentioned: str | None = Field(
        description=(
            "The trailer category the customer named or implied, in THEIR words - the Python "
            "side canonicalizes it. Null when they named none, or said 'not sure' / 'any' / "
            "'I don't know'. Never guess a category to fill this in."
        )
    )
    is_category_info_only: bool = Field(
        description="True when they are ASKING ABOUT a category rather than choosing it."
    )
    extracted: ExtractedFields = Field(description="Structured fields from the latest message.")
    slot_answers: list[SlotAnswer] = Field(
        description=(
            "One entry per qualification slot the message answers, carrying the customer's "
            "RAW wording. Always fill this when they answer a question - the raw text is what "
            "the deterministic parsers read."
        )
    )
    contact: ContactInfo = Field(description="Contact details stated in this message.")
    inventory_lookup: InventoryLookup = Field(description="Specific-inventory lookup identifiers.")
    haul_classification: HaulClassification = Field(description="Which cargo traits the named cargo has.")
    keep_fields_answer: Literal["all", "none", "some"] | None = Field(
        description="Only when a keep-filters question is pending; null otherwise."
    )
    kept_fields: list[str] = Field(description="Specific fields to keep when the answer is 'some'.")
    category_confirm_answer: Literal["yes", "no"] | None = Field(
        description=(
            "Only when a category-switch suggestion is pending: 'yes' to switch to the "
            "suggested category, 'no' to stay on the current one. Null otherwise."
        )
    )
    answered_current_question: bool = Field(
        description=(
            "True only when the message actually answers the pending qualification question. "
            "A counter-question, a non-sequitur, or an answer to a DIFFERENT field is False."
        )
    )
    user_question_to_answer: str | None = Field(
        description="A question of theirs that needs answering this turn, verbatim; else null."
    )
    faq_key: Literal[
        "contact_human", "financing", "trade_in", "service_parts", "store_info"
    ] | None = Field(
        description=(
            "Which of the five standard questions they asked, when they asked one: wanting a "
            "person (contact_human), financing, a trade-in, service or parts, or where we are "
            "and when we are open (store_info). Null when the message asks none of them. Set "
            "it whenever they ask, even alongside something bigger - the team is told about "
            "every one of these."
        )
    )
    listing_reference: int | None = Field(description="1-based shown-listing reference, if any.")
    dropped_fields: list[str] = Field(description="Fields the customer asked to drop.")

    # --- reply pieces: there is no second call, so the prose is authored here ---
    acknowledgement: str = Field(
        description=(
            "One short, natural sentence acknowledging what they just said. No question in "
            "it - the question is a separate field. Empty string when nothing needs "
            "acknowledging."
        )
    )
    answer_to_customer_question: str | None = Field(
        description=(
            "Answer to user_question_to_answer, grounded ONLY in the dealership facts, "
            "categories and brands given in the system prompt. Null when they asked nothing. "
            "Never invent stock, prices, brands or categories."
        )
    )
    next_question_slot: str | None = Field(
        description=(
            "The slot you think should be asked next, from the remaining-required list in "
            "the state block. Python has the final say and will override you."
        )
    )
    next_question_text: str | None = Field(
        description=(
            "Your natural phrasing of the question for next_question_slot. Used only if it "
            "matches the slot Python independently selects."
        )
    )
