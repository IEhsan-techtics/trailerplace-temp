"""Changing category, keeping or dropping filters, and the haul-item suggestion."""
from __future__ import annotations

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


def start_with_filters(fake_llm):
    """A Dump session carrying two real answers."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel", "payload_capacity": "6000 lbs"}))
    run_turn("s1", "gravel, 6000 lbs")


# ------------------------------------------------------------- the keep question (S12)
def test_changing_category_asks_whether_to_keep_collected_filters(fake_llm):
    start_with_filters(fake_llm)

    fake_llm.push(turn_output(category_mentioned="equipment trailer", intent="category_change"))
    result = run_turn("s1", "actually make it an equipment trailer")

    state = state_after()
    assert state["pending_keep_filters"] is not None
    assert state["category"] == "Dump", "not switched until they answer"
    assert "keep those" in result["assistant_text"]


def test_only_filters_with_real_values_are_mentioned(fake_llm):
    """Brief S12: offering to keep a blank is noise."""
    start_with_filters(fake_llm)

    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    result = run_turn("s1", "equipment trailer instead")

    text = result["assistant_text"]
    assert "6,000" in text
    assert "width" not in text and "None" not in text


def test_the_keep_question_never_offers_to_keep_the_haul_item(fake_llm):
    """What they are hauling is what DEFINES the category. Offering to keep it invites the
    customer to preserve the very thing that just changed, and it is a required question
    for the new category anyway."""
    start_with_filters(fake_llm)

    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    result = run_turn("s1", "equipment trailer instead")

    assert "gravel" not in result["assistant_text"]
    assert "haul item" not in result["assistant_text"].lower()


def test_the_old_haul_item_is_dropped_and_asked_again(fake_llm):
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="all"))
    run_turn("s1", "keep them")

    state = state_after()
    assert "haul_item" not in state["slots"], "gravel is not an answer for an Equipment trailer"
    assert "haul_item" in state["required_slots"]


def test_a_haul_item_restated_in_the_same_message_survives(fake_llm):
    """The usual way a category change arrives: "actually I need to move a tractor"."""
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="all", slots={"haul_item": "a tractor"}))
    run_turn("s1", "keep them, I'm moving a tractor")

    assert state_after()["slots"]["haul_item"] == "a tractor"


def test_keeping_everything_carries_the_filters_over(fake_llm):
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="all"))
    run_turn("s1", "yes keep them")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["slots"]["payload_capacity"] == 6000.0


def test_starting_fresh_drops_them(fake_llm):
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="none"))
    run_turn("s1", "no, start over")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["slots"] == {}


def test_keeping_a_subset_keeps_only_what_they_named(fake_llm):
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "equipment trailer instead")

    fake_llm.push(turn_output(keep_fields_answer="some", kept_fields=["payload_capacity"]))
    run_turn("s1", "just the weight")

    state = state_after()
    assert state["slots"] == {"payload_capacity": 6000.0}


def test_changing_with_nothing_collected_switches_without_asking(fake_llm):
    """There is nothing to keep, so the question would be noise."""
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "actually equipment")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["pending_keep_filters"] is None


# ------------------------------------------------- the haul-item suggestion (your addition)
def test_a_haul_item_suiting_another_category_offers_a_switch(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="utility", intent="category_selection"))
    run_turn("s1", "utility trailer")

    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    result = run_turn("s1", "a skid steer")

    state = state_after()
    assert state["pending_category_switch"]["suggested"] == "Equipment"
    assert "Equipment" in result["assistant_text"]
    assert "switch" in result["assistant_text"].lower()


def test_saying_yes_switches_without_asking_about_filters(fake_llm):
    """Your instruction: on a suggested switch, do NOT ask the keep-filters question."""
    fake_llm.push(turn_output(category_mentioned="utility", intent="category_selection"))
    run_turn("s1", "utility trailer")
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    run_turn("s1", "a skid steer")

    fake_llm.push(turn_output(category_confirm_answer="yes"))
    result = run_turn("s1", "yes please")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["pending_keep_filters"] is None, "no keep question on a suggested switch"
    assert state["slots"]["haul_item"] == "a skid steer", "their answer carried over"
    assert "keep those" not in result["assistant_text"]


def test_saying_no_stays_put_and_never_suggests_it_again(fake_llm):
    fake_llm.push(turn_output(category_mentioned="utility", intent="category_selection"))
    run_turn("s1", "utility trailer")
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    run_turn("s1", "a skid steer")

    fake_llm.push(turn_output(category_confirm_answer="no"))
    run_turn("s1", "no thanks")

    state = state_after()
    assert state["category"] == "Utility"
    assert state["pending_category_switch"] is None
    assert "Utility->Equipment" in state["rejected_switches"]

    # Saying the same thing again must not re-open it.
    fake_llm.push(turn_output(slots={"haul_item": "a skid steer"}))
    run_turn("s1", "still a skid steer")
    assert state_after()["pending_category_switch"] is None


# --------------------------------------------- axles and features in the keep question
def start_with_axles_and_features(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(
        slots={"haul_item": "gravel", "total_axle_capacity_lbs": "14,000 lbs total",
               "axle_count": "tandem"},
        extracted={"non_metadata_features": ["ramps"]},
    ))
    run_turn("s1", "gravel, 14,000 lbs total on a tandem, with ramps")


def test_axles_and_features_are_named_in_plain_words(fake_llm):
    start_with_axles_and_features(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    text = run_turn("s1", "actually an equipment trailer")["assistant_text"]

    assert "14,000 lbs of total axle capacity" in text
    assert "2 axles" in text
    assert "ramps" in text
    assert "_" not in text, "never a field name"


def test_keeping_some_can_drop_the_features(fake_llm):
    start_with_axles_and_features(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "actually an equipment trailer")

    fake_llm.push(turn_output(keep_fields_answer="some", kept_fields=["axle_count"]))
    run_turn("s1", "just the axles")

    state = state_after()
    assert state["slots"].get("axle_count") == 2
    assert "total_axle_capacity_lbs" not in state["slots"]
    assert state["non_metadata_features"] == []


def test_keeping_all_keeps_the_features(fake_llm):
    start_with_axles_and_features(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change"))
    run_turn("s1", "actually an equipment trailer")

    fake_llm.push(turn_output(keep_fields_answer="all"))
    run_turn("s1", "keep it all")

    state = state_after()
    assert state["category"] == "Equipment"
    assert state["non_metadata_features"] == ["ramps"]
    assert state["slots"]["total_axle_capacity_lbs"] == 14000.0


def test_the_models_own_line_does_not_claim_the_switch_before_the_keep_question(fake_llm):
    """Live: "we'll switch your search to a 16-foot Flatbed with tandem 7,000 lb axles.
    ... should I keep those, or start fresh?" - kept before it was asked."""
    start_with_filters(fake_llm)
    fake_llm.push(turn_output(category_mentioned="equipment", intent="category_change",
                              acknowledgement="Got it - switching you to Equipment with a 6,000 lb load."))
    text = run_turn("s1", "actually an equipment trailer")["assistant_text"]

    assert text.startswith("Switching to Equipment.")


# ------------------------------------------------- the whole live turn that started this
# Turn 22 of the long-conversation run, on a Dump trailer:
#   USER: gravel and dirt for a landscaping job
#   LUNA: That sounds like a good fit for a Dump trailer. For gravel and dirt for a
#         landscaping job, an Utility trailer is usually the better fit - would you like to
#         switch to that instead?
# Three faults in one reply: the wrong category, two sentences contradicting each other, and
# "an Utility". The model had read it correctly - extracted haul_item was "gravel and dirt".
def test_dump_cargo_described_at_length_does_not_offer_a_switch(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(
        slots={"haul_item": "gravel and dirt for a landscaping job"},
        extracted={"haul_item": "gravel and dirt"},
        acknowledgement="That sounds like a good fit for a Dump trailer.",
    ))
    result = run_turn("s1", "gravel and dirt for a landscaping job")

    state = state_after()
    assert state["pending_category_switch"] is None, "gravel and dirt IS dump cargo"
    assert state["category"] == "Dump"
    assert "Utility" not in result["assistant_text"]
    assert "switch" not in result["assistant_text"].lower()


def test_the_clean_cargo_is_stored_not_the_whole_sentence(fake_llm):
    """The sentence also became the text the search matched on."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(
        slots={"haul_item": "gravel and dirt for a landscaping job"},
        extracted={"haul_item": "gravel and dirt"},
    ))
    run_turn("s1", "gravel and dirt for a landscaping job")

    assert state_after()["slots"]["haul_item"] == "gravel and dirt"


def test_their_wording_is_still_kept_when_the_model_reads_nothing(fake_llm):
    """The model's reading wins only when there IS one - never a slot left empty."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(slots={"haul_item": "whatever the yard sends over"}))
    run_turn("s1", "whatever the yard sends over")

    assert state_after()["slots"]["haul_item"] == "whatever the yard sends over"


def test_the_models_own_line_does_not_contradict_the_switch_question(fake_llm):
    """It writes its line knowing only the category they are on, so it praised the Dump
    trailer in the same breath as asking whether to leave it."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(
        slots={"haul_item": "cattle"},
        acknowledgement="That sounds like a good fit for a Dump trailer.",
    ))
    text = run_turn("s1", "cattle")["assistant_text"]

    assert state_after()["pending_category_switch"]["suggested"] == "Livestock"
    assert "good fit for a Dump" not in text, "it argued with its own question"
    assert text.startswith("For cattle,")


def test_the_switch_question_says_a_utility_not_an_utility(fake_llm):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump trailer", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(slots={"haul_item": "a riding lawnmower"}))
    text = run_turn("s1", "a riding lawnmower")["assistant_text"]

    assert "a Utility trailer" in text
    assert "an Utility" not in text


# -------------------------------------------------- one question, even without a slot
def test_the_category_question_is_not_asked_twice_in_one_reply(fake_llm):
    """Live, turn 2 - the same question, twice, in one breath:

    What type of trailer fits what you need? What type of trailer are you looking for?
    We have Utility, Enclosed, Equipment, Tilt, Livestock, Flatbed and many more - which
    one fits what you need?
    """
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(
        intent="category_exploration",
        answer_to_customer_question=(
            "We carry Utility, Enclosed, Equipment, Dump, Flatbed and many more. Utility "
            "trailers are open general-purpose haulers, Enclosed trailers are lockable and "
            "weatherproof. What type of trailer fits what you need?"
        ),
    ))
    text = run_turn("s1", "what sort of trailers do you carry?")["assistant_text"]

    assert text.count("?") == 1, f"asked more than once: {text}"
    assert "What type of trailer fits what you need?" in text, "the model's wording is kept"


def test_the_category_question_still_goes_out_when_the_model_asks_nothing(fake_llm):
    """The guard drops a duplicate, never the only question in the reply."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(
        intent="category_exploration",
        answer_to_customer_question="We carry a wide range of trailers.",
    ))
    text = run_turn("s1", "what sort of trailers do you carry?")["assistant_text"]

    # The canned menu is one question written as two sentences ("What type of trailer are you
    # looking for? We have ... - which one fits what you need?"), so it is counted by asking,
    # not by question marks.
    assert text.lower().count("type of trailer") == 1
    assert "which one fits what you need?" in text
