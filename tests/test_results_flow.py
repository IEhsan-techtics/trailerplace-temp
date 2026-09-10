"""When results appear, and when they do not (brief S25).

Only two things open the gate: an explicit request, or every required question resolved.
"""
from __future__ import annotations

from src.conversation_store import load_session
from src.graph.build import run_turn
from src.graph.state import from_snapshot

from tests.factories import complete_welcome, turn_output


def state_after(session_id="s1"):
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


# --------------------------------------------------------- explicit request WITH a category
def test_show_me_with_a_category_searches_immediately(fake_llm, no_search):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")

    fake_llm.push(turn_output(intent="skip_all_show_results"))
    result = run_turn("s1", "just show me what you have")

    assert len(no_search) == 1
    assert result["listings"], "listings came back"
    assert "Here's what fits" in result["assistant_text"]


def test_the_search_runs_with_whatever_filters_exist(fake_llm, no_search):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")

    fake_llm.push(turn_output(intent="skip_all_show_results"))
    run_turn("s1", "show me")

    assert no_search[-1]["category"] == "Dump"
    assert no_search[-1]["slots"]["haul_item"] == "gravel"


# ------------------------------------------------------ explicit request WITHOUT a category
def test_show_me_with_no_category_sends_them_to_the_website_and_does_not_search(
    fake_llm, no_search
):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    result = run_turn("s1", "just show me everything you have")

    assert no_search == [], "no search without a category"
    assert result["listings"] == []
    assert "trailerplace.com" in result["assistant_text"]


# ------------------------------------------------------------- completion opens the gate
def test_answering_every_required_question_shows_results(fake_llm, no_search):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")

    assert no_search == [], "not complete yet"

    fake_llm.push(turn_output(slots={"payload_capacity": "3 tons"}))
    result = run_turn("s1", "about 3 tons")

    assert len(no_search) == 1
    assert result["qualification_complete"] is True


def test_declined_questions_still_count_as_resolved(fake_llm, no_search):
    """Giving up on a question must not strand the customer short of results."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")

    fake_llm.push(turn_output(intent="skip_current", answered_current_question=False))
    run_turn("s1", "rather not say")

    assert len(no_search) == 1, "the declined slot completed qualification"


# --------------------------------------------------------------- nothing else opens it
def test_choosing_a_category_alone_does_not_search(fake_llm, no_search):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "I need a dump trailer")
    assert no_search == []


def test_answering_a_question_mid_flow_does_not_search(fake_llm, no_search):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(slots={"haul_item": "gravel"}))
    run_turn("s1", "gravel")
    assert no_search == []


def test_a_general_question_does_not_search(fake_llm, no_search):
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(intent="general_question", user_question_to_answer="where are you?"))
    run_turn("s1", "where are you?")
    assert no_search == []


# ------------------------------------------------------------------- rendering the results
def test_listing_titles_are_unescaped_at_display_time(fake_llm, no_search):
    """22 of 259 rows in trailer_listings still carry raw HTML entities in `title`."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    result = run_turn("s1", "show me")

    assert "P&C" in result["assistant_text"]
    assert "P&amp;C" not in result["assistant_text"]


def test_shown_listings_are_recorded_so_they_are_not_offered_twice(fake_llm, no_search):
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    run_turn("s1", "show me")

    assert state_after()["shown_urls"] == ["https://x/1", "https://x/2"]


def test_no_question_is_appended_to_a_set_of_results(fake_llm, no_search):
    """The listings are the answer. Tacking a question on reads as ignoring them."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    result = run_turn("s1", "show me")

    assert "?" not in result["assistant_text"].split("Here's what fits")[-1]


# ------------------------------------------------------------------- the contact gate
def test_the_first_reply_greets_and_asks_for_contact(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other", acknowledgement="Hi there!"))
    result = run_turn("s1", "Hi, looking for a trailer")

    text = result["assistant_text"]
    assert "Thank you for contacting TrailerPlace" in text
    assert "your name" in text and "email address or phone number" in text


def test_no_results_are_shown_until_contact_has_been_settled(fake_llm, no_search):
    """The lead-generation rule: a visitor who gets listings and leaves is a lost lead, and
    there is no second first message in which to ask who they were."""
    fake_llm.push(
        turn_output(
            category_mentioned="livestock",
            intent="category_selection",
            slots={"length": "20ft", "haul_item": "livestock"},
        )
    )
    result = run_turn("s1", "Hello, looking for a 20ft trailer to haul my livestock")

    assert no_search == [], "no search ran"
    assert result["listings"] == []
    assert "your name" in result["assistant_text"]


def test_giving_only_a_name_is_followed_by_asking_for_the_number(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "Hi, looking for a trailer")

    fake_llm.push(turn_output(intent="contact_info_provided", name="Ibrahim"))
    result = run_turn("s1", "my name is Ibrahim")

    text = result["assistant_text"]
    assert "Thank you for sharing your name, Ibrahim" in text
    assert "email address or phone number" in text
    assert "optional" in text


def test_giving_only_a_number_is_followed_by_asking_for_the_name(fake_llm):
    fake_llm.push(turn_output(intent="contact_info_provided", phone="03304388550"))
    result = run_turn("s1", "my number is 03304388550")
    assert "your name" in result["assistant_text"]


def test_completing_the_details_opens_the_conversation(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "Hi")
    fake_llm.push(turn_output(intent="contact_info_provided", name="Ibrahim"))
    run_turn("s1", "my name is Ibrahim")

    fake_llm.push(turn_output(intent="contact_info_provided", phone="03304388550"))
    result = run_turn("s1", "my number is 03304388550")

    text = result["assistant_text"]
    assert "Great to have your contact info, Ibrahim!" in text
    assert "What type of trailer are you looking for?" in text
    assert "and many more" in text


def test_the_completion_greeting_is_said_once(fake_llm):
    fake_llm.push(turn_output(intent="contact_info_provided", name="Ibrahim", phone="0330"))
    run_turn("s1", "I'm Ibrahim, 0330")

    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    result = run_turn("s1", "a dump trailer")
    assert "great to hear from you" not in result["assistant_text"].lower()


def test_the_gate_stops_asking_after_two_fruitless_turns(fake_llm, no_search):
    """Twice is persistence, three times is pestering."""
    for message in ("Hi", "hmm"):
        fake_llm.push(turn_output(intent="smalltalk_other"))
        run_turn("s1", message)

    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    result = run_turn("s1", "a dump trailer")

    assert "your name" not in result["assistant_text"], "it stopped asking"
    assert state_after()["category"] == "Dump", "and got on with helping them"


def test_progress_earns_another_ask(fake_llm):
    """A customer who gives their name has engaged; asking once more is not nagging."""
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "Hi")
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hmm")
    fake_llm.push(turn_output(intent="contact_info_provided", name="Ibrahim"))
    result = run_turn("s1", "I'm Ibrahim")

    assert "email address or phone number" in result["assistant_text"]


# ----------------------------------------------------------------------- declining
def test_declining_up_front_is_respected_and_the_flow_continues(fake_llm):
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "Hi, looking for a livestock trailer")

    fake_llm.push(turn_output(intent="contact_declined", declined=True))
    run_turn("s1", "I'd rather not share that")

    fake_llm.push(turn_output(category_mentioned="livestock", intent="category_selection"))
    result = run_turn("s1", "livestock trailer")

    assert "your name" not in result["assistant_text"]
    assert state_after()["category"] == "Livestock"


def test_declining_the_second_half_is_respected(fake_llm):
    """They gave a name and will not give a number. That is an answer, not a gap to press."""
    fake_llm.push(turn_output(intent="contact_info_provided", name="Ibrahim"))
    run_turn("s1", "I'm Ibrahim")

    fake_llm.push(turn_output(intent="contact_declined", declined=True))
    result = run_turn("s1", "I'd rather not give my number")

    assert "email address or phone number" not in result["assistant_text"]
    assert state_after()["contact"]["declined"] is True


def test_a_declined_session_can_still_reach_results(fake_llm, no_search):
    fake_llm.push(turn_output(intent="contact_declined", declined=True))
    run_turn("s1", "no thanks")
    fake_llm.push(turn_output(category_mentioned="dump", intent="category_selection"))
    run_turn("s1", "dump trailer")
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    result = run_turn("s1", "just show me")

    assert result["listings"], "declining does not cost them the results"


def test_a_question_in_the_first_message_is_answered_alongside_the_greeting(fake_llm):
    """The gate holds the SEARCH, not the conversation. A customer who opens with a
    question and is answered with a form is being processed, not helped."""
    fake_llm.push(
        turn_output(
            intent="faq",
            user_question_to_answer="what are your hours?",
            answer_to_customer_question=(
                "We don't list our hours here - give the team a call on 979-532-1486 "
                "and they'll confirm."
            ),
        )
    )
    result = run_turn("s1", "hi, what are your hours?")

    text = result["assistant_text"]
    assert "Thank you for contacting TrailerPlace" in text
    assert "979-532-1486" in text, "their question was answered"
    assert "your name" in text, "and the details were still requested"
    assert text.index("979-532-1486") < text.index("your name"), "answer comes first"


def test_the_models_greeting_is_used_when_it_asks_for_the_details(fake_llm):
    fake_llm.push(
        turn_output(
            intent="smalltalk_other",
            acknowledgement=(
                "Thank you for contacting TrailerPlace. I see you're looking for a trailer, "
                "and I'm here to help!"
            ),
            next_question_text=(
                "Could you please provide your name and either your email or phone number?"
            ),
        )
    )
    result = run_turn("s1", "hi")

    text = result["assistant_text"]
    assert "I see you're looking for a trailer" in text
    assert text.count("Could you please provide your name") == 1, "no template appended on top"


def test_the_written_greeting_takes_over_when_the_model_forgets_to_ask(fake_llm):
    """The one reply the conversation cannot afford to get wrong, so it is checked."""
    fake_llm.push(turn_output(intent="smalltalk_other", acknowledgement="Hey!"))
    result = run_turn("s1", "hi")

    text = result["assistant_text"]
    assert "Thank you for contacting TrailerPlace" in text
    assert "your name" in text


def test_the_models_welcome_is_used_when_it_greets_them(fake_llm, no_search):
    fake_llm.push(
        turn_output(
            intent="category_selection",
            category_mentioned="livestock",
            slots={"length": "20ft"},
            name="Ibrahim",
            phone="03304388550",
            acknowledgement=(
                "Thank you for contacting TrailerPlace, Ibrahim! A 20 ft livestock trailer "
                "it is."
            ),
        )
    )
    result = run_turn("s1", "Ibrahim here, 03304388550. Looking for a 20ft livestock")

    text = result["assistant_text"]
    assert "Thank you for contacting TrailerPlace, Ibrahim!" in text
    assert "A 20 ft livestock trailer it is." in text
    assert result["listings"], "and they still got their results"


def test_the_written_welcome_takes_over_when_the_model_does_not_greet(fake_llm, no_search):
    """The opening is the one line every conversation is guaranteed to contain."""
    fake_llm.push(
        turn_output(
            intent="category_selection",
            category_mentioned="livestock",
            slots={"length": "20ft"},
            name="Ibrahim",
            phone="03304388550",
            acknowledgement="Got it, noted.",
        )
    )
    result = run_turn("s1", "Ibrahim here, 03304388550. Looking for a 20ft livestock")
    assert "Thank you for contacting TrailerPlace, Ibrahim!" in result["assistant_text"]


def test_a_later_completion_keeps_both_the_thanks_and_the_acknowledgement(fake_llm):
    """Mid-conversation there is no opening to guarantee, so nothing is discarded."""
    fake_llm.push(turn_output(intent="smalltalk_other"))
    run_turn("s1", "hi")

    fake_llm.push(
        turn_output(intent="contact_info_provided", name="Ibrahim", phone="0330",
                    acknowledgement="Noted your number.")
    )
    result = run_turn("s1", "Ibrahim, 0330")

    text = result["assistant_text"]
    assert "Great to have your contact info, Ibrahim!" in text
    assert "Noted your number." in text


def test_a_friendly_but_non_standard_welcome_does_not_replace_the_opening(fake_llm):
    """"Welcome, Ibrahim - we're glad to help" is warm but is not the opening the
    dealership asked for, and turn one is the only chance to say it."""
    fake_llm.push(
        turn_output(
            intent="contact_info_provided", name="Ibrahim", email="i@x.ai",
            acknowledgement="Welcome, Ibrahim - we're glad to help with your trailer search.",
        )
    )
    result = run_turn("s1", "hello, my name is Ibrahim and email is i@x.ai")
    assert "Thank you for contacting TrailerPlace, Ibrahim!" in result["assistant_text"]


def test_asking_what_types_exist_is_answered_not_redirected(fake_llm, no_search):
    """A customer asking what you sell is exploring, not asking to be sent to the website.

    The model reads it as a results request, so the reply used to carry the answer, the
    whole category list and the website line - three ways of saying the same thing.
    """
    complete_welcome(fake_llm)
    fake_llm.push(
        turn_output(
            intent="skip_all_show_results",
            user_question_to_answer="what trailer types do you have?",
            answer_to_customer_question=(
                "We carry Dump, Utility, Equipment and Enclosed trailers among others."
            ),
        )
    )
    result = run_turn("s1", "show me all of the trailer types that you have")

    text = result["assistant_text"]
    assert "We carry Dump, Utility" in text, "their question was answered"
    assert "browse our full inventory" not in text, "not sent away as well"
    assert text.count("https://www.trailerplace.com") == 0


def test_show_me_trailers_with_no_category_still_goes_to_the_website(fake_llm, no_search):
    """The rule the fix above must not break (S25)."""
    complete_welcome(fake_llm)
    fake_llm.push(turn_output(intent="skip_all_show_results"))
    result = run_turn("s1", "just show me everything you have")

    assert no_search == []
    assert "trailerplace.com" in result["assistant_text"]


def test_the_models_type_question_is_used(fake_llm):
    """The model writes it - it can pick types that fit what they already said."""
    fake_llm.push(
        turn_output(
            intent="contact_info_provided", name="Ibrahim", phone="0330",
            next_question_text=(
                "What type of trailer are you looking for? We have Utility, Dump, "
                "Equipment and many more - which one fits?"
            ),
        )
    )
    result = run_turn("s1", "Ibrahim, 0330")
    assert "which one fits?" in result["assistant_text"]


def test_the_written_question_stands_in_when_the_model_asks_nothing(fake_llm):
    fake_llm.push(turn_output(intent="contact_info_provided", name="Ibrahim", phone="0330"))
    result = run_turn("s1", "Ibrahim, 0330")

    text = result["assistant_text"]
    assert "What type of trailer are you looking for?" in text
    assert "and many more" in text


def test_a_question_reciting_the_whole_catalogue_is_replaced(fake_llm):
    """A guard, not a policy: the prompt asks for four to six and the model still read out
    all thirteen. It keeps its own wording whenever it stays within the limit."""
    from src.domain import categories

    fake_llm.push(
        turn_output(
            intent="contact_info_provided", name="Ibrahim", phone="0330",
            next_question_text="We carry " + ", ".join(categories.CANONICAL_CATEGORIES) + ".",
        )
    )
    result = run_turn("s1", "Ibrahim, 0330")

    named = [c for c in categories.CANONICAL_CATEGORIES if c in result["assistant_text"]]
    assert len(named) <= 6, f"named {len(named)}: {named}"


def test_the_menu_leads_with_the_categories_that_have_the_most_choice():
    """Alphabetical order led with Aluminum, one of the thinnest ranges we carry."""
    from src.graph.nodes.greeting import _menu_sample

    sample = _menu_sample()
    # Under the test catalogue Equipment is stocked by the most makes, so it leads. The
    # point is that BREADTH orders the list, not the alphabet.
    assert sample[0] == "Equipment", sample
    assert sample.index("Equipment") < sample.index("Enclosed"), sample
