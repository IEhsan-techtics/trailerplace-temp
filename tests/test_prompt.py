"""The prompt: short, grounded in real data, and honest about what is already known."""
from __future__ import annotations

from src.graph.nodes import greeting
from src.graph.state import new_state
from src.llm.prompt import build_messages, state_block, system_prompt
from src.tools.category import set_trailer_category


def qualified_state(**kwargs):
    state = new_state("s1")
    set_trailer_category(state, "Dump")
    state.update(kwargs)
    return state


# --------------------------------------------------------------------------- the static half
# Raised from 12,000 when the inventory-lookup identifier rules were added: what a stock
# number is and is not cannot be stated accurately in fewer words, and getting it wrong sends
# a load weight to the matcher as a stock number.
#
# The ceiling exists to catch prompt growth nobody decided on, not to forbid growth. It is
# cheaper than it looks - the block is identical on every turn, so it is lru_cached in-process
# AND hits the provider's prompt cache - but it is still the per-turn floor, so moving it
# should stay a deliberate act with a reason written down.
#
# MEASURED UNDER THE TEST FIXTURE, which stubs the catalogue down to 6 makes. Production
# reads 20 out of trailer_listings, and the brand block grows with it - the same prompt is
# ~2,200 chars longer live than it is here. The headroom below absorbs that, and it is why
# this number is not simply "current size, rounded up".
#
# Raised from 17,000 for the one-question rule. That rule took the LIVE prompt to 17,034 -
# over the old ceiling - while this fixture measured well under it, which is exactly the gap
# the paragraph above warns about.
#
# Raised from 19,000 for the CARGO TRAITS section (~1,100 chars), which the question rules
# render from their trait definitions. It replaces a model judgement that used to be spread
# through the prompt as category-specific rules, and it grows with every trait an admin
# adds - which is exactly the growth this ceiling should catch.
#
# Lowered from 20,500 by the prompt rewrite: every rule stated once as situation -> action,
# the scripted answers shared with the reply prompt, and the category term lists dropped
# (Python maps the customer's words itself). Live it measured 12,630.
#
# Raised from 13,500 for the non_metadata_features rule (~580 chars). Without it the model
# filed cargo and uses as features ("scissor lift", "mobile coffee business"), and each one
# sent every candidate listing to the gpt-5-nano reranker for a ranking nothing could match.
#
# Raised from 14,000 for the AXLES rule (~750 chars; live 14,303). Without it the model called
# a bare "14,000 lbs of axle capacity" TOTAL in the same reply that asked per axle or total -
# half of the clarification questions in the live axles run contradicted themselves.
MAX_SYSTEM_PROMPT_CHARS = 15_400


def test_the_system_prompt_stays_short():
    """It is sent on every turn. A live probe showed input tokens dominating the cost, so
    length here is a running bill, not a style preference.

    NOTE: this measures the prompt built from the SIX fixture makes in conftest, not the
    real catalogue. Against the live database the same prompt is about 250 characters
    longer - measured at 15775 on 2026-09-22 - so this ceiling is a floor on the real
    one, not the real one. Use ``python -c "from src.llm.prompt import system_prompt;
    print(len(system_prompt()))"`` against a live .env to see what is actually sent.
    """
    prompt = system_prompt()
    assert len(prompt) < MAX_SYSTEM_PROMPT_CHARS, f"system prompt is {len(prompt)} chars"


def test_the_system_prompt_is_cached_so_the_catalogue_is_not_re_read_per_turn():
    assert system_prompt() is system_prompt()


def test_it_names_the_real_dealership():
    prompt = system_prompt()
    assert "TrailerPlace" in prompt
    assert "Wharton, TX" in prompt
    assert "979-532-1486" in prompt


def test_it_carries_the_category_menu_with_use_cases():
    """Brief S5: categories are presented with what they are for."""
    prompt = system_prompt()
    assert "hydraulic beds for gravel" in prompt


def test_it_states_the_rules_python_also_enforces():
    prompt = system_prompt()
    assert "SMALLEST" in prompt          # S15
    # S17: units are converted in Python (src/domain/quantities.py) now; the model is told
    # to report the unit it heard rather than do the arithmetic itself.
    assert "Do not convert units" in prompt
    assert "Bumper Pull" in prompt       # S21
    assert "one axle" in prompt.lower()  # per-axle semantics


def test_it_never_advertises_a_category_we_do_not_stock(monkeypatch):
    from src.domain import categories as categories_module

    monkeypatch.setattr(categories_module, "_advertised_categories", lambda: ("Dump", "Utility"))
    system_prompt.cache_clear()
    try:
        prompt = system_prompt()
        assert "- Dump:" in prompt
        assert "- Concession:" not in prompt
    finally:
        system_prompt.cache_clear()


# ------------------------------------------------------------------------ the dynamic half
def test_a_fresh_session_says_no_category_and_asks_for_contact():
    block = state_block(new_state("s1"))
    assert "not chosen yet" in block
    assert "Contact details not asked for yet" in block


def test_known_values_are_listed_with_a_do_not_ask_again_instruction():
    state = qualified_state()
    state["slots"] = {"length": 20.0, "haul_item": "gravel"}
    block = state_block(state)
    assert "NEVER ask about these again" in block
    assert "length = 20" in block          # not "20.0" - it is prose
    assert "haul_item = gravel" in block


def test_remaining_questions_are_listed():
    state = qualified_state()
    state["slots"] = {"haul_item": "gravel"}
    block = state_block(state)
    assert "Still to ask:" in block
    assert "haul_item" not in block.split("Still to ask:")[1].splitlines()[0]


def test_declined_slots_are_marked_do_not_raise_again():
    state = qualified_state(declined_slots=["payload_capacity"])
    assert "Do not raise them again: payload_capacity" in state_block(state)


def test_an_invalid_value_is_explained_in_plain_words():
    state = qualified_state(invalid_retry_slot="payload_capacity", invalid_retry_reason="negative")
    block = state_block(state)
    assert "negative number" in block

    state = qualified_state(invalid_retry_slot="axle_count", invalid_retry_reason="axle_range")
    assert "1 to 4 axles" in state_block(state)


def test_a_pending_category_switch_is_explained():
    state = qualified_state(
        pending_category_switch={"suggested": "Equipment", "from_haul_item": "a tractor"}
    )
    block = state_block(state)
    assert "Equipment" in block and "a tractor" in block
    assert "category_confirm_answer" in block


def test_a_pending_keep_filters_question_is_explained():
    state = qualified_state(pending_keep_filters={"new_category": "Equipment"})
    assert "keep_fields_answer" in state_block(state)


def test_a_declined_contact_is_never_asked_for_again():
    state = new_state("s1")
    state["contact"]["declined"] = True
    assert "Never ask again" in state_block(state)


def test_contact_already_on_file_is_not_asked_for_again():
    state = new_state("s1")
    state["contact"].update({"name": "Dave", "email": "d@x.com"})
    block = state_block(state)
    assert "Do not ask again" in block
    assert "name, email" in block


def test_the_model_is_told_to_ask_again_when_python_still_wants_it_asked():
    """The two used to disagree. Told "never ask again" while the gate was still open, the
    model obediently did not - and compose stitched its own template onto the answer."""
    state = new_state("s1")
    state["turn_index"] = 3
    state["contact"].update({"asked": True, "last_asked_turn": 1})

    block = state_block(state)
    assert "Ask again" in block and "LAST line" in block


def test_a_spent_gate_tells_the_model_to_drop_it():
    """Two fruitless asks is the end of it, whatever they do."""
    state = new_state("s1")
    state["turn_index"] = 5
    state["contact"].update({"asked": True, "asks_without_progress": 2, "last_asked_turn": 3})

    assert "Never ask again" in state_block(state)


def test_the_model_is_not_told_to_ask_the_turn_after_it_just_did():
    state = new_state("s1")
    state["turn_index"] = 2
    state["contact"].update({"asked": True, "last_asked_turn": 1})

    assert "Never ask again" in state_block(state)


# ------------------------------------------------------------------------------- assembly
def test_messages_put_the_state_block_last_before_the_new_message():
    state = new_state("s1")
    state["messages"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    messages = build_messages(state, "I need a dump trailer")

    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "I need a dump trailer"}
    assert "WHERE THIS CONVERSATION IS NOW" in messages[-2]["content"]
    assert [m["content"] for m in messages[1:3]] == ["hi", "hello"]


def test_malformed_transcript_entries_are_skipped_not_crashed_on():
    state = new_state("s1")
    state["messages"] = [{"role": "system", "content": "x"}, {"role": "user"}, {}]
    messages = build_messages(state, "hi")
    assert len(messages) == 3  # system prompt, state block, the new message


def test_it_forbids_reciting_whole_lists():
    """A live probe had it read all nineteen makes out, junk rows included."""
    prompt = system_prompt()
    assert "Never recite a whole list" in prompt


def test_the_gooseneck_ambiguity_is_explained():
    """It is both a hitch type and a make on four Livestock rows in trailer_listings."""
    prompt = system_prompt()
    assert "BOTH a hitch type and a trailer brand" in prompt


def test_a_pending_gooseneck_clarification_is_surfaced():
    state = qualified_state(pending_gooseneck_clarification="the gooseneck trailer")
    block = state_block(state)
    assert "gooseneck HITCH" in block and "Gooseneck the BRAND" in block


def test_it_tells_the_model_to_extract_fields_before_a_category_exists():
    """Brief S11. Python keeps pre-category values, but only what the model hands over."""
    prompt = system_prompt()
    assert "STILL FILL IN EVERY FIELD" in prompt
    assert "category or no category" in prompt


def test_the_model_is_told_to_write_the_greeting():
    """The model writes a better welcome than a template - it can use their name and answer
    what they asked. greeting.py is the fallback when its text does not carry one."""
    prompt = system_prompt()
    assert "THEIR FIRST MESSAGE" in prompt
    assert "word for word" in prompt
    assert "Message one only" in prompt
    # The line itself is quoted by the first-turn state block, not here: it is needed on
    # exactly one turn, and the static prompt is sent on every one of them.
    assert "Thank you for contacting TrailerPlace" not in prompt


def test_the_opening_is_demanded_whatever_the_first_message_says():
    """Read as a rule about greetings, it was skipped by a customer who opened with a
    question: "what do you guys sell?" got the catalogue and no hello at all."""
    prompt = system_prompt()
    assert "No first message skips it" in prompt
    assert "what do you guys sell?" in prompt, "the very message that broke it"


def test_the_first_turn_state_block_repeats_the_opening_verbatim():
    """Last in the prompt and specific to this turn - the strongest place to put it."""
    state = new_state("s1")
    state["turn_index"] = 1
    block = state_block(state)

    assert "This is their FIRST message" in block
    assert greeting.OPENING in block

    state["turn_index"] = 2
    assert "FIRST message" not in state_block(state), "said once, then never again"


def test_it_states_the_opening_hours_it_was_given():
    """The hours are a fact now, so the bot may state them."""
    prompt = system_prompt()
    assert "8:00 AM to 6:00 PM" in prompt


def test_it_refuses_to_invent_which_days_we_open():
    """We were given the times and nothing else. "Mon-Sat" is exactly the kind of plausible
    detail that gets a customer driving to a closed lot."""
    prompt = system_prompt()
    assert "never name days" in prompt.lower()
    for day in ("Monday", "Saturday", "Sunday", "weekday"):
        assert day not in prompt




# ------------------------------------------------------------------ the question rules
def test_it_lists_the_cargo_traits_from_the_rules():
    prompt = system_prompt()
    assert "CARGO TRAITS" in prompt
    assert "- lightweight:" in prompt and "golf cart" in prompt
    assert "- large_or_heavy:" in prompt and "skid steer" in prompt


def test_a_new_rules_version_rebuilds_the_prompt_once():
    import json

    from src.rules import store
    from src.rules.models import validate_document

    raw = json.loads(store.SEED_PATH.read_text(encoding="utf-8"))
    raw["cargo_traits"].append({"key": "livestock", "label": "Animals", "definition": "Live animals.",
                                "examples": ["cattle", "goats"]})
    doc, errors = validate_document(raw)
    assert doc is not None, errors
    before = system_prompt()
    store.set_override(doc, version=42)
    after = system_prompt()
    assert "- livestock: Live animals." in after and "- livestock:" not in before
    assert system_prompt() is after


def test_the_configured_wording_of_every_remaining_question_is_given():
    """Live test: without this the model phrased questions itself, and a rewording saved in
    the control panel never reached a customer."""
    state = qualified_state()
    state["slots"] = {"haul_item": "gravel"}
    block = state_block(state)
    assert "use exactly these words" in block
    assert "payload_capacity: \"What's the rough haul weight per load?\"" in block


# The response schema rides along with every call too, and it used to be bigger than the
# prompt: every field restated a rule the prompt already gave, plus a pydantic title per
# field. It measured 20,677 chars before the trim and 13,470 after.
#
# Raised from 14,000 for listing_url and shared_link_interest (14,146): a customer pasting one
# of our listing links, or a Facebook or Instagram post, had no field to land in.
MAX_RESPONSE_SCHEMA_CHARS = 14_500


def test_the_response_schema_stays_short():
    import json

    from openai.lib._pydantic import to_strict_json_schema

    from src.llm.schemas import ChatbotTurnOutput

    schema = json.dumps(to_strict_json_schema(ChatbotTurnOutput))
    assert len(schema) < MAX_RESPONSE_SCHEMA_CHARS, f"response schema is {len(schema)} chars"
    assert '"title"' not in schema, "titles are dead weight the model reads on every call"
