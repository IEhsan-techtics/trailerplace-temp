"""The prompt: short, grounded in real data, and honest about what is already known."""
from __future__ import annotations

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
MAX_SYSTEM_PROMPT_CHARS = 14_000


def test_the_system_prompt_stays_short():
    """It is sent on every turn. A live probe showed input tokens dominating the cost, so
    length here is a running bill, not a style preference."""
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
    assert "3 tons" in prompt            # S17
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


def test_contact_asked_once_and_ignored_is_not_asked_again():
    """The soft-opener decision: asked once, then dropped whatever they do."""
    state = new_state("s1")
    state["contact"]["asked"] = True
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
    what they asked. greeting.py is the fallback when its text does not ask for the
    details."""
    prompt = system_prompt()
    assert "YOU write this reply" in prompt
    assert "Thank you for contacting TrailerPlace" in prompt
    assert "FIRST message only" in prompt


def test_it_refuses_to_invent_opening_hours():
    """The reference document has no hours, so there are none to state."""
    prompt = system_prompt()
    assert "state none" in prompt
    assert "opening hours" in prompt
