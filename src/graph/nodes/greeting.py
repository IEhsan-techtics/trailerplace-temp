"""The opening of every conversation, and the contact gate that holds it.

Contact details are the one thing the conversation cannot recover later: a visitor who gets
their listings and leaves is a lost lead, and there is no second first message in which to
ask. So no search runs and no qualification question goes out until we have asked at least
once, and the gate keeps asking for whichever half is still missing - a name without a
number is not a lead, and neither is a number without a name.

Three things end the gate:

* both halves arrive (a name AND an email or phone);
* they decline, at which point we never raise it again;
* two asks pass with nothing new, at which point we stop pressing and get on with helping
  them. Progress resets that budget - a customer who gives their name has engaged, and
  asking once more for the number is reasonable rather than nagging.

The wording is deterministic. It is a business decision the dealership has made, and a
greeting that varies run to run is worse than one that reads the same every time.
"""
from __future__ import annotations

from src.domain import categories, company

# How many categories to name before "and many more". Enough to show the range, few enough
# to stay a sentence rather than a list.
_MENU_SAMPLE = 6

# Asks that produced nothing new before the gate gives up. Same ceiling as a qualification
# question, for the same reason: twice is persistence, three times is pestering.
MAX_CONTACT_ASKS = 2

# Turns that must pass before asking a second time. Asked on turn 1 and again on turn 2, the
# two requests read as one form being filled in rather than a conversation - and the customer
# has barely had a chance to answer the first before the second arrives. With a gap, the
# second ask lands after they have actually told us something about what they need.
CONTACT_ASK_TURN_GAP = 2


def has_name(contact: dict) -> bool:
    return bool((contact.get("name") or "").strip())


def has_reach(contact: dict) -> bool:
    """A way to follow up: either channel will do."""
    return bool(contact.get("email") or contact.get("phone"))


def contact_is_complete(contact: dict) -> bool:
    """The dealership's bar for a lead: a name AND one way to reach them."""
    return has_name(contact) and has_reach(contact)


def contact_gate_applies(state: dict) -> bool:
    """True when the reply should carry the contact request rather than only the flow."""
    contact = state.get("contact") or {}
    if contact.get("declined") or contact_is_complete(contact):
        return False
    if int(contact.get("asks_without_progress") or 0) >= MAX_CONTACT_ASKS:
        return False
    last_asked = int(contact.get("last_asked_turn") or 0)
    if not last_asked:
        return True
    return int(state.get("turn_index") or 0) - last_asked >= CONTACT_ASK_TURN_GAP


def note_asked(state: dict) -> None:
    """Record that this turn asked for their details.

    One place, because three of them do it - the gate reply, the one-time opener, and the
    check that spots the agent asking in its own words - and a count kept in two of the
    three is how a customer ends up being asked a third time.
    """
    contact = state.setdefault("contact", {})
    contact["asked"] = True
    contact["asks_without_progress"] = int(contact.get("asks_without_progress") or 0) + 1
    contact["last_asked_turn"] = int(state.get("turn_index") or 0)


def _menu_sample() -> list[str]:
    """The handful of categories to name, broadest first.

    Ordered by how many makes stock each one, so the customer hears the categories with the
    most choice behind them rather than whichever happens to sort first alphabetically -
    "Aluminum, Car Hauler, Equipment..." led with one of the thinnest ranges we carry.
    Derived from the catalogue, so it re-orders itself as stock changes and can never name
    a category we have sold out of.
    """
    from src.domain.brands import load_make_inventory

    stocked = list(categories._advertised_categories())
    inventory = load_make_inventory()
    breadth = {category: 0 for category in stocked}
    for makes_categories in inventory.categories_by_make.values():
        for category in makes_categories:
            if category in breadth:
                breadth[category] += 1
    return sorted(stocked, key=lambda c: (-breadth[c], stocked.index(c)))


def _category_menu_sentence() -> str:
    """A sample of what we carry, never the whole list.

    Written here rather than left to the model, which recited all thirteen even with an
    explicit rule against it. A wall of names is not an answer to "what do you have?".
    """
    stocked = _menu_sample()
    sample = ", ".join(stocked[:_MENU_SAMPLE])
    tail = " and many more" if len(stocked) > _MENU_SAMPLE else ""
    return (
        f"What type of trailer are you looking for? We have {sample}{tail} - "
        f"which one fits what you need?"
    )


# The opening line of every conversation, whatever the first message contained. A customer
# who has already given their name still gets it; so does one who opened with a full spec.
# It is the only sentence in the product that is guaranteed to be said, so it carries both
# jobs at once: who they have reached, and that someone is going to help them.
OPENING = (
    f"Thank you for contacting {company.NAME}. I see you're looking for a trailer, "
    f"and I'm here to help!"
)


# A blank line between the greeting and the request. Written as a constant because the
# shell that generated this file collapses escapes inside string literals.
_GAP = chr(10) * 2


def _is_first_turn(state: dict) -> bool:
    return int(state.get("turn_index") or 0) <= 1


def is_first_turn(state: dict) -> bool:
    """The welcome turn. Public because compose has to know it too."""
    return _is_first_turn(state)


def owns_the_turn(state: dict) -> bool:
    """True when the contact request IS the reply, rather than riding along with it.

    Only the welcome turn. There, the request is the whole point and a qualification
    question alongside it would compete with it.

    Every later ask rides along instead: it goes on the end of whatever the reply was
    already saying. It used to own those turns too, and the cost was hidden until the gate
    stopped asking on consecutive turns - a customer who answered "-500 lbs" on turn three
    got the contact request INSTEAD of being told the weight looked wrong, because the gate
    took the turn and the flow never ran.
    """
    return _is_first_turn(state)


def gate_lead(state: dict) -> str:
    """The opening line, or "" when there is nothing to open with.

    On the first turn it always leads, whatever the message contained. On later turns it
    only does when they have just given us something to thank them for - repeating "thank
    you for contacting us" three messages in is the mark of a script, not a conversation.

    It used to return "Before we go on -" when there was nothing. That is a lead-in to a
    REQUEST, and this text does not always precede one: with an answer between it and the
    request it became a dangling fragment in front of an explanation about livestock
    trailers. Now there is simply no lead, and whatever comes next opens the reply.
    """
    contact = state.get("contact") or {}
    name = (contact.get("name") or "").strip()
    if _is_first_turn(state):
        return OPENING
    if name:
        return f"Thank you for sharing your name, {name}."
    if has_reach(contact):
        return "Thank you for that."
    return ""


def gate_ask(state: dict) -> str:
    """The request itself, addressed to whichever half is still missing.

    Called optional on purpose - it is a courtesy, not a toll gate, and saying so is what
    keeps a reluctant customer in the conversation.
    """
    contact = state.get("contact") or {}
    if has_name(contact):
        return (
            "Could you please provide your email address or phone number so our team can "
            "follow up with you? It's optional, but it would be helpful."
        )
    if has_reach(contact):
        return "Could I take your name as well, so our team knows who they're following up with?"
    return (
        "Could you please provide your name and an email address or phone number so our "
        "team can follow up with you? It's optional, but it would be helpful."
    )


def gate_reply(state: dict, answer: str = "") -> str:
    """Greeting, then anything they actually asked, then the contact request.

    The fallback, not the normal path: the model writes this reply itself and is told in the
    state block to ask for whatever is missing. This is what goes out when it did not.

    Their question is answered BEFORE the request, on purpose. A customer who opens with
    "what are your hours?" and is answered with a form is being processed, not helped - and
    the whole reason the gate is polite about being optional is that a customer who feels
    handled leaves.
    """
    parts = [part for part in (gate_lead(state), (answer or "").strip()) if part]
    parts.append(gate_ask(state))
    return _GAP.join(parts)


def completion_greeting(state: dict) -> str:
    """Said once, on the turn their details become complete.

    On the first turn that is the opening line itself - a customer who introduced
    themselves fully still gets the same welcome as everyone else, with their name in it.
    """
    contact = state.get("contact") or {}
    name = (contact.get("name") or "").strip()
    if _is_first_turn(state):
        if name:
            return (
                f"Thank you for contacting {company.NAME}, {name}! I see you're looking "
                f"for a trailer, and I'm here to help."
            )
        return OPENING
    return f"Great to have your contact info, {name}!" if name else "Great, thank you!"


def orientation_question(state: dict) -> str:
    """What follows once contact is settled and they have named no category yet."""
    return _category_menu_sentence()
