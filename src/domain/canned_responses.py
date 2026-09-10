"""The fixed answers, transcribed from company_and_email_scenarios.md.

Two families, and the difference decides whether an email goes out:

* FAQ answers are ours to give: the customer gets a complete answer on the spot and keeps
  their place in the qualification flow, with no second model call. The team is still told -
  someone asking about financing is a lead worth following up - but through apply, not
  through the escalate tool.
* Escalation/team-request text accompanies an ``escalate`` tool call: we could not settle it,
  so a person has been told.

Every one of them names the phone number, deliberately. They all fire on a turn we could not
finish ourselves, so the number is the one thing the customer must not be left without -
"our sales team can help with that" leaves them with no way to reach anyone.
"""
from __future__ import annotations

from src.domain import company

# Re-exported from company.py rather than restated. They had drifted: this file said
# "https://trailerplace.com" and company.website() said "https://www.trailerplace.com", so
# the same conversation quoted two different addresses depending on which line answered - and
# this copy ignored the TRAILERPLACE_WEBSITE override entirely.
PHONE = company.PHONE
HOURS = company.HOURS
WEBSITE = company.website()

# The five things people ask that we answer from a script. Answering one is NOT an escalation.
FAQ_ANSWERS: dict[str, str] = {
    "contact_human": (
        f"You can reach our team at {PHONE}. Happy to keep helping with your trailer search too."
    ),
    "financing": (
        f"We offer financing. Call {PHONE} to speak with our finance team, and I can keep "
        "helping narrow down the right trailer."
    ),
    "trade_in": f"Our sales team handles trade-in appraisals. Call {PHONE}.",
    "service_parts": f"Our service and parts team can help. Reach them at {PHONE}.",
    "store_info": (
        f"We're located in Wharton, TX and open {HOURS}. Call {PHONE} or visit {WEBSITE}. "
        "We also offer financing and delivery."
    ),
}

# What we say once the escalate tool has recorded something for the team.
#
# The escalation line NEVER offers to keep helping them shop. That text used to end "In the
# meantime, I can keep helping you narrow down the right trailer" - and because canned text
# arrives as an instruction, a customer who opened with "I have a complaint against you guys"
# was answered with the full trailer catalogue and "which type do you want to go with?".
# What we say once the team really has been told. Only correct when the email is on its way -
# see ESCALATION_PENDING for the far more common case where it is still waiting on contact
# details, and ESCALATION_DECLINED for a customer who does not want to be contacted.
ESCALATION_ANSWERS: dict[str, str] = {
    "complaint": (
        "I'm sorry to hear that. I've noted it and passed it to our team - they'll reach out "
        f"to you, and you can also reach them directly on {PHONE}."
    ),
    "team_request": (
        "Thanks, I shared that request with the team so they can help you with it. If you'd "
        f"rather not wait, our sales team is on {PHONE}."
    ),
    "listing_interest": (
        "Your interest in that trailer has been logged and our team can follow up. In the "
        f"meantime, feel free to visit {WEBSITE} or call {PHONE}."
    ),
}

# The same three, for a request we are holding because we cannot reach them yet.
#
# The sent wording claims the request has already gone to the team. Said on the turn it is
# only stashed - which is most of them, since a complaint usually arrives long before a phone
# number - that is a promise we have not yet kept, and the customer has no reason to answer
# the question that follows if the job already sounds done. These say what is actually true,
# and lead into the request for details instead of making it look like an afterthought.
ESCALATION_PENDING: dict[str, str] = {
    "complaint": (
        "I'm sorry to hear that. I've noted it, and our team will reach out as soon as I "
        f"know how to put them in touch with you - you can also call them on {PHONE} right "
        "now if you'd rather not wait."
    ),
    "team_request": (
        "Thanks - I've noted that down for the team. They can pick it up as soon as I know "
        f"how to reach you, and our sales team is on {PHONE} if you'd rather go direct."
    ),
    "listing_interest": (
        "I've noted your interest in that trailer. Our team can follow up as soon as I know "
        f"how to reach you - in the meantime, have a look at {WEBSITE} or call {PHONE}."
    ),
}

# And for someone who told us they do not want to be contacted. We promise nothing and ask
# for nothing: the number is the only thing left to give them.
ESCALATION_DECLINED: dict[str, str] = {
    "complaint": (
        "I'm sorry to hear that. Our team can put it right for you - the quickest way is to "
        f"call them on {PHONE}."
    ),
    "team_request": (
        f"Our sales team can help you with that directly - give them a call on {PHONE}."
    ),
    "listing_interest": (
        f"You can see that trailer at {WEBSITE}, or call our team on {PHONE} to ask about it."
    ),
}

_BY_STATUS = {
    "sent": ESCALATION_ANSWERS,
    "stashed": ESCALATION_PENDING,
    "dropped": ESCALATION_DECLINED,
}


def escalation_answer(key: str, status: str) -> str:
    """The canned line for this situation, worded for what actually happened to the email.

    ``status`` is what ``team_notify.record`` returned: "sent", "stashed" or "dropped".
    """
    answers = _BY_STATUS.get(status, ESCALATION_PENDING)
    return answers.get(key, answers["team_request"])


# The request for a missing name / phone / email lives in src/tools/team_notify.py, which
# words it around the piece we are actually missing rather than asking for both every time.

CANNED_RESPONSES = {**FAQ_ANSWERS, **ESCALATION_ANSWERS}
