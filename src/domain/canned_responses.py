"""The fixed answers, transcribed from company_and_email_scenarios.md.

Two families, and the difference decides whether an email goes out:

* FAQ answers are ours to give. The customer gets a complete answer and the team hears
  nothing - a financing question is not a lead until they ask us to DO something.
* Escalation/team-request text accompanies an ``escalate`` tool call: we could not settle it,
  so a person has been told.

Every one of them names the phone number, deliberately. They all fire on a turn we could not
finish ourselves, so the number is the one thing the customer must not be left without -
"our sales team can help with that" leaves them with no way to reach anyone.
"""
from __future__ import annotations

PHONE = "979-532-1486"
HOURS = "8:00 AM to 6:00 PM"
WEBSITE = "https://trailerplace.com"

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

# The request for a missing name / phone / email lives in src/tools/team_notify.py, which
# words it around the piece we are actually missing rather than asking for both every time.

CANNED_RESPONSES = {**FAQ_ANSWERS, **ESCALATION_ANSWERS}
