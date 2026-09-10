"""The dealership facts the assistant is allowed to state.

Transcribed from ``company_and_email_scenarios.md``, company-info section only. The email
scenarios in that same file are deliberately NOT represented here: email is out of scope
for this build, and a constant sitting here unused is an invitation to wire it up.

Anything about *inventory* - which categories, which brands - is absent on purpose too.
That comes from ``brands.stocked_categories()`` and ``categories.py``, which read the same
table the search reads, so the assistant cannot promise stock the search cannot return.
"""
from __future__ import annotations

from src.config import settings

NAME = "TrailerPlace"
LOCATION = "Wharton, TX"
PHONE = "979-532-1486"

SERVICES = (
    "trailer sales",
    "financing",
    "delivery",
    "trade-in appraisals",
    "service and parts",
)

# Handled by people, not the bot: these questions get the phone number rather than an answer.
HUMAN_HANDLED = ("financing", "trade-ins", "service", "parts")


def website() -> str:
    """The public site. Falls back to the canonical URL when the env var is unset, so the
    "have a look at the website" reply can never go out with an empty link in it."""
    return settings.trailerplace_website or "https://www.trailerplace.com"


def company_facts_block() -> str:
    """The prompt block. Every fact here is checkable against the reference document."""
    return "\n".join(
        [
            "DEALERSHIP FACTS (the only business facts you may state):",
            f"- Name: {NAME}",
            f"- Location: {LOCATION}",
            f"- Phone: {PHONE}",
            f"- Website: {website()}",
            f"- Services: {', '.join(SERVICES)}",
            "",
            f"Financing, trade-ins, service and parts are handled by people, not by you. "
            f"Answer those with the phone number ({PHONE}) rather than details.",
            "Never state a price, a delivery date, a restock date or a stock level that is "
            "not in a listing you were given this turn.",
        ]
    )


def website_redirect_line() -> str:
    """What to say when they want to see everything and have chosen no category (S25).

    No search runs in that situation - with no category there is nothing to narrow, and a
    dump of the whole lot is not an answer - so they are pointed at the full catalogue.
    """
    return (
        f"You can browse our full inventory any time at {website()} - "
        f"or tell me what you'll be hauling and I'll narrow it down for you here."
    )
