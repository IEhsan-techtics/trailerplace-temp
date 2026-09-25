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
# The only location - confirmed with the dealership, and said to customers as such. Live, a
# customer asked "Do you have a location in San Antonio?" and was given the Wharton address
# and hours without ever hearing the answer, because the bot was never told there is no other.
LOCATION = "Wharton, TX"
PHONE = "979-532-1486"
# Stated to customers, so it is a fact and not a guess. Days are deliberately not asserted:
# we were given the times and nothing else, and inventing "Mon-Sat" would be exactly the kind
# of plausible detail that gets a customer driving to a closed lot.
HOURS = "8:00 AM to 6:00 PM"

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
            "DEALERSHIP FACTS (the only business facts you may state)",
            f"- {NAME}, {LOCATION}. Phone {PHONE}. Website {website()}.",
            f"- {LOCATION} is our ONLY location. Asked about another city -> say so first: \"We "
            f"don't have a location in San Antonio - we're only in {LOCATION}.\" Then the details, "
            "and that we deliver.",
            f"- Open {HOURS}. State the times only - we were not told which days, so never "
            "name days of the week.",
            f"- Services: {', '.join(SERVICES)}. Financing, trade-ins, service and parts are "
            "handled by people: give the phone number, not details.",
            "- Never state a price, delivery date, restock date or stock level that is not in "
            "a listing you were given this turn.",
        ]
    )


# The five questions the bot answers from a script instead of escalating. Written once and
# shared by both prompts, so the analysis call and the reply pass can never drift apart.
# The key is the faq_key the analysis call returns; the team is emailed about every one.
STANDARD_ANSWERS = (
    ("financing", "Financing",
     f"We offer financing. Call {PHONE} to speak with our finance team, and I can keep "
     "helping narrow down the right trailer."),
    ("trade_in", "Trade-ins", f"Our sales team handles trade-in appraisals. Call {PHONE}."),
    ("service_parts", "Service or parts", f"Our service and parts team can help. Reach them at {PHONE}."),
    ("store_info", "Where we are / hours",
     f"We're located in {LOCATION} and open {HOURS}. Call {PHONE} or visit {{website}}. "
     "We also offer financing and delivery."),
    ("contact_human", "Wanting a person",
     f"You can reach our team at {PHONE}. Happy to keep helping with your trailer search too."),
)


def standard_answers_block(with_keys: bool = False) -> str:
    """The scripted answers. ``with_keys`` adds the faq_key the analysis call must set."""
    lines = ["THE FIVE STANDARD QUESTIONS - answer them yourself with this script, keeping "
             "its meaning and the phone number. Never escalate them."]
    for key, label, answer in STANDARD_ANSWERS:
        tag = f" (faq_key {key})" if with_keys else ""
        lines.append(f'- {label}{tag}: "{answer.format(website=website())}"')
    return "\n".join(lines)


def website_redirect_line() -> str:
    """What to say when they want to see everything and have chosen no category (S25).

    No search runs in that situation - with no category there is nothing to narrow, and a
    dump of the whole lot is not an answer - so they are pointed at the full catalogue.
    """
    return (
        f"You can browse our full inventory any time at {website()} - "
        f"or tell me what you'll be hauling and I'll narrow it down for you here."
    )
