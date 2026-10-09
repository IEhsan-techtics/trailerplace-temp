"""A dated event the dealership wants every new customer told about.

Two pieces, both switched off by the clock alone - nobody has to remember to remove them:

* ``announcement`` - a separate message after our first reply to a NEW conversation. Python
  writes it, not the model: it is the dealership's own news, and the dates and offers have
  to be exactly right every time.
* ``prompt_block`` - a dedicated prompt section, so the model can answer questions about
  the event from these facts and nothing else.

Times are the dealership's own (America/Chicago). After ``ends`` neither piece exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from src.domain import company

TEXAS = ZoneInfo("America/Chicago")


@dataclass(frozen=True)
class Event:
    name: str
    starts: datetime
    ends: datetime
    offers: tuple[str, ...]


GRAND_OPENING = Event(
    name="our 10-year anniversary and the grand opening of our new location",
    starts=datetime(2026, 10, 10, 10, 0, tzinfo=TEXAS),
    ends=datetime(2026, 10, 10, 14, 0, tzinfo=TEXAS),
    offers=(
        "Enter to win a brand-new Diamond C PSA trailer - no purchase necessary, but they "
        "must be there to enter.",
        "Our manufacturers will be on site with one-day rebates: save up to $1,000 on a trailer.",
        "Buy any trailer and a truck bed together and get free installation.",
        "Food and fun - bring the family.",
    ),
)

CURRENT = GRAND_OPENING


def _clock(moment: datetime) -> str:
    """"10 AM", "2 PM" - strftime's no-padding flag differs between Linux and Windows."""
    return f"{moment.hour % 12 or 12} {'AM' if moment.hour < 12 else 'PM'}"


def _now(now: datetime | None) -> datetime:
    return (now or datetime.now(TEXAS)).astimezone(TEXAS)


def active(now: datetime | None = None, event: Event = CURRENT) -> bool:
    """Until the event is over. False from its last minute on, with no deploy needed."""
    return _now(now) < event.ends


def _when(now: datetime, event: Event) -> str:
    """"this Saturday, October 10" - or "today" once it is the day itself."""
    day = event.starts
    if now.date() == day.date():
        return "today"
    return f"this {day:%A}, {day:%B} {day.day},"


def announcement(now: datetime | None = None, event: Event = CURRENT) -> str:
    """The message a new customer gets after our first reply, or "" once it is over."""
    now = _now(now)
    if not active(now, event):
        return ""
    when = _when(now, event)
    return (
        f"By the way, {when} from {_clock(event.starts)} to {_clock(event.ends)} we're celebrating "
        f"{event.name} at {company.ADDRESS}! You can enter to win a brand-new Diamond C PSA "
        "trailer (no purchase necessary, just be there), save up to $1,000 with manufacturer "
        "rebates on site, and get free installation when you buy a trailer and a truck bed "
        "together. There'll be food and fun too, so bring the family!"
    )


def prompt_block(now: datetime | None = None, event: Event = CURRENT) -> str:
    """The dedicated prompt section, or "" once the event is over."""
    if not active(now, event):
        return ""
    day = event.starts
    offers = "\n".join(f"  - {offer}" for offer in event.offers)
    return "\n".join(
        [
            f"THIS WEEK'S EVENT (true until {day:%A} {day:%B} {day.day}, {_clock(event.ends)}; "
            "after that it is over)",
            f"- What: {event.name[0].upper()}{event.name[1:]}.",
            f"- When: {day:%A}, {day:%B} {day.day}, {day.year}, {_clock(event.starts)} to "
            f"{_clock(event.ends)}.",
            f"- Where: {company.ADDRESS}.",
            "- Offers, that day only:",
            offers,
            "- New customers are told about it in a separate message we send ourselves after "
            "your first reply. Do NOT announce it yourself. When they ask about it, or about "
            "offers, sales or visiting, answer from these facts only - warmly, and invite them.",
            "- Anything about it that is not written here (which manufacturers, the rebate on a "
            "particular trailer, how the winner is picked, RSVPs, other prizes) -> say plainly "
            f"you don't know, and that our team will be happy to guide them at {company.PHONE}. "
            "Never guess or add an offer.",
            "- A question about the event is answered in words, by you, from this section. It "
            "is never a trailer search or a lookup (\"which manufacturers will be there?\" is not "
            "a brand search), and never a request for our team (not team_request_escalation) - "
            "the phone number is the whole answer to what you don't know. Then carry on with "
            "their trailer.",
        ]
    )
