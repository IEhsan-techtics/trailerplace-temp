"""Live scenario run: three scripted customers per stocked category, against a running backend.

    python scripts/scenario_run.py [--base-url http://127.0.0.1:8801] [--workers 6] [--only Dump,Utility]

Real model, real database - every turn goes through POST /chat exactly as the frontend
sends it, and the session state is read back from Postgres after each turn to see what
the bot actually decided.

The customer is scripted but not blind: each turn it answers whatever question the bot
really asked (the pending slot), so a conversation that takes a different route still
gets sensible answers. Three scenarios per category:

* happy  - plain, clean answers in feet and pounds.
* units  - the same information in other units and wordings (metres, inches, tons, kg,
           number words, ranges), checked against what Python parsed.
* vague  - the first measurement question is dodged twice (a counter-question, then an
           off-topic question), so it must be dropped after two asks; the next one gets
           "whatever works", which must be taken as no preference at once.

These customers decline contact details in their first message, so they send no email.

--scripted rules,category,flow,features,email (or all) adds hand-written conversations: the
question rules (light-load weight skip, width for large cargo, defaults), category switches
and recommendations, non-metadata features and the gpt-5-nano reranker, and one conversation
per kind of team email. The email ones GIVE
contact details, so their notifications really go out - to the inbox .env names.

Writes one markdown log with a summary table and every conversation in full to
logs/scenario_run_<timestamp>.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.conversation_store import load_session  # noqa: E402
from src.graph.state import MAX_ASKS_PER_SLOT, from_snapshot  # noqa: E402

OPENING = "Hi, I'd rather not give my name or contact details, I'm just browsing."
MAX_TURNS = 12

# How each category is asked for, and what the customer is hauling in the happy and units
# runs. Cargo is chosen so it does not also imply a DIFFERENT category (a "skid steer" on a
# Tilt would open a switch-to-Equipment question and test that instead).
CATEGORIES: dict[str, dict[str, str]] = {
    "Aluminum": {"ask": "I'm looking for an aluminum trailer", "cargo": "tools and equipment for work"},
    "Car Hauler": {"ask": "I need a car hauler", "cargo": "a Ford Mustang"},
    "Equipment": {"ask": "I need an equipment trailer", "cargo": "a Bobcat T770 skid steer"},
    "Enclosed": {"ask": "I want an enclosed trailer", "cargo": "moving boxes and furniture"},
    "Utility": {"ask": "I need a utility trailer", "cargo": "lawn mowers and landscaping gear"},
    "Fiber": {"ask": "I need a fiber trailer", "cargo": "fiber splicing work"},
    "Race Trailer": {"ask": "I'm after a race trailer", "cargo": "a sprint racer and spare parts"},
    "Roll Off": {"ask": "I need a roll off trailer", "cargo": "construction debris"},
    "Flatbed": {"ask": "I'm looking for a flatbed", "cargo": "steel pipe and lumber"},
    "Dump": {"ask": "I need a dump trailer", "cargo": "gravel and dirt"},
    "Tilt": {"ask": "I want a tilt trailer", "cargo": "a small forklift"},
    "Livestock": {"ask": "I need a livestock trailer", "cargo": "cattle"},
    "Concession": {"ask": "I want a concession trailer", "cargo": "a food business"},
}

# Clean answers.
HAPPY: dict[str, str] = {
    "payload_capacity": "about 5,000 lbs",
    "length": "20 ft",
    "width": "7 ft wide",
    "height": "7 ft tall",
    "hitch_type": "bumper pull",
    "cargo_size": "about 12 ft long and 6 ft wide",
    "bin_size": "20 yard bins",
    "base_category": "a utility trailer",
    "tank_capacity": "500 gallons",
}

# Other units and wordings, with what Python should have stored. Rotated by category so
# the run as a whole covers every variant.
UNITS: dict[str, list[tuple[str, float | str | None]]] = {
    "payload_capacity": [
        ("around 2.5 tons", 5000.0),
        ("roughly 2,000 kg", 4409.2),
        ("three and a half thousand pounds", 3500.0),
        ("somewhere between 4000 and 6000 lbs", 4000.0),
        ("a ton and a half", 3000.0),
        ("half a ton", 1000.0),
    ],
    "length": [
        ("about 6 meters", 19.69),
        ("240 inches", 20.0),
        ("twenty-two feet", 22.0),
        ("18 to 20 foot", 18.0),
    ],
    "width": [("2.2 meters", 7.22), ("84 inches", 7.0), ("seven and a half feet", 7.5), ("7 and a half feet", 7.5)],
    "height": [("2 meters", 6.56), ("80 inches", 6.67)],
    "hitch_type": [("goose neck please", "Gooseneck"), ("the ball hitch kind, bumper pull", "Bumper Pull")],
    "cargo_size": [("about 4m by 2m", 13.12), ("144 x 72 inches", 12.0), ("144 x 72", 12.0)],
    "bin_size": [("15 cubic yards", 15.0), ("twenty yard containers", 20.0)],
    "base_category": [("an enclosed one", "Enclosed"), ("a car hauler", "Car Hauler")],
    "tank_capacity": [("about 1,900 litres", None)],
}

DODGES = [
    "Why do you need to know that?",
    "Hmm, not sure. By the way, what are your opening hours?",
]
VAGUE = "Whatever works, I don't really have a preference."

# Slots that can be dodged or answered vaguely in the vague run - a measurement or a choice,
# not the free-text "what are you hauling".
MEASURED = ("payload_capacity", "length", "width", "height", "cargo_size", "bin_size", "hitch_type", "base_category")


@dataclass
class Turn:
    user: str
    bot: str
    seconds: float
    pending: str | None
    slots: dict[str, Any]
    note: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    listings: list[str] = field(default_factory=list)   # URLs shown this turn, in order


@dataclass
class Result:
    category: str
    scenario: str
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(ok for _, ok, _ in self.checks)


def _state(session_id: str) -> dict[str, Any]:
    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


class Customer:
    """Answers whatever the bot asked, in the style of one scenario."""

    def __init__(self, category: str, scenario: str, index: int,
                 overrides: dict[str, str] | None = None) -> None:
        self.category = category
        self.scenario = scenario
        self.index = index
        self.overrides = overrides or {}         # scripted runs: slot -> the answer to give
        self.dodged: str | None = None           # the slot dodged twice (vague run)
        self.dodges_used = 0
        self.vague_slot: str | None = None       # the slot answered "whatever works"
        self.expected: dict[str, Any] = {}       # units run: slot -> what should be stored

    def answer(self, slot: str) -> str:
        if slot in self.overrides:
            return self.overrides[slot]
        cargo = CATEGORIES.get(self.category, {}).get("cargo", "general cargo")
        if slot == "haul_item":
            if self.scenario == "vague" and not self._has_measured_question():
                return self._dodge(slot)
            return {"vague": "just some random stuff"}.get(self.scenario, cargo)

        if self.scenario == "units" and slot in UNITS:
            options = UNITS[slot]
            text, expected = options[self.index % len(options)]
            self.expected[slot] = expected
            return text

        if self.scenario == "vague" and slot in MEASURED:
            if self.dodged in (None, slot) and self.dodges_used < 2:
                return self._dodge(slot)
            if self.vague_slot is None and slot != self.dodged:
                self.vague_slot = slot
                return VAGUE

        return HAPPY.get(slot, "no particular preference")

    def _dodge(self, slot: str) -> str:
        self.dodged = slot
        line = DODGES[self.dodges_used]
        self.dodges_used += 1
        return line

    def _has_measured_question(self) -> bool:
        # Only Fiber asks nothing but the free-text question, so there the dodge lands on it.
        return self.category != "Fiber"


def run_one(base_url: str, category: str, scenario: str, index: int, log_lock: threading.Lock) -> Result:
    session_id = str(uuid.uuid4())
    result = Result(category, scenario, session_id)
    customer = Customer(category, scenario, index)
    http = httpx.Client(timeout=300)

    def say(text: str, note: str = "") -> dict[str, Any]:
        started = time.time()
        response = http.post(f"{base_url}/chat", json={"message": text, "session_id": session_id})
        response.raise_for_status()
        seconds = time.time() - started  # the /chat call only, not the state read below
        body = response.json()
        state = _state(session_id)
        result.turns.append(
            Turn(text, body.get("assistant_text") or "", seconds,
                 state.get("pending_slot"), dict(state.get("slots") or {}), note,
                 body.get("usage") or {},
                 [str(item.get("url") or "") for item in body.get("listings") or [] if isinstance(item, dict)])
        )
        return state

    try:
        say(OPENING)
        state = say(CATEGORIES[category]["ask"])
        state, asked_after_answered = _drive(say, state, customer, switch="no")
        result.final = _final(state)
        _check(result, customer, state, asked_after_answered)
    except Exception as exc:  # noqa: BLE001 - a broken run is a result, not a crash
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        http.close()
    with log_lock:
        mark = "PASS" if result.passed else "FAIL"
        print(f"  {mark}  {category:<13} {scenario:<6} {len(result.turns):>2} turns", flush=True)
    return result


def _drive(say, state: dict[str, Any], customer: Customer, switch: str) -> tuple[dict[str, Any], list[str]]:
    """Answer whatever the bot asks until listings are shown. Returns the state and every
    slot that was asked again after it already held a value.

    ``switch`` is how a suggested category switch is answered: "yes" or "no".
    """
    asked_after_answered: list[str] = []
    for _ in range(MAX_TURNS):
        if state.get("results_shown"):
            break
        offer = state.get("pending_category_switch")
        if offer:
            if switch == "yes":
                state = say("Yes, that sounds better - let's switch to that.",
                            f"accepted a switch to {offer.get('suggested')}")
            else:
                state = say(f"No thanks, I'll stick with a {state.get('category')} trailer.",
                            f"declined a switch to {offer.get('suggested')}")
            continue
        if state.get("pending_keep_filters"):
            state = say("Keep everything I told you.", "keep-filters question")
            continue
        if state.get("pending_gooseneck_clarification"):
            state = say("I meant the gooseneck hitch.", "gooseneck clarification")
            continue
        pending = state.get("pending_slot")
        if not pending:
            state = say("Can you show me what you have?", "bot asked no question - nudged")
            continue
        if (state.get("slots") or {}).get(pending) not in (None, "", []):
            # The bot is asking for something it already holds.
            asked_after_answered.append(pending)
        state = say(customer.answer(pending), f"answering {pending}")
    return state, asked_after_answered


def _final(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: state.get(key) for key in (
            "category", "slots", "declined_slots", "asked_counts", "required_slots",
            "rule_skipped", "rule_defaults", "cargo_traits", "results_shown",
            "rejected_switches", "unavailable_requests", "pending_email_actions", "contact",
        )
    }


def _check(result: Result, customer: Customer, state: dict[str, Any], repeats: list[str]) -> None:
    add = result.checks.append
    counts = state.get("asked_counts") or {}
    declined = state.get("declined_slots") or []
    slots = state.get("slots") or {}

    add(("reached listings", bool(state.get("results_shown")), f"{len(result.turns)} turns"))
    nudges = sum(1 for t in result.turns if "nudged" in t.note)
    add(("listings shown without having to ask", nudges == 0,
         f"customer had to ask {nudges}x after the bot stopped asking questions" if nudges else ""))
    over = {slot: n for slot, n in counts.items() if n > MAX_ASKS_PER_SLOT}
    add(("no question asked more than twice", not over, str(over) if over else str(counts)))
    add(("never re-asked an answered question", not repeats, ", ".join(repeats)))
    multi = [t.bot[:80] for t in result.turns[1:] if "http" not in t.bot and t.bot.count("?") > 1]
    add(("one question per reply", not multi, f"{len(multi)} replies with 2+ '?'"))

    if customer.scenario == "units":
        for slot, expected in customer.expected.items():
            got = slots.get(slot)
            if expected is None:
                add((f"{slot} stored", got is not None, f"got {got!r}"))
            elif isinstance(expected, float):
                ok = isinstance(got, (int, float)) and abs(float(got) - expected) <= max(0.35, expected * 0.03)
                add((f"{slot} parsed", ok, f"expected ~{expected}, got {got!r}"))
            else:
                ok = expected in (got if isinstance(got, list) else [got])
                add((f"{slot} parsed", ok, f"expected {expected!r}, got {got!r}"))

    if customer.scenario == "vague":
        if customer.dodged:
            slot = customer.dodged
            ok = slot in declined and counts.get(slot) == MAX_ASKS_PER_SLOT and slot not in slots
            add((f"dodged {slot} dropped after 2 asks", ok, f"asked {counts.get(slot)}, declined={slot in declined}"))
        if customer.vague_slot:
            slot = customer.vague_slot
            ok = slot in declined and counts.get(slot) == 1
            add((f"'whatever works' on {slot} taken as no preference", ok,
                 f"asked {counts.get(slot)}, declined={slot in declined}"))


# ------------------------------------------------------------------ scripted scenarios
# Hand-written conversations for behaviour the per-category runs never reach: the question
# rules (light-load weight skip, the width question for large cargo, category defaults),
# category routing (suggested switches, a recommendation with no category named) and every
# kind of email the bot sends to the team.
#
# The email scenarios GIVE contact details, because nothing is sent without them. The
# notifications go to whichever inbox .env names (RECIPIENT_EMAIL / EMAIL_TO), so check it
# points at a test inbox before running --scripted email.

QUALIFY = "<answer the questions until listings are shown>"
TESTER = "Hi, I'm Sam Tester. My email is sam.tester@example.com and my phone is 555-0100."


@dataclass
class Script:
    group: str
    name: str
    steps: list[str]
    answers: dict[str, str] = field(default_factory=dict)   # slot -> answer, during QUALIFY
    switch: str = "no"                                      # how a suggested switch is answered
    skipped: tuple[str, ...] = ()        # skipped by a rule, never asked
    asked: tuple[str, ...] = ()          # must be asked
    not_asked: tuple[str, ...] = ()      # must never be asked
    traits: tuple[str, ...] = ()         # cargo traits the model must have tagged
    defaults: dict[str, Any] = field(default_factory=dict)
    category: str | None = None          # the category at the end
    offered: str | None = None           # a switch to this category must have been suggested
    emails: tuple[str, ...] = ()         # event_type prefixes that must reach the outbox, sent
    no_emails: bool = False              # nothing may reach the outbox
    stash_at: int | None = None          # after this step a request is held, nothing sent yet
    reply_has: tuple[tuple[int, str], ...] = ()   # (step, text) the bot's reply must contain
    no_listings_at: tuple[int, ...] = ()           # steps whose reply must carry no listings
    features: tuple[str, ...] = ()       # each word must be in a stored non-metadata feature
    not_features: tuple[str, ...] = ()   # none of these words may be in one
    no_features: bool = False            # the feature list must end empty
    rerank: bool | None = None           # the gpt-5-nano feature reranker must (not) have run
    top_has: str | None = None           # the first listing shown must carry this word
    slots_equal: dict[str, Any] = field(default_factory=dict)  # slot -> value it must end with
    absent: tuple[str, ...] = ()         # slots that must end unset
    reply_lacks: tuple[tuple[int, str], ...] = ()  # (step, text) the reply must NOT contain


def _bump(cargo: str) -> dict[str, str]:
    return {"haul_item": cargo}


SCRIPTS: list[Script] = [
    # --- question rules ---------------------------------------------------------------
    Script("rules", "utility, riding mower - weight skipped",
           [OPENING, "I need a utility trailer", QUALIFY], answers=_bump("a riding lawn mower"),
           skipped=("payload_capacity",), traits=("lightweight",), category="Utility"),
    Script("rules", "utility, two ATVs - weight skipped",
           [OPENING, "I need a utility trailer", QUALIFY], answers=_bump("two ATVs"),
           skipped=("payload_capacity",), traits=("lightweight",), category="Utility"),
    Script("rules", "utility, camping gear and kayaks - weight skipped",
           [OPENING, "Looking for a small utility trailer", QUALIFY],
           answers=_bump("camping gear and a couple of kayaks"),
           skipped=("payload_capacity",), traits=("lightweight",), category="Utility"),
    Script("rules", "utility, bricks - weight asked",
           [OPENING, "I need a utility trailer", QUALIFY],
           answers=_bump("pallets of bricks and concrete blocks"),
           asked=("payload_capacity",), category="Utility"),
    Script("rules", "equipment, mini excavator - width asked",
           [OPENING, "I need an equipment trailer", QUALIFY], answers=_bump("a mini excavator"),
           asked=("width",), traits=("large_or_heavy",), category="Equipment"),
    Script("rules", "car hauler, pickup truck - width asked",
           [OPENING, "I need a car hauler", QUALIFY], answers=_bump("a Ford F-250 pickup truck"),
           asked=("width",), traits=("large_or_heavy",), category="Car Hauler"),
    Script("rules", "tilt, scissor lift - width asked",
           [OPENING, "I want a tilt trailer", QUALIFY], answers=_bump("a scissor lift"),
           asked=("width",), traits=("large_or_heavy",), category="Tilt"),
    Script("rules", "equipment, generator and hand tools - no width",
           [OPENING, "I need an equipment trailer", QUALIFY],
           answers=_bump("a small generator and hand tools"),
           not_asked=("width",), category="Equipment"),
    Script("rules", "dump, skid steer - width rule excludes Dump",
           [OPENING, "I need a dump trailer", QUALIFY], answers=_bump("crushed concrete and a skid steer"),
           not_asked=("width",), category="Dump"),
    Script("rules", "flatbed - width defaulted, never asked",
           [OPENING, "I'm looking for a flatbed", QUALIFY], answers=_bump("steel pipe and lumber"),
           not_asked=("width",), defaults={"width": 8.0}, category="Flatbed"),

    # --- category routing -------------------------------------------------------------
    Script("category", "utility + skid steer -> Equipment, accepted",
           [OPENING, "I need a utility trailer", QUALIFY], answers=_bump("a Bobcat skid steer"),
           switch="yes", offered="Equipment", category="Equipment", asked=("width",)),
    Script("category", "flatbed + Camaro -> Car Hauler, accepted",
           [OPENING, "I'm looking for a flatbed", QUALIFY], answers=_bump("my Chevy Camaro"),
           switch="yes", offered="Car Hauler", category="Car Hauler"),
    Script("category", "enclosed + horses -> Livestock, declined",
           [OPENING, "I want an enclosed trailer", QUALIFY], answers=_bump("three horses"),
           switch="no", offered="Livestock", category="Enclosed"),
    Script("category", "no category named - cattle",
           [OPENING, "What kind of trailer would you recommend for moving cattle?", QUALIFY],
           switch="yes", category="Livestock", asked=("length",)),

    # --- when listings appear -------------------------------------------------------
    # 1. Category and every required answer in one message: listings at once.
    # 2. Category, then the answers: listings when the last one is in.
    # 3. "Just show me" DURING the questions skips the rest - but a request on the turn that
    #    picks the category only picks it; the questions start first.
    Script("flow", "1. everything in one message - listings at once",
           [OPENING, "I need a 20 ft livestock trailer"],
           not_asked=("length",), category="Livestock", reply_has=((1, "http"),)),
    Script("flow", "1. use case + answer in one message - listings at once",
           [OPENING, "I need a trailer for my food truck business, about 16 ft long"],
           not_asked=("length",), category="Concession", reply_has=((1, "http"),)),
    Script("flow", "2. category, then answers",
           [OPENING, "I need a dump trailer", QUALIFY], answers=_bump("gravel and dirt"),
           asked=("haul_item", "payload_capacity"), category="Dump"),
    Script("flow", "3. show me mid-questions skips the rest",
           [OPENING, "I need a dump trailer", "Just show me what you have"],
           asked=("haul_item",), not_asked=("payload_capacity",), category="Dump",
           reply_has=((2, "http"),)),
    Script("flow", "3. recommendation picks category, then asks first",
           [OPENING, "Can you recommend a trailer for hauling gravel? Just show me options.",
            "Enough questions, just show me the trailers"],
           category="Dump", reply_has=((2, "http"),), no_listings_at=(1,)),
    Script("category", "no category named - excavator",
           [OPENING, "I need something to move my mini excavator between job sites", QUALIFY],
           answers=_bump("a mini excavator"), switch="yes", category="Equipment"),
    Script("category", "no category named - gravel",
           [OPENING, "I need a trailer I can tip to unload gravel and mulch", QUALIFY],
           answers=_bump("gravel and mulch"), switch="yes", category="Dump"),
    Script("category", "no category named - food truck business",
           [OPENING, "I'm starting a mobile coffee business and need a trailer for it", QUALIFY],
           switch="yes", category="Concession"),

    # --- non-metadata features and the gpt-5-nano reranker ---------------------------
    # Real equipment must be stored and must pull matching trailers to the top; cargo, a
    # use, axles, a hitch, a size or a colour must never be stored, and then nano must not
    # run at all. Each feature below is on only SOME of that category's stock (tarp: 12 of
    # 24 Dump, winch: 16 of 38 Flatbed, insulation: 10 of 48 Enclosed, butterfly gates: 6
    # of 12 Livestock), so the ranking has something to do.
    Script("features", "dump with a tarp -> tarp trailers first",
           [OPENING, "I need a dump trailer with a tarp system", QUALIFY],
           answers=_bump("gravel"), features=("tarp",), rerank=True, top_has="tarp",
           category="Dump"),
    Script("features", "flatbed with a winch -> winch trailers first",
           [OPENING, "Looking for a flatbed that has a winch", QUALIFY],
           answers=_bump("steel pipe"), features=("winch",), rerank=True, top_has="winch",
           category="Flatbed"),
    Script("features", "insulated enclosed -> insulation matched by meaning",
           [OPENING, "I want an insulated enclosed trailer", QUALIFY],
           answers=_bump("tools and equipment that have to stay dry"), features=("insul",),
           rerank=True, top_has="insul", category="Enclosed"),
    Script("features", "livestock with butterfly gates",
           [OPENING, "I need a livestock trailer with butterfly gates", QUALIFY],
           answers=_bump("cattle"), features=("butterfly",), rerank=True, top_has="butterfly",
           category="Livestock"),
    Script("features", "feature given mid-questions is kept",
           [OPENING, "I want a tilt trailer", "Oh, and it needs to have a winch", QUALIFY],
           answers=_bump("a small tractor"), features=("winch",), rerank=True,
           top_has="winch", category="Tilt"),
    Script("features", "torsion axles and a winch kept; count, rating, cargo dropped",
           [OPENING, "I need an equipment trailer with tandem 7000 lb torsion axles and a winch "
                     "to haul my Bobcat", QUALIFY],
           answers=_bump("a Bobcat skid steer"), features=("torsion", "winch"),
           not_features=("bobcat", "tandem", "7000", "7,000"), rerank=True, category="Equipment"),
    Script("features", "cargo is not a feature - scissor lift",
           [OPENING, "I need a tilt trailer to haul a scissor lift", QUALIFY],
           answers=_bump("a scissor lift"), no_features=True, not_features=("scissor",),
           rerank=False, category="Tilt"),
    Script("features", "a business is not a feature - coffee",
           [OPENING, "I'm starting a mobile coffee business and need a concession trailer", QUALIFY],
           no_features=True, not_features=("coffee", "business"), rerank=False,
           category="Concession"),
    Script("features", "size, colour, hitch and axles are not features",
           [OPENING, "I need a black 20 ft gooseneck flatbed with tandem 7000 lb axles", QUALIFY],
           answers=_bump("lumber"), no_features=True, rerank=False, category="Flatbed"),

    # --- axle capacity: per axle or total, how many, and what it is not -------------
    # The axle answers are scripted rather than left to QUALIFY: the auto-customer answers
    # the slot questions, and "per axle or total?" / "how many axles?" are not slots.
    Script("axles", "unclear capacity -> per axle, then tandem",
           [OPENING, "I need a dump trailer for gravel with 14,000 lbs of axle capacity",
            "per axle", "tandem", QUALIFY],
           slots_equal={"axle_capacity": 14000.0, "axle_count": 2}, absent=("total_axle_capacity_lbs",),
           not_asked=("payload_capacity",), reply_has=((1, "per axle, or the total"),
                                                        (2, "how many axles")),
           category="Dump"),
    Script("axles", "unclear capacity -> total, no count question",
           [OPENING, "Looking for a flatbed with 14,000 lbs of axle capacity",
            "that's the total across both", QUALIFY],
           answers=_bump("lumber"), slots_equal={"total_axle_capacity_lbs": 14000.0},
           absent=("axle_capacity",), not_asked=("payload_capacity",),
           reply_has=((1, "per axle, or the total"),), reply_lacks=((2, "how many axles"),),
           category="Flatbed"),
    Script("axles", "unclear twice -> dropped, never asked again",
           [OPENING, "I need a utility trailer with 10,000 lbs of axle capacity",
            "hmm, I'm not sure", "no idea honestly", QUALIFY],
           answers=_bump("furniture and boxes"),
           absent=("axle_capacity", "total_axle_capacity_lbs"),
           reply_has=((1, "per axle, or the total"), (2, "per axle, or the total")),
           reply_lacks=((3, "per axle, or the total"),), category="Utility"),
    Script("axles", "7,000 lb axles -> per axle at once, bare 'Tandem.'",
           [OPENING, "I want a tilt trailer with 7,000 lb axles", "Tandem.", QUALIFY],
           answers=_bump("a small tractor"),
           slots_equal={"axle_capacity": 7000.0, "axle_count": 2},
           not_asked=("payload_capacity",), reply_has=((1, "how many axles"),),
           reply_lacks=((1, "per axle, or the total"),), category="Tilt"),
    Script("axles", "5 axles -> one to four, asked again",
           [OPENING, "I need a dump trailer with 7000 lb axles", "5 axles", "two", QUALIFY],
           answers=_bump("gravel"), slots_equal={"axle_capacity": 7000.0, "axle_count": 2},
           reply_has=((2, "one to four"),), category="Dump"),
    Script("axles", "tandem 5200 lb torsion axles -> count, rating, feature",
           [OPENING, "Equipment trailer please, tandem 5200 lb torsion axles", QUALIFY],
           answers=_bump("a mini excavator"),
           slots_equal={"axle_capacity": 5200.0, "axle_count": 2},
           features=("torsion",), not_features=("tandem", "5200"),
           not_asked=("payload_capacity",), reply_lacks=((1, "how many axles"),),
           category="Equipment"),
    Script("axles", "quadruple axles -> 4, no rating so weight still asked",
           [OPENING, "I need a flatbed with quadruple axles", QUALIFY],
           answers=_bump("steel beams"), slots_equal={"axle_count": 4},
           asked=("payload_capacity",), category="Flatbed"),
    Script("axles", "weight given to the count question -> not a count, asked again",
           [OPENING, "I need a dump trailer for gravel with 7000 lb axles", "about 5,000 lbs",
            "tandem", QUALIFY],
           slots_equal={"axle_capacity": 7000.0, "axle_count": 2},
           reply_has=((1, "how many axles"),),
           reply_lacks=((2, "one to four"), (2, "didn't catch")), category="Dump"),
    Script("axles", "'5k axles' first mention -> per axle, never asked which",
           [OPENING, "I need a dump trailer for gravel with 5k axles", "tandem", QUALIFY],
           slots_equal={"axle_capacity": 5000.0, "axle_count": 2}, absent=("total_axle_capacity_lbs",),
           not_asked=("payload_capacity",), reply_has=((1, "how many axles"),),
           reply_lacks=((1, "per axle, or the total"),), category="Dump"),
    Script("axles", "'5k axles' as the load weight -> not a load",
           [OPENING, "I need a dump trailer", "gravel", "5k axles", QUALIFY],
           absent=("payload_capacity",), slots_equal={"axle_capacity": 5000.0},
           category="Dump"),

    # --- emails to the team ------------------------------------------------------------
    Script("email", "FAQ - financing", [TESTER, "Do you offer financing on your trailers?"],
           emails=("faq_-_financing",)),
    Script("email", "FAQ - trade in", [TESTER, "Can I trade in my old utility trailer?"],
           emails=("faq_-_trade_in",)),
    Script("email", "FAQ - service and parts", [TESTER, "Do you do trailer repairs and sell parts?"],
           emails=("faq_-_service_parts",)),
    Script("email", "FAQ - store info", [TESTER, "What are your opening hours and where are you located?"],
           emails=("faq_-_store_info",)),
    Script("email", "FAQ - talk to a person", [TESTER, "Can I speak to a real person please?"],
           emails=("faq_-_contact_human",)),
    Script("email", "complaint -> escalation",
           [TESTER, "I bought a trailer from you last month and the brakes already failed. I'm really unhappy about it."],
           emails=("escalation",)),
    Script("email", "callback request",
           [TESTER, "Can someone call me back tomorrow morning about a dump trailer?"],
           emails=("team_request",)),
    Script("email", "delivery and quote",
           [TESTER, "Can you deliver a trailer to Houston? I'd like a quote including delivery."],
           emails=("team_request",)),
    Script("email", "unavailable - boat trailer", [TESTER, "Do you sell boat trailers?"],
           emails=("trailer_type_not_in_stock",), reply_has=((1, "don't currently have"),)),
    Script("email", "unavailable - camper", [TESTER, "I'm looking for a camper for my family."],
           emails=("trailer_type_not_in_stock",)),
    Script("email", "listing interest after results",
           [TESTER, "I need a dump trailer", QUALIFY,
            "I really like the first one you showed me. Can someone get in touch about buying it?"],
           answers=_bump("gravel and dirt"), emails=("listing_interest",)),
    Script("email", "stashed until contact, then sent",
           ["Hi there", "Do you offer financing?", TESTER],
           emails=("faq_-_financing",), stash_at=1),
    Script("email", "several requests batch in one go",
           ["Hello", "Do you sell boat trailers?", "Also, can I trade in my old trailer?", TESTER],
           emails=("trailer_type_not_in_stock", "faq_-_trade_in"), stash_at=2),
    Script("email", "contact declined - nothing sent",
           [OPENING, "Do you offer financing?", "Do you sell boat trailers?",
            "Can someone call me back about a dump trailer?"],
           no_emails=True),
]


def _outbox(session_id: str, wait: float = 60.0) -> list[dict[str, Any]]:
    """This session's notifications, waiting for the background drain to send them."""
    from src import db
    from src.db_models import ChatbotOutbox

    deadline = time.time() + wait
    while True:
        with db.get_session_factory()() as sql:
            rows = [
                {"event_type": r.event_type, "status": r.status,
                 "subject": (r.payload or {}).get("subject")}
                for r in sql.query(ChatbotOutbox).filter(ChatbotOutbox.session_id == uuid.UUID(session_id))
            ]
        if all(r["status"] != "pending" for r in rows) or time.time() > deadline:
            return rows
        time.sleep(3)


def _was_asked(result: Result, state: dict[str, Any], slot: str) -> bool:
    return bool((state.get("asked_counts") or {}).get(slot)) or any(t.pending == slot for t in result.turns)


def run_script(base_url: str, script: Script, log_lock: threading.Lock) -> Result:
    session_id = str(uuid.uuid4())
    result = Result(script.group, script.name, session_id)
    customer = Customer(script.category or "", "happy", 0, overrides=script.answers)
    http = httpx.Client(timeout=300)
    repeats: list[str] = []
    stash_seen: tuple[bool, int] | None = None

    def say(text: str, note: str = "") -> dict[str, Any]:
        started = time.time()
        response = http.post(f"{base_url}/chat", json={"message": text, "session_id": session_id})
        response.raise_for_status()
        seconds = time.time() - started  # the /chat call only, not the state read below
        body = response.json()
        state = _state(session_id)
        result.turns.append(
            Turn(text, body.get("assistant_text") or "", seconds,
                 state.get("pending_slot"), dict(state.get("slots") or {}), note,
                 body.get("usage") or {},
                 [str(item.get("url") or "") for item in body.get("listings") or [] if isinstance(item, dict)])
        )
        return state

    try:
        state: dict[str, Any] = {}
        for index, step in enumerate(script.steps):
            if step == QUALIFY:
                state, more = _drive(say, state, customer, script.switch)
                repeats += more
            else:
                state = say(step)
            if script.stash_at == index:
                stash_seen = (bool(state.get("pending_email_actions")), len(_outbox(session_id, wait=0)))
        result.final = _final(state)
        _check_script(result, script, state, repeats, stash_seen)
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        http.close()
    with log_lock:
        mark = "PASS" if result.passed else "FAIL"
        print(f"  {mark}  {script.group:<9} {script.name:<50} {len(result.turns):>2} turns", flush=True)
    return result


def _listing_text(url: str) -> str:
    """Everything a listing says about its equipment, lower-cased - read from the catalogue,
    because the /chat response only carries a trimmed pitch list."""
    from src import db
    from src.db_models import TrailerListingRow

    with db.get_session_factory()() as sql:
        row = sql.query(TrailerListingRow).filter(TrailerListingRow.url == url).first()
    if row is None:
        return ""
    features = row.features if isinstance(row.features, list) else [row.features or ""]
    return " | ".join([str(row.title or ""), *map(str, features), str(row.match_evidence_text or "")]).lower()


def _check_features(result: Result, script: Script, state: dict[str, Any]) -> None:
    add = result.checks.append
    stored = [str(f) for f in state.get("non_metadata_features") or []]
    lowered = [f.lower() for f in stored]
    result.final["non_metadata_features"] = stored
    for word in script.features:
        add((f"feature '{word}' stored", any(word in f for f in lowered), f"features={stored}"))
    for word in script.not_features:
        add((f"'{word}' not a feature", not any(word in f for f in lowered), f"features={stored}"))
    if script.no_features:
        add(("no features stored", not stored, f"features={stored}"))

    reranks = sum(int(t.usage.get("feature_reranks") or 0) for t in result.turns)
    nano_turns = [t for t in result.turns if t.usage.get("feature_reranks")]
    if script.rerank is not None:
        detail = f"nano batches={reranks}" + (
            f", on a {nano_turns[0].seconds:.1f}s turn" if nano_turns else "")
        add((f"nano reranker {'ran' if script.rerank else 'did not run'}",
             (reranks > 0) == script.rerank, detail))

    if script.top_has:
        shown = next((t.listings for t in result.turns if t.listings), [])
        texts = [_listing_text(url) for url in shown]
        with_it = sum(script.top_has in text for text in texts)
        add((f"first listing has '{script.top_has}'",
             bool(texts) and script.top_has in texts[0],
             f"{with_it} of {len(texts)} shown have it"))


def _check_script(result: Result, script: Script, state: dict[str, Any], repeats: list[str],
                  stash_seen: tuple[bool, int] | None) -> None:
    add = result.checks.append
    counts = state.get("asked_counts") or {}
    skipped = state.get("rule_skipped") or {}
    traits = state.get("cargo_traits") or []
    slots = state.get("slots") or {}

    over = {slot: n for slot, n in counts.items() if n > MAX_ASKS_PER_SLOT}
    add(("no question asked more than twice", not over, str(over) if over else ""))
    add(("never re-asked an answered question", not repeats, ", ".join(repeats)))
    # The category menu is one question in two sentences ("What type...? ...which one fits?").
    multi = [t.bot[:80] for t in result.turns[1:] if "http" not in t.bot and t.bot.count("?") > 1
             and "which one fits" not in t.bot]
    add(("one question per reply", not multi, f"{len(multi)} replies with 2+ '?'"))

    if QUALIFY in script.steps:
        add(("reached listings", bool(state.get("results_shown")), f"{len(result.turns)} turns"))
    for trait in script.traits:
        add((f"cargo tagged {trait}", trait in traits, f"traits={traits}"))
    for slot in script.skipped:
        add((f"{slot} skipped by a rule", slot in skipped and not _was_asked(result, state, slot),
             f"rule_skipped={skipped}, asked={counts.get(slot, 0)}"))
    for slot in script.asked:
        add((f"{slot} asked", _was_asked(result, state, slot), f"asked_counts={counts}"))
    for slot in script.not_asked:
        add((f"{slot} not asked", not _was_asked(result, state, slot), f"asked_counts={counts}"))
    for slot, value in script.defaults.items():
        add((f"{slot} defaulted to {value}", slots.get(slot) == value, f"got {slots.get(slot)!r}"))
    if script.offered:
        notes = " | ".join(t.note for t in result.turns if "switch" in t.note)
        add((f"suggested switching to {script.offered}", script.offered in notes, notes or "no switch offered"))
    if script.category:
        add((f"category ends as {script.category}", state.get("category") == script.category,
             f"got {state.get('category')!r}"))
    for step in script.no_listings_at:
        reply = result.turns[step].bot if step < len(result.turns) else ""
        add((f"no listings on reply {step + 1}", "http" not in reply, reply[:120]))
    for slot, value in script.slots_equal.items():
        add((f"{slot} = {value}", slots.get(slot) == value, f"got {slots.get(slot)!r}"))
    for slot in script.absent:
        add((f"{slot} not stored", slots.get(slot) in (None, "", []), f"got {slots.get(slot)!r}"))
    for step, text in script.reply_lacks:
        reply = result.turns[step].bot if step < len(result.turns) else ""
        add((f"reply {step + 1} does not say '{text}'",
             text.lower() not in reply.lower().replace("’", "'"), reply[:120]))
    for step, text in script.reply_has:
        reply = result.turns[step].bot if step < len(result.turns) else ""
        add((f"reply {step + 1} says '{text}'", text.lower() in reply.lower().replace("’", "'"), ""))

    if script.features or script.not_features or script.no_features or script.rerank is not None:
        _check_features(result, script, state)

    if stash_seen is not None:
        held, sent = stash_seen
        add(("held until contact given", held and sent == 0, f"stashed={held}, outbox rows={sent}"))

    if script.emails or script.no_emails:
        rows = _outbox(result.session_id)
        summary = ", ".join(f"{r['event_type']} ({r['status']})" for r in rows) or "none"
        result.final["outbox"] = rows
        if script.no_emails:
            add(("no email queued", not rows, summary))
        for prefix in script.emails:
            hit = [r for r in rows if r["event_type"].startswith(prefix)]
            add((f"email '{prefix}' sent", bool(hit) and all(r["status"] == "sent" for r in hit), summary))
        dupes = len(rows) - len({r["event_type"] for r in rows})
        add(("no duplicate emails", dupes == 0, summary if dupes else ""))


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile; 0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _performance(results: list[Result]) -> list[str]:
    """Latency, tokens and prompt-cache hits, split by whether the reply pass ran.

    A turn with ``reply_seconds`` ran the tool-using reply pass (listings, a lookup or an
    escalation); every other turn is the single analysis call. The two have very different
    costs, so one blended number would hide which of them changed.
    """
    turns = [t for r in results for t in r.turns if t.usage]
    if not turns:
        return []
    groups = {
        "Q&A turns (one call)": [t for t in turns if not t.usage.get("reply_seconds")],
        "Reply-pass turns (listings / tools)": [t for t in turns if t.usage.get("reply_seconds")],
        "All turns": turns,
    }
    lines = [
        "",
        "## Performance",
        "",
        "| Turns | n | Latency median | p90 | max | Analysis call median | Reply pass median "
        "| Input tokens / turn | Output tokens / turn | Cached share of input |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, group in groups.items():
        if not group:
            continue
        secs = [t.seconds for t in group]
        analysis = [t.usage.get("analysis_seconds", 0.0) for t in group]
        reply = [t.usage.get("reply_seconds", 0.0) for t in group if t.usage.get("reply_seconds")]
        inp = sum(t.usage.get("prompt_tokens", 0) for t in group)
        out = sum(t.usage.get("completion_tokens", 0) for t in group)
        cached = sum(t.usage.get("cached_tokens", 0) for t in group)
        lines.append(
            f"| {name} | {len(group)} | {_pct(secs, 0.5):.1f}s | {_pct(secs, 0.9):.1f}s "
            f"| {max(secs):.1f}s | {_pct(analysis, 0.5):.1f}s "
            f"| {(f'{_pct(reply, 0.5):.1f}s' if reply else '-')} "
            f"| {inp / len(group):,.0f} | {out / len(group):,.0f} "
            f"| {(cached / inp * 100 if inp else 0):.0f}% |"
        )
    return lines


def _write_log(results: list[Result], path: Path, started: datetime, seconds: float) -> None:
    passed = sum(r.passed for r in results)
    turns = sum(len(r.turns) for r in results)
    lines = [
        f"# Luna scenario run - {started:%Y-%m-%d %H:%M}",
        "",
        f"{passed}/{len(results)} conversations passed every check - {turns} turns in "
        f"{seconds / 60:.1f} min. Real model, real database. The per-category customers decline "
        "contact; the scripted email scenarios give it, so their notifications were sent.",
        "",
        "| Category | Scenario | Result | Turns | Failed checks |",
        "|---|---|---|---|---|",
    ]
    lines += _performance(results)
    for r in results:
        failed = "; ".join(f"{name} ({detail})" for name, ok, detail in r.checks if not ok) or (r.error or "")
        lines.append(f"| {r.category} | {r.scenario} | {'PASS' if r.passed else 'FAIL'} | {len(r.turns)} | {failed} |")
    for r in results:
        lines += ["", "---", "", f"## {r.category} - {r.scenario} - {'PASS' if r.passed else 'FAIL'}",
                  "", f"Session `{r.session_id}`", ""]
        if r.error:
            lines += [f"**Error:** {r.error}", ""]
        for name, ok, detail in r.checks:
            lines.append(f"- {'✅' if ok else '❌'} {name}" + (f" - {detail}" if detail else ""))
        lines.append("")
        for number, t in enumerate(r.turns, 1):
            note = f"  _({t.note})_" if t.note else ""
            lines += [
                f"**{number}. Customer:** {t.user}{note}",
                "",
                "**Bot** " + f"({t.seconds:.1f}s):",
                "",
                *[f"> {line}" if line else ">" for line in t.bot.splitlines()],
                "",
                f"<sub>next question: `{t.pending}` · slots: `{json.dumps(t.slots, default=str)}`</sub>",
                "",
            ]
        lines += ["**Final state:**", "", "```json", json.dumps(r.final, indent=2, default=str), "```"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8801"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", default="", help="comma-separated categories")
    parser.add_argument("--scenarios", default="happy,units,vague",
                        help="per-category scenarios; pass an empty string for none")
    parser.add_argument("--scripted", default="",
                        help="scripted groups to add: rules,category,email or all")
    parser.add_argument("--match", default="",
                        help="comma-separated words; keep only scripted conversations whose name has one")
    args = parser.parse_args()

    health = httpx.get(f"{args.base_url}/health", timeout=10).json()
    if not health.get("persistence"):
        print("The backend is not persisting sessions - state cannot be read back. Aborting.")
        return 2

    categories = [c for c in CATEGORIES if not args.only or c in args.only.split(",")]
    scenarios = [s for s in args.scenarios.split(",") if s]
    jobs: list[Any] = [(c, s, i) for i, c in enumerate(categories) for s in scenarios]
    groups = {"rules", "category", "email", "flow", "features", "axles"} if args.scripted == "all" else set(filter(None, args.scripted.split(",")))
    words = [w.strip().lower() for w in args.match.split(",") if w.strip()]
    jobs += [script for script in SCRIPTS if script.group in groups
             and (not words or any(w in script.name.lower() for w in words))]
    started = datetime.now()
    print(f"Running {len(jobs)} conversations against {args.base_url} ({health.get('model')})")
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda job: run_script(args.base_url, job, lock) if isinstance(job, Script)
            else run_one(args.base_url, *job, lock),
            jobs,
        ))
    path = ROOT / "logs" / f"scenario_run_{started:%Y-%m-%d_%H%M}.md"
    _write_log(results, path, started, time.time() - t0)
    print(f"\n{sum(r.passed for r in results)}/{len(results)} passed. Log: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
