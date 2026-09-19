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

Every customer declines contact details in their first message, so no escalation, FAQ
or listing-interest email can reach the sales team.

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

    def __init__(self, category: str, scenario: str, index: int) -> None:
        self.category = category
        self.scenario = scenario
        self.index = index
        self.dodged: str | None = None           # the slot dodged twice (vague run)
        self.dodges_used = 0
        self.vague_slot: str | None = None       # the slot answered "whatever works"
        self.expected: dict[str, Any] = {}       # units run: slot -> what should be stored

    def answer(self, slot: str) -> str:
        cargo = CATEGORIES[self.category]["cargo"]
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
        body = response.json()
        state = _state(session_id)
        result.turns.append(
            Turn(text, body.get("assistant_text") or "", time.time() - started,
                 state.get("pending_slot"), dict(state.get("slots") or {}), note)
        )
        return state

    asked_after_answered: list[str] = []
    try:
        say(OPENING)
        state = say(CATEGORIES[category]["ask"])
        for _ in range(MAX_TURNS):
            if state.get("results_shown"):
                break
            if state.get("pending_category_switch"):
                state = say(f"No thanks, I'll stick with a {category} trailer.", "declined a category switch")
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
        result.final = {
            key: state.get(key) for key in (
                "category", "slots", "declined_slots", "asked_counts", "required_slots",
                "rule_skipped", "rule_defaults", "cargo_traits", "results_shown",
            )
        }
        _check(result, customer, state, asked_after_answered)
    except Exception as exc:  # noqa: BLE001 - a broken run is a result, not a crash
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        http.close()
    with log_lock:
        mark = "PASS" if result.passed else "FAIL"
        print(f"  {mark}  {category:<13} {scenario:<6} {len(result.turns):>2} turns", flush=True)
    return result


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


def _write_log(results: list[Result], path: Path, started: datetime, seconds: float) -> None:
    passed = sum(r.passed for r in results)
    turns = sum(len(r.turns) for r in results)
    lines = [
        f"# Luna scenario run - {started:%Y-%m-%d %H:%M}",
        "",
        f"{passed}/{len(results)} conversations passed every check - {turns} turns in "
        f"{seconds / 60:.1f} min. Real model, real database; every customer declined contact.",
        "",
        "| Category | Scenario | Result | Turns | Failed checks |",
        "|---|---|---|---|---|",
    ]
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
    parser.add_argument("--scenarios", default="happy,units,vague")
    args = parser.parse_args()

    health = httpx.get(f"{args.base_url}/health", timeout=10).json()
    if not health.get("persistence"):
        print("The backend is not persisting sessions - state cannot be read back. Aborting.")
        return 2

    categories = [c for c in CATEGORIES if not args.only or c in args.only.split(",")]
    scenarios = args.scenarios.split(",")
    jobs = [(c, s, i) for i, c in enumerate(categories) for s in scenarios]
    started = datetime.now()
    print(f"Running {len(jobs)} conversations against {args.base_url} ({health.get('model')})")
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: run_one(args.base_url, *job, lock), jobs))
    path = ROOT / "logs" / f"scenario_run_{started:%Y-%m-%d_%H%M}.md"
    _write_log(results, path, started, time.time() - t0)
    print(f"\n{sum(r.passed for r in results)}/{len(results)} passed. Log: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
