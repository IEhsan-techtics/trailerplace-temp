"""One long conversation, one session, every behaviour the bot has - watched end to end.

    .venv/Scripts/python.exe scripts/long_conversation.py

Not a pass/fail run. ``scenario_run.py`` already checks behaviour one short conversation at
a time, each in its own session. This does the opposite: ONE customer, ONE session, forty-odd
turns, walking through the FAQs, a trailer we do not stock, a recommendation, qualification,
results, more results, three kinds of lookup, a shared Facebook link, a category switch, the
axle questions, a dodged question and a complaint - in the order a real person would hit them.

What that catches and the short runs cannot: state carried between scenarios. A contact ask
that keeps firing after they answered. An email that goes out twice because the earlier
phase already queued it. Filters from the livestock half leaking into the dump half. A reply
that reads fine alone and absurd after the twelve turns before it.

TWO FILES COME OUT, both under logs/:

    long_conversation_<stamp>.txt    the conversation and nothing else - read this first
    long_conversation_<stamp>.log    every log record the turn produced, in order

The transcript carries a turn number and a phase heading, and marks each email where it
happened (sent, or held because we cannot reach them yet). When something in it looks wrong,
grep the .log for `TURN 17` and the whole turn is there: the analysis, the slots, the search,
the outbox row, the send.

MODES

    (default)            in-process, calling run_turn() exactly as the API does. The .log
                         gets everything, because the work happens in this process.
    --base-url URL       POST /chat against a running backend instead. Closer to production,
                         but the bot's own logs are then in the SERVER's console, not here -
                         the .log only holds what this script can see from outside.

EMAILS ARE REAL. Every alert goes to whatever .env names (RECIPIENT_EMAIL / EMAIL_TO), and a
full run sends a dozen or so. Check it points at a test inbox, or pass --no-email to have
them recorded and logged but never sent (in-process mode only).

USEFUL FLAGS

    --list               print the phases and exit
    --phases 1,4,9       run only these (the session still starts empty, so a phase that
                         needs contact details or a shown listing may behave differently)
    --contact-early      give name and number in the first message instead of phase 3
    --no-email           record the alerts, do not send them
    --log-level DEBUG    how much goes in the .log (default DEBUG for src, INFO elsewhere)
    --sql                add every SQL statement to the .log
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("longchat")

# The sentinel for "answer whatever it asks until it shows me trailers".
QUALIFY = "<qualify>"

# Answers the scripted customer gives when the bot asks for a slot. Anything not named here
# falls back to NO_PREFERENCE, which is a legitimate answer and keeps the run moving rather
# than stalling on a slot this script has never seen.
ANSWERS: dict[str, str] = {
    "haul_item": "cattle, usually eight to ten head at a time",
    "payload_capacity": "about 12,000 lbs",
    "length": "24 ft",
    "width": "7 ft wide",
    "height": "7 ft tall",
    "hitch_type": "gooseneck",
    "cargo_size": "about 16 ft long and 6 ft wide",
    "bin_size": "20 yard bins",
    "base_category": "a utility trailer",
    "tank_capacity": "500 gallons",
}
NO_PREFERENCE = "Whatever you'd normally recommend."

# The second half of the conversation is a different trailer for a different job, so the
# answers change with it. Phases carry their own overrides on top of ANSWERS.
DUMP_ANSWERS = {
    "haul_item": "gravel and dirt for a landscaping job",
    "payload_capacity": "around 14,000 lbs",
    "length": "16 ft",
}

# Replaced with the first listing URL the bot has shown so far, wherever it appears in a step.
SHOWN = "{SHOWN}"

TESTER = ("I'm Sam Tester by the way - sam.tester@example.com, "
          "and my number is 555-0100.")


@dataclass
class Phase:
    """One stretch of the conversation, and what it is there to show."""
    title: str
    watch: str                      # what to look at in the reply, printed above the turns
    steps: list[str]
    answers: dict[str, str] = field(default_factory=dict)
    switch: str = "yes"             # how a suggested category switch gets answered here


PHASES: list[Phase] = [
    Phase(
        "Opening, with nobody to reach",
        "the greeting, and the first question it chooses to ask",
        ["Hi there", "Just looking around for now - what sort of trailers do you carry?"],
    ),
    Phase(
        "Two FAQs before we can reach them",
        "both alerts should be HELD, not sent - and the reply should start asking who they are",
        ["Do you offer financing on your trailers?",
         "And what are your opening hours? Where are you located?"],
    ),
    Phase(
        "They give their details",
        "everything held above should go out now, in one go, and the asking should stop",
        [TESTER],
    ),
    Phase(
        "A trailer we do not stock",
        "it should say we do not have them, say the team was told, and list what we DO have",
        ["Do you sell boat trailers?"],
    ),
    Phase(
        "No category named - a recommendation",
        "it should work out Livestock from the job, not ask them to pick from a menu",
        ["Fair enough. I'm hauling cattle to the auction barn most weekends - what would you "
         "recommend for that?"],
    ),
    Phase(
        "Qualifying, then the first results",
        "one question per reply, nothing asked twice, and a Results Shown alert at the end",
        [QUALIFY],
    ),
    Phase(
        "More of the same results",
        "a second batch, different trailers, and a SECOND alert - not a repeat of the first",
        ["Can you show me a few more like those?"],
    ),
    Phase(
        "Lookup: a stock number",
        "one card, the right one, with its price - and an alert saying results were shown",
        ["Quick question - is stock number 13779 still available?"],
    ),
    Phase(
        "Lookup: a make and model, then a year and make",
        "both should resolve to real stock; the second is a search, not a single card",
        ["How much is the Aluma 8220H XL tilt?",
         "Do you have any 2026 Galyean trailers?"],
    ),
    Phase(
        "Lookup: a stock number that does not exist",
        "it must say so plainly and invent nothing - no price, no card",
        ["What about stock number 99999?"],
    ),
    Phase(
        "Interest in one they were shown",
        "a Listing Interest alert naming that URL, and a callback request for the team",
        ["Let's go back to the cattle trailers. I really like this one: " + SHOWN,
         "Can someone call me tomorrow morning about it?"],
    ),
    Phase(
        "A link from Facebook",
        "an alert that says Facebook post link - we cannot price it, and must not pretend to",
        ["Also, I saw this one on Facebook - how much is it? "
         "https://www.facebook.com/share/p/1AbCdEfGh/"],
    ),
    Phase(
        "They change trailer entirely",
        "it should ask what to keep, and NOT carry the livestock answers into a dump trailer",
        ["Different subject - I also need a dump trailer for a landscaping job.",
         "Just keep the length, drop the rest."],
        answers=DUMP_ANSWERS,
    ),
    Phase(
        "An axle rating that could mean two things",
        "per axle or total, then how many - and the load-weight question should be skipped",
        ["It needs 14,000 lbs of axle capacity.", "Per axle.", "Tandem.", QUALIFY],
        answers=DUMP_ANSWERS,
    ),
    Phase(
        "A question dodged twice",
        "asked at most twice, then dropped for good - it must not keep chasing it",
        ["Why do you need to know that?", "Hmm, no idea. What colours do these come in?",
         QUALIFY],
        answers=DUMP_ANSWERS,
    ),
    Phase(
        "A complaint",
        "an Escalation alert, and a reply that does not argue with them",
        ["Something else - I bought a trailer from you last month and the brakes already "
         "failed. I'm pretty unhappy about it."],
    ),
    Phase(
        "The rest of the FAQs",
        "a trade-in, a delivery quote and a human - three different alerts, no repeats",
        ["Can I trade in my old utility trailer against one of these?",
         "Can you deliver to Houston? I'd want a quote with delivery included.",
         "Actually, can I just speak to a real person?"],
    ),
    Phase(
        "Goodbye",
        "it should close cleanly and not open a new question on the way out",
        ["That's everything for now, thanks."],
    ),
]


# --------------------------------------------------------------------------- the two files
class Transcript:
    """The readable file. Written and flushed as it goes, so a crash still leaves a record."""

    def __init__(self, path: Path, session_id: str, started: datetime, mode: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8")
        self._write(
            "=" * 78,
            f" LUNA - one long conversation - {started:%Y-%m-%d %H:%M}",
            f" session {session_id}",
            f" {mode}",
            "=" * 78,
            "",
            " Every turn is numbered. If a reply looks wrong, search the .log file beside",
            " this one for 'TURN <n>' and the whole turn is there.",
            "",
        )

    def phase(self, number: int, phase: Phase) -> None:
        self._write("", "", "-" * 78,
                    f" PHASE {number}. {phase.title}",
                    f" watch for: {phase.watch}",
                    "-" * 78, "")

    def turn(self, number: int, user: str, bot: str, seconds: float, note: str) -> None:
        self._write(f"[turn {number}]" + (f"  ({note})" if note else ""),
                    f"YOU:  {user}",
                    "")
        for line in (bot or "(no reply)").splitlines():
            self._write(f"LUNA: {line}" if line else "LUNA:")
        self._write("", f"      ({seconds:.1f}s)", "")

    def email(self, line: str) -> None:
        self._write(f"      >>> {line}")

    def note(self, line: str) -> None:
        self._write(f"      --- {line}")

    def close(self, summary: list[str]) -> None:
        self._write("", "", "=" * 78, " SUMMARY", "=" * 78, "", *summary, "")
        self.handle.close()

    def _write(self, *lines: str) -> None:
        self.handle.write("\n".join(lines) + "\n")
        self.handle.flush()


def _start_logging(path: Path, level: str, sql: bool) -> None:
    """Everything, in order, in one file - and a quiet line or two on the console.

    Third-party loggers are pinned whatever the level: httpcore at DEBUG prints every HTTP
    frame of every model call, and SQLAlchemy prints every statement twice over. Both bury
    the turn in noise that is almost never what went wrong. --sql brings the queries back.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    to_file = logging.FileHandler(path, encoding="utf-8")
    to_file.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    to_file.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-34s %(message)s", datefmt="%H:%M:%S"
    ))
    root.addHandler(to_file)

    to_console = logging.StreamHandler(sys.stdout)
    to_console.setLevel(logging.WARNING)
    to_console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(to_console)

    for noisy in ("httpcore", "httpx", "urllib3", "openai", "asyncio", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO if sql else logging.WARNING)


# ------------------------------------------------------------------------------ the emails
class Mailbox:
    """What the team was told, and whether it actually left the building.

    Read from ``chatbot_outbox`` for this session, so it is the real record and not this
    script's idea of one. Each poll returns only what is new since the last.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.seen: set[str] = set()
        self.enabled = True
        try:
            from src import conversation_store
            self.enabled = conversation_store.persistence_enabled()
        except Exception:  # noqa: BLE001 - no database is a reason to say so, not to stop
            self.enabled = False

    def new_rows(self, wait: float = 20.0) -> list[dict[str, Any]]:
        """Rows added since the last call, waited on until none are still pending."""
        if not self.enabled:
            return []
        deadline = time.time() + wait
        while True:
            rows = self._rows()
            fresh = [r for r in rows if r["id"] not in self.seen]
            if not fresh or all(r["status"] != "pending" for r in fresh) or time.time() > deadline:
                self.seen.update(r["id"] for r in fresh)
                return fresh
            time.sleep(2)

    def _rows(self) -> list[dict[str, Any]]:
        from src import db
        from src.db_models import ChatbotOutbox

        try:
            with db.get_session_factory()() as sql:
                return [
                    {"id": str(r.event_id), "event_type": r.event_type, "status": r.status,
                     "subject": (r.payload or {}).get("subject") or "",
                     "body": (r.payload or {}).get("body") or "",
                     "error": r.last_error or ""}
                    for r in sql.query(ChatbotOutbox)
                    .filter(ChatbotOutbox.session_id == uuid.UUID(self.session_id))
                    .order_by(ChatbotOutbox.created_at)
                ]
        except Exception:  # noqa: BLE001
            log.exception("Could not read the outbox - the emails below may be incomplete")
            return []


def _silence_email() -> list[tuple[str, str]]:
    """Record what would have been sent instead of sending it. Returns the growing list."""
    from src.tools import email_sender

    sent: list[tuple[str, str]] = []

    def _fake(subject: str, body: str) -> bool:
        sent.append((subject, body))
        log.info("EMAIL NOT SENT (--no-email) | subject=%r\n%s", subject, body)
        return True

    email_sender.send_email = _fake  # type: ignore[assignment]
    return sent


# -------------------------------------------------------------------------------- the turn
def _make_say(mode: str, base_url: str, session_id: str) -> Callable[[str], dict[str, Any]]:
    """One function that sends a message, whichever end of the app we are talking to."""
    if mode == "http":
        import httpx

        client = httpx.Client(timeout=300)

        def say_http(text: str) -> dict[str, Any]:
            response = client.post(f"{base_url}/chat",
                                   json={"message": text, "session_id": session_id})
            response.raise_for_status()
            return response.json()

        return say_http

    from src.graph.build import run_turn

    def say_local(text: str) -> dict[str, Any]:
        return run_turn(session_id, text)

    return say_local


def _settle(mode: str) -> None:
    """Wait for the turn to be on disk before reading the state back.

    The commit runs off the reply path (src/turn_saver.py), so without this the state read
    below can be a turn behind and the script answers a question the bot has moved past.
    """
    if mode == "http":
        time.sleep(1.5)     # the server's saver is in the server; this is the best we can do
        return
    from src import turn_saver
    turn_saver.drain(timeout=30)


def _state(session_id: str) -> dict[str, Any]:
    from src.conversation_store import load_session
    from src.graph.state import from_snapshot

    snapshot, _conversation, _lead = load_session(session_id)
    return from_snapshot(session_id, snapshot)


# The questions that are not slots. Each is a state key the bot sets when it is waiting on
# something, and the answer this scripted customer gives to it.
def _pending_question(state: dict[str, Any], switch: str) -> tuple[str, str] | None:
    if state.get("pending_category_switch"):
        suggested = (state["pending_category_switch"] or {}).get("suggested")
        if switch == "yes":
            return f"Yes, a {suggested} sounds right - let's do that.", f"accepting a switch to {suggested}"
        return f"No, I'll stick with the {state.get('category')} trailer.", f"declining a switch to {suggested}"
    if state.get("pending_keep_filters"):
        return "Keep what I've already told you.", "keep-filters question"
    if state.get("pending_gooseneck_clarification"):
        return "I meant the gooseneck hitch.", "gooseneck clarification"
    if state.get("pending_axle_basis"):
        return "That's per axle.", "per axle or total"
    if state.get("pending_axle_count"):
        return "Tandem.", "how many axles"
    return None


def _answer_for(slot: str, answers: dict[str, str]) -> str:
    return answers.get(slot) or ANSWERS.get(slot) or NO_PREFERENCE


# --------------------------------------------------------------------------------- the run
@dataclass
class Turn:
    number: int
    phase: int
    user: str
    bot: str
    seconds: float
    note: str
    state: dict[str, Any]
    usage: dict[str, Any]
    listings: list[str]
    emails: list[dict[str, Any]]


def run(args: argparse.Namespace) -> int:
    session_id = args.session or str(uuid.uuid4())
    started = datetime.now()
    stamp = f"{started:%Y-%m-%d_%H%M}"
    logs = ROOT / "logs"
    _start_logging(logs / f"long_conversation_{stamp}.log", args.log_level, args.sql)

    mode = "http" if args.base_url else "local"
    where = (f"POST /chat at {args.base_url} - the bot's own logs are in the SERVER console"
             if mode == "http" else "in-process run_turn() - every log below is this turn's")
    if args.no_email and mode == "local":
        _silence_email()
        where += " | --no-email: alerts recorded, not sent"

    if mode == "local":
        from src import warmup
        log.info("WARMING UP")
        warmup.warm_everything()

    transcript = Transcript(logs / f"long_conversation_{stamp}.txt", session_id, started, where)
    say = _make_say(mode, args.base_url, session_id)
    mailbox = Mailbox(session_id)
    if not mailbox.enabled:
        transcript.note("No database: the emails below cannot be read back from the outbox.")

    phases = [(i, p) for i, p in enumerate(PHASES, 1)
              if not args.phases or i in args.phases]
    print(f"Session {session_id}")
    print(f"{len(phases)} phases, {mode} mode. Watch: {transcript.path}\n")

    turns: list[Turn] = []
    state: dict[str, Any] = {}
    number = 0
    first_url = ""

    def one_turn(text: str, phase_number: int, note: str = "") -> dict[str, Any]:
        """Send one message, record everything about it, return the state after it."""
        nonlocal number, state, first_url
        number += 1
        log.info("=" * 100)
        log.info("TURN %d  (phase %d)  <<< %s", number, phase_number, text)
        log.info("=" * 100)

        clock = time.time()
        body = say(text)
        seconds = time.time() - clock
        reply = body.get("assistant_text") or ""
        listings = [str(item.get("url") or "") for item in body.get("listings") or []
                    if isinstance(item, dict)]
        first_url = first_url or next((u for u in listings if u), "")

        _settle(mode)
        state = _state(session_id)
        emails = mailbox.new_rows()

        log.info("TURN %d  >>> %s", number, reply)
        log.info("TURN %d  listings: %s", number, listings or "none")
        log.info("TURN %d  usage: %s", number, json.dumps(body.get("usage") or {}, default=str))
        log.info("TURN %d  state: %s", number, json.dumps(_interesting(state), indent=2, default=str))
        for row in emails:
            log.info("TURN %d  EMAIL %s [%s] %s\n%s",
                     number, row["status"], row["event_type"], row["subject"], row["body"])

        transcript.turn(number, text, reply, seconds, note)
        held = state.get("pending_email_actions") or []
        for row in emails:
            mark = {"sent": "EMAIL SENT"}.get(row["status"], f"EMAIL {row['status'].upper()}")
            transcript.email(f"{mark} [{row['event_type']}] {row['subject']}")
            first_line = next((l for l in row["body"].splitlines() if l.strip()), "")
            if first_line:
                transcript.email(f"    {first_line.strip()}")
            if row["error"]:
                transcript.email(f"    error: {row['error']}")
        if held:
            reasons = ", ".join(sorted({str(a.get("reason") or "?") for a in held}))
            transcript.email(f"EMAIL HELD, no way to reach them yet ({len(held)}): {reasons}")
        if listings:
            transcript.note(f"{len(listings)} trailer(s) shown")

        turns.append(Turn(number, phase_number, text, reply, seconds, note,
                          _interesting(state), body.get("usage") or {}, listings, emails))
        return state

    try:
        for index, phase in phases:
            log.info("PHASE %d: %s", index, phase.title)
            transcript.phase(index, phase)
            print(f"  phase {index}. {phase.title}", flush=True)

            for step in phase.steps:
                if step == QUALIFY:
                    _qualify(one_turn, state, phase, index, args.max_qualify_turns)
                    continue
                if index == 1 and args.contact_early and step == phase.steps[0]:
                    step = f"{step}. {TESTER}"
                if SHOWN in step:
                    step = step.replace(SHOWN, first_url or "(nothing has been shown yet)")
                state = one_turn(step, index)
    except KeyboardInterrupt:
        log.warning("Stopped by hand after turn %d", number)
        transcript.note("stopped by hand")
    except Exception as exc:  # noqa: BLE001 - the transcript so far is worth keeping
        log.exception("The run stopped on turn %d", number)
        transcript.note(f"STOPPED: {type(exc).__name__}: {exc}")

    if mode == "local":
        from src import turn_saver
        turn_saver.drain(timeout=60)
    late = mailbox.new_rows(wait=30)
    for row in late:
        transcript.email(f"AFTER THE LAST TURN: {row['status']} [{row['event_type']}] {row['subject']}")

    transcript.close(_summary(turns, session_id, state))
    log.info("FINAL STATE\n%s", json.dumps(state, indent=2, default=str))
    print(f"\n{number} turns. Transcript: {transcript.path}")
    print(f"Full log:  {transcript.path.with_suffix('.log')}")
    return 0


def _qualify(one_turn, state: dict[str, Any], phase: Phase, index: int, budget: int) -> None:
    """Answer whatever it asks until it runs out of questions or shows trailers."""
    for _ in range(budget):
        other = _pending_question(state, phase.switch)
        if other:
            state = one_turn(other[0], index, other[1])
            continue
        slot = state.get("pending_slot")
        if not slot:
            log.info("QUALIFY: no question left to answer")
            return
        state = one_turn(_answer_for(slot, phase.answers), index, f"answering {slot}")
    log.warning("QUALIFY: gave up after %d turns, it is still asking", budget)


_WATCH_KEYS = (
    "category", "slots", "required_slots", "pending_slot", "asked_counts", "declined_slots",
    "rule_skipped", "rule_defaults", "cargo_traits", "non_metadata_features",
    "pending_category_switch", "pending_keep_filters", "pending_axle_basis",
    "pending_axle_count", "results_shown", "unavailable_requests", "pending_email_actions",
    "contact", "qualification_complete", "turn_index",
)


def _interesting(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state.get(key) for key in _WATCH_KEYS if state.get(key) not in (None, [], {})}


def _summary(turns: list[Turn], session_id: str, state: dict[str, Any]) -> list[str]:
    """A few lines at the foot of the transcript - the things worth checking at a glance."""
    if not turns:
        return ["nothing ran"]
    emails = [row for t in turns for row in t.emails]
    kinds: dict[str, int] = {}
    for row in emails:
        kinds[row["event_type"]] = kinds.get(row["event_type"], 0) + 1
    seconds = [t.seconds for t in turns]
    two_questions = [t.number for t in turns if "http" not in t.bot and t.bot.count("?") > 1]
    over_asked = {s: n for s, n in (state.get("asked_counts") or {}).items() if n > 2}

    lines = [
        f"{len(turns)} turns, {sum(seconds):.0f}s total, {sum(seconds) / len(seconds):.1f}s "
        f"average, {max(seconds):.1f}s slowest (turn {max(turns, key=lambda t: t.seconds).number}).",
        f"{len(emails)} email(s) to the team:",
    ]
    lines += [f"    {count} x {kind}" for kind, count in sorted(kinds.items())] or ["    none"]
    lines += [
        "",
        f"Trailers shown on turns: {', '.join(str(t.number) for t in turns if t.listings) or 'none'}",
        f"Replies with more than one question: {two_questions or 'none'}",
        f"Questions asked more than twice: {over_asked or 'none'}",
        f"Anything still held, unsent: {len(state.get('pending_email_actions') or [])}",
        f"Ended on category {state.get('category')!r}, "
        f"slots {json.dumps(state.get('slots') or {}, default=str)}",
        "",
        f"Session {session_id} - open it in the UI with ?chat_session={session_id}",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="",
                        help="POST /chat here instead of running in-process")
    parser.add_argument("--session", default="", help="carry on an existing session id")
    parser.add_argument("--phases", default="", help="comma-separated phase numbers")
    parser.add_argument("--list", action="store_true", help="print the phases and exit")
    parser.add_argument("--contact-early", action="store_true",
                        help="give contact details in the first message")
    parser.add_argument("--no-email", action="store_true",
                        help="record the alerts, never send them (in-process only)")
    parser.add_argument("--log-level", default="DEBUG")
    parser.add_argument("--sql", action="store_true", help="log every SQL statement too")
    parser.add_argument("--max-qualify-turns", type=int, default=10)
    args = parser.parse_args()

    if args.list:
        for number, phase in enumerate(PHASES, 1):
            print(f"{number:>3}. {phase.title}")
            print(f"     watch: {phase.watch}")
            print(f"     {len(phase.steps)} step(s)")
        return 0

    args.phases = {int(p) for p in args.phases.replace(" ", "").split(",") if p}
    if args.no_email and args.base_url:
        print("--no-email only works in-process; the server does its own sending. Dropping it.")
        args.no_email = False
    if args.base_url:
        import httpx

        health = httpx.get(f"{args.base_url}/health", timeout=20).json()
        if not health.get("persistence"):
            print("That backend is not persisting sessions, so nothing can be read back.")
            return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
