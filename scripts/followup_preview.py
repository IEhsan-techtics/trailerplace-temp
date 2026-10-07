"""What would the follow-up agent send? Read-only: it sends nothing and writes nothing.

    python scripts/followup_preview.py [--days 6] [--attempt 1] [--limit 40]

Takes the real Messenger conversations of the last few days that ended on our message, treats
each as due, and prints the model's decision and the follow-up it would write. Timing and the
24-hour window are ignored on purpose, so a week of conversations can be read in one go.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.followup import decide as deciding  # noqa: E402
from src.followup import store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=float, default=6)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    now = store.now_utc()
    candidates = store.find_candidates(now, ignore_timing=True, lookback_hours=args.days * 24)
    print(f"{len(candidates)} quiet Messenger conversations in the last {args.days:g} days\n")
    sent = 0
    for number, candidate in enumerate(candidates[: args.limit], 1):
        candidate.attempt = args.attempt
        decision = deciding.decide(candidate, now)
        sent += decision.send
        users = [m for m in candidate.conversation if m.get("role") == "user"]
        ours = [m for m in candidate.conversation if m.get("role") == "assistant"]
        print("=" * 100)
        print(f"#{number}  {candidate.contact.get('name') or 'unknown'}  |  {len(users)} message(s) from them  "
              f"|  quiet {candidate.hours_silent(now):.0f} h  |  attempt {candidate.attempt}")
        if users:
            print("  THEM (last): " + _short(users[-1].get("content")))
        if ours:
            print("  US   (last): " + _short(ours[-1].get("content")))
        print(f"  -> {'SEND' if decision.send else 'NO FOLLOW-UP'}  [{decision.scenario}]  {decision.reason}")
        if decision.send:
            print(textwrap.indent(decision.message or "", "     | "))
    print("=" * 100)
    print(f"\n{sent} of {min(len(candidates), args.limit)} would get follow-up {args.attempt}.")
    return 0


def _short(text: object, limit: int = 220) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


if __name__ == "__main__":
    sys.exit(main())
