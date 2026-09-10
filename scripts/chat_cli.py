"""Terminal REPL against run_turn(). Real model calls, real search.

    .venv/Scripts/python.exe scripts/chat_cli.py

Commands: /state shows the session state, /new starts a fresh session, /quit exits.
"""
from __future__ import annotations

import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import conversation_store  # noqa: E402
from src.graph.build import run_turn  # noqa: E402
from src.graph.state import from_snapshot  # noqa: E402


def _show_state(session_id: str) -> None:
    snapshot, _conversation, lead_id = conversation_store.load_session(session_id)
    state = from_snapshot(session_id, snapshot)
    print(f"  session   {session_id}")
    print(f"  lead      {lead_id}")
    print(f"  category  {state.get('category')}")
    print(f"  slots     {state.get('slots')}")
    print(f"  asked     {state.get('asked_counts')}")
    print(f"  declined  {state.get('declined_slots')}")
    print(f"  pending   {state.get('pending_slot')}")
    print(f"  contact   {state.get('contact')}")
    print(f"  complete  {state.get('qualification_complete')}")


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"), format="%(levelname)s %(message)s")
    session_id = str(uuid.uuid4())

    print("TrailerPlace chatbot. /state, /new, /quit\n")
    while True:
        try:
            message = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not message:
            continue
        if message in {"/quit", "/exit"}:
            return
        if message == "/state":
            _show_state(session_id)
            continue
        if message == "/new":
            session_id = str(uuid.uuid4())
            print(f"  new session {session_id}")
            continue

        result = run_turn(session_id, message)
        print(f"\nbot > {result['assistant_text']}\n")
        usage = result.get("usage") or {}
        print(
            f"      [{usage.get('chat_completions')} call, "
            f"{usage.get('total_tokens')} tokens, "
            f"category={result.get('category')}, "
            f"complete={result.get('qualification_complete')}]\n"
        )


if __name__ == "__main__":
    main()
