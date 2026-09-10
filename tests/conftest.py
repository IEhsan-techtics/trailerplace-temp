"""Shared fixtures. Nothing here touches the network or the database.

The single-LLM-call design is what makes this possible: every business rule lives in
deterministic Python downstream of one structured output, so a test supplies that output
directly and asserts on the state that results.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain import brands  # noqa: E402


# A small, fixed stand-in for what load_make_inventory() reads out of trailer_listings.
# Real makes and real categories, so a test asserting "we never advertise a brand we do not
# stock" is asserting something meaningful, but frozen so the suite does not change
# behaviour when the lot does.
FIXTURE_MAKES = {
    "Load Trail": ("Dump", "Equipment", "Flatbed", "Utility"),
    "PJ Trailers": ("Car Hauler", "Dump", "Equipment", "Tilt"),
    "Big Tex": ("Car Hauler", "Equipment", "Utility"),
    "Cargo Craft": ("Enclosed",),
    "Diamond C": ("Dump", "Equipment", "Tilt"),
    # Really in trailer_listings, on four Livestock rows: a hitch type sitting in the
    # manufacturer column. Kept in the fixture because the ambiguity it creates is one the
    # chatbot has to handle, and a fixture without it tests a catalogue we do not have.
    "Gooseneck": ("Livestock",),
}


@pytest.fixture(autouse=True)
def stub_make_inventory(monkeypatch):
    """Keep the brand tables off the live Azure database.

    ``brands.load_make_inventory`` is lru_cached and falls back to listings.xlsx, so an
    un-stubbed unit test either opens a Postgres connection over the network or parses a
    workbook - one test took 6.2 s doing exactly that before this existed.
    """
    inventory = brands.MakeInventory(
        canonical_makes=tuple(sorted(FIXTURE_MAKES)),
        categories_by_make=dict(FIXTURE_MAKES),
        filter_values_by_make={make: (make,) for make in FIXTURE_MAKES},
    )
    # Cleared BEFORE patching, not after: this fixture is torn down ahead of monkeypatch,
    # so a teardown call would land on the plain lambda, which has no cache_clear.
    brands.load_make_inventory.cache_clear()
    monkeypatch.setattr(brands, "load_make_inventory", lambda: inventory)


@pytest.fixture
def no_database(monkeypatch):
    """Force the persistence layer onto its in-memory fallback."""
    from src import db

    monkeypatch.setattr(db, "database_enabled", lambda: False)


@pytest.fixture(autouse=True)
def memory_store(monkeypatch):
    """Every test runs against the in-memory conversation store, never Postgres."""
    from src import conversation_store

    conversation_store.reset_memory()
    monkeypatch.setattr(conversation_store, "persistence_enabled", lambda: False)
    yield
    conversation_store.reset_memory()


@pytest.fixture
def fake_llm(monkeypatch):
    """Script the single model call.

    Returns a recorder whose ``queue`` holds the outputs to hand back in order, and whose
    ``calls`` counts invocations - which is how the one-call-per-turn rule is asserted.
    """
    from src.graph import build as build_module
    from src.llm import usage

    class Recorder:
        def __init__(self):
            self.queue = []
            self.calls = 0
            self.states_seen = []

        def push(self, output):
            self.queue.append(output)
            return output

        def __call__(self, state, user_message):
            self.calls += 1
            self.states_seen.append(dict(state))
            # Counted exactly as the real client does, so usage assertions are meaningful.
            usage.record_completion("gpt-5.6-luna", 100, 20, purpose="chat_turn")
            if self.queue:
                return self.queue.pop(0)
            from src.llm.client import empty_output

            return empty_output("no scripted output")

    recorder = Recorder()
    monkeypatch.setattr(build_module, "analyze_turn", recorder)
    return recorder


@pytest.fixture(autouse=True)
def no_search(monkeypatch):
    """Keep search_node off the live catalogue; record what it was asked for.

    Autouse on purpose: any turn that completes qualification calls search_node, so a test
    that merely answers every question would otherwise open a connection to Azure. Tests
    that care about the search request take the fixture by name and read the list.
    """
    from src.graph import build as build_module

    calls = []

    def _fake_search(state):
        calls.append({"category": state.get("category"), "slots": dict(state.get("slots") or {})})
        outcome = state.setdefault("turn_outcome", {})
        outcome["search_ran"] = True
        outcome["listings"] = [
            {"title": "2026 P&amp;C Car Hauler", "url": "https://x/1", "price_display": "$7,995"},
            {"title": "2025 Diamond C Dump", "url": "https://x/2", "price_display": "$12,500"},
        ]
        outcome["result_count"] = 2
        return state

    monkeypatch.setattr(build_module, "search_node", _fake_search)
    return calls
