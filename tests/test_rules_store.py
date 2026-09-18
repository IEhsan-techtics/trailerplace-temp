"""Where the rules come from: the seed, the active DB version, and what happens when that goes wrong.

Runs the store's real SQL against in-memory SQLite - the table's document column is JSON
there and JSONB on Postgres, which is the only difference.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import settings
from src.db_models import ChatbotQuestionRules
from src.rules import store
from src.rules.models import validate_document
from src.rules.store import SEED_PATH, SEED_VERSION, seed_document


def _doc(mutate=lambda raw: None):
    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    doc, errors = validate_document(raw)
    assert doc is not None, errors
    return doc


def _disable_width_rule(raw):
    next(r for r in raw["rules"] if r["id"] == "ask-width-large-cargo")["enabled"] = False


@pytest.fixture
def sqlite_rules(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ChatbotQuestionRules.__table__.create(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(store, "_enabled", lambda: True)
    monkeypatch.setattr(store, "_sessions", lambda: factory)
    return factory


def test_persistence_off_serves_the_seed():
    assert store.current_rules() is seed_document()
    assert store.rules_version() == SEED_VERSION


def test_an_empty_table_serves_the_seed(sqlite_rules):
    assert store.current_rules() is seed_document()
    assert store.rules_version() == SEED_VERSION


def test_a_saved_version_is_served_at_once_by_the_process_that_saved_it(sqlite_rules):
    version = store.save_version(_doc(_disable_width_rule), note="no width", author="test")
    assert version == 1
    assert store.rules_version() == 1
    width = next(r for r in store.current_rules().rules if r.id == "ask-width-large-cargo")
    assert width.enabled is False


def test_another_process_picks_up_a_new_version_after_the_refresh_interval(sqlite_rules):
    object.__setattr__(settings, "rules_refresh_seconds", 3600.0)
    try:
        store.current_rules()                      # seed, and the clock starts
        # Another process saves: straight into the table, no invalidate() here.
        with sqlite_rules() as sql, sql.begin():
            sql.add(ChatbotQuestionRules(document=_doc(_disable_width_rule).model_dump(mode="json"), is_active=True))
        assert store.rules_version() == SEED_VERSION   # still inside the interval
        store.invalidate()                              # the interval running out
        assert store.rules_version() == 1
    finally:
        object.__setattr__(settings, "rules_refresh_seconds", 30.0)


def test_an_invalid_active_document_keeps_the_last_good_one(sqlite_rules):
    store.save_version(_doc(_disable_width_rule))
    assert store.rules_version() == 1
    with sqlite_rules() as sql, sql.begin():
        sql.execute(update(ChatbotQuestionRules).values(is_active=False))
        sql.add(ChatbotQuestionRules(document={"categories": {"Boat": {}}}, is_active=True))
    store.invalidate()
    assert store.rules_version() == 1
    assert next(r for r in store.current_rules().rules if r.id == "ask-width-large-cargo").enabled is False


def test_a_database_error_keeps_the_last_good_document(sqlite_rules, monkeypatch):
    store.save_version(_doc(_disable_width_rule))
    assert store.rules_version() == 1

    def broken():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(store, "_sessions", broken)
    store.invalidate()
    assert store.rules_version() == 1


def test_rolling_back_to_an_earlier_version(sqlite_rules):
    first = store.save_version(seed_document(), note="as shipped")
    second = store.save_version(_doc(_disable_width_rule), note="no width")
    assert store.rules_version() == second
    store.activate(first)
    assert store.rules_version() == first
    versions = store.list_versions()
    assert [(v.version, v.is_active) for v in versions] == [(second, False), (first, True)]


def test_only_one_version_is_ever_active(sqlite_rules):
    store.save_version(seed_document())
    store.save_version(seed_document())
    store.save_version(seed_document(), activate=False)
    assert sum(v.is_active for v in store.list_versions()) == 1
    assert store.rules_version() == 2


def test_activating_a_missing_version_fails(sqlite_rules):
    with pytest.raises(store.RulesStoreError):
        store.activate(99)


def test_saving_needs_the_database():
    with pytest.raises(store.RulesStoreError):
        store.save_version(seed_document())
