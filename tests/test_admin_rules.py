"""The admin API a control panel attaches to. Store SQL runs on in-memory SQLite."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import admin_rules
from src.db_models import ChatbotQuestionRules
from src.rules import store

TOKEN = {"X-Admin-Token": "s3cret"}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(admin_rules, "settings", replace(admin_rules.settings, admin_api_token="s3cret"))
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ChatbotQuestionRules.__table__.create(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(store, "_enabled", lambda: True)
    monkeypatch.setattr(store, "_sessions", lambda: factory)
    app = FastAPI()
    app.include_router(admin_rules.router)
    return TestClient(app)


def seed_raw():
    return json.loads(store.SEED_PATH.read_text(encoding="utf-8"))


def without_width_rule():
    raw = seed_raw()
    next(r for r in raw["rules"] if r["id"] == "ask-width-large-cargo")["enabled"] = False
    return raw


def test_mounted_only_when_a_token_is_configured():
    import main

    mounted = any(getattr(route, "path", "").startswith("/admin") for route in main.app.routes)
    assert mounted == bool(main.settings.admin_api_token)


def test_a_missing_or_wrong_token_is_refused(api):
    assert api.get("/admin/rules").status_code == 401
    assert api.get("/admin/rules", headers={"X-Admin-Token": "nope"}).status_code == 401


def test_an_empty_table_reports_the_seed(api):
    body = api.get("/admin/rules", headers=TOKEN).json()
    assert body["version"] == 0 and body["source"] == "seed"
    assert "Utility" in body["document"]["categories"]


def test_the_catalog_lists_what_a_form_needs(api):
    body = api.get("/admin/rules/catalog", headers=TOKEN).json()
    assert "Flatbed" in body["categories"]
    assert {"slot": "width", "kind": "width_ft"} in body["slots"]
    assert {"slot": "haul_item", "kind": None} in body["slots"]
    assert [t["key"] for t in body["traits"]] == ["lightweight", "large_or_heavy"]
    assert body["actions"] == ["skip_question", "ask_question", "set_default"]


def test_validate_reports_errors_without_saving(api):
    raw = seed_raw()
    raw["rules"][2]["value"] = "wide"
    body = api.post("/admin/rules/validate", json={"document": raw}, headers=TOKEN).json()
    assert body["valid"] is False and any("not a usable value" in e for e in body["errors"])
    assert api.get("/admin/rules/versions", headers=TOKEN).json()["versions"] == []


def test_an_invalid_document_is_refused_with_its_errors(api):
    raw = seed_raw()
    raw["categories"]["Boat"] = {"required": []}
    response = api.put("/admin/rules", json={"document": raw}, headers=TOKEN)
    assert response.status_code == 422
    assert any("unknown category" in e for e in response.json()["detail"]["errors"])


def test_save_then_read_round_trips_and_goes_live(api):
    response = api.put("/admin/rules", json={"document": without_width_rule(), "note": "no width",
                                              "author": "ibrahim"}, headers=TOKEN)
    assert response.status_code == 200 and response.json()["version"] == 1

    body = api.get("/admin/rules", headers=TOKEN).json()
    assert body["version"] == 1 and body["source"] == "database"
    width = next(r for r in body["document"]["rules"] if r["id"] == "ask-width-large-cargo")
    assert width["enabled"] is False
    assert store.rules_version() == 1, "the bot serves it on the next turn"

    versions = api.get("/admin/rules/versions", headers=TOKEN).json()["versions"]
    assert versions[0]["note"] == "no width" and versions[0]["created_by"] == "ibrahim"


def test_rolling_back(api):
    api.put("/admin/rules", json={"document": seed_raw()}, headers=TOKEN)
    api.put("/admin/rules", json={"document": without_width_rule()}, headers=TOKEN)
    assert api.post("/admin/rules/versions/1/activate", headers=TOKEN).status_code == 200
    assert api.get("/admin/rules", headers=TOKEN).json()["version"] == 1
    assert api.post("/admin/rules/versions/99/activate", headers=TOKEN).status_code == 404


def test_reading_one_stored_version(api):
    api.put("/admin/rules", json={"document": seed_raw()}, headers=TOKEN)
    body = api.get("/admin/rules/versions/1", headers=TOKEN).json()
    assert body["valid"] is True and "Utility" in body["document"]["categories"]
    assert api.get("/admin/rules/versions/5", headers=TOKEN).status_code == 404


def test_saving_without_a_database_is_a_clear_error(api, monkeypatch):
    monkeypatch.setattr(store, "_enabled", lambda: False)
    response = api.put("/admin/rules", json={"document": seed_raw()}, headers=TOKEN)
    assert response.status_code == 409
