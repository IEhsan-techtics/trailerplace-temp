"""The admin API a control panel uses to edit the question rules.

Every write replaces the WHOLE document with a new version - there is no endpoint that
edits one rule in place. That keeps the server simple (validate, save, activate) and gives
the panel history and rollback for free.

Guarded by one shared token (``ADMIN_API_TOKEN``, sent as ``X-Admin-Token``). With no
token configured, main.py does not mount this router at all, so an unconfigured deploy
has no admin surface to probe.
"""
from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.domain import slot_map
from src.domain.categories import CANONICAL_CATEGORIES
from src.rules import store
from src.rules.models import ACTIONS, CONDITION_KEYS, FALLBACK_CATEGORY, POSITIONS, validate_document


def _require_token(x_admin_token: str | None = Header(default=None)) -> None:
    expected = settings.admin_api_token
    if not expected or not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="invalid admin token")


router = APIRouter(prefix="/admin/rules", tags=["admin"], dependencies=[Depends(_require_token)])


class SaveRequest(BaseModel):
    document: dict[str, Any]
    note: str | None = Field(default=None, max_length=2000)
    author: str | None = Field(default=None, max_length=255)


class ValidateRequest(BaseModel):
    document: dict[str, Any]


@router.get("")
def get_active_rules() -> dict[str, Any]:
    """The document the bot is serving right now, and where it came from."""
    version = store.rules_version()
    return {
        "version": version,
        "source": "seed" if version == store.SEED_VERSION else "database",
        "document": store.current_rules().model_dump(mode="json"),
    }


@router.get("/catalog")
def get_catalog() -> dict[str, Any]:
    """Everything a panel needs to build its dropdowns."""
    rules = store.current_rules()
    slots = sorted(rules.known_slots())
    return {
        "categories": list(CANONICAL_CATEGORIES),
        "fallback_category": FALLBACK_CATEGORY,
        # kind None means free text; anything else is parsed like a customer's answer.
        "slots": [{"slot": slot, "kind": slot_map.slot_value_kind(slot)} for slot in slots],
        "traits": [{"key": trait.key, "label": trait.label} for trait in rules.cargo_traits],
        "actions": list(ACTIONS),
        "positions": list(POSITIONS),
        "conditions": list(CONDITION_KEYS),
    }


@router.post("/validate")
def validate(body: ValidateRequest) -> dict[str, Any]:
    _doc, errors = validate_document(body.document)
    return {"valid": not errors, "errors": errors}


@router.put("")
def save(body: SaveRequest) -> dict[str, Any]:
    """Validate, store as a new version, and make it live."""
    doc, errors = validate_document(body.document)
    if doc is None:
        raise HTTPException(status_code=422, detail={"errors": errors})
    try:
        version = store.save_version(doc, note=body.note, author=body.author)
    except store.RulesStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"version": version, "active": True}


@router.get("/versions")
def versions() -> dict[str, Any]:
    return {
        "active_version": store.rules_version(),
        "versions": [
            {
                "version": info.version,
                "is_active": info.is_active,
                "note": info.note,
                "created_by": info.created_by,
                "created_at": info.created_at.isoformat() if info.created_at else None,
            }
            for info in store.list_versions()
        ],
    }


@router.get("/versions/{version}")
def version_document(version: int) -> dict[str, Any]:
    raw = store.get_version(version)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"version {version} not found")
    _doc, errors = validate_document(raw)
    return {"version": version, "document": raw, "valid": not errors, "errors": errors}


@router.post("/versions/{version}/activate")
def activate(version: int) -> dict[str, Any]:
    """Roll back (or forward) to a stored version."""
    try:
        store.activate(version)
    except store.RulesStoreError as exc:
        status = 404 if "does not exist" in str(exc) else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return {"version": version, "active": True}
