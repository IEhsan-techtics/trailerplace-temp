"""Where the live rules document comes from, and where the admin API saves new ones.

The active row of ``chatbot_question_rules`` when persistence is on, otherwise - and
whenever that row is missing, unreadable or invalid - the ``seed.json`` shipped with the
code. A bad document in the DB never takes the bot down: it logs, and the last good
document (or the seed) keeps serving.

Reading is cheap on purpose. A turn asks for the rules several times, so the document is
held in memory and the DB is only asked "which version is active?" once every
``RULES_REFRESH_SECONDS``; the document itself is fetched only when that number changes.
A save through this module resets the clock, so the process that saved sees it at once.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from src.config import settings
from src.rules.models import RulesDocument, validate_document

logger = logging.getLogger(__name__)

SEED_PATH = Path(__file__).with_name("seed.json")
# The version number the seed reports. Saved versions start at 1.
SEED_VERSION = 0


class RulesStoreError(RuntimeError):
    """A write could not be made - the DB is off, or the version does not exist."""


@dataclass(frozen=True)
class VersionInfo:
    version: int
    is_active: bool
    note: str | None
    created_by: str | None
    created_at: datetime | None


@lru_cache(maxsize=1)
def seed_document() -> RulesDocument:
    doc, errors = validate_document(json.loads(SEED_PATH.read_text(encoding="utf-8")))
    if doc is None:
        # The seed is code, reviewed and tested - this only happens to a broken checkout.
        raise RuntimeError(f"src/rules/seed.json is invalid: {errors}")
    return doc


# ------------------------------------------------------------------------- DB plumbing
def _enabled() -> bool:
    from src.conversation_store import persistence_enabled

    return persistence_enabled()


_schema_ready = False


def _sessions():
    global _schema_ready
    from src import db

    if not _schema_ready:
        # The same create-if-missing the conversation store runs; the rules may be read
        # before any conversation has been saved.
        db.ensure_schema()
        _schema_ready = True
    return db.get_session_factory()


def _model():
    from src.db_models import ChatbotQuestionRules

    return ChatbotQuestionRules


# ----------------------------------------------------------------------------- reading
class _Cache:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.version: int = SEED_VERSION
        self.doc: RulesDocument | None = None
        self.checked_at: float | None = None
        # An active version that failed validation. Remembered so it is not re-fetched
        # and re-logged on every refresh while the last good document keeps serving.
        self.rejected_version: int | None = None


_cache = _Cache()
_override: tuple[int, RulesDocument] | None = None


def _refresh_locked() -> None:
    now = time.monotonic()
    if _cache.checked_at is not None and now - _cache.checked_at < settings.rules_refresh_seconds:
        return
    _cache.checked_at = now

    Model = _model()
    try:
        with _sessions()() as sql:
            active = sql.execute(select(Model.version).where(Model.is_active.is_(True))).scalar_one_or_none()
            if active is None:
                if _cache.version != SEED_VERSION:
                    logger.info("RULES no active version - serving the seed")
                _cache.version, _cache.doc = SEED_VERSION, seed_document()
                return
            if active == _cache.version or active == _cache.rejected_version:
                return
            raw = sql.execute(select(Model.document).where(Model.version == active)).scalar_one()
    except Exception:
        logger.warning("RULES could not read chatbot_question_rules - keeping version %s", _cache.version, exc_info=True)
        return

    doc, errors = validate_document(raw)
    if doc is None:
        _cache.rejected_version = active
        logger.error("RULES version %s is invalid, keeping version %s: %s", active, _cache.version, errors)
        return
    logger.info("RULES loaded version %s (was %s)", active, _cache.version)
    _cache.version, _cache.doc, _cache.rejected_version = active, doc, None


def _current() -> tuple[int, RulesDocument]:
    if _override is not None:
        return _override
    if not _enabled():
        return SEED_VERSION, seed_document()
    with _cache.lock:
        _refresh_locked()
        return _cache.version, _cache.doc or seed_document()


def current_rules() -> RulesDocument:
    return _current()[1]


def rules_version() -> int:
    return _current()[0]


def invalidate() -> None:
    """Check the DB again on the next read instead of waiting out the refresh interval."""
    with _cache.lock:
        _cache.checked_at = None


def set_override(doc: RulesDocument | None, version: int = -1) -> None:
    """Serve ``doc`` instead of the stored rules (tests). None goes back to normal."""
    global _override
    _override = None if doc is None else (version, doc)


def reset() -> None:
    """Forget everything cached (tests)."""
    global _cache, _override
    _cache = _Cache()
    _override = None


# ----------------------------------------------------------------------------- writing
def _require_db() -> None:
    if not _enabled():
        raise RulesStoreError("rules can only be saved when the database is configured")


def save_version(doc: RulesDocument, note: str | None = None, author: str | None = None,
                 activate: bool = True) -> int:
    """Store ``doc`` as a new version, and make it the live one unless told otherwise."""
    _require_db()
    Model = _model()
    with _sessions()() as sql, sql.begin():
        if activate:
            sql.execute(update(Model).where(Model.is_active.is_(True)).values(is_active=False))
        row = Model(document=doc.model_dump(mode="json"), is_active=activate, note=note, created_by=author)
        sql.add(row)
        sql.flush()
        version = row.version
    logger.info("RULES saved version %s active=%s by=%s note=%r", version, activate, author, note)
    invalidate()
    return version


def activate(version: int) -> None:
    """Make an earlier version live again. The document is re-validated first: the code
    may have moved on since it was saved (a category removed, a slot renamed)."""
    _require_db()
    Model = _model()
    with _sessions()() as sql, sql.begin():
        raw = sql.execute(select(Model.document).where(Model.version == version)).scalar_one_or_none()
        if raw is None:
            raise RulesStoreError(f"version {version} does not exist")
        doc, errors = validate_document(raw)
        if doc is None:
            raise RulesStoreError(f"version {version} is no longer valid: {'; '.join(errors)}")
        sql.execute(update(Model).where(Model.is_active.is_(True)).values(is_active=False))
        sql.execute(update(Model).where(Model.version == version).values(is_active=True))
    logger.info("RULES activated version %s", version)
    invalidate()


def list_versions(limit: int = 100) -> list[VersionInfo]:
    if not _enabled():
        return []
    Model = _model()
    with _sessions()() as sql:
        rows = sql.execute(
            select(Model.version, Model.is_active, Model.note, Model.created_by, Model.created_at)
            .order_by(Model.version.desc())
            .limit(limit)
        ).all()
    return [VersionInfo(*row) for row in rows]


def get_version(version: int) -> dict[str, Any] | None:
    """A stored document exactly as saved, valid or not."""
    if not _enabled():
        return None
    Model = _model()
    with _sessions()() as sql:
        return sql.execute(select(Model.document).where(Model.version == version)).scalar_one_or_none()
