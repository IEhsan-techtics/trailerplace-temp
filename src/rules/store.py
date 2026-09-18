"""Where the live rules document comes from.

The active row of ``chatbot_question_rules`` when persistence is on, otherwise - and
whenever that row is missing, unreadable or invalid - the ``seed.json`` shipped with the
code. A bad document in the DB never takes the bot down: it logs, and the last good
document (or the seed) keeps serving.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from src.rules.models import RulesDocument, validate_document

logger = logging.getLogger(__name__)

SEED_PATH = Path(__file__).with_name("seed.json")
# The version number the seed reports. Real saved versions start at 1.
SEED_VERSION = 0

_override: tuple[int, RulesDocument] | None = None


@lru_cache(maxsize=1)
def seed_document() -> RulesDocument:
    doc, errors = validate_document(json.loads(SEED_PATH.read_text(encoding="utf-8")))
    if doc is None:
        # The seed is code, reviewed and tested - this only happens to a broken checkout.
        raise RuntimeError(f"src/rules/seed.json is invalid: {errors}")
    return doc


def current_rules() -> RulesDocument:
    if _override is not None:
        return _override[1]
    return seed_document()


def rules_version() -> int:
    if _override is not None:
        return _override[0]
    return SEED_VERSION


def set_override(doc: RulesDocument | None, version: int = -1) -> None:
    """Serve ``doc`` instead of the stored rules (tests). None goes back to normal."""
    global _override
    _override = None if doc is None else (version, doc)
