"""Existence checks, so a migration can run against an already-populated database.

Luna's tables were created by ``Base.metadata.create_all`` long before Alembic existed
here, so the live database holds every table while carrying no ``alembic_version`` row
at all. Running ``alembic upgrade head`` there replays from the baseline, which would
collide with tables that are already present.

Every revision therefore asks before it acts: create what is missing, leave what exists
untouched. That makes ``upgrade head`` safe to run repeatedly, on a blank database and
on the live one alike.

Import from a revision as ``from src.migration_utils import has_table`` - alembic.ini
sets ``prepend_sys_path = .``, so the repository root is importable.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


def _inspector() -> sa.Inspector:
    # Built per call, never cached: a revision inspects the schema again after its own
    # DDL, and a stale inspector would describe the database as it looked before the
    # revision started.
    return sa.inspect(op.get_bind())


def has_table(table: str) -> bool:
    return _inspector().has_table(table)


def has_column(table: str, column: str) -> bool:
    if not has_table(table):
        return False
    return any(existing["name"] == column for existing in _inspector().get_columns(table))


def has_index(table: str, index: str) -> bool:
    if not has_table(table):
        return False
    inspector = _inspector()
    if any(existing["name"] == index for existing in inspector.get_indexes(table)):
        return True
    # A UNIQUE CONSTRAINT is backed by an index of the same name but is not reported by
    # get_indexes, so a unique index created either way is found.
    return any(existing["name"] == index for existing in inspector.get_unique_constraints(table))
