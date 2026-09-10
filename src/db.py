"""Database engine/session helpers for durable chatbot persistence."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from src.config import settings
from src.db_models import Base

logger = logging.getLogger(__name__)

_ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


def database_enabled() -> bool:
    return all([settings.host, settings.pguser, settings.password, settings.database, settings.port])


def _database_url() -> str:
    password = quote_plus(settings.password)
    return f"postgresql+psycopg://{settings.pguser}:{password}@{settings.host}:{settings.port}/{settings.database}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    if not database_enabled():
        raise RuntimeError("Database settings are incomplete")
    connect_args = {} if settings.host in {"localhost", "127.0.0.1", "::1"} else {"sslmode": "require"}
    return create_engine(_database_url(), connect_args=connect_args, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def ensure_schema() -> None:
    """Create any table declared in the models but missing from the database.

    create_all defaults to checkfirst=True, so existing tables are left exactly
    as they are — no column, index or row is touched.
    """
    Base.metadata.create_all(get_engine())


def run_migrations() -> None:
    """Bring the database up to date (M8: boot-time when DB_AUTO_CREATE=1).

    Two steps, because neither alone is sufficient:

    1. ``alembic upgrade head`` applies outstanding revisions. Every revision is
       written to skip objects that already exist, so this is safe on a database
       whose tables were created by create_all() and never stamped.
    2. ``ensure_schema()`` then creates anything still missing. Alembic alone
       cannot do this: once alembic_version reads head it considers its work
       done, so a table dropped afterwards would never come back.

    Together: whatever is missing gets created, whatever exists is left alone.

    alembic/env.py resolves the engine through get_engine(), so no URL is passed here.
    """
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config(str(_ALEMBIC_INI)), "head")
    ensure_schema()


def ping() -> bool:
    """True when the database answers a trivial query. Never raises."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database ping failed")
        return False
