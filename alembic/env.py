"""Alembic's entry point.

The engine is NOT built from a URL in alembic.ini: it comes from ``src.db.get_engine``,
so a migration connects with exactly the credentials, SSL mode and pooling the running
bot uses. There is one source of connection truth, and no password in a committed file.

There is ONE history for this database, in the standard ``alembic_version`` table, and
Luna continues it rather than starting its own. The chatbot_* tables and trailer_listings
were built by the revisions in ``alembic/versions`` - they came from the New Prompt
codebase, which shares this Azure Postgres server, and the live table is stamped at their
head. Luna's own future changes are appended on top of that, with the next number in the
same sequence.

The revisions up to that head are therefore HISTORY, not work to do: the live database has
already run them, and every one of them checks before it acts, so replaying them on a fresh
database produces the same schema without colliding on a fresh one.

The consequence to keep in mind: two codebases share this sequence. A new revision here and
a new revision there, both parented on the same head, leave the database with two heads and
``upgrade head`` refuses to run. Check the current head before adding one.
"""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv

from src.db import get_engine
from src.db_models import Base

load_dotenv()

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    # Offline mode emits SQL without a connection, and every revision here asks the live
    # database what already exists. There is nothing to ask, so this cannot work.
    raise RuntimeError("Offline migrations are not configured for this project")


def run_migrations_online() -> None:
    with get_engine().connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
