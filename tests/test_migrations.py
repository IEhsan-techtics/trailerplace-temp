"""The migration history: linear, importable, and safe to run against a live database.

None of these open a database. What they check is what only goes wrong on the day you
finally run ``upgrade head`` against production - a broken chain, two heads, or a revision
that would collide with a table that is already there.

The history is shared. These revisions built the chatbot_* tables and trailer_listings,
they came from the New Prompt codebase on the same Azure server, and Luna continues the
same sequence in the same ``alembic_version`` table rather than starting one of its own.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "alembic" / "versions"

# The head a new revision must be parented on.
CURRENT_HEAD = "20260925_0010"
# Luna's own first revision.
LUNA_FIRST = "20260920_0009"
# What the live database was stamped at when Luna picked the history up.
SHARED_HEAD = "20260905_0008"


@pytest.fixture(scope="module")
def script():
    return ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))


def test_there_is_exactly_one_head(script):
    """Two heads and ``upgrade head`` refuses to run at all.

    The likeliest way to get two is the shared sequence: a revision added here and another
    added in the other codebase, both parented on the same head.
    """
    assert script.get_heads() == [CURRENT_HEAD]


def test_the_chain_is_unbroken_back_to_one_root(script):
    revisions = list(script.walk_revisions())
    assert len(revisions) >= 10, "no migrations found - check script_location"
    assert revisions[-1].down_revision is None, "exactly one root"
    assert all(rev.module for rev in revisions), "every revision imports cleanly"


def test_luna_writes_to_the_shared_version_table(script):
    """One database, one history. A private version table would fork it silently."""
    env = (ROOT / "alembic" / "env.py").read_text(encoding="utf-8")
    assert "version_table" not in env, "the standard alembic_version table is the one"


def test_every_revision_asks_before_it_acts():
    """The live database held these tables long before it was ever stamped, and the older
    ones were built by create_all. An unguarded CREATE or ADD COLUMN would collide."""
    for path in VERSIONS.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for kind in re.findall(r"op\.(create_table|create_index|add_column)\(", source):
            assert re.search(r"has_(table|column|index)\(", source), (
                f"{path.name} runs {kind} without checking whether it already exists"
            )


def test_the_turn_receipt_and_the_state_snapshot_are_in_the_history():
    """The two columns that make a conversation resumable and a turn replayable."""
    source = (VERSIONS / "20260703_0002_durable_chat_state.py").read_text(encoding="utf-8")
    assert "state_snapshot" in source and "chatbot_turns" in source


def test_the_inbound_queue_is_in_the_history():
    source = (VERSIONS / "20260904_0007_inbound_message_queue.py").read_text(encoding="utf-8")
    assert "uq_chatbot_inbound_external_id" in source, "the duplicate guard is the point"
    assert "ix_chatbot_inbound_pending" in source, "the drain's only query"


def test_lunas_own_revisions_continue_the_shared_sequence(script):
    """Not a fork: Luna's first revision is parented on the head the database already had."""
    assert script.get_revision(LUNA_FIRST).down_revision == SHARED_HEAD


def test_the_idle_timers_are_in_the_history():
    source = (VERSIONS / "20260925_0010_idle_timers.py").read_text(encoding="utf-8")
    assert "chatbot_idle_timers" in source
    assert "ix_chatbot_idle_timers_due" in source, "the sweep's only query"


def test_the_listing_photo_a_messenger_card_needs_is_in_the_history():
    source = (VERSIONS / "20260905_0008_listing_image_url.py").read_text(encoding="utf-8")
    assert "image_url" in source


def test_every_table_luna_models_was_created_by_some_revision():
    """A model with no revision behind it is a table that only exists where create_all ran."""
    from src.db_models import Base

    history = "\n".join(path.read_text(encoding="utf-8") for path in VERSIONS.glob("*.py"))
    for table in Base.metadata.tables:
        assert table in history, f"{table} is modelled but no revision creates it"
