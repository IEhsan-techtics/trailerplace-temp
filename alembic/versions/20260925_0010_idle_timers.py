"""The 5-minute rule's timers: one row per conversation, armed while a question waits.

A quiet customer who has chosen a category and been asked a question is shown that
category's trailers after five minutes, whatever we know by then. Every turn rewrites the
row in the same transaction as the turn, and the sweep - called every minute - fires the
armed rows whose time has come. See src/idle_timer.py.

Guarded like every revision here, so running it twice does nothing the second time.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from src.migration_utils import has_index, has_table

revision = "20260925_0010"
down_revision = "20260920_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not has_table("chatbot_idle_timers"):
        op.create_table(
            "chatbot_idle_timers",
            sa.Column("session_id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("channel", sa.String(32), nullable=False),
            sa.Column("channel_id", sa.String(255), nullable=True),
            sa.Column("category", sa.String(64), nullable=False),
            sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("status", sa.String(16), nullable=False, server_default="armed"),
            sa.Column("fired_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
    if not has_index("chatbot_idle_timers", "ix_chatbot_idle_timers_due"):
        op.create_index("ix_chatbot_idle_timers_due", "chatbot_idle_timers", ["status", "due_at"])


def downgrade() -> None:
    if not has_table("chatbot_idle_timers"):
        return
    if has_index("chatbot_idle_timers", "ix_chatbot_idle_timers_due"):
        op.drop_index("ix_chatbot_idle_timers_due", table_name="chatbot_idle_timers")
    op.drop_table("chatbot_idle_timers")
