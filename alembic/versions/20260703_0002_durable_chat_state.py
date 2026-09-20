"""Persist complete chatbot state and idempotent turns."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from src.migration_utils import has_column, has_index, has_table

revision = "20260703_0002"
down_revision = "20260525_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Guarded so this revision can run against a database whose tables were
    # created by create_all() and never stamped with a revision.
    for column in (
        sa.Column("state_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("state_schema_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("state_version", sa.Integer(), server_default="0", nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    ):
        if not has_column("chatbot_conversations", column.name):
            op.add_column("chatbot_conversations", column)

    if not has_table("chatbot_turns"):
        op.create_table(
            "chatbot_turns",
            sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chatbot_conversations.session_id", ondelete="CASCADE"), primary_key=True),
            sa.Column("turn_id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("request_message", sa.Text(), nullable=False),
            sa.Column("response", postgresql.JSONB(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
    if not has_table("chatbot_outbox"):
        op.create_table(
            "chatbot_outbox",
            sa.Column("event_id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
            sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("event_key", sa.String(128), nullable=False),
            sa.Column("event_type", sa.String(64), nullable=False),
            sa.Column("payload", postgresql.JSONB(), nullable=False),
            sa.Column("status", sa.String(16), server_default="pending", nullable=False),
            sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
            sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("session_id", "turn_id", "event_key", name="uq_chatbot_outbox_event"),
        )
    if not has_index("chatbot_outbox", "ix_chatbot_outbox_session_id"):
        op.create_index("ix_chatbot_outbox_session_id", "chatbot_outbox", ["session_id"])


def downgrade() -> None:
    if has_index("chatbot_outbox", "ix_chatbot_outbox_session_id"):
        op.drop_index("ix_chatbot_outbox_session_id", table_name="chatbot_outbox")
    if has_table("chatbot_outbox"):
        op.drop_table("chatbot_outbox")
    if has_table("chatbot_turns"):
        op.drop_table("chatbot_turns")
    for column in ("closed_at", "state_version", "state_schema_version", "state_snapshot"):
        if has_column("chatbot_conversations", column):
            op.drop_column("chatbot_conversations", column)
