"""The follow-up agent's log: one row per decision about a quiet Messenger customer.

Written before the model is asked or anything is sent, so the unique key on
(session_id, anchor_turn_id, attempt) is what keeps two runs from messaging the same
customer twice. See src/followup.

Guarded like every revision here, so running it twice does nothing the second time.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from src.migration_utils import has_index, has_table

revision = "20261006_0012"
down_revision = "20260929_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not has_table("chatbot_followups"):
        op.create_table(
            "chatbot_followups",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                      server_default=sa.text("gen_random_uuid()")),
            sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("psid", sa.String(255), nullable=False),
            sa.Column("anchor_turn_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("attempt", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(16), nullable=False),
            sa.Column("scenario", sa.String(64), nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("session_id", "anchor_turn_id", "attempt", name="uq_chatbot_followups_attempt"),
        )
    if not has_index("chatbot_followups", "ix_chatbot_followups_session_id"):
        op.create_index("ix_chatbot_followups_session_id", "chatbot_followups", ["session_id"])


def downgrade() -> None:
    if not has_table("chatbot_followups"):
        return
    if has_index("chatbot_followups", "ix_chatbot_followups_session_id"):
        op.drop_index("ix_chatbot_followups_session_id", table_name="chatbot_followups")
    op.drop_table("chatbot_followups")
