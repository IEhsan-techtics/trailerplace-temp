"""Shared inbound-message queue, so a customer's messages are answered in order.

Ordering used to be a per-process deque, which is only correct while exactly one
instance is running. Behind a load balancer two messages sent a second apart can land
on different instances, and nothing decided which went first.

This table is the shared queue. The webhook records the message here and acknowledges;
whichever instance wins the per-customer advisory lock drains it oldest first, ordered
by ``sent_at`` - the timestamp FACEBOOK assigned, which is the customer's real send
order regardless of how the deliveries were routed.

The unique constraint on (channel, external_id) is also the duplicate guard: Meta
reuses ``mid`` on every retry, so a redelivery collides here and is discarded before
any work is done. That check is durable, unlike the in-process one it backs up.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from src.migration_utils import has_index, has_table

revision = "20260904_0007"
down_revision = "20260826_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Guarded like every revision here, so it is safe against a database whose tables
    # were built by create_all() and never stamped.
    if not has_table("chatbot_inbound_messages"):
        op.create_table(
            "chatbot_inbound_messages",
            sa.Column("message_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
            sa.Column("channel", sa.String(32), nullable=False, server_default="messenger"),
            sa.Column("session_id", sa.String(255), nullable=False),
            sa.Column("external_id", sa.String(255), nullable=False),
            sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("body", sa.Text(), nullable=False),
            sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
            sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("channel", "external_id", name="uq_chatbot_inbound_external_id"),
        )
    if not has_index("chatbot_inbound_messages", "ix_chatbot_inbound_pending"):
        # Covers the drain's only query: oldest pending message for one customer.
        op.create_index(
            "ix_chatbot_inbound_pending",
            "chatbot_inbound_messages",
            ["session_id", "status", "sent_at"],
        )


def downgrade() -> None:
    if not has_table("chatbot_inbound_messages"):
        return
    if has_index("chatbot_inbound_messages", "ix_chatbot_inbound_pending"):
        op.drop_index("ix_chatbot_inbound_pending", table_name="chatbot_inbound_messages")
    op.drop_table("chatbot_inbound_messages")
