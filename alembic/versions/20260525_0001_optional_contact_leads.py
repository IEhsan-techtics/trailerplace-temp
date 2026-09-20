"""Allow anonymous leads with optional contact details.

Revision ID: 20260525_0001
Revises:
Create Date: 2026-05-25
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260525_0001"
down_revision = "a001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("chatbot_leads"):
        # Fresh database: create the tables outright.
        op.create_table(
            "chatbot_leads",
            sa.Column(
                "lead_id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column("psid", sa.String(length=255), nullable=True),
            sa.Column("name", sa.String(length=255), nullable=True),
            sa.Column("phone_number", sa.String(length=64), nullable=True),
            sa.Column("email", sa.String(length=255), nullable=True),
            sa.Column("lead_type", sa.String(length=16), nullable=False),
            sa.Column(
                "contact_status",
                sa.String(length=32),
                nullable=False,
                server_default="missing_contact",
            ),
            sa.Column("item_of_interest", sa.Text(), nullable=False),
            sa.CheckConstraint(
                "lead_type IN ('hard', 'soft')",
                name="ck_chatbot_leads_lead_type",
            ),
        )
        op.create_table(
            "chatbot_conversations",
            sa.Column("session_id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "lead_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("chatbot_leads.lead_id", ondelete="RESTRICT"),
                nullable=False,
            ),
            sa.Column("conversation", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
        op.create_index(
            "ix_chatbot_conversations_lead_id",
            "chatbot_conversations",
            ["lead_id"],
            unique=False,
        )
        return

    # chatbot_leads already exists. Only bring across what is genuinely absent —
    # a database created by create_all() is already in the target shape, and
    # re-running the alter/backfill below would rewrite live rows for nothing.
    lead_columns = {
        column["name"]: column for column in inspector.get_columns("chatbot_leads")
    }
    for name, column_type in (("name", sa.String(length=255)), ("phone_number", sa.String(length=64))):
        if lead_columns.get(name) is not None and not lead_columns[name]["nullable"]:
            op.alter_column("chatbot_leads", name, existing_type=column_type, nullable=True)

    if "contact_status" not in lead_columns:
        op.add_column(
            "chatbot_leads",
            sa.Column(
                "contact_status",
                sa.String(length=32),
                nullable=False,
                server_default="missing_contact",
            ),
        )
        # Backfill ONLY the column this revision just introduced. Running it
        # unconditionally would overwrite contact_status on every existing lead.
        op.execute(
            "UPDATE chatbot_leads "
            "SET contact_status = CASE "
            "WHEN NULLIF(TRIM(COALESCE(phone_number, '')), '') IS NOT NULL "
            "OR NULLIF(TRIM(COALESCE(email, '')), '') IS NOT NULL "
            "THEN 'contact_available' ELSE 'missing_contact' END"
        )
    if not inspector.has_table("chatbot_conversations"):
        op.create_table(
            "chatbot_conversations",
            sa.Column("session_id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "lead_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("chatbot_leads.lead_id", ondelete="RESTRICT"),
                nullable=False,
            ),
            sa.Column("conversation", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
        op.create_index(
            "ix_chatbot_conversations_lead_id",
            "chatbot_conversations",
            ["lead_id"],
            unique=False,
        )


def downgrade() -> None:
    op.execute("UPDATE chatbot_leads SET name = COALESCE(name, 'Unknown')")
    op.execute("UPDATE chatbot_leads SET phone_number = COALESCE(phone_number, '')")
    op.drop_column("chatbot_leads", "contact_status")
    op.alter_column("chatbot_leads", "phone_number", existing_type=sa.String(length=64), nullable=False)
    op.alter_column("chatbot_leads", "name", existing_type=sa.String(length=255), nullable=False)
