"""The question-rules document, versioned, with exactly one live at a time.

Luna's first revision of its own, and it is catching up rather than changing anything: the
table has been in use since the rules admin shipped, but it was created by
``Base.metadata.create_all`` and so existed only where that had run. Written down here, a
database built from the migrations alone gets it too.

Every save is a new row, and the partial unique index is what guarantees the bot can never
be left with two live documents - two admins saving at the same moment is otherwise exactly
how that happens. An empty table is normal and means the bot serves src/rules/seed.json.

Guarded like every revision here, so on the live database - where the table is already
there, with its versions in it - this does nothing at all.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from src.migration_utils import has_index, has_table

revision = "20260920_0009"
down_revision = "20260905_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not has_table("chatbot_question_rules"):
        op.create_table(
            "chatbot_question_rules",
            sa.Column("version", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("document", postgresql.JSONB(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("note", sa.Text(), nullable=True),
            sa.Column("created_by", sa.String(255), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
    if not has_index("chatbot_question_rules", "uq_chatbot_question_rules_one_active"):
        # Partial, on purpose: any number of inactive versions, never two active ones.
        op.create_index(
            "uq_chatbot_question_rules_one_active",
            "chatbot_question_rules",
            ["is_active"],
            unique=True,
            postgresql_where=sa.text("is_active"),
        )


def downgrade() -> None:
    if not has_table("chatbot_question_rules"):
        return
    if has_index("chatbot_question_rules", "uq_chatbot_question_rules_one_active"):
        op.drop_index("uq_chatbot_question_rules_one_active", table_name="chatbot_question_rules")
    op.drop_table("chatbot_question_rules")
