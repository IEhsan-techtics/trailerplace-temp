"""Which bot queued each team email, so only that bot sends it.

Local runs and the Azure bot share chatbot_outbox, and each sent every pending row it found:
a local test's email went out through Azure's settings, to Transax. Nullable, because rows
queued before this column have no origin; the deployed bot sends those.

Guarded like every revision here, so running it twice does nothing the second time.
"""
import sqlalchemy as sa
from alembic import op

from src.migration_utils import has_column

revision = "20260929_0011"
down_revision = "20260925_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not has_column("chatbot_outbox", "origin"):
        op.add_column("chatbot_outbox", sa.Column("origin", sa.String(64), nullable=True))


def downgrade() -> None:
    if has_column("chatbot_outbox", "origin"):
        op.drop_column("chatbot_outbox", "origin")
