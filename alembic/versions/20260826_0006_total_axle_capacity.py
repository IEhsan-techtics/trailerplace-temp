"""Total axle capacity: what the axles carry between them.

``axle_capacity`` is a PER-AXLE rating, which is the number the dealer publishes
but not the one a customer asks about. Two 7,500 lb axles carry 15,000 lb, and
until now working that out meant multiplying two columns at read time on every
listing card and every rerank.

Stored rather than computed on read because both inputs are already settled at
ingest: the count and the per-axle rating are written in the same pass, and a
generated column would tie the arithmetic to Postgres.

Null unless BOTH parts are known. A listing stating a capacity but no count
yields nothing here rather than a total that quietly assumes two axles.
"""
from alembic import op
import sqlalchemy as sa

from src.migration_utils import has_column, has_table

revision = "20260826_0006"
down_revision = "20260826_0005"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("total_axle_capacity_lbs_num", sa.Float()),
)


def upgrade() -> None:
    if not has_table("trailer_listings"):
        # Nothing to alter. 0003 owns creation; a create_all-built database already
        # has these columns from the model, which has_column below detects.
        return
    for name, column_type in _COLUMNS:
        if not has_column("trailer_listings", name):
            op.add_column("trailer_listings", sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    if not has_table("trailer_listings"):
        return
    for name, _column_type in reversed(_COLUMNS):
        if has_column("trailer_listings", name):
            op.drop_column("trailer_listings", name)
