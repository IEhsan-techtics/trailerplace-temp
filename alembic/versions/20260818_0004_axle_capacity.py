"""Axle capacity on trailer listings: raw display string plus its parsed pounds.

The workbook has always carried an ``axle_capacity`` column; ingest simply never read it.
The value is a PER-AXLE rating (a two-axle trailer rated 3500 here has a 7000 lb GVWR), so
it is stored on its own rather than folded into the payload/GVWR numbers.
"""
from alembic import op
import sqlalchemy as sa

from src.migration_utils import has_column, has_table

revision = "20260818_0004"
down_revision = "20260728_0003"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("axle_capacity", sa.Text()),
    ("axle_capacity_lbs_num", sa.Float()),
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
