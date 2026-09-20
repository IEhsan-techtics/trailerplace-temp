"""Axle count on trailer listings.

The scraper now extracts how many axles a trailer has, recovering it from the
title, model name and feature lines when the listing carries no axle fields of
its own. 94 of 262 listings have no ``axle capacity`` field at all, so without
this their axle data had nowhere to go.

One column, not the raw/parsed pair that ``axle_capacity`` and the dimensions
use. A capacity has a display form ("10000 lbs") genuinely distinct from its
number; a count does not - "2" and 2 say the same thing, and a second copy would
only be something to keep in step. The raw text already lives in ``axles``.
"""
from alembic import op
import sqlalchemy as sa

from src.migration_utils import has_column, has_table

revision = "20260826_0005"
down_revision = "20260818_0004"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("axle_count", sa.Integer()),
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
