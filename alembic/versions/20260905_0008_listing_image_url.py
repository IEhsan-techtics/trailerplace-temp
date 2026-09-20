"""The listing's lead photo, so a channel can build a card.

Messenger renders a rich preview when a person pastes a URL, but not when a bot
sends one through the Send API — the preview is a client-side courtesy, not a
property of the message, and there is no flag that turns it on. The way to get a
card is to send a generic template, and a template has to be handed its picture.

Nothing in the catalogue carried one: the scraper read the WordPress admin form
but dropped ``wpp_manage_inventory_image_link[]`` on the floor, so neither the
workbook nor this table had an image anywhere. The scraper now keeps the first
entry, absolute, and ingest writes it here.

One image, not the gallery. A card shows a single picture, so the rest would be
rows nothing reads.

Null where the listing has no photo — 26 of 448 on the April scrape.
"""
from alembic import op
import sqlalchemy as sa

from src.migration_utils import has_column, has_table

revision = "20260905_0008"
down_revision = "20260904_0007"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("image_url", sa.Text()),
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
