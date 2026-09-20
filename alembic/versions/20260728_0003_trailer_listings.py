"""Searchable trailer inventory table, replacing the Pinecone index."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from src.migration_utils import has_index, has_table

revision = "20260728_0003"
down_revision = "20260703_0002"
branch_labels = None
depends_on = None

_INDEXES = (
    ("ix_trailer_listings_stock_number", ["stock_number"], False),
    ("ix_trailer_listings_category", ["category"], False),
    ("ix_trailer_listings_category_make", ["category", "make"], False),
    ("ix_trailer_listings_category_length", ["category", "length_ft_num"], False),
    ("ix_trailer_listings_url", ["url"], True),
)


def upgrade() -> None:
    if has_table("trailer_listings"):
        # Already present (created here previously, or by create_all). Leave the
        # table and its data alone; only add indexes that are genuinely missing.
        _create_missing_indexes()
        return

    op.create_table(
        "trailer_listings",
        sa.Column("listing_id", sa.String(128), primary_key=True),
        sa.Column("stock_number", sa.String(64), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("condition", sa.String(64), nullable=True),
        sa.Column("category", sa.String(64), nullable=True),
        sa.Column("subcategory", sa.String(64), nullable=True),
        sa.Column("make", sa.String(64), nullable=True),
        sa.Column("color", sa.String(64), nullable=True),
        sa.Column("hitch_type", sa.String(32), nullable=True),
        sa.Column("price", sa.Numeric(12, 2), nullable=True),
        sa.Column("price_display", sa.String(32), nullable=True),
        sa.Column("year", sa.String(16), nullable=True),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("trim", sa.String(128), nullable=True),
        sa.Column("length", sa.Text(), nullable=True),
        sa.Column("width", sa.Text(), nullable=True),
        sa.Column("height", sa.Text(), nullable=True),
        sa.Column("axles", sa.Text(), nullable=True),
        sa.Column("gvwr", sa.Text(), nullable=True),
        sa.Column("payload_capacity", sa.Text(), nullable=True),
        sa.Column("length_ft_num", sa.Float(), nullable=True),
        sa.Column("width_ft_num", sa.Float(), nullable=True),
        sa.Column("height_ft_num", sa.Float(), nullable=True),
        sa.Column("gvwr_lbs_num", sa.Float(), nullable=True),
        sa.Column("payload_lbs_num", sa.Float(), nullable=True),
        sa.Column("trailer_material", sa.String(64), nullable=True),
        sa.Column("floor", sa.String(64), nullable=True),
        sa.Column("features", postgresql.JSONB(), nullable=True),
        sa.Column("match_evidence_text", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("info_json_source", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    _create_missing_indexes()


def _create_missing_indexes() -> None:
    for name, columns, unique in _INDEXES:
        if not has_index("trailer_listings", name):
            op.create_index(name, "trailer_listings", columns, unique=unique)


def downgrade() -> None:
    for name, _columns, _unique in reversed(_INDEXES):
        if has_index("trailer_listings", name):
            op.drop_index(name, table_name="trailer_listings")
    if has_table("trailer_listings"):
        op.drop_table("trailer_listings")
