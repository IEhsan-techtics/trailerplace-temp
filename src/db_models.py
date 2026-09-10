"""SQLAlchemy ORM models for chatbot persistence."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ChatbotLead(Base):
    __tablename__ = "chatbot_leads"
    __table_args__ = (
        CheckConstraint("lead_type IN ('hard', 'soft')", name="ck_chatbot_leads_lead_type"),
    )

    lead_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    psid: Mapped[str | None] = mapped_column(String(255), nullable=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lead_type: Mapped[str] = mapped_column(String(16), nullable=False)
    contact_status: Mapped[str] = mapped_column(String(32), nullable=False, default="missing_contact", server_default="missing_contact")
    item_of_interest: Mapped[str] = mapped_column(Text, nullable=False)

    conversations: Mapped[list["ChatbotConversation"]] = relationship(back_populates="lead")


class ChatbotConversation(Base):
    __tablename__ = "chatbot_conversations"

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    lead_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbot_leads.lead_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    conversation: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    state_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    state_schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    lead: Mapped[ChatbotLead] = relationship(back_populates="conversations")
    # The unit of work orders inserts by RELATIONSHIP, not by the raw foreign key. Without
    # this, a session's FIRST turn — the one flush where the conversation row and its turn
    # row are both new — could emit the chatbot_turns INSERT first and violate
    # chatbot_turns_session_id_fkey. It worked most of the time purely by luck of ordering.
    turns: Mapped[list["ChatbotTurn"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class ChatbotTurn(Base):
    __tablename__ = "chatbot_turns"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbot_conversations.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    turn_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    request_message: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation: Mapped[ChatbotConversation] = relationship(back_populates="turns")


class TrailerListingRow(Base):
    """Searchable inventory, one row per workbook listing.

    Replaces the Pinecone index: the five hard gates that used to be metadata
    filters ($eq on category/make/hitch_type/subcategory, $gte on
    length_ft_num) are plain columns here, and search is a SELECT rather than a
    vector query. There is no embedding column — ranking is handled entirely by
    the fit reranker (dimensions) and the feature reranker (gpt-5-nano).

    Every column but the primary key is nullable. Pinecone stripped nulls and
    empty strings from metadata before upsert, so any field could be absent on
    any vector, and the read path already coalesces each one. A NOT NULL here
    would reject workbook rows the old pipeline accepted.
    """

    __tablename__ = "trailer_listings"
    __table_args__ = (
        Index("ix_trailer_listings_category", "category"),
        Index("ix_trailer_listings_category_make", "category", "make"),
        Index("ix_trailer_listings_category_length", "category", "length_ft_num"),
        Index("ix_trailer_listings_url", "url", unique=True),
    )

    # build_vector_id()'s value, reused so ingest stays idempotent across runs.
    listing_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    stock_number: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)

    condition: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subcategory: Mapped[str | None] = mapped_column(String(64), nullable=True)
    make: Mapped[str | None] = mapped_column(String(64), nullable=True)
    color: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hitch_type: Mapped[str | None] = mapped_column(String(32), nullable=True)

    price: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    price_display: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Text, not integer: Pinecone stored year as a string and the search path
    # passes it through to the card untouched.
    year: Mapped[str | None] = mapped_column(String(16), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    trim: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Raw display strings ("24 ft 0 in", "9990 lbs") beside their parsed
    # numerics. The fit reranker re-parses the raw strings; the numerics back
    # the SQL gates. Both are kept so neither side has to guess.
    length: Mapped[str | None] = mapped_column(Text, nullable=True)
    width: Mapped[str | None] = mapped_column(Text, nullable=True)
    height: Mapped[str | None] = mapped_column(Text, nullable=True)
    axles: Mapped[str | None] = mapped_column(Text, nullable=True)
    gvwr: Mapped[str | None] = mapped_column(Text, nullable=True)
    # PER-AXLE rating, not a total: a two-axle trailer rated 3500 here has a 7000 lb GVWR.
    axle_capacity: Mapped[str | None] = mapped_column(Text, nullable=True)
    # How many axles. One column, not the raw/parsed pair the dimensions use:
    # "2" and 2 carry identical information, so a display string would only be a
    # second copy to keep in step. The raw text already lives in `axles`.
    axle_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload_capacity: Mapped[str | None] = mapped_column(Text, nullable=True)

    length_ft_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    width_ft_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    height_ft_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    gvwr_lbs_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload_lbs_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    # No index: axle capacity is a rerank signal, never a WHERE clause.
    axle_capacity_lbs_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    # count x per-axle rating: two 7,500 lb axles carry 15,000 lb between them.
    # Derived, and only when both parts are known - see ingest.build_record.
    total_axle_capacity_lbs_num: Mapped[float | None] = mapped_column(Float, nullable=True)

    trailer_material: Mapped[str | None] = mapped_column(String(64), nullable=True)
    floor: Mapped[str | None] = mapped_column(String(64), nullable=True)

    features: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    # The only evidence the feature reranker (gpt-5-nano) and its deterministic
    # fallback ever see. Capped at MATCH_EVIDENCE_TEXT_MAX_CHARS by ingest.
    match_evidence_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # SHA-256 of the canonical row payload; drives skip-unchanged on re-ingest.
    # NULL simply forces that row to be rewritten.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    info_json_source: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class ChatbotOutbox(Base):
    __tablename__ = "chatbot_outbox"
    __table_args__ = (UniqueConstraint("session_id", "turn_id", "event_key", name="uq_chatbot_outbox_event"),)

    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    turn_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    event_key: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", server_default="pending")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
