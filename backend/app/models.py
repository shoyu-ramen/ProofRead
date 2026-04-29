"""SQLAlchemy 2.0 ORM models for the v1 schema.

See SPEC.md §0 for the full schema and v1.9 for which tables back each
endpoint. Types are kept portable: ``sqlalchemy.Uuid`` and ``sa.JSON``
both decay to TEXT on SQLite and to UUID/JSONB on Postgres.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255))
    ttb_basic_permit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    billing_plan: Mapped[str] = mapped_column(String(32), default="starter")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True)
    role: Mapped[str] = mapped_column(String(32), default="producer")
    company_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("companies.id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id")
    )
    beverage_type: Mapped[str] = mapped_column(String(16))
    container_size_ml: Mapped[int] = mapped_column(Integer)
    container_size_source: Mapped[str] = mapped_column(String(16), default="user")
    is_imported: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="uploading")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ScanImage(Base):
    __tablename__ = "scan_images"
    __table_args__ = (UniqueConstraint("scan_id", "surface"),)

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    scan_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("scans.id")
    )
    surface: Mapped[str] = mapped_column(String(16))
    s3_key: Mapped[str] = mapped_column(String(512))
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class OCRResultRow(Base):
    __tablename__ = "ocr_results"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    scan_image_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("scan_images.id")
    )
    provider: Mapped[str] = mapped_column(String(32))
    raw_json: Mapped[dict] = mapped_column(JSON)
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    ms: Mapped[int] = mapped_column(Integer)


class ExtractedFieldRow(Base):
    __tablename__ = "extracted_fields"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    scan_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("scans.id")
    )
    field_id: Mapped[str] = mapped_column(String(64))
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    source_image_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("scan_images.id"), nullable=True
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    scan_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("scans.id")
    )
    overall: Mapped[str] = mapped_column(String(16))
    rule_version: Mapped[str] = mapped_column(String(32))
    image_quality: Mapped[str] = mapped_column(String(16), default="good")
    image_quality_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    extractor: Mapped[str] = mapped_column(String(32), default="ocr")
    # External-source match (TTB COLA approval, etc.) populated by the
    # finalize hook from `services/enrichment.py`. Nullable because
    # the lookup is gated and best-effort. Shape is the dict form of
    # `app.services.external.types.ExternalMatch` (`source`,
    # `source_id`, `brand`, `approval_date`, `source_url`, etc.).
    external_match_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RuleResultRow(Base):
    __tablename__ = "rule_results"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    report_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("reports.id")
    )
    rule_id: Mapped[str] = mapped_column(String(128))
    rule_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    finding: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation: Mapped[str] = mapped_column(String(64))
    fix_suggestion: Mapped[str | None] = mapped_column(Text, nullable=True)
    bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    flag_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("scan_images.id"), nullable=True
    )
    # Which scan_image surface (front/back/...) the rule's evidence was
    # read from. Mirrors the verify-path `RuleResult.surface` so the
    # mobile report screen can highlight the captured image when the
    # user taps a result.
    surface: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # AI-generated one-sentence plain-language explanation for fail /
    # advisory rules, contextualised to THIS scan's extracted values.
    # Populated by the finalize hook from `services/enrichment.py`.
    # Nullable because (a) passing rules don't carry an explanation,
    # (b) the explanation service is gated by `explanation_enabled` and
    # may be off, and (c) generation is best-effort and can fail open.
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)


class LabelCacheEntry(Base):
    """L3 perceptual cache row — survives process restart.

    Sits underneath the in-process L2 ``ReverseLookupCache``: a verify
    request that misses L1 (byte-exact) and L2 (in-process dhash) probes
    this table before paying for the cold VLM path. The candidate set is
    pre-filtered by ``(beverage_type, panel_count)``; per-panel Hamming
    distance is computed in Python on the candidate rows. Storing the
    enrichment payloads (TTB COLA match, AI rule explanations) on the
    same row keeps the persisted entry as the single source of truth
    when a sibling workstream produces them post-extraction.
    """

    __tablename__ = "label_cache"
    __table_args__ = (
        Index("ix_label_cache_bev_panels", "beverage_type", "panel_count"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    beverage_type: Mapped[str] = mapped_column(String(16), index=True)
    panel_count: Mapped[int] = mapped_column(Integer, index=True)
    # Comma-separated hex of 64-bit dhash ints, one per panel, in panel
    # order. Decoded back to a tuple by ``signature_from_hex`` at lookup
    # time so per-panel Hamming distance can be computed in Python.
    signature_hex: Mapped[str] = mapped_column(String(128))
    extraction_json: Mapped[dict] = mapped_column(JSON)
    external_match_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    explanations_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
