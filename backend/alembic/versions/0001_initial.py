"""initial v1 schema

Revision ID: 0001
Revises:
Create Date: 2026-04-28 00:00:00

Creates all v1 tables: companies, users, scans, scan_images,
ocr_results, extracted_fields, reports, rule_results.

Uses portable ``sa.Uuid`` and ``sa.JSON`` so the migration runs cleanly
against an empty Postgres or empty SQLite — the dialect picks the right
underlying type (UUID/JSONB on PG, CHAR(32)/TEXT on SQLite).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("ttb_basic_permit", sa.String(length=64), nullable=True),
        sa.Column(
            "billing_plan", sa.String(length=32), nullable=False, server_default="starter"
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column(
            "role", sa.String(length=32), nullable=False, server_default="producer"
        ),
        sa.Column(
            "company_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("companies.id"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "scans",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("beverage_type", sa.String(length=16), nullable=False),
        sa.Column("container_size_ml", sa.Integer(), nullable=False),
        sa.Column(
            "container_size_source",
            sa.String(length=16),
            nullable=False,
            server_default="user",
        ),
        sa.Column(
            "is_imported",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "status", sa.String(length=16), nullable=False, server_default="uploading"
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "scan_images",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "scan_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("scans.id"),
            nullable=False,
        ),
        sa.Column("surface", sa.String(length=16), nullable=False),
        sa.Column("s3_key", sa.String(length=512), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column(
            "captured_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("scan_id", "surface", name="uq_scan_images_scan_surface"),
    )

    op.create_table(
        "ocr_results",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "scan_image_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("scan_images.id"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("ms", sa.Integer(), nullable=False),
    )

    op.create_table(
        "extracted_fields",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "scan_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("scans.id"),
            nullable=False,
        ),
        sa.Column("field_id", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "source_image_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("scan_images.id"),
            nullable=True,
        ),
    )

    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "scan_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("scans.id"),
            nullable=False,
        ),
        sa.Column("overall", sa.String(length=16), nullable=False),
        sa.Column("rule_version", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "rule_results",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "report_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("reports.id"),
            nullable=False,
        ),
        sa.Column("rule_id", sa.String(length=128), nullable=False),
        sa.Column("rule_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("finding", sa.Text(), nullable=True),
        sa.Column("expected", sa.Text(), nullable=True),
        sa.Column("citation", sa.String(length=64), nullable=False),
        sa.Column("fix_suggestion", sa.Text(), nullable=True),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column(
            "image_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("scan_images.id"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("rule_results")
    op.drop_table("reports")
    op.drop_table("extracted_fields")
    op.drop_table("ocr_results")
    op.drop_table("scan_images")
    op.drop_table("scans")
    op.drop_table("users")
    op.drop_table("companies")
