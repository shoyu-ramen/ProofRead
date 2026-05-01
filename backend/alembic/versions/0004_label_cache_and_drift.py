"""label_cache table + reports.external_match_json + rule_results.explanation

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-01 12:00:00

Three additions in one migration so production picks them up in a single
``alembic upgrade head`` run:

  * ``label_cache`` — the L3 perceptual cache row backing
    ``LabelCacheEntry``. Survives process restart and accumulates
    verified labels across deploys. Pre-filtered on
    ``(beverage_type, panel_count)`` (composite index) so the lookup
    candidate set stays small even at six-figure corpus sizes.

  * ``reports.external_match_json`` — TTB COLA match payload populated
    by the verify-path enrichment hook. Nullable; missing on rows
    written before the enrichment shipped.

  * ``rule_results.explanation`` — AI-generated one-sentence
    explanation for ``fail`` / ``advisory`` rule results. Nullable;
    missing on rows written before the explanation feature shipped or
    when the call-site disabled it.

The columns were declared on the SQLAlchemy models alongside the
features that introduced them, but no Alembic revision recorded them.
This migration brings the DB schema into agreement so any production
deploy that ran ``alembic upgrade head`` against an existing database
ends up with the columns present.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "label_cache",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("beverage_type", sa.String(length=16), nullable=False),
        sa.Column("panel_count", sa.Integer(), nullable=False),
        sa.Column("signature_hex", sa.String(length=128), nullable=False),
        sa.Column("extraction_json", sa.JSON(), nullable=False),
        sa.Column("external_match_json", sa.JSON(), nullable=True),
        sa.Column("explanations_json", sa.JSON(), nullable=True),
        sa.Column(
            "hit_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_label_cache_beverage_type", "label_cache", ["beverage_type"]
    )
    op.create_index(
        "ix_label_cache_panel_count", "label_cache", ["panel_count"]
    )
    op.create_index(
        "ix_label_cache_bev_panels",
        "label_cache",
        ["beverage_type", "panel_count"],
    )

    op.add_column(
        "reports",
        sa.Column("external_match_json", sa.JSON(), nullable=True),
    )
    op.add_column(
        "rule_results",
        sa.Column("explanation", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rule_results", "explanation")
    op.drop_column("reports", "external_match_json")
    op.drop_index("ix_label_cache_bev_panels", table_name="label_cache")
    op.drop_index("ix_label_cache_panel_count", table_name="label_cache")
    op.drop_index("ix_label_cache_beverage_type", table_name="label_cache")
    op.drop_table("label_cache")
