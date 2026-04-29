"""backfill drifted columns on reports + rule_results

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-29 14:50:00

Adds five columns that exist in the SQLAlchemy models but were never
recorded in ``0001_initial.py``. Without this migration any production
database that ran the initial migration is missing these columns and
will fail on the next write.

- reports.image_quality (String(16), default "good")
- reports.image_quality_notes (Text, nullable)
- reports.extractor (String(32), default "ocr")
- rule_results.is_flagged (Boolean, default false)
- rule_results.flag_comment (Text, nullable)

``server_default`` values are baked in so existing rows get sensible
backfills without a separate UPDATE pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "reports",
        sa.Column(
            "image_quality",
            sa.String(length=16),
            nullable=False,
            server_default="good",
        ),
    )
    op.add_column(
        "reports",
        sa.Column("image_quality_notes", sa.Text(), nullable=True),
    )
    op.add_column(
        "reports",
        sa.Column(
            "extractor",
            sa.String(length=32),
            nullable=False,
            server_default="ocr",
        ),
    )
    op.add_column(
        "rule_results",
        sa.Column(
            "is_flagged",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "rule_results",
        sa.Column("flag_comment", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rule_results", "flag_comment")
    op.drop_column("rule_results", "is_flagged")
    op.drop_column("reports", "extractor")
    op.drop_column("reports", "image_quality_notes")
    op.drop_column("reports", "image_quality")
