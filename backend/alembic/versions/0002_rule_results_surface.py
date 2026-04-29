"""add rule_results.surface column

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-29 14:30:00

Adds the `surface` column to `rule_results` so the report DTO can carry
which scan_image surface (front / back / panel_N) the rule's evidence
was read from. The column is nullable with default null — existing rows
keep their pre-migration semantics (no surface attribution); rows
written after the migration carry the value populated by the engine
via `RuleResult.surface`.

The model side landed in task #7 (`backend/app/models.py:166`); this
migration brings the schema into agreement so production deploys with
existing `rule_results` data don't break on the next finalize.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "rule_results",
        sa.Column("surface", sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rule_results", "surface")
