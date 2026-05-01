"""label_cache: brand_name_normalized + first_frame_signature_hex

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-01 13:00:00

Two new nullable columns on ``label_cache`` to back the known-label
recognition feature:

  * ``brand_name_normalized`` — lower-cased, stripped brand name lifted
    from the cached extraction's ``fields.brand_name.value``. Indexed
    so the new ``lookup_by_brand`` path stays a single-key probe.
  * ``first_frame_signature_hex`` — single-panel dhash of the camera
    frame the user gave the detect-container gate, stored as a
    16-char lowercase hex string. Stamped after the panorama
    completes (``enrich_verdict``) with an ``IS NULL`` guard so the
    first observed frame wins. Not indexed at v1 corpus sizes — the
    composite ``ix_label_cache_bev_panels`` index already narrows the
    candidate set; the in-Python Hamming pass scans those.

Pattern mirrors ``0004_label_cache_and_drift.py`` exactly: nullable
add_column, named index, symmetric downgrade.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "label_cache",
        sa.Column("brand_name_normalized", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_label_cache_brand_name_normalized",
        "label_cache",
        ["brand_name_normalized"],
    )
    op.add_column(
        "label_cache",
        sa.Column("first_frame_signature_hex", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("label_cache", "first_frame_signature_hex")
    op.drop_index(
        "ix_label_cache_brand_name_normalized", table_name="label_cache"
    )
    op.drop_column("label_cache", "brand_name_normalized")
