"""Alembic migration 0005 round-trip test.

Pins the upgrade/downgrade contract for the known-label feature's two
new columns. Calls the migration's ``upgrade`` / ``downgrade``
functions directly through ``alembic.migration.MigrationContext`` —
keeps the test self-contained instead of shelling out to the alembic
CLI (which isn't always on PATH in dev / CI).

Skipped if the installed ``alembic`` package can't be located (the
local ``backend/alembic/`` directory shadows it as a namespace package
when ``sys.path`` includes the cwd; in environments where alembic isn't
``pip install``-ed at all the test is a no-op).

The schema diff this asserts:

  * ``label_cache.brand_name_normalized`` — new String(255) NULL column
  * ``ix_label_cache_brand_name_normalized`` — new index on that column
  * ``label_cache.first_frame_signature_hex`` — new String(32) NULL
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_revision(revision: str):
    """Import a migration module by file path, bypassing the local
    ``alembic/`` namespace shadow on ``sys.path``."""
    versions_dir = REPO_ROOT / "alembic" / "versions"
    matches = list(versions_dir.glob(f"{revision}_*.py"))
    if not matches:
        raise FileNotFoundError(
            f"no migration file for revision {revision!r} under {versions_dir}"
        )
    module_path = matches[0]
    module_name = f"_test_migration_{revision}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _migration_alembic_or_skip():
    """Skip the test when ``alembic.migration`` can't be imported.

    Probes by absolute path so the local ``backend/alembic/`` namespace
    package doesn't fool the spec lookup. Some dev interpreters don't
    have alembic installed (production does); in that case we skip
    rather than fail the suite.
    """
    try:
        from alembic.migration import MigrationContext  # noqa: F401
        from alembic.operations import Operations  # noqa: F401
    except ImportError:
        pytest.skip("alembic package not importable in this interpreter")


def _columns(engine, table: str) -> dict:
    return {c["name"]: c for c in inspect(engine).get_columns(table)}


def _index_names(engine, table: str) -> set[str]:
    return {idx["name"] for idx in inspect(engine).get_indexes(table)}


def test_migration_0005_round_trip(tmp_path):
    _migration_alembic_or_skip()
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    db_path = tmp_path / "alembic.db"
    db_url = f"sqlite:///{db_path}"

    rev_0005 = _load_revision("0005")

    # Build the prerequisite ``label_cache`` table the migration
    # expects to find. Mirrors the columns 0004 creates so the test
    # doesn't need to chain through every prior migration to assert
    # the 0005-only behavior.
    engine = create_engine(db_url)
    metadata = sa.MetaData()
    sa.Table(
        "label_cache",
        metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("beverage_type", sa.String(16), nullable=False),
        sa.Column("panel_count", sa.Integer, nullable=False),
        sa.Column("signature_hex", sa.String(128), nullable=False),
        sa.Column("extraction_json", sa.JSON, nullable=False),
        sa.Column("external_match_json", sa.JSON, nullable=True),
        sa.Column("explanations_json", sa.JSON, nullable=True),
        sa.Column("hit_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("last_seen_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    metadata.create_all(engine)

    # 1. Run upgrade() — columns + index land.
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            rev_0005.upgrade()

    columns = _columns(engine, "label_cache")
    assert "brand_name_normalized" in columns
    assert columns["brand_name_normalized"]["nullable"] is True
    assert "first_frame_signature_hex" in columns
    assert columns["first_frame_signature_hex"]["nullable"] is True

    assert (
        "ix_label_cache_brand_name_normalized" in _index_names(engine, "label_cache")
    )

    # 2. Run downgrade() — columns + index disappear.
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            rev_0005.downgrade()

    columns = _columns(engine, "label_cache")
    assert "brand_name_normalized" not in columns
    assert "first_frame_signature_hex" not in columns
    assert (
        "ix_label_cache_brand_name_normalized"
        not in _index_names(engine, "label_cache")
    )

    # 3. Re-run upgrade() — idempotent so a deploy that rolls back and
    #    forward never trips a "column already exists" conflict.
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            rev_0005.upgrade()
    columns = _columns(engine, "label_cache")
    assert "brand_name_normalized" in columns
    assert "first_frame_signature_hex" in columns

    engine.dispose()
