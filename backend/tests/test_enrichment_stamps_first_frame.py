"""Tests for the first-frame + brand stamping behavior in
``enrich_verdict``.

The known-label feature stamps two columns onto the L3 row after the
upsert resolves to an entry_id:

  * ``first_frame_signature_hex`` — passed in from the finalize route
  * ``brand_name_normalized`` — derived from the extraction's
    fields.brand_name.value

Both writes are guarded by ``IS NULL`` inside ``PersistedLabelCache``
so a label scanned multiple times never silently rewrites an existing
hash. These tests pin the contract end-to-end against an in-memory
SQLite-backed L3 cache.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.config import settings
from app.db import configure_engine, dispose_engine, get_session_factory
from app.models import Base, LabelCacheEntry
from app.rules.types import ExtractedField
from app.services.enrichment import enrich_verdict
from app.services.persisted_cache import PersistedLabelCache
from app.services.verify import VerifyReport
from app.services.vision import VisionExtraction


@pytest.fixture
async def cache_db(tmp_path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/enrich_first_frame.db"
    monkeypatch.setattr(settings, "database_url", db_url)
    # Disable both enrichment side-effects by default — these tests
    # only care about the L3 stamp behavior. Tests that need
    # explanations or external lookup re-enable them locally.
    monkeypatch.setattr(settings, "explanation_enabled", False)
    monkeypatch.setattr(settings, "ttb_cola_lookup_enabled", False)
    engine = configure_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield db_url
    await dispose_engine()


def _extraction(brand: str = "Anytown Ale") -> VisionExtraction:
    return VisionExtraction(
        fields={
            "brand_name": ExtractedField(value=brand, confidence=0.95),
            "country_of_origin": ExtractedField(value="USA", confidence=0.9),
        },
        unreadable=[],
        raw_response="{}",
        image_quality="good",
        beverage_type_observed="beer",
    )


def _report(brand: str = "Anytown Ale") -> VerifyReport:
    return VerifyReport(
        overall="pass",
        rule_results=[],
        extracted={"brand_name": {"value": brand, "confidence": 0.95}},
        unreadable_fields=[],
        image_quality="good",
        elapsed_ms=42,
        raw_extraction=_extraction(brand),
    )


@pytest.mark.asyncio
async def test_enrich_stamps_first_frame_signature(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0xCAFE,)

    result = await enrich_verdict(
        report=_report(),
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        persisted_cache=cache,
        persisted_hit=None,
        signature=sig,
        first_frame_signature_hex="deadbeefcafef00d",
    )
    assert result.persisted_entry_id is not None

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, result.persisted_entry_id)
        assert row is not None
        assert row.first_frame_signature_hex == "deadbeefcafef00d"


@pytest.mark.asyncio
async def test_enrich_first_frame_stamp_idempotent(cache_db):
    """A second enrich_verdict on the same row, with a different first
    frame hex, must NOT overwrite the original — the IS NULL guard
    inside ``stamp_first_frame_signature`` makes the stamp single-write."""
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0xCAFE,)

    first = await enrich_verdict(
        report=_report(),
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        persisted_cache=cache,
        persisted_hit=None,
        signature=sig,
        first_frame_signature_hex="aaaaaaaaaaaaaaaa",
    )
    second = await enrich_verdict(
        report=_report(),
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        persisted_cache=cache,
        persisted_hit=None,
        signature=sig,
        first_frame_signature_hex="bbbbbbbbbbbbbbbb",
    )

    assert first.persisted_entry_id == second.persisted_entry_id
    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, first.persisted_entry_id)
        assert row is not None
        assert row.first_frame_signature_hex == "aaaaaaaaaaaaaaaa"


@pytest.mark.asyncio
async def test_enrich_stamps_brand_name_normalized(cache_db):
    """``enrich_verdict`` must restamp the brand on the row even when
    ``upsert`` already filled it (covers older rows that pre-date the
    column where the upsert path didn't write the field)."""
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0xBEEF,)

    result = await enrich_verdict(
        report=_report(brand="Sierra Nevada"),
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        persisted_cache=cache,
        persisted_hit=None,
        signature=sig,
    )

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, result.persisted_entry_id)
        assert row is not None
        assert row.brand_name_normalized == "sierra nevada"


@pytest.mark.asyncio
async def test_enrich_no_first_frame_hex_leaves_column_null(cache_db):
    """Calling enrich_verdict without a first_frame_signature_hex (the
    /v1/verify path) must NOT touch the column — verify uploads have no
    detect-container frame to stamp."""
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0xFEED,)

    result = await enrich_verdict(
        report=_report(),
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        persisted_cache=cache,
        persisted_hit=None,
        signature=sig,
        first_frame_signature_hex=None,
    )

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, result.persisted_entry_id)
        assert row is not None
        assert row.first_frame_signature_hex is None


@pytest.mark.asyncio
async def test_enrich_handles_existing_l3_hit(cache_db):
    """When the upstream supplied a ``persisted_hit``, the upsert is
    skipped (we already have an entry_id). The stamp helpers still
    apply against the existing row, so a label whose first scan went
    through the verify path (no first_frame) and second scan through
    the detect-container path (with first_frame) ends up stamped on
    the second pass."""
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0x1,)

    # Pre-seed a row with no first_frame_signature_hex.
    first = await enrich_verdict(
        report=_report(),
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        persisted_cache=cache,
        persisted_hit=None,
        signature=sig,
        first_frame_signature_hex=None,
    )

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, first.persisted_entry_id)
        assert row is not None
        assert row.first_frame_signature_hex is None

    # Re-run via a persisted_hit (mimics a warm L3 read). Stamp should
    # now populate the column.
    from app.services.persisted_cache import PersistedHit, signature_from_hex

    async with factory() as session:
        row = await session.get(LabelCacheEntry, first.persisted_entry_id)
        from app.services.persisted_cache import extraction_from_dict

        existing_extraction = extraction_from_dict(row.extraction_json)
        hit = PersistedHit(
            entry_id=row.id,
            extraction=existing_extraction,
            external_match=None,
            explanations=None,
            min_distance=0,
            signature=signature_from_hex(row.signature_hex),
        )

    second = await enrich_verdict(
        report=_report(),
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        persisted_cache=cache,
        persisted_hit=hit,
        signature=sig,
        first_frame_signature_hex="deadbeefcafef00d",
    )
    assert second.persisted_entry_id == first.persisted_entry_id

    async with factory() as session:
        row = await session.get(LabelCacheEntry, first.persisted_entry_id)
        assert row.first_frame_signature_hex == "deadbeefcafef00d"
