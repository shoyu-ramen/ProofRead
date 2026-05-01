"""Tests for the known-label additions to ``PersistedLabelCache``.

Covers ``lookup_by_brand`` + ``lookup_by_first_frame`` + the
brand-name-normalized stamp on ``upsert``. SPEC §0.5 fail-honest filter
gets its own coverage so a downgraded extraction is never returned as
recognition fodder.

The DB harness mirrors the per-test SQLite + schema-reset pattern from
``test_persisted_cache.py`` — same fixture name, same async lifecycle.
"""

from __future__ import annotations

import uuid

import pytest

from app.config import settings
from app.db import configure_engine, dispose_engine, get_session_factory
from app.models import Base, LabelCacheEntry
from app.rules.types import ExtractedField
from app.services.persisted_cache import (
    PersistedLabelCache,
    signature_to_hex,
)
from app.services.vision import VisionExtraction


@pytest.fixture
async def cache_db(tmp_path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/known_label_cache.db"
    monkeypatch.setattr(settings, "database_url", db_url)
    engine = configure_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield db_url
    await dispose_engine()


def _extraction(
    *, brand: str = "Sierra Nevada", brand_confidence: float = 0.95
) -> VisionExtraction:
    return VisionExtraction(
        fields={
            "brand_name": ExtractedField(
                value=brand,
                bbox=(10, 20, 100, 50),
                confidence=brand_confidence,
                source_image_id="panorama",
            ),
            "country_of_origin": ExtractedField(
                value="USA", bbox=None, confidence=0.9
            ),
            "net_contents": ExtractedField(
                value="12 FL OZ", bbox=None, confidence=0.95
            ),
        },
        unreadable=[],
        raw_response="{}",
        image_quality="good",
        image_quality_notes=None,
        beverage_type_observed="beer",
    )


# ---------------------------------------------------------------------------
# upsert stamps brand_name_normalized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_stamps_brand_name_normalized(cache_db):
    """Upsert must lift fields.brand_name.value from the extraction
    JSON, lower-case + strip, and persist it to the indexed column so a
    later ``lookup_by_brand`` succeeds without a manual stamp call."""
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0xCAFE,)
    entry_id = await cache.upsert(
        signature=sig,
        beverage_type="beer",
        extraction=_extraction(brand="Sierra Nevada"),
    )

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, entry_id)
        assert row is not None
        assert row.brand_name_normalized == "sierra nevada"


@pytest.mark.asyncio
async def test_upsert_handles_missing_brand_in_extraction(cache_db):
    """When the extraction has no readable brand_name, the column stays
    NULL — we never write empty/whitespace strings to the index."""
    cache = PersistedLabelCache(hamming_threshold=6)
    extraction = VisionExtraction(
        fields={
            "country_of_origin": ExtractedField(
                value="USA", bbox=None, confidence=0.9
            ),
        },
        unreadable=["brand_name"],
        raw_response="{}",
        image_quality="good",
        image_quality_notes=None,
        beverage_type_observed="beer",
    )
    entry_id = await cache.upsert(
        signature=(0x1234,),
        beverage_type="beer",
        extraction=extraction,
    )
    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, entry_id)
        assert row is not None
        assert row.brand_name_normalized is None


@pytest.mark.asyncio
async def test_upsert_normalizes_whitespace_and_case(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0xABCD,),
        beverage_type="beer",
        extraction=_extraction(brand="  DOGFISH HEAD  "),
    )
    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, entry_id)
        assert row is not None
        assert row.brand_name_normalized == "dogfish head"


# ---------------------------------------------------------------------------
# lookup_by_brand
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_by_brand_exact_case_insensitive(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(0xCAFE,),
        beverage_type="beer",
        extraction=_extraction(brand="Sierra Nevada"),
    )

    hit = await cache.lookup_by_brand("beer", "  SIERRA NEVADA  ")
    assert hit is not None
    assert hit.extraction.fields["brand_name"].value == "Sierra Nevada"
    assert hit.min_distance == 0


@pytest.mark.asyncio
async def test_lookup_by_brand_returns_none_on_miss(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(0xCAFE,),
        beverage_type="beer",
        extraction=_extraction(brand="Sierra Nevada"),
    )
    assert await cache.lookup_by_brand("beer", "Dogfish Head") is None


@pytest.mark.asyncio
async def test_lookup_by_brand_filters_by_beverage_type(cache_db):
    """Two rows with the same brand text — querying "spirits" must only
    return the spirits row, not the beer one. Same brand, different
    rule sets."""
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="House"),
    )
    spirits_extraction = VisionExtraction(
        fields={
            "brand_name": ExtractedField(
                value="House", bbox=None, confidence=0.9
            ),
        },
        unreadable=[],
        raw_response="{}",
        image_quality="good",
        beverage_type_observed="spirits",
    )
    spirits_id = await cache.upsert(
        signature=(0x2,),
        beverage_type="spirits",
        extraction=spirits_extraction,
    )

    hit = await cache.lookup_by_brand("spirits", "House")
    assert hit is not None
    assert hit.entry_id == spirits_id


@pytest.mark.asyncio
async def test_lookup_by_brand_none_beverage_type_queries_all(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    beer_id = await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )

    hit = await cache.lookup_by_brand(None, "anytown")
    assert hit is not None
    assert hit.entry_id == beer_id


@pytest.mark.asyncio
async def test_lookup_by_brand_skips_low_confidence_extraction(cache_db):
    """SPEC §0.5: a row whose cached brand_name was extracted with low
    confidence is ineligible — returning it would let a disputed read
    short-circuit the panorama scan."""
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Brewco", brand_confidence=0.3),
    )
    assert await cache.lookup_by_brand("beer", "Brewco") is None


@pytest.mark.asyncio
async def test_lookup_by_brand_prefers_high_confidence_over_low(cache_db):
    """When two rows share a brand text — one downgraded, one good —
    the good one wins. The low-confidence row is filtered out, the
    second candidate is returned."""
    cache = PersistedLabelCache(hamming_threshold=6)
    bad_id = await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Shared", brand_confidence=0.3),
    )
    good_id = await cache.upsert(
        signature=(0x2,),
        beverage_type="beer",
        extraction=_extraction(brand="Shared", brand_confidence=0.9),
    )

    hit = await cache.lookup_by_brand("beer", "shared")
    assert hit is not None
    assert hit.entry_id == good_id
    assert hit.entry_id != bad_id


@pytest.mark.asyncio
async def test_lookup_by_brand_bumps_hit_counter(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    await cache.lookup_by_brand("beer", "Anytown")
    await cache.lookup_by_brand("beer", "Anytown")

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, entry_id)
        assert row is not None
        assert row.hit_count == 2


@pytest.mark.asyncio
async def test_lookup_by_brand_empty_string_returns_none(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    assert await cache.lookup_by_brand("beer", "") is None
    assert await cache.lookup_by_brand("beer", "   ") is None


# ---------------------------------------------------------------------------
# lookup_by_first_frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_by_first_frame_exact_match(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0xCAFE,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    await cache.stamp_first_frame_signature(entry_id, "deadbeefcafef00d")

    hit = await cache.lookup_by_first_frame("deadbeefcafef00d", "beer")
    assert hit is not None
    assert hit.entry_id == entry_id
    assert hit.min_distance == 0


@pytest.mark.asyncio
async def test_lookup_by_first_frame_within_threshold(cache_db):
    """1-bit drift on the first-frame hash must still hit the cached
    entry — that's the perceptual match the threshold buys us."""
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0xCAFE,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    await cache.stamp_first_frame_signature(entry_id, "0")

    # Query: 0b1 → Hamming(0, 1) = 1
    hit = await cache.lookup_by_first_frame("1", "beer")
    assert hit is not None
    assert hit.entry_id == entry_id
    assert hit.min_distance == 1


@pytest.mark.asyncio
async def test_lookup_by_first_frame_above_threshold_misses(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0xCAFE,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    await cache.stamp_first_frame_signature(entry_id, "0")
    # Query: 0xFF → Hamming(0, 0xFF) = 8 > threshold 6.
    miss = await cache.lookup_by_first_frame("ff", "beer")
    assert miss is None


@pytest.mark.asyncio
async def test_lookup_by_first_frame_filters_by_beverage_type(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    beer_id = await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    spirits_id = await cache.upsert(
        signature=(0x2,),
        beverage_type="spirits",
        extraction=_extraction(brand="Other"),
    )
    await cache.stamp_first_frame_signature(beer_id, "abc")
    await cache.stamp_first_frame_signature(spirits_id, "abc")

    hit = await cache.lookup_by_first_frame("abc", "spirits")
    assert hit is not None
    assert hit.entry_id == spirits_id


@pytest.mark.asyncio
async def test_lookup_by_first_frame_skips_unstamped_rows(cache_db):
    """Rows without a first_frame_signature_hex must never match — the
    query column is NULL, the lookup short-circuits. Without this,
    every row (where first_frame is NULL) could be erroneously
    selected by the WHERE-clause."""
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    miss = await cache.lookup_by_first_frame("abc", "beer")
    assert miss is None


@pytest.mark.asyncio
async def test_lookup_by_first_frame_skips_low_confidence(cache_db):
    """SPEC §0.5: a low-confidence cached extraction is ineligible for
    first-frame lookup the same way it is for brand lookup."""
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Bad", brand_confidence=0.3),
    )
    await cache.stamp_first_frame_signature(entry_id, "abc")
    miss = await cache.lookup_by_first_frame("abc", "beer")
    assert miss is None


@pytest.mark.asyncio
async def test_lookup_by_first_frame_invalid_hex_returns_none(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    await cache.stamp_first_frame_signature(entry_id, "abc")
    assert await cache.lookup_by_first_frame("not-hex", "beer") is None
    assert await cache.lookup_by_first_frame("", "beer") is None


@pytest.mark.asyncio
async def test_lookup_by_first_frame_picks_closest(cache_db):
    cache = PersistedLabelCache(hamming_threshold=10)
    a_id = await cache.upsert(
        signature=(0x10,),
        beverage_type="beer",
        extraction=_extraction(brand="Distant"),
    )
    b_id = await cache.upsert(
        signature=(0x11,),
        beverage_type="beer",
        extraction=_extraction(brand="Close"),
    )
    await cache.stamp_first_frame_signature(a_id, "0")
    await cache.stamp_first_frame_signature(b_id, "ff")

    # Query: 0b101 → Hamming(0, 5)=2, Hamming(0xFF, 5)=6
    hit = await cache.lookup_by_first_frame("5", "beer")
    assert hit is not None
    assert hit.entry_id == a_id
    assert hit.min_distance == 2


# ---------------------------------------------------------------------------
# stamp helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stamp_first_frame_idempotent(cache_db):
    """The IS NULL guard means the second stamp call is a no-op — the
    first observed frame wins so the L3 row carries the earliest hash
    we ever saw for that label."""
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    await cache.stamp_first_frame_signature(entry_id, "abcdef")
    await cache.stamp_first_frame_signature(entry_id, "deadbeef")

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, entry_id)
        assert row is not None
        assert row.first_frame_signature_hex == "abcdef"


@pytest.mark.asyncio
async def test_stamp_brand_name_idempotent_on_existing_row(cache_db):
    """When the upsert already filled brand_name_normalized, the
    explicit stamp method does not overwrite it (the IS NULL guard
    prevents silent rewrites)."""
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    await cache.stamp_brand_name_normalized(entry_id, "Different Brand")

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, entry_id)
        assert row is not None
        assert row.brand_name_normalized == "anytown"


@pytest.mark.asyncio
async def test_stamp_brand_name_fills_null_column(cache_db):
    """When the upsert didn't fill the column (older row, no extracted
    brand_name) the explicit stamp populates it. Used by enrichment
    after the cold path to backfill rows that pre-date the column."""
    cache = PersistedLabelCache(hamming_threshold=6)
    factory = get_session_factory()
    async with factory() as session:
        entry = LabelCacheEntry(
            id=uuid.uuid4(),
            beverage_type="beer",
            panel_count=1,
            signature_hex=signature_to_hex((0x1,)),
            extraction_json={"fields": {}},
            brand_name_normalized=None,
        )
        session.add(entry)
        await session.commit()
        entry_id = entry.id

    await cache.stamp_brand_name_normalized(entry_id, "Backfill Brand")

    async with factory() as session:
        row = await session.get(LabelCacheEntry, entry_id)
        assert row is not None
        assert row.brand_name_normalized == "backfill brand"


@pytest.mark.asyncio
async def test_stamp_brand_name_handles_none(cache_db):
    """A None / empty brand_name argument is a no-op."""
    cache = PersistedLabelCache(hamming_threshold=6)
    entry_id = await cache.upsert(
        signature=(0x1,),
        beverage_type="beer",
        extraction=_extraction(brand="Anytown"),
    )
    await cache.stamp_brand_name_normalized(entry_id, None)
    await cache.stamp_brand_name_normalized(entry_id, "   ")

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, entry_id)
        assert row is not None
        assert row.brand_name_normalized == "anytown"
