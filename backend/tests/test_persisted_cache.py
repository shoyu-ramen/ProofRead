"""Unit tests for the L3 persisted perceptual cache.

Pins the module's contract: signature encoding round-trip, Hamming
arithmetic, lookup hit/miss semantics across single- and multi-panel
signatures, beverage-type scoping, enrichment write-back, and
hit-counter durability. The DB harness uses the same SQLite + schema
reset pattern as ``test_db_persistence.py`` so the tests run offline
and in full isolation.
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
    extraction_from_dict,
    extraction_to_dict,
    hamming,
    signature_from_hex,
    signature_to_hex,
)
from app.services.vision import VisionExtraction

# ---------------------------------------------------------------------------
# DB harness — separate from db_setup because we don't need the auth-stub
# user/company seed for cache-only tests; the cache table has no FKs.
# ---------------------------------------------------------------------------


@pytest.fixture
async def cache_db(tmp_path, monkeypatch):
    """Create a clean SQLite schema for one test and tear it down after.

    Mirrors the engine-override pattern in conftest.py's ``db_setup``
    but uses an async fixture so the schema reset can run inside the
    test's event loop instead of a sync ``asyncio.run`` from a sync
    fixture (which would conflict with pytest-asyncio's auto mode).
    """
    db_url = f"sqlite+aiosqlite:///{tmp_path}/persisted_cache.db"
    monkeypatch.setattr(settings, "database_url", db_url)
    engine = configure_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield db_url
    await dispose_engine()


def _extraction(brand: str = "Old Tom") -> VisionExtraction:
    return VisionExtraction(
        fields={
            "brand_name": ExtractedField(
                value=brand,
                bbox=(10, 20, 100, 50),
                confidence=0.95,
                source_image_id="panorama",
            ),
        },
        unreadable=["country_of_origin"],
        raw_response='{"brand_name":{"value":"' + brand + '"}}',
        image_quality="good",
        image_quality_notes=None,
        beverage_type_observed="spirits",
    )


# ---------------------------------------------------------------------------
# signature_to_hex / signature_from_hex
# ---------------------------------------------------------------------------


def test_signature_hex_round_trip():
    sig = (0xDEADBEEF, 0x1234567890ABCDEF, 0x0)
    encoded = signature_to_hex(sig)
    assert signature_from_hex(encoded) == sig


def test_signature_hex_uses_lowercase():
    """Lower-case canonical encoding so dhash-exact matching by string
    equality on ``upsert`` is stable across writes."""
    encoded = signature_to_hex((0xABCDEF,))
    assert encoded == encoded.lower()
    assert encoded == "abcdef"


def test_signature_hex_handles_single_panel():
    assert signature_from_hex(signature_to_hex((0xCAFE,))) == (0xCAFE,)


def test_signature_from_hex_empty_string():
    assert signature_from_hex("") == ()


# ---------------------------------------------------------------------------
# hamming
# ---------------------------------------------------------------------------


def test_hamming_zero_inputs_zero_distance():
    assert hamming(0, 0) == 0


def test_hamming_full_bit_flip():
    assert hamming(0xFF, 0) == 8


def test_hamming_xor_popcount_semantics():
    assert hamming(0b1010, 0b0101) == 4
    assert hamming(0xFFFFFFFFFFFFFFFF, 0) == 64


# ---------------------------------------------------------------------------
# extraction round-trip
# ---------------------------------------------------------------------------


def test_extraction_to_dict_round_trip():
    """A round-trip through the JSON column form must preserve every
    field the verify path consumes downstream."""
    original = _extraction("RoundTrip")
    rebuilt = extraction_from_dict(extraction_to_dict(original))
    assert rebuilt.fields["brand_name"].value == "RoundTrip"
    assert rebuilt.fields["brand_name"].bbox == (10, 20, 100, 50)
    assert rebuilt.fields["brand_name"].confidence == 0.95
    assert rebuilt.fields["brand_name"].source_image_id == "panorama"
    assert rebuilt.unreadable == ["country_of_origin"]
    assert rebuilt.image_quality == "good"
    assert rebuilt.beverage_type_observed == "spirits"


# ---------------------------------------------------------------------------
# PersistedLabelCache.lookup / upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_validates_threshold(cache_db):
    with pytest.raises(ValueError):
        PersistedLabelCache(hamming_threshold=-1)
    with pytest.raises(ValueError):
        PersistedLabelCache(hamming_threshold=65)


@pytest.mark.asyncio
async def test_upsert_then_lookup_byte_exact(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0xDEADBEEF,)
    entry_id = await cache.upsert(
        signature=sig, beverage_type="spirits", extraction=_extraction("Exact")
    )
    assert isinstance(entry_id, uuid.UUID)

    hit = await cache.lookup(signature=sig, beverage_type="spirits")
    assert hit is not None
    assert hit.entry_id == entry_id
    assert hit.min_distance == 0
    assert hit.extraction.fields["brand_name"].value == "Exact"
    assert hit.signature == sig


@pytest.mark.asyncio
async def test_lookup_within_threshold_hits(cache_db):
    """A 1-bit drift on a single panel must still hit the cached entry —
    that is the whole point of the perceptual layer over byte-exact."""
    cache = PersistedLabelCache(hamming_threshold=6)
    sig_stored = (0,)
    sig_query = (0b1,)  # exactly one bit flip
    await cache.upsert(
        signature=sig_stored, beverage_type="spirits", extraction=_extraction()
    )

    hit = await cache.lookup(signature=sig_query, beverage_type="spirits")
    assert hit is not None
    assert hit.min_distance == 1


@pytest.mark.asyncio
async def test_lookup_above_threshold_misses(cache_db):
    """7 bit-flips against a default threshold of 6 must miss — the
    threshold is a hard cutoff, not a ranking hint."""
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(0,), beverage_type="spirits", extraction=_extraction()
    )

    miss = await cache.lookup(signature=(0b1111111,), beverage_type="spirits")
    assert miss is None


@pytest.mark.asyncio
async def test_multipanel_all_panels_within_threshold_hits(cache_db):
    """Worst-of-N gating: every panel must independently sit within the
    threshold for the entry to count as a hit."""
    cache = PersistedLabelCache(hamming_threshold=6)
    stored = (0x0, 0xFF00, 0xAB, 0xCD)
    query = (0x1, 0xFF01, 0xAA, 0xCD)  # each panel ≤ 2 bit flips
    await cache.upsert(
        signature=stored, beverage_type="beer", extraction=_extraction("Beer4Panel")
    )

    hit = await cache.lookup(signature=query, beverage_type="beer")
    assert hit is not None
    assert hit.extraction.fields["brand_name"].value == "Beer4Panel"
    # Worst-case panel hamming across the four pairs.
    assert hit.min_distance == max(
        hamming(s, q) for s, q in zip(stored, query, strict=True)
    )


@pytest.mark.asyncio
async def test_multipanel_one_panel_above_threshold_misses(cache_db):
    """Front matches but back is wildly different → no hit. Prevents
    cross-promoting verdicts between bottles that share a front face."""
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(0x0, 0xFF00, 0xAB, 0xCD),
        beverage_type="beer",
        extraction=_extraction(),
    )

    miss = await cache.lookup(
        signature=(0x0, 0xFF00, 0xAB, 0xFFFFFFFFFFFFFFFF),
        beverage_type="beer",
    )
    assert miss is None


@pytest.mark.asyncio
async def test_beverage_type_scope_does_not_match_across_types(cache_db):
    """A beer extraction is not a legitimate spirits answer — the rule
    sets differ, so the cache is scoped by beverage type even when the
    perceptual signature is identical."""
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0xCAFE,)
    await cache.upsert(
        signature=sig, beverage_type="spirits", extraction=_extraction()
    )

    miss = await cache.lookup(signature=sig, beverage_type="beer")
    assert miss is None


@pytest.mark.asyncio
async def test_panel_count_scope_does_not_match_across_counts(cache_db):
    """A 2-panel signature against a 1-panel cached entry must miss —
    the merged extraction's surface IDs don't translate cleanly across
    panel counts."""
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(1, 2), beverage_type="spirits", extraction=_extraction()
    )

    miss = await cache.lookup(signature=(1,), beverage_type="spirits")
    assert miss is None


@pytest.mark.asyncio
async def test_lookup_picks_closest_match_when_multiple_eligible(cache_db):
    """Lowest worst-panel Hamming wins — verdict-reuse fidelity when
    several similar labels are persisted at the same time."""
    cache = PersistedLabelCache(hamming_threshold=10)
    await cache.upsert(
        signature=(0,), beverage_type="spirits", extraction=_extraction("Distant")
    )
    await cache.upsert(
        signature=(0xFF,), beverage_type="spirits", extraction=_extraction("Close")
    )

    # query=0b101 → hamming(0b101, 0)=2, hamming(0b101, 0xFF)=6
    hit = await cache.lookup(signature=(0b101,), beverage_type="spirits")
    assert hit is not None
    assert hit.extraction.fields["brand_name"].value == "Distant"
    assert hit.min_distance == 2


@pytest.mark.asyncio
async def test_lookup_with_none_in_signature_misses(cache_db):
    """A None panel in the signature means dhash failed upstream; the
    lookup is unreliable and we fall through to the cold path."""
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(0,), beverage_type="spirits", extraction=_extraction()
    )

    miss = await cache.lookup(
        signature=(None,),  # type: ignore[arg-type]
        beverage_type="spirits",
    )
    assert miss is None


@pytest.mark.asyncio
async def test_upsert_dhash_exact_updates_in_place(cache_db):
    """Same signature + same beverage type → update existing row, not
    insert a duplicate. The cold path occasionally produces a
    higher-confidence extraction on a second look at the same image
    (e.g. after warm cache, after model rev), so we let the second
    cold-path read replace the first."""
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0xBEEF,)
    first_id = await cache.upsert(
        signature=sig, beverage_type="spirits", extraction=_extraction("First")
    )
    second_id = await cache.upsert(
        signature=sig, beverage_type="spirits", extraction=_extraction("Second")
    )
    assert first_id == second_id

    hit = await cache.lookup(signature=sig, beverage_type="spirits")
    assert hit is not None
    assert hit.entry_id == first_id
    assert hit.extraction.fields["brand_name"].value == "Second"

    # Sanity: only one row exists for that signature.
    factory = get_session_factory()
    async with factory() as session:
        from sqlalchemy import select as sa_select

        rows = (await session.scalars(sa_select(LabelCacheEntry))).all()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Enrichment write-back (sibling workstreams)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_external_match_visible_on_next_lookup(cache_db):
    cache = PersistedLabelCache(hamming_threshold=0)
    sig = (0xC0FFEE,)
    entry_id = await cache.upsert(
        signature=sig, beverage_type="spirits", extraction=_extraction()
    )

    payload = {"ttb_id": "12345", "matched_brand": "Old Tom"}
    await cache.update_external_match(entry_id, payload)

    hit = await cache.lookup(signature=sig, beverage_type="spirits")
    assert hit is not None
    assert hit.external_match == payload
    assert hit.explanations is None


@pytest.mark.asyncio
async def test_update_explanations_visible_on_next_lookup(cache_db):
    cache = PersistedLabelCache(hamming_threshold=0)
    sig = (0xFEED,)
    entry_id = await cache.upsert(
        signature=sig, beverage_type="beer", extraction=_extraction()
    )

    explanations = {
        "beer.alcohol_content.format": "Reads as 5.5% ABV — meets the format check.",
        "beer.health_warning.present": "GOVERNMENT WARNING is present and verbatim.",
    }
    await cache.update_explanations(entry_id, explanations)

    hit = await cache.lookup(signature=sig, beverage_type="beer")
    assert hit is not None
    assert hit.explanations == explanations


# ---------------------------------------------------------------------------
# Hit counter durability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hit_count_increments_on_lookup_hit(cache_db):
    """Per-row hit_count is the durable lifetime counter — survives
    process restart, surfaced via stats() so an operator can see which
    cached labels are repeatedly valuable."""
    cache = PersistedLabelCache(hamming_threshold=6)
    sig = (0xABCD,)
    entry_id = await cache.upsert(
        signature=sig, beverage_type="wine", extraction=_extraction()
    )

    await cache.lookup(signature=sig, beverage_type="wine")
    await cache.lookup(signature=sig, beverage_type="wine")
    await cache.lookup(signature=sig, beverage_type="wine")

    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(LabelCacheEntry, entry_id)
        assert row is not None
        assert row.hit_count == 3


@pytest.mark.asyncio
async def test_stats_aggregates_entries_and_hits(cache_db):
    cache = PersistedLabelCache(hamming_threshold=6)
    await cache.upsert(
        signature=(1,), beverage_type="spirits", extraction=_extraction("A")
    )
    await cache.upsert(
        signature=(2,), beverage_type="spirits", extraction=_extraction("B")
    )

    await cache.lookup(signature=(1,), beverage_type="spirits")
    await cache.lookup(signature=(1,), beverage_type="spirits")
    await cache.lookup(signature=(2,), beverage_type="spirits")
    # Signature with all 64 bits set is Hamming-distance 63/62 from
    # the stored 1 / 2 — well above threshold so this lookup misses.
    await cache.lookup(
        signature=(0xFFFFFFFFFFFFFFFF,), beverage_type="spirits"
    )

    stats = await cache.stats()
    assert stats.total_entries == 2
    assert stats.total_hits == 3
