"""Unit tests for the perceptual-hash reverse-image lookup cache.

Service-level integration with the verify orchestrator lives in
`test_verify_reverse_lookup.py`. These tests pin the module's invariants
in isolation: dhash determinism, Hamming-distance hit/miss semantics,
beverage-type / panel-count scoping, LRU eviction, cache-isolation
under caller mutation, stats accuracy.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.rules.types import ExtractedField
from app.services.reverse_lookup import (
    DEFAULT_HAMMING_THRESHOLD,
    ReverseLookupCache,
    compute_dhash,
    compute_dhash_bytes,
    hamming,
)
from app.services.vision import VisionExtraction
from tests.conftest import _make_synthetic_png


# ---------------------------------------------------------------------------
# dhash + hamming
# ---------------------------------------------------------------------------


def _png_bytes(text: str = "TEST", **kwargs) -> bytes:
    return _make_synthetic_png(text=text, **kwargs)


def test_dhash_returns_64_bit_int_from_pil_image():
    img = Image.open(io.BytesIO(_png_bytes()))
    h = compute_dhash(img)
    assert h is not None
    assert isinstance(h, int)
    # 64-bit unsigned space (8×8 grid → 64 booleans).
    assert 0 <= h < (1 << 64)


def test_dhash_deterministic_for_same_image():
    img_bytes = _png_bytes()
    h1 = compute_dhash_bytes(img_bytes)
    h2 = compute_dhash_bytes(img_bytes)
    assert h1 == h2 and h1 is not None


def test_dhash_invariant_to_jpeg_re_encode():
    """dhash on the PIL image is identical regardless of the encoding
    of the source bytes — the whole point of the reverse-lookup cache.

    A re-encoded JPEG of the same artwork must dhash to a Hamming
    distance well below the default threshold (≤6 over 64 bits) so a
    designer's "re-export at 90 % quality" hits the cached verdict.
    """
    src = Image.open(io.BytesIO(_png_bytes()))

    buf_high = io.BytesIO()
    src.save(buf_high, format="JPEG", quality=95)
    h_high = compute_dhash_bytes(buf_high.getvalue())

    buf_low = io.BytesIO()
    src.save(buf_low, format="JPEG", quality=60)
    h_low = compute_dhash_bytes(buf_low.getvalue())

    assert h_high is not None and h_low is not None
    # dhash quantises to 8×8 grayscale, which collapses JPEG block
    # artifacts. Quality 60 introduces visible blockiness on a
    # synthetic noisy frame so a couple of bits of drift is realistic;
    # the binding requirement is that the Hamming distance lands well
    # below the 6-bit default threshold so a designer's "re-export at
    # different quality" still hits the cache. Comfortably below 6
    # is the contract we ship.
    assert hamming(h_high, h_low) < DEFAULT_HAMMING_THRESHOLD


def test_dhash_distinguishes_clearly_different_images():
    """Two visually-different labels must dhash far apart — well above
    threshold so the cache doesn't conflate them as the same label.

    `_make_synthetic_png`'s `text` parameter only rasterises ~30 px at
    the top edge of an 1800-px-wide frame, which dhash's 8×8 grid
    collapses to one or two cell differences. Use the `flat` /
    `bright` toggles instead — those rewrite a meaningful share of
    the frame and produce dhash signatures that diverge at the bit
    level the way real distinct labels would."""
    h_a = compute_dhash_bytes(_png_bytes())
    h_b = compute_dhash_bytes(_png_bytes(flat=True))
    assert h_a is not None and h_b is not None
    assert hamming(h_a, h_b) > DEFAULT_HAMMING_THRESHOLD


def test_dhash_returns_none_on_garbage_bytes():
    """Decode failure must NOT throw. The orchestrator treats None as
    "skip reverse-lookup" and falls through to the cold path —
    perceptual hashing is a speed optimisation, never a correctness path."""
    assert compute_dhash_bytes(b"not a real image") is None
    assert compute_dhash_bytes(b"") is None


def test_hamming_basics():
    assert hamming(0, 0) == 0
    assert hamming(0, 1) == 1
    assert hamming(0xFFFFFFFFFFFFFFFF, 0) == 64
    # XOR + popcount semantics.
    assert hamming(0b1010, 0b0101) == 4


# ---------------------------------------------------------------------------
# ReverseLookupCache.get / put
# ---------------------------------------------------------------------------


def _extraction(brand: str = "Old Tom") -> VisionExtraction:
    return VisionExtraction(
        fields={
            "brand_name": ExtractedField(
                value=brand,
                bbox=None,
                confidence=0.95,
                source_image_id="panorama",
            ),
        },
        unreadable=[],
        raw_response="{}",
        image_quality="good",
        image_quality_notes=None,
        beverage_type_observed="spirits",
    )


def test_put_then_exact_match_hits():
    cache = ReverseLookupCache(max_entries=8, hamming_threshold=6)
    sig = (0xDEADBEEF,)
    cache.put(signature=sig, beverage_type="spirits", extraction=_extraction())

    hit = cache.get(signature=sig, beverage_type="spirits")
    assert hit is not None
    assert hit.min_distance == 0
    assert hit.extraction.fields["brand_name"].value == "Old Tom"


def test_within_threshold_hits():
    cache = ReverseLookupCache(max_entries=8, hamming_threshold=6)
    sig_a = (0,)
    sig_b = (0b111111,)  # exactly 6 bits → at threshold, must hit
    cache.put(signature=sig_a, beverage_type="spirits", extraction=_extraction())

    hit = cache.get(signature=sig_b, beverage_type="spirits")
    assert hit is not None
    assert hit.min_distance == 6


def test_just_above_threshold_misses():
    cache = ReverseLookupCache(max_entries=8, hamming_threshold=6)
    sig_a = (0,)
    sig_b = (0b1111111,)  # 7 bits → just over the cutoff, must miss
    cache.put(signature=sig_a, beverage_type="spirits", extraction=_extraction())

    miss = cache.get(signature=sig_b, beverage_type="spirits")
    assert miss is None


def test_beverage_type_scopes_lookups():
    """Same signature but different beverage_type must not cross-match.

    A beer-rule extraction is not a legitimate spirits-rule answer; the
    rule sets are different and the field semantics ("class_type" means
    different things on a beer vs. spirits label) don't translate
    cleanly. Scoping by beverage_type keeps the cache honest."""
    cache = ReverseLookupCache(max_entries=8, hamming_threshold=6)
    sig = (0xCAFE,)
    cache.put(signature=sig, beverage_type="spirits", extraction=_extraction())

    miss = cache.get(signature=sig, beverage_type="beer")
    assert miss is None


def test_panel_count_scopes_lookups():
    """A 2-panel and 1-panel signature must not cross-match — the merged
    extraction's `source_image_id` shape ("panel_N" vs "panorama")
    differs, and reusing across counts would break the surface-tagging
    contract the API surfaces to the UI."""
    cache = ReverseLookupCache(max_entries=8, hamming_threshold=6)
    cache.put(signature=(1, 2), beverage_type="spirits", extraction=_extraction())

    miss = cache.get(signature=(1,), beverage_type="spirits")
    assert miss is None


def test_multipanel_lookup_requires_all_panels_within_threshold():
    """Worst-of-N gating: if any panel exceeds the Hamming threshold,
    the whole entry misses. This prevents a "front matches but back
    doesn't" cross-promotion where the two physical bottles share a
    front face but carry different back-panel content."""
    cache = ReverseLookupCache(max_entries=8, hamming_threshold=6)
    cache.put(
        signature=(0, 0), beverage_type="spirits", extraction=_extraction()
    )

    # Panel 0 matches exactly, panel 1 is way off.
    miss = cache.get(
        signature=(0, 0xFFFFFFFFFFFFFFFF), beverage_type="spirits"
    )
    assert miss is None


def test_picks_closest_match_when_multiple_eligible():
    """When multiple cache entries are within threshold, the closest
    (lowest worst-panel Hamming) wins. Determines verdict reuse fidelity
    when several similar labels are resident at the same time."""
    cache = ReverseLookupCache(max_entries=8, hamming_threshold=10)
    cache.put(
        signature=(0,), beverage_type="spirits", extraction=_extraction("Distant")
    )  # distance 5
    cache.put(
        signature=(0xFF,),
        beverage_type="spirits",
        extraction=_extraction("Close"),
    )  # distance 7 from query 0xFE

    # Pick the one closest to a query that's "in between".
    hit = cache.get(signature=(0b101,), beverage_type="spirits")
    assert hit is not None
    # query=0b101 → hamming(0b101, 0)=2, hamming(0b101, 0xFF)=6
    # closest is the first entry → Distant
    assert hit.extraction.fields["brand_name"].value == "Distant"
    assert hit.min_distance == 2


def test_signature_with_none_entry_is_a_miss():
    """A None in the signature means dhash failed on at least one
    panel; the lookup is unreliable and we must fall through to the
    cold path. Also bumps the miss counter so dashboards reflect the
    failed-hash mode."""
    cache = ReverseLookupCache(max_entries=8, hamming_threshold=6)
    cache.put(signature=(0,), beverage_type="spirits", extraction=_extraction())

    miss = cache.get(signature=(None,), beverage_type="spirits")  # type: ignore[arg-type]
    assert miss is None
    stats = cache.stats()
    assert stats.misses == 1
    assert stats.hits == 0


# ---------------------------------------------------------------------------
# LRU + capacity
# ---------------------------------------------------------------------------


def test_lru_evicts_oldest_when_full():
    cache = ReverseLookupCache(max_entries=2, hamming_threshold=0)
    cache.put(signature=(1,), beverage_type="spirits", extraction=_extraction("A"))
    cache.put(signature=(2,), beverage_type="spirits", extraction=_extraction("B"))
    cache.put(signature=(3,), beverage_type="spirits", extraction=_extraction("C"))

    # A was evicted at insert time.
    assert cache.get(signature=(1,), beverage_type="spirits") is None
    # B and C survive.
    assert cache.get(signature=(2,), beverage_type="spirits") is not None
    assert cache.get(signature=(3,), beverage_type="spirits") is not None


def test_hit_promotes_entry_to_lru_head():
    """Touched entries don't get evicted on the next insert — same
    LRU promotion semantics `verify_cache.VerifyCache` uses, so a
    repeat-hitter stays resident even when the cache is hot."""
    cache = ReverseLookupCache(max_entries=2, hamming_threshold=0)
    cache.put(signature=(1,), beverage_type="spirits", extraction=_extraction("A"))
    cache.put(signature=(2,), beverage_type="spirits", extraction=_extraction("B"))

    # Touch A — bumps to LRU head. Cache order: [B, A].
    assert cache.get(signature=(1,), beverage_type="spirits") is not None

    # Insert C — evicts the oldest (B).
    cache.put(signature=(3,), beverage_type="spirits", extraction=_extraction("C"))

    # B was evicted, A and C survive.
    assert cache.get(signature=(1,), beverage_type="spirits") is not None
    assert cache.get(signature=(2,), beverage_type="spirits") is None
    assert cache.get(signature=(3,), beverage_type="spirits") is not None


def test_clear_drops_everything():
    cache = ReverseLookupCache(max_entries=4)
    cache.put(signature=(1,), beverage_type="spirits", extraction=_extraction())
    cache.get(signature=(1,), beverage_type="spirits")  # bump hit counter
    cache.get(signature=(2,), beverage_type="spirits")  # bump miss counter

    cache.clear()
    stats = cache.stats()
    assert stats.size == 0
    assert stats.hits == 0
    assert stats.misses == 0


def test_constructor_validation():
    with pytest.raises(ValueError):
        ReverseLookupCache(max_entries=0)
    with pytest.raises(ValueError):
        ReverseLookupCache(hamming_threshold=-1)
    with pytest.raises(ValueError):
        ReverseLookupCache(hamming_threshold=65)


# ---------------------------------------------------------------------------
# Mutation isolation
# ---------------------------------------------------------------------------


def test_cache_isolates_caller_mutations_to_extracted_fields():
    """A caller mutating the returned extraction's confidence (which
    the verify orchestrator does, capping it at the surface verdict)
    must not contaminate future hits. Same fail-honestly invariant
    `verify_cache` enforces — the cache's stored entry is the cold
    path's verdict, not the side-effects of any subsequent caller."""
    cache = ReverseLookupCache(max_entries=4, hamming_threshold=0)
    cache.put(signature=(1,), beverage_type="spirits", extraction=_extraction())

    hit_a = cache.get(signature=(1,), beverage_type="spirits")
    assert hit_a is not None
    # Aggressively mutate the materialised field.
    hit_a.extraction.fields["brand_name"].confidence = 0.1
    hit_a.extraction.fields["brand_name"].value = "MUTATED"
    hit_a.extraction.fields.clear()

    hit_b = cache.get(signature=(1,), beverage_type="spirits")
    assert hit_b is not None
    assert hit_b.extraction.fields["brand_name"].value == "Old Tom", (
        "cache returned mutated extraction — snapshot isolation is broken"
    )
    assert hit_b.extraction.fields["brand_name"].confidence == 0.95


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_track_hits_and_misses():
    cache = ReverseLookupCache(max_entries=4, hamming_threshold=0)
    cache.put(signature=(1,), beverage_type="spirits", extraction=_extraction())

    cache.get(signature=(1,), beverage_type="spirits")  # hit
    cache.get(signature=(1,), beverage_type="spirits")  # hit
    cache.get(signature=(99,), beverage_type="spirits")  # miss
    cache.get(signature=(1,), beverage_type="beer")  # miss (scope)

    s = cache.stats()
    assert s.hits == 2
    assert s.misses == 2
    assert s.size == 1
    assert s.hamming_threshold == 0
