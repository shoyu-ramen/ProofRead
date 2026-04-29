"""Cache layer tests for `/v1/verify`.

Covers:

  * Cold-then-warm: an identical request hits the cache and returns the
    same verdict in well under 50 ms (the iterative-design budget).
  * Identity of the verdict: a hit returns the same `overall`,
    `rule_results`, `image_quality`, and extracted-fields summary as
    the cold path produced.
  * Invalidation on inputs that change the verdict: a different
    beverage type, container size, imported flag, or producer claim
    must miss.
  * Invalidation on rule edits: a bumped rule version invalidates
    every entry that depended on it.
  * `cache_hit` flag is False on cold and True on warm.
  * Disabled cache (max_entries=0) leaves behavior unchanged.
"""

from __future__ import annotations

import time

import pytest

from app.rules import loader as rules_loader
from app.services.verify import VerifyInput, verify
from app.services.verify_cache import VerifyCache, make_cache_key
from app.services.vision import MockVisionExtractor
from tests.conftest import _make_synthetic_png

CANONICAL_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)

_GOOD_PNG = _make_synthetic_png()

_PASS_FIXTURE = {
    "brand_name": "Old Tom Distillery",
    "class_type": "Kentucky Straight Bourbon Whiskey",
    "alcohol_content": "45% Alc./Vol. (90 Proof)",
    "net_contents": "750 mL",
    "name_address": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
    "health_warning": CANONICAL_WARNING,
}

_BOURBON_APPLICATION = {
    "producer_record": {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45",
        "net_contents": "750 mL",
        "name_address": "Old Tom Distilling Co., Bardstown, Kentucky",
        "country_of_origin": "USA",
    }
}


def _bourbon_input(**overrides) -> VerifyInput:
    base = dict(
        image_bytes=_GOOD_PNG,
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=_BOURBON_APPLICATION,
    )
    base.update(overrides)
    return VerifyInput(**base)


@pytest.fixture
def cache() -> VerifyCache:
    return VerifyCache(max_entries=64)


@pytest.fixture
def extractor() -> MockVisionExtractor:
    return MockVisionExtractor(_PASS_FIXTURE)


# ---------------------------------------------------------------------------
# Cold → warm
# ---------------------------------------------------------------------------


def test_cold_miss_then_warm_hit(cache: VerifyCache, extractor: MockVisionExtractor):
    """First request: cold path, no cache_hit. Second request with the same
    inputs: cache_hit=True and the verdict is identical."""
    inp = _bourbon_input()

    cold = verify(inp, extractor=extractor, cache=cache)
    warm = verify(inp, extractor=extractor, cache=cache)

    assert cold.cache_hit is False
    assert warm.cache_hit is True
    assert cold.overall == warm.overall
    assert [r.rule_id for r in cold.rule_results] == [
        r.rule_id for r in warm.rule_results
    ]
    assert [r.status for r in cold.rule_results] == [
        r.status for r in warm.rule_results
    ]
    assert cold.extracted == warm.extracted
    assert cold.image_quality == warm.image_quality

    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.size == 1


def test_cache_hit_meets_50ms_budget(cache: VerifyCache):
    """The whole point: warm-path verify resolves in <50 ms.

    Uses `MockVisionExtractor` so the cold path itself is fast (no
    network); the assertion is on the warm path, where we want to
    confirm the rule fingerprint + hash + LRU lookup come in well
    under the iterative-design budget.
    """
    extractor = MockVisionExtractor(_PASS_FIXTURE)
    inp = _bourbon_input()

    # Prime.
    verify(inp, extractor=extractor, cache=cache)

    elapsed: list[int] = []
    for _ in range(5):
        t0 = time.monotonic()
        warm = verify(inp, extractor=extractor, cache=cache)
        elapsed.append(int((time.monotonic() - t0) * 1000))
        assert warm.cache_hit is True

    # All five warm paths must clear the budget. Use a slightly looser
    # threshold for noise tolerance on shared CI runners — the goal is
    # "no order-of-magnitude regressions", not bit-exact perf.
    assert max(elapsed) < 50, (
        f"cache-hit p100 {max(elapsed)} ms exceeds 50 ms budget; samples={elapsed}"
    )


def test_cache_hit_does_not_call_extractor(cache: VerifyCache):
    """A warm read must NOT invoke the vision extractor; that's the whole
    cost we're skipping. We assert this by counting how many times the
    mock's `extract` method was called."""

    class _CountingExtractor:
        def __init__(self) -> None:
            self.calls = 0
            self._inner = MockVisionExtractor(_PASS_FIXTURE)

        def extract(self, image_bytes, media_type="image/png", **kwargs):
            self.calls += 1
            return self._inner.extract(image_bytes, media_type=media_type, **kwargs)

    extractor = _CountingExtractor()
    inp = _bourbon_input()

    verify(inp, extractor=extractor, cache=cache)
    verify(inp, extractor=extractor, cache=cache)
    verify(inp, extractor=extractor, cache=cache)

    assert extractor.calls == 1, (
        f"expected one cold call; got {extractor.calls}. The cache is failing to short-circuit."
    )


# ---------------------------------------------------------------------------
# Invalidation: the inputs that go into the cache key
# ---------------------------------------------------------------------------


def test_different_beverage_type_misses(cache: VerifyCache, extractor):
    verify(_bourbon_input(), extractor=extractor, cache=cache)
    other = _bourbon_input(beverage_type="beer")
    second = verify(other, extractor=extractor, cache=cache)
    assert second.cache_hit is False


def test_different_container_size_misses(cache: VerifyCache, extractor):
    verify(_bourbon_input(), extractor=extractor, cache=cache)
    second = verify(
        _bourbon_input(container_size_ml=375), extractor=extractor, cache=cache
    )
    assert second.cache_hit is False


def test_different_imported_flag_misses(cache: VerifyCache, extractor):
    verify(_bourbon_input(), extractor=extractor, cache=cache)
    second = verify(
        _bourbon_input(is_imported=True), extractor=extractor, cache=cache
    )
    assert second.cache_hit is False


def test_different_producer_record_misses(cache: VerifyCache, extractor):
    """Changing the claim is a legitimate verdict-changing input — the
    rule engine cross-checks the claim against the label, so a new claim
    can flip a PASS to a WARN even on the same image."""
    verify(_bourbon_input(), extractor=extractor, cache=cache)
    different_claim = _bourbon_input(
        application={
            "producer_record": {
                **_BOURBON_APPLICATION["producer_record"],
                "brand_name": "Different Brand",
            }
        }
    )
    second = verify(different_claim, extractor=extractor, cache=cache)
    assert second.cache_hit is False


def test_different_image_bytes_misses(cache: VerifyCache, extractor):
    verify(_bourbon_input(), extractor=extractor, cache=cache)
    different = _bourbon_input(image_bytes=_make_synthetic_png(text="OTHER"))
    second = verify(different, extractor=extractor, cache=cache)
    assert second.cache_hit is False


def test_application_key_order_is_irrelevant(cache: VerifyCache, extractor):
    """JSON object ordering must not affect the hash — otherwise two
    callers with the same data but different dict-construction order
    would miss each other."""
    record_a = {"producer_record": {"brand_name": "Old Tom", "class_type": "Bourbon"}}
    record_b = {"producer_record": {"class_type": "Bourbon", "brand_name": "Old Tom"}}

    verify(_bourbon_input(application=record_a), extractor=extractor, cache=cache)
    second = verify(
        _bourbon_input(application=record_b), extractor=extractor, cache=cache
    )
    assert second.cache_hit is True


# ---------------------------------------------------------------------------
# Rule-set invalidation
# ---------------------------------------------------------------------------


def test_cache_key_changes_when_rules_change(cache: VerifyCache):
    """Cache keys must include a rule-set fingerprint — bumping a rule
    version is a verdict-changing edit and should invalidate."""
    from dataclasses import replace

    rules_v1 = rules_loader.load_rules(beverage_type="spirits")
    key_v1 = make_cache_key(
        panels=[(_GOOD_PNG, "image/png")],
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=_BOURBON_APPLICATION,
        rules=rules_v1,
    )

    # Simulate a rule edit: bump one rule's version.
    rules_v2 = [replace(rules_v1[0], version=rules_v1[0].version + 1)] + list(
        rules_v1[1:]
    )
    key_v2 = make_cache_key(
        panels=[(_GOOD_PNG, "image/png")],
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=_BOURBON_APPLICATION,
        rules=rules_v2,
    )
    assert key_v1 != key_v2


# ---------------------------------------------------------------------------
# Cache plumbing
# ---------------------------------------------------------------------------


def test_lru_evicts_oldest_when_full():
    cache = VerifyCache(max_entries=2)
    extractor = MockVisionExtractor(_PASS_FIXTURE)
    a = _bourbon_input(image_bytes=_make_synthetic_png(text="A"))
    b = _bourbon_input(image_bytes=_make_synthetic_png(text="B"))
    c = _bourbon_input(image_bytes=_make_synthetic_png(text="C"))

    verify(a, extractor=extractor, cache=cache)  # miss, cache=[a]
    verify(b, extractor=extractor, cache=cache)  # miss, cache=[a, b]
    verify(c, extractor=extractor, cache=cache)  # miss, evicts a, cache=[b, c]

    # Touch b — LRU bookkeeping bumps it to the end. Cache is now [c, b].
    again_b = verify(b, extractor=extractor, cache=cache)
    assert again_b.cache_hit is True

    # a was evicted at insert time, so this is a fresh miss. The miss
    # *also* pushes a back in, evicting the now-oldest entry (c).
    again_a = verify(a, extractor=extractor, cache=cache)
    assert again_a.cache_hit is False

    # c was evicted by the previous step's insert.
    again_c = verify(c, extractor=extractor, cache=cache)
    assert again_c.cache_hit is False


def test_cache_isolates_caller_mutations(cache: VerifyCache, extractor):
    """A caller mutating the returned report must NOT corrupt the cached
    snapshot. This is the latent-bug guard: the cold path constructs
    fresh lists today, but if any future code path mutated
    `report.rule_results` after `verify()` returned, a shallow-copy cache
    would propagate the corruption to every subsequent hit."""
    inp = _bourbon_input()

    cold = verify(inp, extractor=extractor, cache=cache)
    cold.rule_results.clear()  # simulate an aggressive caller
    cold.extracted.clear()
    cold.unreadable_fields.append("evil")

    warm = verify(inp, extractor=extractor, cache=cache)

    assert warm.cache_hit is True
    assert len(warm.rule_results) > 0, "cache returned mutated rule_results"
    assert "evil" not in warm.unreadable_fields, (
        "cache returned mutated unreadable_fields"
    )
    assert warm.extracted, "cache returned emptied extracted dict"


def test_no_cache_path_unchanged():
    """Calling verify() without a cache must work exactly as before — no
    accidental dependency on the cache being supplied."""
    extractor = MockVisionExtractor(_PASS_FIXTURE)
    report = verify(_bourbon_input(), extractor=extractor)
    assert report.overall == "pass"
    assert report.cache_hit is False


def test_cache_internal_storage_is_immutable_snapshot(cache: VerifyCache, extractor):
    """Item #4 freeze pattern: the cache's internal entry is a frozen
    snapshot, not a mutable VerifyReport. A direct attempt to reassign
    on the snapshot raises FrozenInstanceError — the read shape is
    locked by construction rather than by recursive copy.

    This is the "isolation by construction" guarantee that lets us drop
    the deepcopy: if the snapshot itself can't be mutated, neither
    a returned-report mutation nor a future cold-path bug can leak
    state across cache hits.
    """
    from dataclasses import FrozenInstanceError

    from app.services.verify_cache import _Snapshot

    inp = _bourbon_input()
    verify(inp, extractor=extractor, cache=cache)

    # Reach into the internal LRU and confirm the stored entry is a
    # _Snapshot, not a VerifyReport.
    [stored_key] = list(cache._cache.keys())
    snap = cache._cache[stored_key]
    assert isinstance(snap, _Snapshot)

    # The frozen dataclass refuses attribute reassignment.
    with pytest.raises(FrozenInstanceError):
        snap.overall = "fail"
    # rule_results is a tuple, not a list — no .append / .clear available.
    assert isinstance(snap.rule_results, tuple)
    with pytest.raises(AttributeError):
        snap.rule_results.append("evil")  # type: ignore[attr-defined]


def test_zero_capacity_cache_disabled_at_api_layer():
    """When `verify_cache_max_entries=0`, `get_verify_cache()` returns
    None and the verify call runs without any caching. Verified via the
    factory rather than by passing 0 to the constructor (which raises)."""
    from app.api import verify as verify_api
    from app.config import settings

    original = settings.verify_cache_max_entries
    try:
        settings.verify_cache_max_entries = 0
        verify_api._reset_verify_cache()
        assert verify_api.get_verify_cache() is None
    finally:
        settings.verify_cache_max_entries = original
        verify_api._reset_verify_cache()
