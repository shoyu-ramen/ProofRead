"""Integration tests: reverse-image lookup wired into the verify orchestrator.

Service-internal cache mechanics live in `test_reverse_lookup.py`. These
tests pin the *orchestrator* contract: a perceptually-similar resubmission
of a previously-verified image hits the lookup, skips the VLM call, and
still produces a fresh rule-engine verdict that respects the current
request's container size / imported flag / claim / rule fingerprint.

Why a separate suite from `test_verify_cache.py`: the byte-exact
`VerifyCache` short-circuits the whole pipeline (sensor pre-check, VLM,
rules) and returns a frozen verdict; the reverse-lookup cache short-
circuits only the VLM call and leaves the rule engine to evaluate
against the *current* request's parameters. The two layers do different
work and need different invariants pinned.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.services.reverse_lookup import ReverseLookupCache
from app.services.verify import VerifyInput, verify
from app.services.verify_cache import VerifyCache
from app.services.vision import MockVisionExtractor
from tests.conftest import _make_synthetic_png

CANONICAL_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


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


# Reusable PNG + a different-bytes-but-perceptually-equivalent
# version. The PNG and JPEG carry the same pixel content (we round-
# trip through PIL → JPEG); dhash should treat them as the same
# image.
_PNG_BYTES = _make_synthetic_png()


def _jpeg_of_png(png_bytes: bytes, *, quality: int = 85) -> bytes:
    """Re-encode a PNG as JPEG without changing pixel content meaningfully.

    Mimics the "designer re-exported the same artwork at a different
    quality / different format" workflow: same source pixels, different
    upload bytes — exactly the case the byte-exact cache misses and
    the reverse-lookup cache catches."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _bourbon_input(*, image_bytes: bytes = _PNG_BYTES, **overrides) -> VerifyInput:
    base = dict(
        image_bytes=image_bytes,
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=_BOURBON_APPLICATION,
    )
    base.update(overrides)
    return VerifyInput(**base)


@pytest.fixture
def reverse_cache() -> ReverseLookupCache:
    return ReverseLookupCache(max_entries=64, hamming_threshold=6)


class _CountingExtractor:
    """Wrap MockVisionExtractor so tests can pin extractor-call counts.

    The cache is pointless if we still pay the VLM call; this is the
    direct measurement that the reverse-lookup short-circuit fired.
    """

    def __init__(self, fixture=_PASS_FIXTURE):
        self.calls = 0
        self._inner = MockVisionExtractor(fixture)

    def extract(self, image_bytes, media_type="image/png", **kwargs):
        self.calls += 1
        return self._inner.extract(image_bytes, media_type=media_type, **kwargs)


# ---------------------------------------------------------------------------
# Cold-then-perceptually-similar: the headline contract
# ---------------------------------------------------------------------------


def test_perceptual_resubmit_skips_vlm_call(reverse_cache):
    """The whole point: a re-encoded version of the same image must hit
    the reverse-lookup cache and skip the VLM call entirely.

    The byte-exact verify cache misses (different bytes, different
    SHA-256), but the reverse-lookup cache identifies the perceptual
    sameness and reuses the cold path's extraction. The rule engine
    runs fresh on the cached extraction and produces the same verdict.
    """
    extractor = _CountingExtractor()

    # Cold path: PNG → cache the extraction
    cold = verify(_bourbon_input(), extractor=extractor, reverse_cache=reverse_cache)
    assert cold.reverse_lookup_hit is False
    assert extractor.calls == 1

    # Re-encode as JPEG → different bytes, same pixels → reverse-hit
    jpeg_bytes = _jpeg_of_png(_PNG_BYTES, quality=85)
    warm = verify(
        _bourbon_input(image_bytes=jpeg_bytes, media_type="image/jpeg"),
        extractor=extractor,
        reverse_cache=reverse_cache,
    )
    assert warm.reverse_lookup_hit is True
    assert warm.cache_hit is False, (
        "reverse-lookup hit must NOT be conflated with the byte-exact cache_hit"
    )
    assert extractor.calls == 1, (
        f"VLM call leaked through reverse-lookup; got {extractor.calls} calls"
    )

    # Verdict identity: the rule-engine output must be the same when
    # the same extraction feeds the same rule context.
    assert warm.overall == cold.overall
    assert [r.rule_id for r in warm.rule_results] == [
        r.rule_id for r in cold.rule_results
    ]
    assert [r.status for r in warm.rule_results] == [
        r.status for r in cold.rule_results
    ]


def test_byte_exact_cache_takes_priority_over_reverse_lookup(reverse_cache):
    """The byte-exact cache is the first gate; an identical-bytes
    resubmission must hit it (not reverse-lookup) so we get the full
    short-circuit (sensor pre-check + rule engine + cache deserialise)
    rather than the partial short-circuit reverse-lookup gives."""
    cache = VerifyCache(max_entries=8)
    extractor = _CountingExtractor()
    inp = _bourbon_input()

    cold = verify(
        inp, extractor=extractor, cache=cache, reverse_cache=reverse_cache
    )
    warm = verify(
        inp, extractor=extractor, cache=cache, reverse_cache=reverse_cache
    )

    assert cold.reverse_lookup_hit is False
    assert cold.cache_hit is False
    assert warm.cache_hit is True
    # Critical: byte-exact takes priority. `reverse_lookup_hit` must
    # NOT also be set on the byte-exact warm path — they are different
    # cache layers and operators tune them on different signals.
    assert warm.reverse_lookup_hit is False
    assert extractor.calls == 1


def test_reverse_lookup_reruns_rule_engine_with_current_imported_flag(reverse_cache):
    """A reverse-lookup hit must re-run the rule engine against the
    current request's parameters, not replay the cached run's verdict.

    This is the load-bearing difference from the byte-exact cache:
    the reverse cache reuses the *extraction* (the model's read of
    the label) but evaluates it against the *current* request's
    container size, imported flag, claim. Same image / extraction,
    but `is_imported=True` activates the country-of-origin presence
    rule that's `applies_if: is_imported == True` in the spirits
    rule set — so a previously-passing run must FAIL on the warm
    re-eval if no country-of-origin field was extracted."""
    # Use a fixture that DOES NOT have country_of_origin so toggling
    # `is_imported` causes the country-of-origin rule to FAIL.
    fixture_no_origin = dict(_PASS_FIXTURE)
    extractor = _CountingExtractor(fixture=fixture_no_origin)

    # Cold path with is_imported=False → country-of-origin rule
    # doesn't apply → pass.
    cold = verify(_bourbon_input(), extractor=extractor, reverse_cache=reverse_cache)
    assert cold.overall == "pass"
    assert extractor.calls == 1

    # Same image, but is_imported=True. The country-of-origin rule
    # should now apply and FAIL because the (cached) extraction has
    # no country_of_origin field.
    warm = verify(
        _bourbon_input(
            image_bytes=_jpeg_of_png(_PNG_BYTES),
            media_type="image/jpeg",
            is_imported=True,
        ),
        extractor=extractor,
        reverse_cache=reverse_cache,
    )
    assert warm.reverse_lookup_hit is True
    assert extractor.calls == 1, "VLM was re-run despite reverse-hit"

    # Pin that the rule engine ran fresh: a country-of-origin rule
    # must have been evaluated against the new is_imported=True
    # context. We expect a non-pass overall on the warm path.
    assert warm.overall != "pass", (
        "rule engine did not re-evaluate against the new is_imported flag — "
        "country-of-origin rule should now apply and fail"
    )
    coo_warm = [r for r in warm.rule_results if "country_of_origin" in r.rule_id]
    assert coo_warm, "country-of-origin rule missing from warm verdict"


def test_reverse_lookup_does_not_promote_unreadable_runs(reverse_cache):
    """Don't poison the cache with extractions that have no fields —
    those came from an unreadable frame, and reusing an empty
    extraction would defeat the cache rather than helping. The cold-
    path orchestrator gates the promotion; this test pins the
    contract."""
    extractor = _CountingExtractor(fixture={})  # empty fixture → all unreadable

    verify(_bourbon_input(), extractor=extractor, reverse_cache=reverse_cache)

    stats = reverse_cache.stats()
    assert stats.size == 0, (
        "unreadable extraction was promoted; the cache will return empty "
        "extractions on perceptual matches"
    )


def test_disabled_reverse_cache_is_a_no_op():
    """When `reverse_cache=None` the verify path runs unchanged — every
    call is cold, the VLM is called every time, and there's no
    reverse-lookup state to leak into the response."""
    extractor = _CountingExtractor()

    verify(_bourbon_input(), extractor=extractor)
    jpeg_bytes = _jpeg_of_png(_PNG_BYTES)
    second = verify(
        _bourbon_input(image_bytes=jpeg_bytes, media_type="image/jpeg"),
        extractor=extractor,
    )

    assert extractor.calls == 2, (
        "reverse-lookup behaviour should be inert when reverse_cache is None"
    )
    assert second.reverse_lookup_hit is False


def test_reverse_lookup_does_not_cross_beverage_types(reverse_cache):
    """The cache scope guarantees: a cached spirits extraction cannot
    serve a beer request, even if the upload is perceptually identical.
    The rule sets are different and a cross-scope reuse would surface
    a spirits extraction against beer rules — broken contract."""
    spirits_extractor = _CountingExtractor()
    beer_extractor = _CountingExtractor(
        fixture={
            "brand_name": "Anytown Ale",
            "class_type": "India Pale Ale",
            "alcohol_content": "5.5% ABV",
            "net_contents": "12 FL OZ",
            "name_address": "Brewed by Anytown Brewing Co., Anytown, ST",
            "health_warning": CANONICAL_WARNING,
        }
    )

    # Cold path on spirits.
    verify(
        _bourbon_input(),
        extractor=spirits_extractor,
        reverse_cache=reverse_cache,
    )
    assert spirits_extractor.calls == 1

    # Same image, beer beverage_type — must miss reverse-lookup and
    # invoke the beer extractor.
    beer_app = {
        "producer_record": {
            "brand_name": "Anytown Ale",
            "class_type": "India Pale Ale",
        }
    }
    verify(
        VerifyInput(
            image_bytes=_PNG_BYTES,
            media_type="image/png",
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            application=beer_app,
        ),
        extractor=beer_extractor,
        reverse_cache=reverse_cache,
    )
    assert beer_extractor.calls == 1, (
        "reverse-lookup leaked across beverage_type scope"
    )


def test_perceptual_resubmit_at_different_jpeg_quality(reverse_cache):
    """A designer's typical iteration: same artwork, exported at
    quality 95 the first time, quality 70 the second. Different
    bytes, same pixels at the dhash quantisation grid. Must hit."""
    extractor = _CountingExtractor()

    high_q = _jpeg_of_png(_PNG_BYTES, quality=95)
    low_q = _jpeg_of_png(_PNG_BYTES, quality=70)
    assert high_q != low_q, "test setup failure: bytes should differ"

    verify(
        _bourbon_input(image_bytes=high_q, media_type="image/jpeg"),
        extractor=extractor,
        reverse_cache=reverse_cache,
    )
    warm = verify(
        _bourbon_input(image_bytes=low_q, media_type="image/jpeg"),
        extractor=extractor,
        reverse_cache=reverse_cache,
    )

    assert warm.reverse_lookup_hit is True
    assert extractor.calls == 1


def test_visually_distinct_image_misses_reverse_lookup(reverse_cache):
    """Sanity test for the perceptual scope: a clearly-different image
    must NOT hit the cache, even when other scope attributes match.
    Otherwise we'd be reusing one label's extraction for an unrelated
    label — wrong-pass territory.

    We construct a second image with bars at completely different
    vertical positions so dhash's row-by-row gradient comparisons
    diverge. Both images pass sensor pre-check (the bars give the
    Laplacian filter contrast to latch onto)."""
    import io as _io
    import random as _random

    from PIL import Image as _Image
    from PIL import ImageDraw as _ImageDraw

    extractor = _CountingExtractor()

    verify(_bourbon_input(), extractor=extractor, reverse_cache=reverse_cache)
    assert extractor.calls == 1

    # Build a frame with bars at different horizontal positions and
    # varying widths. Same overall layout as the synthetic PNG (so
    # sensor pre-check passes), distinct dhash signature.
    width, height = 1800, 1200
    img = _Image.new("RGB", (width, height), color=(220, 230, 240))
    draw = _ImageDraw.Draw(img)
    rng = _random.Random(0xBADF00D)
    for _ in range(3000):
        x = rng.randrange(width)
        y = rng.randrange(height)
        v = rng.randrange(40, 215)
        draw.rectangle((x, y, x + 4, y + 4), fill=(v, v, v))
    # Bars in completely different positions from the default fixture.
    for x in range(150, width - 150, 200):
        draw.rectangle((x, 200, x + 80, height - 200), fill=(20, 20, 30))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    different = buf.getvalue()

    verify(
        _bourbon_input(image_bytes=different),
        extractor=extractor,
        reverse_cache=reverse_cache,
    )
    assert extractor.calls == 2, (
        "reverse-lookup matched a perceptually-distinct image; threshold "
        "is too loose or the cache is leaking across content"
    )


def test_panel_count_isolation(reverse_cache):
    """A 1-panel verify and a 2-panel verify on the same upload bytes
    must not cross-match. The merged-extraction's `source_image_id`
    shape differs (`panorama` vs. `panel_N`), so reusing across
    counts breaks the surface contract the API hands to the UI."""
    from app.services.verify import Panel

    extractor = _CountingExtractor()

    # Cold path: 1-panel
    verify(_bourbon_input(), extractor=extractor, reverse_cache=reverse_cache)
    assert extractor.calls == 1

    # Same first panel, plus a second panel — different panel count
    # → must miss reverse-lookup.
    inp_2p = _bourbon_input()
    inp_2p.extra_panels = [Panel(image_bytes=_PNG_BYTES, media_type="image/png")]
    verify(inp_2p, extractor=extractor, reverse_cache=reverse_cache)
    assert extractor.calls == 3, (
        "reverse-lookup cross-matched across panel counts; "
        f"got {extractor.calls} extractor calls (expected 1 cold + 2 panels)"
    )
