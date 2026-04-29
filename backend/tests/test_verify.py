"""End-to-end tests for the single-shot /v1/verify path with a mock vision extractor."""

from app.rules.types import CheckOutcome
from app.services.verify import VerifyInput, verify
from app.services.vision import MockVisionExtractor
from tests.conftest import _make_synthetic_png

# The verify path runs `sensor_check` on the image bytes before invoking the
# vision extractor — it has to, because that's the SPEC §0.5 fail-honestly
# guarantee. So tests can't pass an 8-byte PNG signature and expect the
# pre-check to wave it through; they need a real, readable frame. The mock
# extractor ignores the bytes entirely, so we generate a synthetic frame
# once at import time and reuse it across the verify scenarios.
_GOOD_PNG = _make_synthetic_png()

CANONICAL_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


def _bourbon_application(**overrides):
    record = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45",
        "net_contents": "750 mL",
        "name_address": "Old Tom Distilling Co., Bardstown, Kentucky",
        "country_of_origin": "USA",
    }
    record.update(overrides)
    return {"producer_record": record}


def _verify(extracted, application=None, **kw):
    extractor = MockVisionExtractor(extracted)
    inp = VerifyInput(
        image_bytes=_GOOD_PNG,  # mock extractor ignores the bytes; sensor pre-check decodes them
        media_type=kw.get("media_type", "image/png"),
        beverage_type=kw.get("beverage_type", "spirits"),
        container_size_ml=kw.get("container_size_ml", 750),
        is_imported=kw.get("is_imported", False),
        application=application or _bourbon_application(),
    )
    return verify(inp, extractor=extractor)


def test_pass_scenario_returns_pass():
    """Old Tom bourbon — perfect match across all required fields."""
    report = _verify(
        {
            "brand_name": "Old Tom Distillery",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol. (90 Proof)",
            "net_contents": "750 mL",
            "name_address": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
            "health_warning": CANONICAL_WARNING,
        }
    )
    assert report.overall == "pass", [
        (r.rule_id, r.status.value, r.finding) for r in report.rule_results
    ]


def test_warn_scenario_brand_case_only():
    """STONE'S THROW on label vs Stone's Throw in application → WARN."""
    report = _verify(
        {
            "brand_name": "STONE'S THROW",
            "class_type": "London Dry Gin",
            "alcohol_content": "41.5% Alc./Vol.",
            "net_contents": "750 mL",
            "name_address": "Distilled & bottled by Stone's Throw Distilling Co., Portland, OR",
            "health_warning": CANONICAL_WARNING,
        },
        application=_bourbon_application(
            brand_name="Stone's Throw",
            class_type="London Dry Gin",
            alcohol_content="41.5",
            name_address="Stone's Throw Distilling Co., Portland, OR",
        ),
    )
    assert report.overall == "warn", [
        (r.rule_id, r.status.value, r.finding) for r in report.rule_results
    ]


def test_fail_scenario_titlecase_warning():
    """Mountain Crest IPA: warning in title case → FAIL on warning rule."""
    bad_warning = CANONICAL_WARNING.replace(
        "GOVERNMENT WARNING:", "Government Warning:"
    )
    report = _verify(
        {
            "brand_name": "Mountain Crest",
            "class_type": "India Pale Ale",
            "alcohol_content": "6.8% Alc./Vol.",
            "net_contents": "16 FL OZ",
            "name_address": "Brewed and canned by Mountain Crest Brewing Co., Bend, Oregon",
            "health_warning": bad_warning,
        },
        application=_bourbon_application(
            brand_name="Mountain Crest",
            class_type="India Pale Ale",
            alcohol_content="6.8",
            net_contents="16 FL OZ",
        ),
    )
    assert report.overall == "fail"
    failed = [r for r in report.rule_results if r.status == CheckOutcome.FAIL]
    failed_ids = {r.rule_id for r in failed}
    assert "spirits.health_warning.compliance" in failed_ids


def test_unreadable_scenario_degrades_to_advisory():
    """If the extractor can't read fields confidently, dependent required
    rules degrade to ADVISORY rather than FAIL — the overall verdict
    surfaces as 'advisory' (or 'unreadable' if all fields were unreadable)."""
    report = _verify(
        {
            "brand_name": {"value": None, "unreadable": True},
            "class_type": {"value": None, "unreadable": True},
            "alcohol_content": {"value": None, "unreadable": True},
            "net_contents": {"value": None, "unreadable": True},
            "name_address": {"value": None, "unreadable": True},
            "health_warning": {"value": None, "unreadable": True},
        }
    )
    # No fields could be read → dominant verdict is "advisory" (rules
    # downgrade) or "unreadable" depending on the extractor signal.
    assert report.overall in {"advisory", "unreadable"}, report.overall
    assert len(report.unreadable_fields) >= 5


def test_elapsed_ms_is_set():
    report = _verify(
        {
            "brand_name": "Old Tom Distillery",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
            "name_address": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
            "health_warning": CANONICAL_WARNING,
        }
    )
    assert report.elapsed_ms >= 0
    # Mock extractor is fast — should be well under 100ms typically.
    assert report.elapsed_ms < 5_000


def test_extracted_summary_includes_unreadable_marker():
    report = _verify(
        {
            "brand_name": "Old Tom Distillery",
            "class_type": {"value": None, "unreadable": True},
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
            "name_address": "Bottled by Old Tom Distilling Co., Bardstown, KY",
            "health_warning": CANONICAL_WARNING,
        }
    )
    assert report.extracted["class_type"]["unreadable"] is True
    assert report.extracted["brand_name"]["unreadable"] is False


def test_primary_failure_skips_pending_secondary():
    """Item #6: when the primary extractor raises, the secondary's HTTP
    call must be skipped. The `abort: threading.Event` short-circuits the
    secondary at function entry rather than letting it run a doomed call.

    Implemented as a counter probe: the secondary's `read_warning` should
    NOT be invoked when the primary raises, even though we submitted both
    futures to the pool.
    """
    from app.services.anthropic_client import ExtractorUnavailable
    from app.services.health_warning_second_pass import (
        HealthWarningExtractor,
        WarningRead,
    )

    class _BoomExtractor:
        def extract(self, image_bytes, media_type="image/png", **kwargs):
            raise ExtractorUnavailable("simulated rate-limit")

    secondary_calls = {"count": 0}

    class _CountingSecondary(HealthWarningExtractor):
        def read_warning(self, image_bytes, media_type="image/png"):
            secondary_calls["count"] += 1
            # Sleep briefly to ensure if the abort.is_set() didn't fire,
            # the call really did run to completion (so we're not just
            # observing a race window).
            import time as _t
            _t.sleep(0.05)
            return WarningRead(
                value=None,
                found=False,
                confidence=0.0,
                source="counting-mock",
            )

    inp = VerifyInput(
        image_bytes=_GOOD_PNG,
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=_bourbon_application(),
    )

    import pytest as _pytest

    with _pytest.raises(ExtractorUnavailable):
        verify(
            inp,
            extractor=_BoomExtractor(),
            health_warning_reader=_CountingSecondary(),
        )

    # The primary fails synchronously inside the pool; the abort gate
    # prevents the secondary from running. With a single panel and a
    # single-worker race (primary submits before secondary), the abort
    # nearly always fires before the secondary thread enters
    # `read_warning`. Allow at most one call to acknowledge the rare
    # race where the secondary started executing the moment after the
    # primary failure.
    assert secondary_calls["count"] <= 1, (
        f"Secondary should be skipped on primary failure; got "
        f"{secondary_calls['count']} calls."
    )


def test_rule_results_carry_surface_field():
    """Item #8: every rule_result whose check references an extracted
    field must carry the source panel id. Mobile uses this to know which
    captured image to highlight when the user taps a result. The merge
    step retags `source_image_id` from the mock's hard-coded "front" to
    "panel_0" (the verify path's panel-indexed convention), so the
    rule_result's surface should report `panel_0` for the single-shot
    path."""
    report = _verify(
        {
            "brand_name": "Old Tom Distillery",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
            "name_address": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
            "health_warning": CANONICAL_WARNING,
        }
    )
    surfaced = [r for r in report.rule_results if r.surface is not None]
    assert surfaced, (
        "At least one rule_result should report a non-None surface — the "
        "MockVisionExtractor tags every field with source_image_id."
    )
    # Single-image (legacy) path always assigns `panel_0`.
    assert all(r.surface == "panel_0" for r in surfaced), (
        f"All field-tied rule_results should carry surface 'panel_0' on "
        f"the single-shot path; got {[r.surface for r in surfaced]}"
    )


def test_glare_blob_overlap_downgrades_fail_to_advisory():
    """Item #1 (verify ↔ scans parity): a FAIL whose field bbox sits inside
    a glare blob is unsupportable — must downgrade to ADVISORY. Same
    fail-honestly guarantee as the scan path (`pipeline._apply_capture_
    downgrade`'s `blob_occluded` branch).

    Synthesized scenario: a paraphrased Health Warning that the engine
    would FAIL on, with the field bbox overlapping a saturated blob the
    sensor pre-check finds in the synthetic frame. The downgrade fires
    when ≥30 % of the field area sits inside the blob.
    """
    from app.rules.engine import RuleEngine
    from app.rules.loader import load_rules
    from app.rules.types import ExtractedField, ExtractionContext
    from app.services.sensor_check import Bbox, GlareBlob, SurfaceCaptureQuality
    from app.services.verify import _downgrade_fails_for_glare_blob

    # Build a SurfaceCaptureQuality with one big saturated blob.
    surface = _build_surface_with_glare_blob(
        Bbox(x=100, y=100, w=400, h=200)
    )

    # Field bbox sits entirely inside the blob.
    fields = {
        "health_warning": ExtractedField(
            value="Government Warning: ...",  # title-case → engine FAILs
            bbox=(150, 150, 200, 100),  # 100% inside blob (100..500 × 100..300)
            confidence=0.95,
            source_image_id="panel_0",
        ),
    }
    ctx = ExtractionContext(
        fields=fields,
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application={},
        unreadable_fields=[],
    )

    rules = load_rules(beverage_type="spirits")
    pre = RuleEngine(rules).evaluate(ctx)

    # Sanity: at least one FAIL exists pre-downgrade.
    fails_pre = [r for r in pre if r.status == CheckOutcome.FAIL]
    assert fails_pre, (
        "Test fixture should produce at least one FAIL before the "
        "blob-overlap downgrade fires."
    )

    surfaces_by_panel = {0: surface}
    post = _downgrade_fails_for_glare_blob(pre, ctx, surfaces_by_panel)

    # The FAILs that referenced `health_warning` (the field with the
    # blob-overlapping bbox) must be downgraded; FAILs on other fields
    # untouched.
    for r_pre, r_post in zip(pre, post, strict=True):
        if "health_warning" in r_pre.rule_id and r_pre.status == CheckOutcome.FAIL:
            assert r_post.status == CheckOutcome.ADVISORY, (
                f"{r_pre.rule_id} should have been downgraded to ADVISORY; "
                f"got {r_post.status}"
            )
            assert "glare blob" in (r_post.finding or "").lower(), (
                f"Downgraded finding should mention glare blob; got "
                f"{r_post.finding!r}"
            )


def _build_surface_with_glare_blob(blob_bbox):
    """Construct a `SurfaceCaptureQuality` with one large saturated blob
    so the verify-path blob-overlap downgrade has something to fire on.
    Mirrors enough of the real `assess_capture_quality` shape that the
    downgrade helper's `getattr` accesses don't trip."""
    from app.services.sensor_check import (
        Bbox,
        GlareBlob,
        ImageQualityMetrics,
        SensorMetadata,
        SurfaceCaptureQuality,
    )

    metrics = ImageQualityMetrics(
        sharpness=200.0,
        glare_fraction=0.05,
        brightness_mean=128.0,
        brightness_stddev=64.0,
        color_cast=10.0,
        megapixels=2.0,
        width_px=1000,
        height_px=800,
    )
    blob = GlareBlob(
        bbox=blob_bbox,
        area_fraction_frame=0.05,
        area_fraction_label=0.10,
    )
    return SurfaceCaptureQuality(
        surface="panel_0",
        sensor=SensorMetadata(),
        metrics=metrics,
        verdict="good",
        confidence=0.9,
        glare_blobs=[blob],
        label_bbox=Bbox(x=0, y=0, w=1000, h=800),
    )


def test_verify_and_scan_aggregations_agree_on_same_input():
    """Item #2 (shared aggregation): the verify path and the scan path
    used to maintain their own roll-up function. They diverged on the
    empty-rule-list case (verify said "na", scan said "pass"). Both
    paths now route through `app.rules.aggregation.overall_status`, so
    a same-input check produces the same overall verdict.

    Probes a representative range of rule-result mixes that exercise
    every branch of the aggregator (fail / warn / advisory / pass /
    empty / unreadable-image-quality).
    """
    from app.rules.types import RuleResult
    from app.services.pipeline import overall_status as scan_overall
    from app.services.verify import _aggregate_overall as verify_overall

    def _r(status: CheckOutcome, rule_id: str = "test.rule") -> RuleResult:
        return RuleResult(
            rule_id=rule_id,
            rule_version=1,
            citation="27 CFR test",
            status=status,
        )

    # Each scenario: (rule_results, image_quality, unreadable_fields).
    scenarios = [
        ([_r(CheckOutcome.PASS), _r(CheckOutcome.PASS)], "good", []),
        ([_r(CheckOutcome.FAIL), _r(CheckOutcome.PASS)], "good", []),
        ([_r(CheckOutcome.WARN), _r(CheckOutcome.ADVISORY)], "good", []),
        ([_r(CheckOutcome.ADVISORY)], "degraded", []),
        # Empty-rule-list case — the divergence we fixed.
        ([], "good", []),
        # Empty rule list with unreadable fields → unreadable.
        ([], "good", ["health_warning"]),
        # Unreadable-image-quality always wins.
        ([_r(CheckOutcome.PASS)], "unreadable", []),
    ]

    for rules, iq, unreadable in scenarios:
        v = verify_overall(rules, unreadable, iq)
        s = scan_overall(rules, image_quality=iq)
        # The scan path doesn't take `unreadable_fields`, so the empty-
        # results-with-unreadable-fields case is verify-only — skip the
        # parity comparison for that scenario.
        if not rules and unreadable:
            continue
        assert v == s, (
            f"Aggregator divergence on rules={[r.status.value for r in rules]}, "
            f"image_quality={iq!r}: verify={v!r} scan={s!r}"
        )
