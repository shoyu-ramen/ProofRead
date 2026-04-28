"""Tests for the SPEC §0.5 extreme-condition behaviors.

Covers, end-to-end:

  * The sensor pre-check rejects fatally degraded frames before OCR
    runs, with a clear "unreadable" verdict and per-rule advisory.
  * The rule engine downgrades required rules to ADVISORY when the
    fields they depend on were reported with low confidence.
  * Producer-record cross-checks distinguish substantive mismatches
    (FAIL) from typography-only differences (WARN), as in the Stone's
    Throw / STONE'S THROW gin example in the prototype.
  * The pipeline wires the Claude vision extractor as the primary path
    and falls back to OCR when the vision call raises.

The Anthropic SDK is stubbed throughout — these tests should run in CI
with no API key and no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import (
    CheckOutcome,
    ExtractedField,
    ExtractionContext,
)
from app.services.extractors.claude_vision import (
    ClaudeVisionExtractor,
    FieldExtraction,
    LabelExtraction,
    ProducerRecord,
)
from app.services.ocr import MockOCRProvider, OCRResult
from app.services.pipeline import ScanInput, process_scan

CANONICAL_HW = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubVision:
    """A vision extractor stub that returns a pre-built ExtractionContext."""

    def __init__(self, ctx: ExtractionContext) -> None:
        self._ctx = ctx
        self.calls = 0

    def extract(self, **kwargs: Any) -> ExtractionContext:
        self.calls += 1
        return self._ctx


class _RaisingVision:
    """Vision extractor that always blows up — used to test OCR fallback."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def extract(self, **kwargs: Any) -> ExtractionContext:
        self.calls += 1
        raise self._exc


def _ocr_fixture(text: str) -> dict:
    return {
        "full_text": text,
        "blocks": [
            {"text": line, "bbox": [0, i * 30, 400, 25], "confidence": 0.99}
            for i, line in enumerate(text.split("\n"))
        ],
    }


def _multi_ocr(front: str, back: str):
    fixtures = {"front": _ocr_fixture(front), "back": _ocr_fixture(back)}

    class _Multi:
        def process(self, image_bytes: bytes, hint: str | None = None) -> OCRResult:
            return MockOCRProvider(fixtures[hint]).process(image_bytes, hint)

    return _Multi()


def _ctx_for_engine(
    fields: dict[str, ExtractedField],
    *,
    unreadable: list[str] | None = None,
    application: dict | None = None,
    is_imported: bool = False,
) -> ExtractionContext:
    return ExtractionContext(
        fields=fields,
        beverage_type="beer",
        container_size_ml=355,
        is_imported=is_imported,
        application=application or {},
        unreadable_fields=unreadable or [],
    )


def _rule(rule_id: str):
    return next(r for r in load_rules("beer") if r.id == rule_id)


def _vision_label_with_warning(warning: str, *, image_quality: str = "good") -> LabelExtraction:
    return LabelExtraction(
        beverage_type_observed="beer",
        image_quality=image_quality,
        image_quality_notes="",
        brand_name=FieldExtraction(value="ANYTOWN ALE", confidence=0.95),
        class_type=FieldExtraction(value="India Pale Ale", confidence=0.92),
        alcohol_content=FieldExtraction(value="5.5% ABV", confidence=0.95),
        net_contents=FieldExtraction(value="12 FL OZ", confidence=0.95),
        name_address=FieldExtraction(
            value="Brewed and bottled by Anytown Brewing Co., Anytown, ST",
            confidence=0.92,
        ),
        country_of_origin=FieldExtraction(value=None, confidence=0.0),
        health_warning=FieldExtraction(value=warning, confidence=0.95),
    )


class _FakeAnthropic:
    """Minimal anthropic-SDK stand-in for ClaudeVisionExtractor unit tests."""

    def __init__(self, scripted: LabelExtraction) -> None:
        self.messages = SimpleNamespace(
            parse=lambda **kw: SimpleNamespace(parsed_output=scripted, usage=None)
        )


# ---------------------------------------------------------------------------
# Rule engine: confidence-aware degradation
# ---------------------------------------------------------------------------


def test_required_rule_downgrades_to_advisory_when_field_unreadable():
    engine = RuleEngine([_rule("beer.health_warning.exact_text")])
    ctx = _ctx_for_engine(
        {"health_warning": ExtractedField(value=CANONICAL_HW, confidence=0.99)},
        unreadable=["health_warning"],
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.ADVISORY
    assert "couldn't verify" in (result.finding or "").lower()
    assert result.fix_suggestion  # canonical fix still surfaced


def test_required_rule_downgrades_when_field_confidence_below_threshold():
    engine = RuleEngine([_rule("beer.brand_name.presence")])
    ctx = _ctx_for_engine(
        {"brand_name": ExtractedField(value="ANYTOWN ALE", confidence=0.4)}
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.ADVISORY
    assert "brand_name" in (result.finding or "")


def test_required_rule_passes_when_field_confidence_just_above_threshold():
    engine = RuleEngine([_rule("beer.brand_name.presence")])
    ctx = _ctx_for_engine(
        {"brand_name": ExtractedField(value="ANYTOWN ALE", confidence=0.7)}
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.PASS


def test_advisory_rule_unaffected_by_unreadable_fields():
    """The size rule is advisory; even if its field is unreadable it must
    not be 'double-downgraded' — it's already advisory."""
    engine = RuleEngine([_rule("beer.health_warning.size")])
    ctx = _ctx_for_engine({}, unreadable=["health_warning"])
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.ADVISORY


def test_missing_field_still_fails_presence_check():
    """A field that's plainly absent should FAIL, not ADVISORY — the user
    needs to know they're missing the element, not that we couldn't read it."""
    engine = RuleEngine([_rule("beer.brand_name.presence")])
    ctx = _ctx_for_engine({})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


# ---------------------------------------------------------------------------
# Rule engine: producer-record cross-check (WARN)
# ---------------------------------------------------------------------------


def test_case_only_brand_mismatch_produces_warn():
    engine = RuleEngine([_rule("beer.brand_name.presence")])
    ctx = _ctx_for_engine(
        {"brand_name": ExtractedField(value="STONE'S THROW", confidence=0.95)},
        application={"producer_record": {"brand": "Stone's Throw"}},
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.WARN
    assert "case-only" in (result.finding or "").lower()


def test_substantive_brand_mismatch_produces_fail():
    engine = RuleEngine([_rule("beer.brand_name.presence")])
    ctx = _ctx_for_engine(
        {"brand_name": ExtractedField(value="DIFFERENT NAME", confidence=0.95)},
        application={"producer_record": {"brand": "Stone's Throw"}},
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


def test_brand_match_produces_pass():
    engine = RuleEngine([_rule("beer.brand_name.presence")])
    ctx = _ctx_for_engine(
        {"brand_name": ExtractedField(value="Stone's Throw", confidence=0.95)},
        application={"producer_record": {"brand": "Stone's Throw"}},
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.PASS


def test_no_producer_record_does_not_force_pass_or_fail():
    """Without a record, cross_reference is a no-op — presence still checked."""
    engine = RuleEngine([_rule("beer.brand_name.presence")])
    ctx = _ctx_for_engine(
        {"brand_name": ExtractedField(value="ANY", confidence=0.95)}
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.PASS


# ---------------------------------------------------------------------------
# Pipeline: sensor pre-check short-circuit
# ---------------------------------------------------------------------------


def test_unreadable_capture_no_rule_returns_fail(synthetic_label_png):
    """A frame the sensor module judges unreadable must NEVER produce a FAIL.

    SPEC §0.5 "fail honestly": a wrong "fail" is the second-worst outcome
    (the user wastes time chasing a phantom problem). Required rules whose
    label content can't be read come back ADVISORY; rules that don't apply
    or have optional content can stay NA / PASS — those are honest answers.
    """
    front = synthetic_label_png(flat=True)  # uniform gray → no contrast
    back = synthetic_label_png(flat=True)
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": front, "back": back},
    )
    ocr = _multi_ocr("ANYTOWN ALE\n5.5% ABV", CANONICAL_HW)
    report = process_scan(scan, ocr)

    assert report.overall == "unreadable"
    assert report.image_quality == "unreadable"
    statuses = {r.status for r in report.rule_results}
    assert CheckOutcome.FAIL not in statuses, (
        "no rule may FAIL when capture is unreadable; "
        f"got: {[(r.rule_id, r.status) for r in report.rule_results]}"
    )
    # The rules whose required label content can't be read should be advisory.
    presence_rules = {
        "beer.brand_name.presence",
        "beer.class_type.presence",
        "beer.net_contents.presence",
        "beer.name_address.presence",
        "beer.health_warning.exact_text",
    }
    by_id = {r.rule_id: r for r in report.rule_results}
    for rid in presence_rules:
        assert by_id[rid].status == CheckOutcome.ADVISORY, (
            f"{rid} should be advisory when frame is unreadable; "
            f"got {by_id[rid].status}: {by_id[rid].finding}"
        )


def test_good_capture_passes_through_to_extractor(synthetic_label_png):
    front = synthetic_label_png()
    back = synthetic_label_png()
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": front, "back": back},
    )
    ocr = _multi_ocr(
        "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ",
        "Brewed and bottled by Anytown Brewing Co., Anytown, ST\n" + CANONICAL_HW,
    )
    report = process_scan(scan, ocr)

    assert report.overall in {"pass", "advisory"}
    assert report.image_quality in {"good", "degraded"}
    assert report.capture_quality is not None


# ---------------------------------------------------------------------------
# Pipeline: vision is preferred, OCR is the fallback
# ---------------------------------------------------------------------------


def test_pipeline_uses_vision_extractor_when_supplied(synthetic_label_png):
    label = _vision_label_with_warning(CANONICAL_HW)
    extractor = ClaudeVisionExtractor(client=_FakeAnthropic(label))
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={
            "front": synthetic_label_png(),
            "back": synthetic_label_png(),
        },
        producer_record=ProducerRecord(brand="Anytown Ale", class_type="India Pale Ale"),
    )

    report = process_scan(scan, ocr=None, vision=extractor)
    assert report.extractor == "claude_opus_4_7"
    assert report.overall in {"pass", "advisory", "warn"}


def test_pipeline_falls_back_to_ocr_when_vision_raises(synthetic_label_png):
    raising = _RaisingVision(RuntimeError("anthropic 503"))
    ocr = _multi_ocr(
        "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ",
        "Brewed and bottled by Anytown Brewing Co., Anytown, ST\n" + CANONICAL_HW,
    )
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={
            "front": synthetic_label_png(),
            "back": synthetic_label_png(),
        },
    )
    report = process_scan(scan, ocr=ocr, vision=raising)
    assert raising.calls == 1
    assert report.extractor == "ocr"
    assert report.overall in {"pass", "advisory"}


def test_vision_failure_without_ocr_propagates(synthetic_label_png):
    raising = _RaisingVision(RuntimeError("anthropic 503"))
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={
            "front": synthetic_label_png(),
            "back": synthetic_label_png(),
        },
    )
    with pytest.raises(RuntimeError):
        process_scan(scan, ocr=None, vision=raising)


def test_pipeline_requires_at_least_one_extractor(synthetic_label_png):
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": synthetic_label_png()},
    )
    with pytest.raises(ValueError):
        process_scan(scan, ocr=None, vision=None)


# ---------------------------------------------------------------------------
# Pipeline: vision result with degraded image_quality propagates honestly
# ---------------------------------------------------------------------------


def test_vision_degraded_image_does_not_pretend_to_pass(synthetic_label_png):
    label = _vision_label_with_warning(CANONICAL_HW, image_quality="degraded")
    label.image_quality_notes = "Foil neck wrap reduces confidence on top half."
    label.brand_name = FieldExtraction(
        value="ANYTOWN ALE", confidence=0.4, note="Foil obscures lower edge"
    )
    extractor = ClaudeVisionExtractor(client=_FakeAnthropic(label))
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={
            "front": synthetic_label_png(),
            "back": synthetic_label_png(),
        },
    )
    report = process_scan(scan, ocr=None, vision=extractor)

    assert report.image_quality in {"degraded", "unreadable"}
    # The brand rule should be advisory because the field's confidence is
    # below the rule-engine threshold — even if the rest of the label is fine.
    brand_rule = next(
        r for r in report.rule_results if r.rule_id == "beer.brand_name.presence"
    )
    assert brand_rule.status == CheckOutcome.ADVISORY


def test_vision_unreadable_image_marks_overall_unreadable(synthetic_label_png):
    label = _vision_label_with_warning(CANONICAL_HW, image_quality="unreadable")
    label.image_quality_notes = "Bottle held at 60° with severe motion blur."
    extractor = ClaudeVisionExtractor(client=_FakeAnthropic(label))
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={
            "front": synthetic_label_png(),
            "back": synthetic_label_png(),
        },
    )
    report = process_scan(scan, ocr=None, vision=extractor)
    assert report.overall == "unreadable"
    # Every rule the engine produced should be advisory — never a wrong-fail.
    assert all(
        r.status in {CheckOutcome.ADVISORY, CheckOutcome.NA}
        for r in report.rule_results
    )


# ---------------------------------------------------------------------------
# Pipeline: producer-record cross-check end-to-end
# ---------------------------------------------------------------------------


def test_pipeline_warns_on_case_only_brand_mismatch(synthetic_label_png):
    """The Stone's Throw gin scenario from the prototype.

    The label prints "STONE'S THROW"; the producer record says "Stone's Throw".
    The pipeline should produce overall="warn" with brand WARN, not FAIL.
    """
    label = _vision_label_with_warning(CANONICAL_HW)
    label.brand_name = FieldExtraction(value="STONE'S THROW", confidence=0.95)
    extractor = ClaudeVisionExtractor(client=_FakeAnthropic(label))

    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={
            "front": synthetic_label_png(),
            "back": synthetic_label_png(),
        },
        producer_record=ProducerRecord(
            brand="Stone's Throw",
            class_type="India Pale Ale",
            container_size_ml=355,
        ),
    )
    report = process_scan(scan, ocr=None, vision=extractor)
    assert report.overall == "warn", (
        f"expected WARN for case-only mismatch; got {report.overall}: "
        f"{[(r.rule_id, r.status) for r in report.rule_results]}"
    )
    brand = next(
        r for r in report.rule_results if r.rule_id == "beer.brand_name.presence"
    )
    assert brand.status == CheckOutcome.WARN


# ---------------------------------------------------------------------------
# Per-condition coverage from the SPEC §0.5 catalog
# ---------------------------------------------------------------------------


def test_severe_blur_marks_capture_unreadable(synthetic_label_png):
    """Shaky-hands / motion-blur path — the most common bar / festival failure."""
    blurred = synthetic_label_png(blur=True)
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": blurred, "back": blurred},
    )
    ocr = _multi_ocr("ANYTOWN ALE", CANONICAL_HW)
    report = process_scan(scan, ocr)
    assert report.image_quality in {"degraded", "unreadable"}
    assert any(
        "blur" in (i or "").lower()
        for s in report.capture_quality.surfaces
        for i in s.issues
    )


def test_extreme_glare_marks_capture_unreadable(synthetic_label_png):
    """Direct-sunlight / flashbulb-bounce path."""
    glared = synthetic_label_png(glare=True)
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": glared, "back": glared},
    )
    ocr = _multi_ocr("ANYTOWN ALE", CANONICAL_HW)
    report = process_scan(scan, ocr)
    assert report.image_quality in {"degraded", "unreadable"}
    surfaces_issues = [i for s in report.capture_quality.surfaces for i in s.issues]
    assert any("glare" in (i or "").lower() for i in surfaces_issues)


def test_very_dark_capture_marks_unreadable(synthetic_label_png):
    """Dim-bar / cellar path (<50 lux)."""
    dim = synthetic_label_png(dark=True)
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": dim, "back": dim},
    )
    ocr = _multi_ocr("ANYTOWN ALE", CANONICAL_HW)
    report = process_scan(scan, ocr)
    assert report.image_quality in {"degraded", "unreadable"}


def test_only_back_unreadable_downgrades_health_warning(synthetic_label_png):
    """Front readable, back unreadable: brand-name rules still pass, but the
    Health Warning rule (which lives on the back) is downgraded — not FAIL."""
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={
            "front": synthetic_label_png(),
            "back": synthetic_label_png(flat=True),
        },
    )
    ocr = _multi_ocr(
        "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ",
        "Brewed and bottled by Anytown Brewing Co., Anytown, ST",
    )
    report = process_scan(scan, ocr)

    by_id = {r.rule_id: r for r in report.rule_results}
    hw = by_id["beer.health_warning.exact_text"]
    assert hw.status == CheckOutcome.ADVISORY, (
        f"health warning should advisory when back is unreadable; got {hw.status}"
    )


def test_sensor_verdict_overrides_extractor_optimism(synthetic_label_png):
    """If the vision agent claims 'good' but the sensor pre-check says
    'degraded', the report must reflect the worse verdict — fail honestly."""
    blurred = synthetic_label_png(blur=True)
    label = _vision_label_with_warning(CANONICAL_HW, image_quality="good")
    extractor = ClaudeVisionExtractor(client=_FakeAnthropic(label))
    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": blurred, "back": blurred},
    )
    report = process_scan(scan, ocr=None, vision=extractor)
    # The sensor said degraded/unreadable; the report must NOT show "good".
    assert report.image_quality in {"degraded", "unreadable"}


# ---------------------------------------------------------------------------
# Real artwork: the four sample labels under artwork/labels/
# ---------------------------------------------------------------------------


def _artwork_dir():
    from pathlib import Path

    candidate = Path(__file__).resolve().parents[2] / "artwork" / "labels"
    return candidate if candidate.is_dir() else None


@pytest.mark.skipif(_artwork_dir() is None, reason="artwork/labels not available")
def test_sample_labels_are_classified_honestly():
    """Smoke-check the sensor pre-check against the four hand-rendered
    sample labels. We assert structural properties — every surface gets
    a verdict and a confidence — without pinning specific verdicts; the
    sensor heuristics drift as we calibrate them and brittle assertions
    on per-label outcomes turn into churn rather than coverage."""
    from app.services.sensor_check import assess_capture_quality

    labels_dir = _artwork_dir()
    files = sorted(labels_dir.glob("*.png"))
    images = {p.stem: p.read_bytes() for p in files}
    report = assess_capture_quality(images)

    assert len(report.surfaces) == len(files), (
        "every sample label should produce a SurfaceCaptureQuality entry"
    )
    for s in report.surfaces:
        assert s.verdict in {"good", "degraded", "unreadable"}
        assert 0.0 <= s.confidence <= 1.0
        # Decoding succeeded — at minimum we got a sane megapixel count.
        assert s.metrics.megapixels > 0

    # The bourbon label is the cleanest rendering of all four — it should
    # never be the worst-graded surface.
    by_surface = report.by_surface()
    bourbon = by_surface.get("01_pass_old_tom_distillery")
    if bourbon is not None:
        assert bourbon.verdict != "unreadable", (
            f"clean bourbon label should not be flagged unreadable; "
            f"got {bourbon.verdict}: {bourbon.issues}"
        )
