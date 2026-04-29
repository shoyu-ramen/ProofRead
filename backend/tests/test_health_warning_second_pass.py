"""Tests for the Government Warning redundant second-pass.

The cross-check is the load-bearing piece: it decides whether a rule
engine FAIL should stand or be downgraded to ADVISORY based on whether
two independent reads of the warning agree.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.rules.types import CheckOutcome
from app.services.health_warning_second_pass import (
    ClaudeHealthWarningExtractor,
    MockHealthWarningExtractor,
    ObstructionSignal,
    WarningRead,
    _parse_response,
    cross_check,
)
from app.services.verify import Panel, VerifyInput, verify
from app.services.vision import MockVisionExtractor

CANONICAL = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)

TITLECASE_PARAPHRASE = (
    "Government Warning: According to the Surgeon General, pregnant women "
    "should avoid alcoholic beverages because of the risk of birth defects. "
    "Drinking impairs your ability to drive and may cause health problems."
)


# ---------------------------------------------------------------------------
# Cross-check decision tree
# ---------------------------------------------------------------------------


def test_both_reads_match_canonical_is_confirmed_compliant():
    primary = WarningRead(value=CANONICAL, found=True, confidence=0.95)
    secondary = WarningRead(value=CANONICAL, found=True, confidence=0.95)
    cc = cross_check(primary, secondary)
    assert cc.outcome == "confirmed_compliant"
    assert cc.edit_distance_between_reads == 0
    assert cc.edit_distance_to_canonical == 0


def test_both_reads_agree_on_paraphrase_is_confirmed_noncompliant():
    primary = WarningRead(value=TITLECASE_PARAPHRASE, found=True, confidence=0.95)
    secondary = WarningRead(value=TITLECASE_PARAPHRASE, found=True, confidence=0.94)
    cc = cross_check(primary, secondary)
    assert cc.outcome == "confirmed_noncompliant"
    assert cc.edit_distance_between_reads == 0
    assert cc.edit_distance_to_canonical and cc.edit_distance_to_canonical > 5


def test_reads_disagree_returns_disagreement():
    primary = WarningRead(value=CANONICAL, found=True, confidence=0.95)
    secondary = WarningRead(value=TITLECASE_PARAPHRASE, found=True, confidence=0.95)
    cc = cross_check(primary, secondary)
    assert cc.outcome == "disagreement"
    assert "couldn't verify" in cc.notes.lower() or "rescan" in cc.notes.lower()


def test_minor_jitter_when_both_reads_match_each_other():
    """Both reads agree on a slight deviation from canonical (e.g. an OCR
    drop of one space). They agree with each other but differ from canonical
    by the same small amount → confirmed_noncompliant."""
    paraphrase = CANONICAL.replace(" women", "women", 1)
    primary = WarningRead(value=paraphrase, found=True, confidence=0.95)
    secondary = WarningRead(value=paraphrase, found=True, confidence=0.95)
    cc = cross_check(primary, secondary)
    assert cc.outcome == "confirmed_noncompliant"
    assert cc.edit_distance_between_reads == 0


def test_primary_matches_canonical_secondary_does_not_is_disagreement():
    """If only one read sees canonical, the trust isn't there to call
    pass — that's a disagreement, not a confirmed verdict."""
    primary = WarningRead(value=CANONICAL, found=True, confidence=0.95)
    secondary = WarningRead(
        value=CANONICAL.replace(" women", "women", 1),
        found=True,
        confidence=0.95,
    )
    cc = cross_check(primary, secondary)
    assert cc.outcome == "disagreement"


def test_only_primary_returns_primary_only():
    primary = WarningRead(value=CANONICAL, found=True, confidence=0.95)
    secondary = WarningRead(value=None, found=False, confidence=0.0)
    cc = cross_check(primary, secondary)
    assert cc.outcome == "primary_only"


def test_no_secondary_at_all_returns_primary_only():
    primary = WarningRead(value=CANONICAL, found=True, confidence=0.95)
    cc = cross_check(primary, None)
    assert cc.outcome == "primary_only"


def test_neither_read_returns_no_warning_present():
    primary = WarningRead(value=None, found=False, confidence=0.0)
    secondary = WarningRead(value=None, found=False, confidence=0.0)
    cc = cross_check(primary, secondary)
    assert cc.outcome == "no_warning_present"


# ---------------------------------------------------------------------------
# JSON response parsing (Claude path)
# ---------------------------------------------------------------------------


def test_parse_response_handles_clean_json():
    raw = (
        '{"value": "GOVERNMENT WARNING: ...", "found": true, "confidence": 0.95}'
    )
    read = _parse_response(raw)
    assert read.found is True
    assert read.value == "GOVERNMENT WARNING: ..."
    assert read.confidence == 0.95


def test_parse_response_strips_markdown_fences():
    raw = '```json\n{"value": "x", "found": true, "confidence": 0.9}\n```'
    read = _parse_response(raw)
    assert read.found is True
    assert read.value == "x"


def test_parse_response_returns_not_found_on_invalid_json():
    raw = "I see a Government Warning but I'd rather chat about it."
    read = _parse_response(raw)
    assert read.found is False
    assert read.value is None
    assert read.raw_response == raw


def test_parse_response_recovers_json_with_trailing_prose():
    """Haiku occasionally appends commentary after the closing fence on
    degraded images, in spite of the system prompt forbidding it.
    Regression: that commentary used to break the strict end-of-string
    fence regex, throwing away an otherwise-perfectly-valid 0.65-conf
    read on every glare-fouled label. The redundancy is the whole point
    of the second pass — discarding readable text here defeats it."""
    raw = (
        "```json\n"
        '{"value": "GOVERNMENT WARNING: ...", '
        '"found": true, "confidence": 0.65}\n'
        "```\n\n"
        "**Note:** The warning text is visible on the label but at an angle "
        "and with some glare, making certain portions slightly difficult to "
        "read with complete certainty."
    )
    read = _parse_response(raw)
    assert read.found is True
    assert read.value == "GOVERNMENT WARNING: ..."
    assert read.confidence == 0.65


def test_parse_response_recovers_json_with_no_fences_and_trailing_prose():
    """Same defence, no Markdown fences in the response."""
    raw = (
        '{"value": "GOVERNMENT WARNING: text", "found": true, "confidence": 0.7}'
        "\n\nThe label is partially obscured."
    )
    read = _parse_response(raw)
    assert read.found is True
    assert read.value == "GOVERNMENT WARNING: text"


def test_parse_response_treats_found_false_as_unread():
    raw = '{"value": "", "found": false, "confidence": 0.2}'
    read = _parse_response(raw)
    assert read.found is False
    assert read.value is None


# ---------------------------------------------------------------------------
# ClaudeHealthWarningExtractor wiring
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, scripted: str) -> None:
        self._scripted = scripted
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._scripted)]
        )


class _FakeClient:
    def __init__(self, scripted: str) -> None:
        self.messages = _FakeMessages(scripted)


def test_claude_second_pass_uses_cached_system_prompt():
    fake = _FakeClient(
        '{"value": "GOVERNMENT WARNING: ...", "found": true, "confidence": 0.92}'
    )
    extractor = ClaudeHealthWarningExtractor(client=fake, model="claude-opus-4-7")
    read = extractor.read_warning(b"fake-png-bytes", media_type="image/png")
    assert read.found is True
    assert read.value == "GOVERNMENT WARNING: ..."
    [call] = fake.messages.calls
    [system_block] = call["system"]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert call["model"] == "claude-opus-4-7"
    assert call["max_tokens"] == 1024
    [content] = call["messages"]
    image_block = next(b for b in content["content"] if b["type"] == "image")
    assert image_block["source"]["media_type"] == "image/png"


# ---------------------------------------------------------------------------
# verify() integration: cross-check downgrades / preserves verdicts
# ---------------------------------------------------------------------------


def _verify_with_warning_reader(*, primary_warning: str, secondary_warning: str | None):
    """End-to-end verify with mock vision + mock second-pass."""
    extractor = MockVisionExtractor(
        {
            "brand_name": "Old Tom Distillery",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol. (90 Proof)",
            "net_contents": "750 mL",
            "name_address": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
            "health_warning": primary_warning,
        }
    )
    second = MockHealthWarningExtractor(value=secondary_warning)
    inp = VerifyInput(
        image_bytes=b"\x89PNG\r\n\x1a\n",
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application={
            "producer_record": {
                "brand_name": "Old Tom Distillery",
                "class_type": "Kentucky Straight Bourbon Whiskey",
                "alcohol_content": "45",
                "net_contents": "750 mL",
                "name_address": "Old Tom Distilling Co., Bardstown, Kentucky",
            }
        },
    )
    return verify(
        inp,
        extractor=extractor,
        health_warning_reader=second,
        skip_capture_quality=True,
    )


def test_verify_disagreement_downgrades_warning_to_advisory():
    """Primary saw canonical; second pass disagreed → downgrade to advisory.
    The system declines to claim a verdict it cannot stand behind."""
    report = _verify_with_warning_reader(
        primary_warning=CANONICAL,
        secondary_warning=TITLECASE_PARAPHRASE,
    )
    warning = next(
        r for r in report.rule_results if "health_warning" in r.rule_id
    )
    assert warning.status == CheckOutcome.ADVISORY
    assert "couldn't verify" in (warning.finding or "").lower()
    assert report.health_warning_cross_check["outcome"] == "disagreement"


def test_verify_confirmed_compliant_leaves_pass_alone():
    report = _verify_with_warning_reader(
        primary_warning=CANONICAL,
        secondary_warning=CANONICAL,
    )
    warning = next(
        r for r in report.rule_results if "health_warning" in r.rule_id
    )
    assert warning.status == CheckOutcome.PASS
    assert report.health_warning_cross_check["outcome"] == "confirmed_compliant"


def test_verify_confirmed_noncompliant_leaves_fail_alone():
    """Both reads agree the label is paraphrased — the FAIL is well-supported
    and must not be downgraded to advisory."""
    report = _verify_with_warning_reader(
        primary_warning=TITLECASE_PARAPHRASE,
        secondary_warning=TITLECASE_PARAPHRASE,
    )
    warning = next(
        r for r in report.rule_results if "health_warning" in r.rule_id
    )
    assert warning.status == CheckOutcome.FAIL
    assert report.health_warning_cross_check["outcome"] == "confirmed_noncompliant"


def test_verify_without_second_pass_reader_still_works():
    """The second pass is opt-in; when no reader is supplied, verify behaves
    the same as before (cross-check field is null)."""
    extractor = MockVisionExtractor(
        {
            "brand_name": "Old Tom Distillery",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
            "name_address": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
            "health_warning": CANONICAL,
        }
    )
    inp = VerifyInput(
        image_bytes=b"\x89PNG\r\n\x1a\n",
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application={
            "producer_record": {
                "brand_name": "Old Tom Distillery",
                "class_type": "Kentucky Straight Bourbon Whiskey",
                "alcohol_content": "45",
                "net_contents": "750 mL",
                "name_address": "Old Tom Distilling Co., Bardstown, Kentucky",
            }
        },
    )
    report = verify(inp, extractor=extractor, skip_capture_quality=True)
    assert report.health_warning_cross_check is None


def test_verify_second_pass_failure_does_not_break_request():
    """A flaky second-pass reader (raises) must not break the verify path —
    the primary verdict stands."""

    class _Boom:
        def read_warning(self, image_bytes, media_type="image/png"):
            raise RuntimeError("anthropic 503")

    extractor = MockVisionExtractor(
        {
            "brand_name": "Old Tom Distillery",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
            "name_address": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
            "health_warning": CANONICAL,
        }
    )
    inp = VerifyInput(
        image_bytes=b"\x89PNG\r\n\x1a\n",
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application={
            "producer_record": {
                "brand_name": "Old Tom Distillery",
                "class_type": "Kentucky Straight Bourbon Whiskey",
                "alcohol_content": "45",
                "net_contents": "750 mL",
                "name_address": "Old Tom Distilling Co., Bardstown, Kentucky",
            }
        },
    )
    report = verify(
        inp,
        extractor=extractor,
        health_warning_reader=_Boom(),
        skip_capture_quality=True,
    )
    # Primary verdict survived; cross_check is None because the second pass blew up.
    assert report.health_warning_cross_check is None
    warning = next(r for r in report.rule_results if "health_warning" in r.rule_id)
    assert warning.status == CheckOutcome.PASS


# ---------------------------------------------------------------------------
# Recall under glare — the warning is detected when present, even when both
# readers struggle. The mission of this section: never let a glare-cascade
# produce a confident "warning is missing" verdict.
# ---------------------------------------------------------------------------


def test_neither_read_with_region_visible_returns_unverifiable_obstructed():
    """Both readers couldn't transcribe text but at least one saw the
    fine-print block in a typical warning location → recall guarantee
    kicks in and the cross-check refuses to claim the warning is
    missing."""
    primary = WarningRead(value=None, found=False, confidence=0.2, region_visible=True)
    secondary = WarningRead(value=None, found=False, confidence=0.1, region_visible=False)
    cc = cross_check(primary, secondary)
    assert cc.outcome == "unverifiable_obstructed"
    assert "warning region" in cc.notes.lower() or "obstruction" in cc.notes.lower()


def test_neither_read_with_external_obstruction_returns_unverifiable_obstructed():
    """No region_visible signals from either reader, but the verify
    orchestrator passed an obstruction signal because the sensor pre-check
    saw glare on the label region. Same recall guarantee applies."""
    primary = WarningRead(value=None, found=False, confidence=0.0)
    secondary = WarningRead(value=None, found=False, confidence=0.0)
    cc = cross_check(
        primary,
        secondary,
        obstruction_signal=ObstructionSignal(
            is_obstructed=True,
            reason="back: glare blobs cover 40% of the label region",
        ),
    )
    assert cc.outcome == "unverifiable_obstructed"
    assert "glare" in cc.notes.lower() or "obstruction" in cc.notes.lower()


def test_neither_read_with_clean_frame_still_no_warning_present():
    """No region_visible AND no obstruction signal: the conclusion that
    the warning is missing is honest, and the cross-check leaves it for
    the engine to FAIL on. We are not making the system unable to ever
    fail on missing warnings — only refusing to do so when glare could
    be hiding it."""
    primary = WarningRead(value=None, found=False, confidence=0.9, region_visible=False)
    secondary = WarningRead(value=None, found=False, confidence=0.9, region_visible=False)
    cc = cross_check(primary, secondary)
    assert cc.outcome == "no_warning_present"


def test_low_confidence_primary_alone_under_obstruction_is_unverifiable():
    """The primary returned a partial low-confidence read and the second
    pass was unavailable. With an obstruction signal, edit-distance to
    canonical is meaningless — we can't tell whether the missing chars
    were the model's or the label's. Refuse to FAIL."""
    primary = WarningRead(
        value="GOVERNMENT WARNING:",  # only the prefix survived
        found=True,
        confidence=0.45,
    )
    cc = cross_check(
        primary,
        None,
        obstruction_signal=ObstructionSignal(
            is_obstructed=True,
            reason="back: glare blobs cover 30% of the label region",
        ),
    )
    assert cc.outcome == "unverifiable_obstructed"


def test_high_confidence_primary_alone_under_obstruction_stays_primary_only():
    """Even with an obstruction signal, a confident full primary read is
    still trustworthy on its own. Only LOW-confidence primary reads are
    escalated — we don't want the obstruction signal to start downgrading
    legitimate single-reader verdicts."""
    primary = WarningRead(value=CANONICAL, found=True, confidence=0.95)
    cc = cross_check(
        primary,
        None,
        obstruction_signal=ObstructionSignal(
            is_obstructed=True,
            reason="front: minor glare in the corner",
        ),
    )
    assert cc.outcome == "primary_only"


def test_secondary_only_low_confidence_under_obstruction_is_unverifiable():
    """Symmetric: secondary partial, primary failed, obstruction signal →
    unverifiable. We don't want the system to flip the verdict on the
    secondary's partial read alone."""
    primary = WarningRead(value=None, found=False, confidence=0.0)
    secondary = WarningRead(
        value="GOVERNMENT WARNING:",
        found=True,
        confidence=0.50,
    )
    cc = cross_check(
        primary,
        secondary,
        obstruction_signal=ObstructionSignal(
            is_obstructed=True,
            reason="back: motion blur over the label",
        ),
    )
    assert cc.outcome == "unverifiable_obstructed"


# ---------------------------------------------------------------------------
# Region-visible JSON parsing
# ---------------------------------------------------------------------------


def test_parse_response_carries_region_visible_when_warning_obscured():
    """The 'I saw the block but couldn't read it' branch — found=false
    with region_visible=true — must round-trip through the parser."""
    raw = (
        '{"value": "", "found": false, "confidence": 0.25, '
        '"region_visible": true}'
    )
    read = _parse_response(raw)
    assert read.found is False
    assert read.region_visible is True


def test_parse_response_infers_region_visible_for_successful_read():
    """A successful read implies region_visible=true even if the model
    forgot to set the field. Recall is preserved."""
    raw = '{"value": "GOVERNMENT WARNING: ...", "found": true, "confidence": 0.92}'
    read = _parse_response(raw)
    assert read.found is True
    assert read.region_visible is True


# ---------------------------------------------------------------------------
# verify() integration: recall under glare
# ---------------------------------------------------------------------------


def _verify_with_warning_glare(
    *,
    primary_warning,
    secondary_value: str | None,
    secondary_region_visible: bool = False,
    secondary_confidence: float = 0.95,
    primary_image_quality_notes: str | None = None,
):
    """End-to-end verify with mocked vision + mocked second-pass, varying
    glare-related signals. Built so we can exercise every branch of the
    recall guarantee from the orchestrator down.

    `primary_warning` may be a string (treated as a clean read), a dict
    matching the MockVisionExtractor field shape (for partial / low-
    confidence reads), or None (primary returned no value).

    The non-warning fields here are deliberately chosen to clear the
    foreign-language adversarial guard with ≥ 3 generic English compliance
    keywords (alcohol, proof, bottled, distilled, company) — without them,
    the `verify` short-circuits through `_unreadable_rule_results` and we
    never exercise the cross-check."""
    extractor_fixture = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% alcohol by volume (90 Proof)",
        "net_contents": "750 mL",
        "name_address": (
            "Distilled and bottled by Old Tom Distilling Company, "
            "Bardstown, Kentucky"
        ),
    }
    if primary_warning is not None:
        extractor_fixture["health_warning"] = primary_warning
    if primary_image_quality_notes:
        extractor_fixture["image_quality_notes"] = primary_image_quality_notes
    extractor = MockVisionExtractor(extractor_fixture)
    second = MockHealthWarningExtractor(
        value=secondary_value,
        confidence=secondary_confidence,
        region_visible=secondary_region_visible,
    )
    inp = VerifyInput(
        image_bytes=b"\x89PNG\r\n\x1a\n",
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application={
            "producer_record": {
                "brand_name": "Old Tom Distillery",
                "class_type": "Kentucky Straight Bourbon Whiskey",
                "alcohol_content": "45",
                "net_contents": "750 mL",
                "name_address": "Old Tom Distilling Co., Bardstown, Kentucky",
            }
        },
    )
    return verify(
        inp,
        extractor=extractor,
        health_warning_reader=second,
        skip_capture_quality=True,
    )


def test_verify_glare_blocks_both_readers_with_region_signal_is_advisory():
    """The user's mission: a label that DOES carry the warning, both
    readers fail to transcribe due to glare, but at least one saw the
    block — engine must produce ADVISORY, never FAIL with 'warning
    missing'."""
    report = _verify_with_warning_glare(
        primary_warning=None,  # primary returned no value
        primary_image_quality_notes=(
            "Specular highlight covers the back-panel warning paragraph; "
            "fine-print block visible but glared out."
        ),
        secondary_value=None,
        secondary_region_visible=True,
        secondary_confidence=0.25,
    )
    warning = next(
        r for r in report.rule_results if "health_warning" in r.rule_id
    )
    assert warning.status == CheckOutcome.ADVISORY, (
        f"Expected ADVISORY (glare-recall guarantee); got {warning.status}. "
        f"finding={warning.finding!r}"
    )
    assert "missing" not in (warning.finding or "").lower()
    cc = report.health_warning_cross_check
    assert cc is not None
    assert cc["outcome"] == "unverifiable_obstructed"


def test_verify_glare_with_partial_primary_low_confidence_is_advisory():
    """Primary saw the prefix only (partial read, low confidence under
    glare). Secondary failed entirely but its notes flag region_visible.
    Cross-check must escalate to unverifiable_obstructed rather than
    take the primary's partial read at face value."""
    report = _verify_with_warning_glare(
        primary_warning={
            "value": "GOVERNMENT WARNING:",
            "confidence": 0.40,
            "unreadable": False,
        },
        secondary_value=None,
        secondary_region_visible=True,
        secondary_confidence=0.25,
    )
    warning = next(
        r for r in report.rule_results if "health_warning" in r.rule_id
    )
    assert warning.status == CheckOutcome.ADVISORY, (
        f"Partial low-conf primary under glare must be advisory; got "
        f"{warning.status}: {warning.finding!r}"
    )


def test_verify_no_glare_signal_warning_missing_still_fails_honestly():
    """Sanity check the inverse: when the primary returned no warning AND
    nothing in the system flags obstruction, the engine still FAILs on
    a missing warning. The recall guarantee is about refusing confident
    wrong-fails under glare, not about disabling the FAIL path entirely."""
    report = _verify_with_warning_glare(
        primary_warning=None,  # primary saw no warning
        primary_image_quality_notes="Label is clean and well-lit.",
        secondary_value=None,
        secondary_region_visible=False,
        secondary_confidence=0.95,
    )
    warning = next(
        r for r in report.rule_results if "health_warning" in r.rule_id
    )
    assert warning.status == CheckOutcome.FAIL


def test_verify_glare_advisory_finding_names_the_obstruction():
    """The advisory finding string should help the user understand WHY —
    not just 'couldn't verify'. It should mention either glare/obstruction
    or the warning region so the user knows what to fix on the rescan."""
    report = _verify_with_warning_glare(
        primary_warning=None,
        secondary_value=None,
        secondary_region_visible=True,
        secondary_confidence=0.20,
    )
    warning = next(
        r for r in report.rule_results if "health_warning" in r.rule_id
    )
    finding_lower = (warning.finding or "").lower()
    assert any(
        cue in finding_lower
        for cue in ("glar", "obstruct", "warning region", "hidden", "obscured")
    ), f"finding should explain the obstruction; got {warning.finding!r}"


# ---------------------------------------------------------------------------
# ObstructionSignal building from sensor pre-check
# ---------------------------------------------------------------------------


def test_obstruction_signal_built_from_glare_blobs_on_label():
    """Glare blobs covering ≥ 5% of the label region produce an
    is_obstructed=True signal. This is the wiring that lets a sensor-
    detected glare blob over the warning area drive the cross-check's
    recall guarantee."""
    from app.services.sensor_check import (
        Bbox,
        CaptureQualityReport,
        GlareBlob,
        ImageQualityMetrics,
        SensorMetadata,
        SurfaceCaptureQuality,
    )
    from app.services.verify import _build_obstruction_signal

    surface = SurfaceCaptureQuality(
        surface="panel_0",
        sensor=SensorMetadata(),
        metrics=ImageQualityMetrics(
            sharpness=200,
            glare_fraction=0.05,
            brightness_mean=120,
            brightness_stddev=40,
            color_cast=10,
            megapixels=8.0,
            width_px=4000,
            height_px=2000,
        ),
        verdict="good",
        confidence=0.85,
        label_bbox=Bbox(x=0, y=0, w=2000, h=1000),
        label_verdict="good",
        glare_blobs=[
            GlareBlob(
                bbox=Bbox(x=0, y=600, w=2000, h=200),
                area_fraction_frame=0.05,
                area_fraction_label=0.20,  # 20 % of label region
            )
        ],
    )
    capture = CaptureQualityReport(
        surfaces=[surface], overall_verdict="good", overall_confidence=0.85
    )
    signal = _build_obstruction_signal(capture)
    assert signal.is_obstructed is True
    assert "glare" in signal.reason.lower()
    assert "20%" in signal.reason


def test_obstruction_signal_clear_on_pristine_capture():
    """A clean label produces an is_obstructed=False signal — the recall
    guarantee only kicks in when there's evidence of obstruction."""
    from app.services.sensor_check import (
        CaptureQualityReport,
        ImageQualityMetrics,
        SensorMetadata,
        SurfaceCaptureQuality,
    )
    from app.services.verify import _build_obstruction_signal

    surface = SurfaceCaptureQuality(
        surface="panel_0",
        sensor=SensorMetadata(),
        metrics=ImageQualityMetrics(
            sharpness=400,
            glare_fraction=0.05,
            brightness_mean=128,
            brightness_stddev=50,
            color_cast=8,
            megapixels=8.0,
            width_px=4000,
            height_px=2000,
        ),
        verdict="good",
        confidence=0.95,
        label_verdict="good",
    )
    capture = CaptureQualityReport(
        surfaces=[surface], overall_verdict="good", overall_confidence=0.95
    )
    signal = _build_obstruction_signal(capture)
    assert signal.is_obstructed is False


def test_obstruction_signal_picks_up_degraded_label_verdict():
    """A degraded label region — even without obvious glare blobs — is
    enough to trigger the guarantee. This catches motion-blur-over-the-
    warning and other label-level degradations that aren't blob-shaped."""
    from app.services.sensor_check import (
        CaptureQualityReport,
        ImageQualityMetrics,
        SensorMetadata,
        SurfaceCaptureQuality,
    )
    from app.services.verify import _build_obstruction_signal

    surface = SurfaceCaptureQuality(
        surface="panel_1",
        sensor=SensorMetadata(),
        metrics=ImageQualityMetrics(
            sharpness=80,
            glare_fraction=0.0,
            brightness_mean=128,
            brightness_stddev=50,
            color_cast=8,
            megapixels=4.0,
            width_px=2000,
            height_px=2000,
        ),
        verdict="degraded",
        confidence=0.55,
        label_verdict="degraded",
        motion_blur_direction="horizontal",
    )
    capture = CaptureQualityReport(
        surfaces=[surface], overall_verdict="degraded", overall_confidence=0.55
    )
    signal = _build_obstruction_signal(capture)
    assert signal.is_obstructed is True
    assert "panel_1" in signal.reason
    assert "degraded" in signal.reason.lower()


# ---------------------------------------------------------------------------
# Backstop downgrade: missing warning + obstruction signal, no second pass
# ---------------------------------------------------------------------------


def test_no_second_pass_missing_warning_under_glare_is_advisory():
    """Even without a second-pass reader, when the primary extractor
    returned no warning AND the capture report shows obstruction, the
    "warning missing" FAIL must downgrade to ADVISORY. This exercises
    `_downgrade_missing_warning_under_obstruction` directly."""
    from app.rules.types import CheckOutcome as _CO
    from app.rules.types import ExtractedField, ExtractionContext, RuleResult
    from app.services.health_warning_second_pass import ObstructionSignal as _OS
    from app.services.verify import _downgrade_missing_warning_under_obstruction

    ctx = ExtractionContext(
        fields={},  # no health_warning extracted
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        unreadable_fields=["health_warning"],
    )
    results = [
        RuleResult(
            rule_id="spirits.health_warning.compliance",
            rule_version=1,
            citation="27 CFR 16.21",
            status=_CO.FAIL,
            finding="Government Warning is missing from the label",
        )
    ]
    obstruction = _OS(
        is_obstructed=True,
        reason="back: glare blobs cover 30% of the label region",
    )
    out = _downgrade_missing_warning_under_obstruction(results, ctx, obstruction)
    [downgraded] = out
    assert downgraded.status == _CO.ADVISORY
    assert "glare" in (downgraded.finding or "").lower()
    assert "reshoot" in (downgraded.finding or "").lower()


def test_foreign_language_guard_suppressed_when_secondary_sees_warning_region():
    """A glared English label often has sparse extracted text — only a
    couple of English keywords survive once the warning paragraph is
    occluded. The foreign-language adversarial guard would normally fire
    on that, short-circuiting through `_unreadable_rule_results` and
    bypassing the cross-check entirely. The recall guarantee says: when
    the second-pass saw the warning region, the label is plausibly
    English-with-obstruction. Suppress the guard so the cross-check runs
    and produces ADVISORY rather than the dread "unreadable" stub."""
    extractor = MockVisionExtractor(
        {
            # Sparse fixture: only one generic English token total
            # ("Co.") — would normally trip the foreign-language guard.
            "brand_name": "ASTRO",
            "class_type": "TYPE",
            "alcohol_content": "12%",
            "net_contents": "750 mL",
            "name_address": "Co.",
        }
    )
    second = MockHealthWarningExtractor(
        value=None,
        confidence=0.20,
        region_visible=True,  # the recall-preserving signal
    )
    inp = VerifyInput(
        image_bytes=b"\x89PNG\r\n\x1a\n",
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application={"producer_record": {}},
    )
    report = verify(
        inp,
        extractor=extractor,
        health_warning_reader=second,
        skip_capture_quality=True,
    )
    # Foreign-language did NOT short-circuit — the cross-check ran.
    assert report.health_warning_cross_check is not None, (
        "Foreign-language guard should be suppressed when the second-pass "
        "saw the warning region; cross-check must still run."
    )
    assert report.health_warning_cross_check["outcome"] == "unverifiable_obstructed"
    warning = next(
        r for r in report.rule_results if "health_warning" in r.rule_id
    )
    assert warning.status == CheckOutcome.ADVISORY


def test_foreign_language_guard_still_fires_when_no_warning_signal():
    """The suppression is targeted: a label with non-English text and no
    warning signals from either reader still gets flagged as foreign-
    language. We're not disabling the guard — only refusing to let it
    swallow glared English labels."""
    extractor = MockVisionExtractor(
        {
            # Plausibly Spanish-only label — no English warning vocabulary,
            # no English compliance keywords, but enough total characters
            # to clear the foreign-language minimum-length gate.
            "brand_name": "ASTRONÓMICA RESERVA",
            "class_type": "VINO TINTO RIOJA DENOMINACION DE ORIGEN",
            "alcohol_content": "13.5% Vol.",
            "net_contents": "750 ml",
            "name_address": "Embotellado en España por Bodegas Astro",
        }
    )
    # Secondary explicitly says: no warning, no region visible — no
    # English-via-warning signals to override the guard.
    second = MockHealthWarningExtractor(
        value=None,
        confidence=0.95,
        region_visible=False,
    )
    inp = VerifyInput(
        image_bytes=b"\x89PNG\r\n\x1a\n",
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application={"producer_record": {}},
    )
    report = verify(
        inp,
        extractor=extractor,
        health_warning_reader=second,
        skip_capture_quality=True,
    )
    # The foreign-language short-circuit fired — the cross-check never ran.
    assert report.overall == "unreadable"
    assert report.health_warning_cross_check is None


# ---------------------------------------------------------------------------
# Beer rule: exact_text path produces the same recall guarantee
# ---------------------------------------------------------------------------


def test_beer_exact_text_missing_warning_under_obstruction_is_advisory():
    """Beer's health_warning rule uses `exact_text` (not warning_compliance).
    The recall backstop is rule-id keyed, not check-type keyed, so beer
    must get the same ADVISORY downgrade under obstruction."""
    extractor = MockVisionExtractor(
        {
            "brand_name": "Anytown Ale",
            "class_type": "India Pale Ale",
            "alcohol_content": "5.5% alcohol by volume",
            "net_contents": "12 fl oz",
            "name_address": (
                "Brewed and bottled by Anytown Brewing Company, Anytown, ST"
            ),
        }
    )
    second = MockHealthWarningExtractor(
        value=None, confidence=0.20, region_visible=True
    )
    inp = VerifyInput(
        image_bytes=b"\x89PNG\r\n\x1a\n",
        media_type="image/png",
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        application={"producer_record": {}},
    )
    report = verify(
        inp,
        extractor=extractor,
        health_warning_reader=second,
        skip_capture_quality=True,
    )
    # Find the beer-specific health_warning rule that uses exact_text.
    warning = next(
        r for r in report.rule_results
        if r.rule_id == "beer.health_warning.exact_text"
    )
    assert warning.status == CheckOutcome.ADVISORY, (
        f"Beer exact_text rule must downgrade to ADVISORY under obstruction; "
        f"got {warning.status}: {warning.finding!r}"
    )


def test_backstop_does_not_downgrade_edit_distance_fail_with_real_text():
    """The backstop must NOT downgrade an edit-distance FAIL on text the
    reader successfully transcribed — that's a substantive non-compliance
    we want to surface, glare or no glare. Only the "I see nothing"
    branch becomes advisory under obstruction."""
    from app.rules.types import CheckOutcome as _CO
    from app.rules.types import ExtractedField, ExtractionContext, RuleResult
    from app.services.health_warning_second_pass import ObstructionSignal as _OS
    from app.services.verify import _downgrade_missing_warning_under_obstruction

    ctx = ExtractionContext(
        fields={
            "health_warning": ExtractedField(
                value="Government Warning: paraphrased text",
                confidence=0.9,
                source_image_id="panel_0",
            )
        },
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
    )
    results = [
        RuleResult(
            rule_id="spirits.health_warning.compliance",
            rule_version=1,
            citation="27 CFR 16.21",
            status=_CO.FAIL,
            finding="Warning differs from the required statement by 47 characters.",
        )
    ]
    obstruction = _OS(
        is_obstructed=True,
        reason="front: minor glare on label corner",
    )
    out = _downgrade_missing_warning_under_obstruction(results, ctx, obstruction)
    [unchanged] = out
    assert unchanged.status == _CO.FAIL, (
        "Substantive edit-distance FAIL on real transcribed text must not be "
        "downgraded; the reader had a legitimate read."
    )
