"""Tests for the Government Warning redundant second-pass.

The cross-check is the load-bearing piece: it decides whether a rule
engine FAIL should stand or be downgraded to ADVISORY based on whether
two independent reads of the warning agree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rules.types import CheckOutcome, ExtractedField, ExtractionContext
from app.services.health_warning_second_pass import (
    ClaudeHealthWarningExtractor,
    CrossCheckResult,
    MockHealthWarningExtractor,
    WarningRead,
    _parse_response,
    cross_check,
)
from app.services.verify import VerifyInput, verify
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
