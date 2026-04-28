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
