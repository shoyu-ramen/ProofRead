"""Tests for the warning-compliance, numeric and volume cross-reference checks."""

from app.rules.checks import (
    check_cross_reference_numeric,
    check_cross_reference_volume,
    check_warning_compliance,
)
from app.rules.types import CheckOutcome, ExtractedField, ExtractionContext


def _ctx(fields=None, application=None) -> ExtractionContext:
    return ExtractionContext(
        fields=fields or {},
        beverage_type="spirits",
        container_size_ml=750,
        application=application or {},
    )


# ---------- warning_compliance ----------

CANONICAL_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


def test_warning_passes_on_canonical():
    ctx = _ctx({"health_warning": ExtractedField(value=CANONICAL_WARNING)})
    res = check_warning_compliance({"field": "health_warning"}, ctx)
    assert res.outcome == CheckOutcome.PASS


def test_warning_passes_on_all_caps_body():
    """The prefix must be ALL CAPS; the body case-insensitively matches."""
    all_caps = CANONICAL_WARNING.upper()  # whole thing in caps including prefix
    ctx = _ctx({"health_warning": ExtractedField(value=all_caps)})
    res = check_warning_compliance({"field": "health_warning"}, ctx)
    assert res.outcome == CheckOutcome.PASS, res.finding


def test_warning_fails_on_title_case_prefix():
    """Jenny's specific concern — title case prefix must FAIL outright."""
    bad = CANONICAL_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    ctx = _ctx({"health_warning": ExtractedField(value=bad)})
    res = check_warning_compliance({"field": "health_warning"}, ctx)
    assert res.outcome == CheckOutcome.FAIL
    assert "capitals" in (res.finding or "").lower()


def test_warning_warns_on_small_body_difference():
    """A handful of character edits → WARN, not FAIL — surfaces for review."""
    typo = CANONICAL_WARNING.replace("women should", "women shoul")  # 1 char drop
    ctx = _ctx({"health_warning": ExtractedField(value=typo)})
    res = check_warning_compliance(
        {"field": "health_warning", "max_body_edit_distance": 5}, ctx
    )
    assert res.outcome == CheckOutcome.WARN


def test_warning_fails_on_large_body_difference():
    paraphrase = (
        "GOVERNMENT WARNING: According to the Surgeon General, drinking is "
        "dangerous for your health and you should not do it while pregnant or driving."
    )
    ctx = _ctx({"health_warning": ExtractedField(value=paraphrase)})
    res = check_warning_compliance(
        {"field": "health_warning", "max_body_edit_distance": 5}, ctx
    )
    assert res.outcome == CheckOutcome.FAIL
    assert "edit distance" in (res.finding or "").lower()


def test_warning_fails_on_missing():
    ctx = _ctx({})
    res = check_warning_compliance({"field": "health_warning"}, ctx)
    assert res.outcome == CheckOutcome.FAIL
    assert "missing" in (res.finding or "").lower()


# ---------- cross_reference_numeric ----------


def _abv_ctx(label_value: str, app_value: str | None) -> ExtractionContext:
    fields = {"alcohol_content": ExtractedField(value=label_value)}
    app = {"producer_record": {"alcohol_content": app_value}} if app_value is not None else {}
    return _ctx(fields, app)


def test_numeric_passes_on_exact_match():
    ctx = _abv_ctx("45.0% Alc./Vol.", "45.0")
    res = check_cross_reference_numeric(
        {"field": "alcohol_content", "record_key": "alcohol_content"}, ctx
    )
    assert res.outcome == CheckOutcome.PASS


def test_numeric_passes_within_tolerance():
    ctx = _abv_ctx("44.99% Alc./Vol.", "45.0")
    res = check_cross_reference_numeric(
        {
            "field": "alcohol_content",
            "record_key": "alcohol_content",
            "tolerance": 0.01,
        },
        ctx,
    )
    assert res.outcome == CheckOutcome.PASS


def test_numeric_passes_on_proof_equivalence():
    """Label shows '90 Proof' first (proof = 2 × ABV) — should still match."""
    ctx = _abv_ctx("90 Proof", "45.0")
    res = check_cross_reference_numeric(
        {
            "field": "alcohol_content",
            "record_key": "alcohol_content",
            "allow_proof_equivalence": True,
        },
        ctx,
    )
    assert res.outcome == CheckOutcome.PASS


def test_numeric_proof_equivalence_can_be_disabled():
    ctx = _abv_ctx("90 Proof", "45.0")
    res = check_cross_reference_numeric(
        {
            "field": "alcohol_content",
            "record_key": "alcohol_content",
            "allow_proof_equivalence": False,
        },
        ctx,
    )
    assert res.outcome == CheckOutcome.FAIL


def test_numeric_fails_on_substantive_mismatch():
    ctx = _abv_ctx("40.0% Alc./Vol.", "45.0")
    res = check_cross_reference_numeric(
        {"field": "alcohol_content", "record_key": "alcohol_content"}, ctx
    )
    assert res.outcome == CheckOutcome.FAIL
    assert "40" in (res.finding or "")
    assert "45" in (res.finding or "")


def test_numeric_passes_when_no_record_and_optional():
    ctx = _abv_ctx("45.0%", None)
    res = check_cross_reference_numeric(
        {"field": "alcohol_content", "record_key": "alcohol_content"}, ctx
    )
    assert res.outcome == CheckOutcome.PASS


# ---------- cross_reference_volume ----------


def _vol_ctx(label_value: str, app_value: str | None) -> ExtractionContext:
    fields = {"net_contents": ExtractedField(value=label_value)}
    app = {"producer_record": {"net_contents": app_value}} if app_value is not None else {}
    return _ctx(fields, app)


def test_volume_passes_on_exact_match():
    ctx = _vol_ctx("750 mL", "750 mL")
    res = check_cross_reference_volume(
        {"field": "net_contents", "record_key": "net_contents"}, ctx
    )
    assert res.outcome == CheckOutcome.PASS


def test_volume_passes_across_unit_conversion():
    """0.75 L on the application equals 750 mL on the label."""
    ctx = _vol_ctx("750 mL", "0.75 L")
    res = check_cross_reference_volume(
        {"field": "net_contents", "record_key": "net_contents"}, ctx
    )
    assert res.outcome == CheckOutcome.PASS


def test_volume_passes_for_fl_oz_to_ml():
    """16 FL OZ ≈ 473 mL — within 1% tolerance."""
    ctx = _vol_ctx("16 FL OZ", "473 mL")
    res = check_cross_reference_volume(
        {"field": "net_contents", "record_key": "net_contents", "tolerance": 0.02}, ctx
    )
    assert res.outcome == CheckOutcome.PASS


def test_volume_fails_on_substantive_mismatch():
    ctx = _vol_ctx("750 mL", "375 mL")
    res = check_cross_reference_volume(
        {"field": "net_contents", "record_key": "net_contents"}, ctx
    )
    assert res.outcome == CheckOutcome.FAIL
