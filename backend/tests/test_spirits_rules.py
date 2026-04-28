"""Tests for the spirits rule definitions."""

from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import CheckOutcome, ExtractedField, ExtractionContext

CANONICAL_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


EXPECTED_SPIRITS_RULES = {
    "spirits.brand_name.matches_application",
    "spirits.class_type.matches_application",
    "spirits.alcohol_content.format",
    "spirits.alcohol_content.matches_application",
    "spirits.net_contents.matches_application",
    "spirits.name_address.presence",
    "spirits.country_of_origin.presence_if_imported",
    "spirits.health_warning.compliance",
}


def _full_context(application_overrides=None, field_overrides=None) -> ExtractionContext:
    """Build a fully-populated 'good' bourbon scan."""
    base_app = {
        "producer_record": {
            "brand_name": "Old Tom Distillery",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45.0",
            "net_contents": "750 mL",
            "name_address": "Old Tom Distilling Co., Bardstown, Kentucky",
            "country_of_origin": "USA",
        }
    }
    if application_overrides:
        base_app["producer_record"].update(application_overrides)

    base_fields = {
        "brand_name": ExtractedField(value="Old Tom Distillery"),
        "class_type": ExtractedField(value="Kentucky Straight Bourbon Whiskey"),
        "alcohol_content": ExtractedField(value="45% Alc./Vol. (90 Proof)"),
        "net_contents": ExtractedField(value="750 mL"),
        "name_address": ExtractedField(
            value="Bottled by Old Tom Distilling Co., Bardstown, Kentucky"
        ),
        "health_warning": ExtractedField(value=CANONICAL_WARNING),
    }
    if field_overrides:
        for k, v in field_overrides.items():
            if v is None:
                base_fields.pop(k, None)
            else:
                base_fields[k] = v

    return ExtractionContext(
        fields=base_fields,
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=base_app,
    )


def test_spirits_rules_load():
    rule_ids = {r.id for r in load_rules("spirits")}
    missing = EXPECTED_SPIRITS_RULES - rule_ids
    assert not missing, f"Missing spirits rules: {missing}"


def test_compliant_bourbon_label_passes_all_rules():
    rules = load_rules("spirits")
    engine = RuleEngine(rules)
    results = engine.evaluate(_full_context())
    failed = [r for r in results if r.status == CheckOutcome.FAIL]
    assert not failed, f"Unexpected failures: {[(r.rule_id, r.finding) for r in failed]}"


def test_brand_case_difference_warns():
    """Label shows ALL CAPS, application has title case — should WARN, not FAIL."""
    rules = [r for r in load_rules("spirits") if r.id == "spirits.brand_name.matches_application"]
    engine = RuleEngine(rules)
    ctx = _full_context(
        field_overrides={"brand_name": ExtractedField(value="OLD TOM DISTILLERY")},
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.WARN, result.finding


def test_brand_substantive_mismatch_fails():
    rules = [r for r in load_rules("spirits") if r.id == "spirits.brand_name.matches_application"]
    engine = RuleEngine(rules)
    ctx = _full_context(
        field_overrides={"brand_name": ExtractedField(value="Different Distillery")},
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


def test_alcohol_proof_equivalence_passes():
    rules = [
        r for r in load_rules("spirits")
        if r.id == "spirits.alcohol_content.matches_application"
    ]
    engine = RuleEngine(rules)
    ctx = _full_context(
        field_overrides={"alcohol_content": ExtractedField(value="90 Proof")},
        application_overrides={"alcohol_content": "45"},
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.PASS


def test_alcohol_substantive_mismatch_fails():
    rules = [
        r for r in load_rules("spirits")
        if r.id == "spirits.alcohol_content.matches_application"
    ]
    engine = RuleEngine(rules)
    ctx = _full_context(
        field_overrides={"alcohol_content": ExtractedField(value="40% Alc./Vol.")},
        application_overrides={"alcohol_content": "45"},
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


def test_net_contents_unit_conversion_passes():
    rules = [r for r in load_rules("spirits") if r.id == "spirits.net_contents.matches_application"]
    engine = RuleEngine(rules)
    ctx = _full_context(
        application_overrides={"net_contents": "0.75 L"},
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.PASS


def test_warning_with_titlecase_prefix_fails():
    rules = [r for r in load_rules("spirits") if r.id == "spirits.health_warning.compliance"]
    engine = RuleEngine(rules)
    bad = CANONICAL_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    ctx = _full_context(field_overrides={"health_warning": ExtractedField(value=bad)})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL
    assert "capitals" in (result.finding or "").lower()


def test_country_of_origin_required_when_imported():
    rules = [
        r for r in load_rules("spirits")
        if r.id == "spirits.country_of_origin.presence_if_imported"
    ]
    engine = RuleEngine(rules)
    ctx = _full_context(field_overrides={"country_of_origin": None})
    ctx.is_imported = True
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


def test_country_of_origin_skipped_for_domestic():
    rules = [
        r for r in load_rules("spirits")
        if r.id == "spirits.country_of_origin.presence_if_imported"
    ]
    engine = RuleEngine(rules)
    ctx = _full_context(field_overrides={"country_of_origin": None})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.NA
