"""Tests for the `spirits.age_statement.format` rule (27 CFR 5.40).

Severity is conditional on the class_type the label declares:

  * Straight whiskey class (e.g. "Kentucky Straight Bourbon Whiskey",
    "Tennessee Straight Whiskey", "Straight Rye Whiskey"): age statement
    is REQUIRED for products bottled at less than four years old. A
    missing or malformed age statement FAILs the report.
  * Anything else: the rule is ADVISORY — a missing age statement is a
    PASS (no requirement), and a present-but-malformed value surfaces
    as an ADVISORY note rather than a FAIL.
"""

from __future__ import annotations

import pytest

from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import CheckOutcome, ExtractedField, ExtractionContext

RULE_ID = "spirits.age_statement.format"


def _ctx(
    *,
    class_type: str | None,
    age_value: str | None,
) -> ExtractionContext:
    fields: dict[str, ExtractedField] = {}
    if class_type is not None:
        fields["class_type"] = ExtractedField(value=class_type, confidence=0.95)
    if age_value is not None:
        fields["age_statement"] = ExtractedField(value=age_value, confidence=0.95)
    return ExtractionContext(
        fields=fields,
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        unreadable_fields=[],
    )


def _evaluate(ctx: ExtractionContext):
    rule = next(r for r in load_rules("spirits") if r.id == RULE_ID)
    [result] = RuleEngine([rule]).evaluate(ctx)
    return result


# ---------------------------------------------------------------------------
# Required path — straight-whiskey class triggers the age requirement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "class_type",
    [
        "Kentucky Straight Bourbon Whiskey",
        "Tennessee Straight Whiskey",
        "Straight Rye Whiskey",
        "STRAIGHT MALT WHISKEY",
        # Single-line "Straight Bourbon" with no state prefix.
        "Straight Bourbon Whiskey",
    ],
)
def test_required_pass_when_age_present_and_well_formed(class_type: str):
    ctx = _ctx(class_type=class_type, age_value="Aged 4 Years")
    assert _evaluate(ctx).status == CheckOutcome.PASS


def test_required_fail_when_age_missing_for_straight_whiskey():
    ctx = _ctx(class_type="Kentucky Straight Bourbon Whiskey", age_value=None)
    result = _evaluate(ctx)
    assert result.status == CheckOutcome.FAIL
    assert "age statement" in (result.finding or "").lower()


def test_required_fail_when_age_value_is_malformed():
    """Free-form text that doesn't match the recognised pattern still fails
    on a straight whiskey class.
    """
    ctx = _ctx(
        class_type="Kentucky Straight Bourbon Whiskey",
        age_value="aged for a long time",
    )
    result = _evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


@pytest.mark.parametrize(
    "age_value",
    [
        "Aged 4 Years",
        "AGED 18 MONTHS",
        "Aged 1 Year",
        "Aged 12 Months",
        "Aged a minimum of 6 Years",
    ],
)
def test_required_pass_on_recognised_age_formats(age_value: str):
    ctx = _ctx(
        class_type="Kentucky Straight Bourbon Whiskey", age_value=age_value
    )
    assert _evaluate(ctx).status == CheckOutcome.PASS


# ---------------------------------------------------------------------------
# Advisory path — non-straight class doesn't trigger the requirement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "class_type",
    [
        "London Dry Gin",
        "Vodka",
        "Silver Tequila",
        "Light Rum",
        # Whiskey words without "Straight" → still advisory.
        "Bourbon Whiskey",
        "Single Malt Whisky",
    ],
)
def test_advisory_pass_when_age_missing_for_non_straight(class_type: str):
    """Gin, vodka, and non-straight whiskey have no age requirement; a
    missing age_statement is a PASS, not an ADVISORY.
    """
    ctx = _ctx(class_type=class_type, age_value=None)
    assert _evaluate(ctx).status == CheckOutcome.PASS


def test_advisory_when_age_present_but_malformed_on_non_straight():
    """A present-but-non-conforming age statement on a non-straight class
    surfaces as ADVISORY so the producer can clean up the wording —
    without failing the report.
    """
    ctx = _ctx(class_type="London Dry Gin", age_value="aged for a while")
    assert _evaluate(ctx).status == CheckOutcome.ADVISORY


def test_advisory_pass_when_age_present_and_formed_on_non_straight():
    """A well-formed age statement on a non-straight class still passes."""
    ctx = _ctx(class_type="London Dry Gin", age_value="Aged 3 Years")
    assert _evaluate(ctx).status == CheckOutcome.PASS


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_missing_class_type_treated_as_non_straight():
    """If the class_type couldn't be read at all, the rule defers to its
    ADVISORY behaviour (treat as non-straight) rather than guessing FAIL.
    """
    ctx = _ctx(class_type=None, age_value=None)
    assert _evaluate(ctx).status == CheckOutcome.PASS


def test_age_unreadable_field_downgrades_to_advisory():
    """If the field is in unreadable_fields, the engine downgrades the
    REQUIRED rule to ADVISORY before the check runs.
    """
    ctx = ExtractionContext(
        fields={
            "class_type": ExtractedField(
                value="Kentucky Straight Bourbon Whiskey", confidence=0.95
            ),
        },
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        unreadable_fields=["age_statement"],
    )
    rule = next(r for r in load_rules("spirits") if r.id == RULE_ID)
    [result] = RuleEngine([rule]).evaluate(ctx)
    assert result.status == CheckOutcome.ADVISORY


def test_age_rule_metadata():
    rule = next(r for r in load_rules("spirits") if r.id == RULE_ID)
    assert rule.version == 1
    assert rule.citation.startswith("https://")
    assert "ecfr.gov" in rule.citation
    assert rule.fix_suggestion is not None
    assert len(rule.fix_suggestion.strip()) > 0
