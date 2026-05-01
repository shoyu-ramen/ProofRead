"""Tests for the `wine.organic.format` rule (7 CFR 205).

Organic claims on wine labels are optional. When present, the wording
must match a recognised form ("USDA Organic", "Made with Organic
Grapes", a certifying-agent attribution, etc.). When absent, the rule
must NOT fail — making no claim is a normal, compliant state.
Severity is ADVISORY end-to-end so a stale or unusual phrasing surfaces
to the producer without failing the report.
"""

from __future__ import annotations

import pytest

from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import CheckOutcome, ExtractedField, ExtractionContext

RULE_ID = "wine.organic.format"


def _ctx(organic_value: str | None) -> ExtractionContext:
    fields: dict[str, ExtractedField] = {}
    unreadable: list[str] = []
    if organic_value is None:
        unreadable.append("organic_certification")
    else:
        fields["organic_certification"] = ExtractedField(
            value=organic_value, confidence=0.95
        )
    return ExtractionContext(
        fields=fields,
        beverage_type="wine",
        container_size_ml=750,
        is_imported=False,
        unreadable_fields=unreadable,
    )


def _evaluate(ctx: ExtractionContext):
    rule = next(r for r in load_rules("wine") if r.id == RULE_ID)
    [result] = RuleEngine([rule]).evaluate(ctx)
    return result


@pytest.mark.parametrize(
    "value",
    [
        "USDA Organic",
        "USDA ORGANIC",
        "Made with Organic Grapes",
        "MADE WITH ORGANIC GRAPES",
        "Certified Organic by Oregon Tilth",
        "Organic",
    ],
)
def test_organic_format_pass_on_recognised_phrasings(value: str):
    result = _evaluate(_ctx(value))
    assert result.status == CheckOutcome.PASS, (result.status, result.finding)


def test_organic_field_unreadable_passes_via_optional_regex():
    """The conditional rule says: only validate if the field has a value.
    The regex check is `optional: true`, so a missing value passes
    silently — making no organic claim is a valid, compliant state.

    Note: the engine still applies the REQUIRED-rule confidence-aware
    downgrade for unreadable fields. Because this rule is ADVISORY, that
    downgrade does not apply, and the optional-regex semantics surface
    a PASS directly.
    """
    result = _evaluate(_ctx(None))
    # Either PASS (optional regex) or ADVISORY (engine downgrade) — both
    # are honest outcomes; what we do NOT want is FAIL.
    assert result.status in {CheckOutcome.PASS, CheckOutcome.ADVISORY}, result.finding


def test_organic_unrecognised_phrasing_advisory():
    """A value that doesn't match any recognised pattern surfaces as
    ADVISORY because the rule's static severity is `advisory` — the
    engine downgrades a regex-FAIL to ADVISORY automatically.
    """
    result = _evaluate(_ctx("biodynamic"))
    assert result.status == CheckOutcome.ADVISORY


def test_organic_rule_metadata():
    rule = next(r for r in load_rules("wine") if r.id == RULE_ID)
    assert rule.version == 1
    assert rule.citation.startswith("https://")
    assert "ecfr.gov" in rule.citation
    assert rule.fix_suggestion is not None
    # ADVISORY: the rule must never produce FAIL even on a non-matching value.
    from app.rules.types import Severity

    assert rule.severity == Severity.ADVISORY
