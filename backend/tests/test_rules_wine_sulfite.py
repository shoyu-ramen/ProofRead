"""Tests for the `wine.sulfite.presence` rule (27 CFR 4.32).

Wine labels for products at or above 10 ppm SO₂ — which covers nearly
every commercial wine — must declare "Contains Sulfites" or an
equivalent. The rule pairs a presence check with a regex that pins the
phrase to a recognised legal form.
"""

from __future__ import annotations

import pytest

from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import CheckOutcome, ExtractedField, ExtractionContext

RULE_ID = "wine.sulfite.presence"


def _ctx(sulfite_value: str | None, *, low_confidence: bool = False) -> ExtractionContext:
    fields: dict[str, ExtractedField] = {}
    unreadable: list[str] = []
    if sulfite_value is None:
        unreadable.append("sulfite_declaration")
    else:
        fields["sulfite_declaration"] = ExtractedField(
            value=sulfite_value,
            confidence=0.4 if low_confidence else 0.95,
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
        "Contains Sulfites",
        "CONTAINS SULFITES",
        "contains sulfites",
        "CONTAINS SULFITES (USED AS A PRESERVATIVE)",
        # Singular variant — TTB accepts both.
        "Contains Sulfite",
        # Embedded in a longer line.
        "Government Warning: ... Contains Sulfites",
    ],
)
def test_sulfite_pass_on_recognised_phrasings(value: str):
    result = _evaluate(_ctx(value))
    assert result.status == CheckOutcome.PASS, result.finding


def test_sulfite_present_with_unrelated_text_fails_regex():
    """A non-empty value that doesn't carry the canonical phrase fails the
    regex check (presence passes but format does not).
    """
    result = _evaluate(_ctx("Vegan Wine"))
    assert result.status == CheckOutcome.FAIL


def test_sulfite_unreadable_field_downgrades_to_advisory():
    """When the model marks the sulfite declaration unreadable (couldn't
    find or read it), the REQUIRED rule downgrades to ADVISORY rather
    than producing a confident wrong-FAIL.
    """
    result = _evaluate(_ctx(None))
    assert result.status == CheckOutcome.ADVISORY


def test_sulfite_low_confidence_field_downgrades_to_advisory():
    result = _evaluate(_ctx("Contains Sulfites", low_confidence=True))
    assert result.status == CheckOutcome.ADVISORY


def test_sulfite_rule_metadata():
    """Spot-check the metadata required by the delivery checklist:
    a numbered version, a citation URL, and a fix_suggestion."""
    rule = next(r for r in load_rules("wine") if r.id == RULE_ID)
    assert rule.version == 1
    assert rule.citation.startswith("https://")
    assert "ecfr.gov" in rule.citation
    assert rule.fix_suggestion is not None
    assert len(rule.fix_suggestion.strip()) > 0
