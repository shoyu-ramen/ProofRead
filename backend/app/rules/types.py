from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    REQUIRED = "required"
    ADVISORY = "advisory"


class CheckOutcome(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ADVISORY = "advisory"
    NA = "na"


# Severity-ordered for aggregation — higher value wins.
_OUTCOME_ORDER = {
    CheckOutcome.NA: 0,
    CheckOutcome.PASS: 1,
    CheckOutcome.ADVISORY: 2,
    CheckOutcome.WARN: 3,
    CheckOutcome.FAIL: 4,
}


def worse(a: CheckOutcome, b: CheckOutcome) -> CheckOutcome:
    """Return whichever outcome is more severe."""
    return a if _OUTCOME_ORDER[a] >= _OUTCOME_ORDER[b] else b


Bbox = tuple[int, int, int, int]


@dataclass
class Check:
    type: str
    params: dict[str, Any]


@dataclass
class Rule:
    id: str
    version: int
    beverage_types: list[str]
    citation: str
    description: str
    severity: Severity
    checks: list[Check]
    fix_suggestion: str | None = None
    applies_if: str | None = None
    exempt_if: str | None = None


@dataclass
class ExtractedField:
    value: str | None
    bbox: Bbox | None = None
    confidence: float = 1.0
    source_image_id: str | None = None


@dataclass
class ExtractionContext:
    fields: dict[str, ExtractedField]
    beverage_type: str
    container_size_ml: int
    is_imported: bool = False
    abv_pct: float | None = None
    raw_ocr_texts: dict[str, str] = field(default_factory=dict)
    application: dict[str, Any] = field(default_factory=dict)
    unreadable_fields: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    outcome: CheckOutcome
    finding: str | None = None
    expected: str | None = None
    bbox: Bbox | None = None


@dataclass
class RuleResult:
    rule_id: str
    rule_version: int
    citation: str
    status: CheckOutcome
    finding: str | None = None
    expected: str | None = None
    fix_suggestion: str | None = None
    bbox: Bbox | None = None
