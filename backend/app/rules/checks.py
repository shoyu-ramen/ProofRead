import re
from collections.abc import Callable

from rapidfuzz.distance import Levenshtein

from app.rules.canonical import load_canonical
from app.rules.types import CheckOutcome, CheckResult, ExtractionContext


def _get_field_value(ctx: ExtractionContext, field: str) -> tuple[str | None, tuple | None]:
    f = ctx.fields.get(field)
    if f is None:
        return None, None
    return f.value, f.bbox


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def check_presence(params: dict, ctx: ExtractionContext) -> CheckResult:
    field = params["field"]
    value, bbox = _get_field_value(ctx, field)
    if value is None or not value.strip():
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=f"Required field '{field}' was not found on the label",
            expected=f"A non-empty value for '{field}'",
        )
    return CheckResult(outcome=CheckOutcome.PASS, bbox=bbox)


def _parse_regex_flags(flags_str: str) -> int:
    """Translate the YAML-friendly `flags: i` shorthand to a `re` bitmask."""
    flags = 0
    if "i" in flags_str.lower():
        flags |= re.IGNORECASE
    return flags


def check_regex(params: dict, ctx: ExtractionContext) -> CheckResult:
    field = params["field"]
    pattern = params["pattern"]
    flags = _parse_regex_flags(params.get("flags", ""))
    optional = params.get("optional", False)

    value, bbox = _get_field_value(ctx, field)
    if value is None or not value.strip():
        if optional:
            return CheckResult(outcome=CheckOutcome.PASS)
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=f"Field '{field}' not found",
            expected=f"A value matching pattern: {pattern}",
        )

    if re.search(pattern, value.strip(), flags):
        return CheckResult(outcome=CheckOutcome.PASS, bbox=bbox)

    return CheckResult(
        outcome=CheckOutcome.FAIL,
        finding=f"Field '{field}' value {value.strip()!r} does not match required format",
        expected=f"A value matching pattern: {pattern}",
        bbox=bbox,
    )


def check_exact_text(params: dict, ctx: ExtractionContext) -> CheckResult:
    field = params["field"]
    canonical_ref = params.get("canonical_ref")
    max_edit_distance = int(params.get("max_edit_distance", 0))

    if canonical_ref:
        canonical_raw = load_canonical(canonical_ref)
    else:
        canonical_raw = params.get("canonical", "")
    canonical = _normalize(canonical_raw)

    value, bbox = _get_field_value(ctx, field)
    if value is None or not value.strip():
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=f"Required text for '{field}' was not found on the label",
            expected=canonical,
        )

    found = _normalize(value)
    distance = Levenshtein.distance(found, canonical)
    if distance <= max_edit_distance:
        return CheckResult(outcome=CheckOutcome.PASS, bbox=bbox)

    return CheckResult(
        outcome=CheckOutcome.FAIL,
        finding=(
            f"Found text differs from required by edit distance {distance} "
            f"(max allowed {max_edit_distance}): {found!r}"
        ),
        expected=canonical,
        bbox=bbox,
    )


def check_advisory_note(params: dict, ctx: ExtractionContext) -> CheckResult:
    return CheckResult(
        outcome=CheckOutcome.ADVISORY,
        finding=params.get("message", "Advisory note"),
    )


def check_cross_reference(params: dict, ctx: ExtractionContext) -> CheckResult:
    """Compare an extracted field against the producer's record value.

    Distinguishes substantive mismatches (FAIL) from typography-only
    differences (WARN) — e.g. label shows "STONE'S THROW" while the
    submission record reads "Stone's Throw". A wrong "fail" wastes the
    user's time chasing a phantom problem (SPEC §0.5); WARN surfaces the
    discrepancy without claiming non-compliance.
    """
    field = params["field"]
    record_key = params["record_key"]
    optional = params.get("optional", True)

    record = (ctx.application or {}).get("producer_record") or {}
    expected_raw = record.get(record_key)
    if expected_raw is None or not str(expected_raw).strip():
        if optional:
            return CheckResult(outcome=CheckOutcome.PASS)
        return CheckResult(
            outcome=CheckOutcome.ADVISORY,
            finding=f"No producer record for '{record_key}' to cross-reference",
        )

    value, bbox = _get_field_value(ctx, field)
    if value is None or not value.strip():
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=(
                f"Field '{field}' missing from label; producer record "
                f"expects {expected_raw!r}"
            ),
            expected=str(expected_raw),
        )

    found_n = _normalize(value)
    expected_n = _normalize(str(expected_raw))

    if found_n == expected_n:
        return CheckResult(outcome=CheckOutcome.PASS, bbox=bbox)

    if found_n.lower() == expected_n.lower():
        return CheckResult(
            outcome=CheckOutcome.WARN,
            finding=(
                f"Case-only mismatch: label shows {found_n!r} but producer "
                f"record is {expected_n!r}. Confirm intent — register the "
                "label exactly as printed if this is intentional."
            ),
            expected=expected_n,
            bbox=bbox,
        )

    return CheckResult(
        outcome=CheckOutcome.FAIL,
        finding=(
            f"Substantive mismatch: label shows {found_n!r} but producer "
            f"record is {expected_n!r}."
        ),
        expected=expected_n,
        bbox=bbox,
    )


_VOLUME_TO_ML = {
    "ml": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "millilitre": 1.0,
    "millilitres": 1.0,
    "cl": 10.0,
    "centiliter": 10.0,
    "centiliters": 10.0,
    "l": 1000.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "litre": 1000.0,
    "litres": 1000.0,
    "fluid ounces": 29.5735,
    "fluid ounce": 29.5735,
    "fl oz": 29.5735,
    "fl. oz.": 29.5735,
    "floz": 29.5735,
    "oz": 29.5735,
}


def _parse_volume_to_ml(text: str) -> float | None:
    if not text:
        return None
    cleaned = re.sub(r"[,\s]", " ", text.lower()).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*([a-z\.\s]+)", cleaned)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except ValueError:
        return None
    unit_part = m.group(2).strip().rstrip(".")
    # Try longest unit names first so "fluid ounces" beats "ounces".
    for unit, factor in sorted(_VOLUME_TO_ML.items(), key=lambda kv: -len(kv[0])):
        if unit_part.startswith(unit):
            return num * factor
    return None


def _extract_first_number(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def check_warning_compliance(params: dict, ctx: ExtractionContext) -> CheckResult:
    """Verify the Government Health Warning.

    Two failure modes are distinguished:
      - The "GOVERNMENT WARNING:" prefix is not present in capitals → FAIL
        (Jenny's specific concern: "people try to get creative with the
        warning... title case instead of all caps. Rejected.")
      - The body text differs from the canonical statement. Tiny edits within
        `max_body_edit_distance` → WARN; larger edits → FAIL.

    Body comparison is case-insensitive so legally-acceptable ALL-CAPS
    renderings of the warning still pass once the prefix is correct.
    """
    field = params["field"]
    canonical_ref = params.get("canonical_ref", "health_warning")
    max_body_edit_distance = int(params.get("max_body_edit_distance", 5))
    required_prefix = params.get("required_prefix", "GOVERNMENT WARNING:")

    canonical = _normalize(load_canonical(canonical_ref))
    value, bbox = _get_field_value(ctx, field)
    if value is None or not value.strip():
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding="Government Warning is missing from the label",
            expected=canonical,
        )

    found = _normalize(value)

    if required_prefix not in found:
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=(
                f"The {required_prefix!r} prefix must appear verbatim in capitals. "
                "Title case (e.g. 'Government Warning:') is not acceptable."
            ),
            expected=canonical,
            bbox=bbox,
        )

    distance = Levenshtein.distance(found.lower(), canonical.lower())
    if distance == 0:
        return CheckResult(outcome=CheckOutcome.PASS, bbox=bbox)
    if distance <= max_body_edit_distance:
        return CheckResult(
            outcome=CheckOutcome.WARN,
            finding=(
                f"Warning text differs from the required statement by {distance} "
                "character(s). Review wording before approving."
            ),
            expected=canonical,
            bbox=bbox,
        )
    return CheckResult(
        outcome=CheckOutcome.FAIL,
        finding=(
            f"Warning text differs from the required statement by edit distance "
            f"{distance} (max allowed {max_body_edit_distance})."
        ),
        expected=canonical,
        bbox=bbox,
    )


def check_cross_reference_numeric(params: dict, ctx: ExtractionContext) -> CheckResult:
    """Compare an extracted number against the producer record.

    Handles ABV ↔ proof equivalence (proof = ABV × 2): if the label shows
    "45% Alc./Vol. (90 Proof)" and the application records 45.0, both
    numbers are recognised as matches.

    `tolerance` is a fractional tolerance (default 1%), applied symmetrically
    around the larger absolute value so 0.005 means ±0.5%.
    """
    field = params["field"]
    record_key = params["record_key"]
    tolerance = float(params.get("tolerance", 0.005))
    allow_proof_equivalence = bool(params.get("allow_proof_equivalence", True))
    optional = params.get("optional", True)

    record = (ctx.application or {}).get("producer_record") or {}
    expected_raw = record.get(record_key)
    if expected_raw is None or str(expected_raw).strip() == "":
        if optional:
            return CheckResult(outcome=CheckOutcome.PASS)
        return CheckResult(
            outcome=CheckOutcome.ADVISORY,
            finding=f"No producer record for '{record_key}' to cross-reference",
        )

    value, bbox = _get_field_value(ctx, field)
    if value is None or not value.strip():
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=(
                f"Field '{field}' missing from label; producer record "
                f"expects {expected_raw!r}"
            ),
            expected=str(expected_raw),
        )

    expected_num = _extract_first_number(str(expected_raw))
    found_num = _extract_first_number(value)
    if expected_num is None or found_num is None:
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=(
                f"Could not parse numeric value: label has {value!r}, "
                f"record has {expected_raw!r}"
            ),
            expected=str(expected_raw),
            bbox=bbox,
        )

    def _within(a: float, b: float) -> bool:
        return abs(a - b) <= tolerance * max(abs(a), abs(b), 1.0)

    if _within(found_num, expected_num):
        return CheckResult(outcome=CheckOutcome.PASS, bbox=bbox)

    if allow_proof_equivalence and (
        _within(found_num, expected_num * 2) or _within(found_num * 2, expected_num)
    ):
        return CheckResult(outcome=CheckOutcome.PASS, bbox=bbox)

    return CheckResult(
        outcome=CheckOutcome.FAIL,
        finding=(
            f"Numeric mismatch: label shows {found_num} (from {value!r}); "
            f"record expects {expected_num} (from {expected_raw!r})."
        ),
        expected=str(expected_raw),
        bbox=bbox,
    )


def check_cross_reference_volume(params: dict, ctx: ExtractionContext) -> CheckResult:
    """Compare an extracted net-contents value against the producer record,
    converting both sides to millilitres so "750 mL" and "0.75 L" match.
    """
    field = params["field"]
    record_key = params["record_key"]
    tolerance = float(params.get("tolerance", 0.01))
    optional = params.get("optional", True)

    record = (ctx.application or {}).get("producer_record") or {}
    expected_raw = record.get(record_key)
    if expected_raw is None or str(expected_raw).strip() == "":
        if optional:
            return CheckResult(outcome=CheckOutcome.PASS)
        return CheckResult(
            outcome=CheckOutcome.ADVISORY,
            finding=f"No producer record for '{record_key}' to cross-reference",
        )

    value, bbox = _get_field_value(ctx, field)
    if value is None or not value.strip():
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=(
                f"Field '{field}' missing from label; producer record "
                f"expects {expected_raw!r}"
            ),
            expected=str(expected_raw),
        )

    found_ml = _parse_volume_to_ml(value)
    expected_ml = _parse_volume_to_ml(str(expected_raw))
    if found_ml is None or expected_ml is None:
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=(
                f"Could not parse volume: label has {value!r}, "
                f"record has {expected_raw!r}"
            ),
            expected=str(expected_raw),
            bbox=bbox,
        )

    if abs(found_ml - expected_ml) <= tolerance * max(found_ml, expected_ml):
        return CheckResult(outcome=CheckOutcome.PASS, bbox=bbox)

    return CheckResult(
        outcome=CheckOutcome.FAIL,
        finding=(
            f"Net contents mismatch: label shows ~{found_ml:.0f} mL "
            f"(from {value!r}); record expects ~{expected_ml:.0f} mL "
            f"(from {expected_raw!r})."
        ),
        expected=str(expected_raw),
        bbox=bbox,
    )


# Words that, paired with "Straight", trigger 27 CFR 5.40's age-statement
# requirement. Any spirit whose `class_type` contains "Straight" plus one
# of these designators must declare an age when bottled under four years
# old. The check below treats the requirement as REQUIRED for that combo
# and as ADVISORY for everything else (so a present-but-malformed age
# statement on, say, a gin still surfaces, just without failing the
# report).
_STRAIGHT_WHISKEY_TRIGGER_WORDS = (
    "whiskey",
    "whisky",
    "bourbon",
    "rye",
    "corn",
    "malt",
    "wheat",
)


def _is_straight_whiskey_class(class_type_value: str | None) -> bool:
    """True when `class_type_value` declares a straight whiskey class.

    27 CFR 5.40 requires an age statement on straight whiskey under four
    years old; the cleanest heuristic is "label says 'straight' AND a
    whiskey designator". Case-insensitive; tolerates spacing and
    punctuation variants ("Kentucky Straight Bourbon Whiskey", "STRAIGHT
    RYE WHISKEY").
    """
    if not class_type_value:
        return False
    text = class_type_value.lower()
    if "straight" not in text:
        return False
    return any(word in text for word in _STRAIGHT_WHISKEY_TRIGGER_WORDS)


def check_age_statement(params: dict, ctx: ExtractionContext) -> CheckResult:
    """Validate a spirits age statement (27 CFR 5.40).

    Severity is conditional on the class_type the label declares:

      * Straight whiskey class (e.g. "Kentucky Straight Bourbon
        Whiskey"): REQUIRED — missing or malformed age FAILs.
      * Anything else: ADVISORY — missing is PASS (no requirement);
        malformed value still surfaces as ADVISORY so the producer can
        clean up the wording without failing the report.

    The rule's static severity is `required` so the engine's
    REQUIRED-rule confidence-aware downgrade still fires when the field
    is unreadable. The check decides FAIL vs ADVISORY directly based on
    what the label actually says.
    """
    field = params["field"]
    class_type_field = params.get("class_type_field", "class_type")
    pattern = params["pattern"]
    flags = _parse_regex_flags(params.get("flags", ""))

    class_type_value, _ = _get_field_value(ctx, class_type_field)
    age_value, age_bbox = _get_field_value(ctx, field)

    is_required = _is_straight_whiskey_class(class_type_value)

    if age_value is None or not age_value.strip():
        # No age statement on the label.
        if is_required:
            return CheckResult(
                outcome=CheckOutcome.FAIL,
                finding=(
                    "Straight whiskey labels require an age statement under "
                    "27 CFR 5.40 when bottled at less than four years old; "
                    "none was found on this label."
                ),
                expected="An age statement (e.g. 'Aged 4 Years').",
            )
        return CheckResult(outcome=CheckOutcome.PASS)

    if re.search(pattern, age_value.strip(), flags):
        return CheckResult(outcome=CheckOutcome.PASS, bbox=age_bbox)

    if is_required:
        return CheckResult(
            outcome=CheckOutcome.FAIL,
            finding=(
                f"Age statement {age_value.strip()!r} does not match the "
                "required 'Aged <number> Years/Months' format (27 CFR 5.40)."
            ),
            expected="A statement matching pattern 'Aged <number> Years/Months'.",
            bbox=age_bbox,
        )

    return CheckResult(
        outcome=CheckOutcome.ADVISORY,
        finding=(
            f"Age statement {age_value.strip()!r} does not match the "
            "recommended 'Aged <number> Years/Months' format. Format is "
            "advisory only for non-straight-whiskey classes."
        ),
        expected="A statement matching pattern 'Aged <number> Years/Months'.",
        bbox=age_bbox,
    )


CHECK_REGISTRY: dict[str, Callable[[dict, ExtractionContext], CheckResult]] = {
    "presence": check_presence,
    "regex": check_regex,
    "exact_text": check_exact_text,
    "advisory_note": check_advisory_note,
    "cross_reference": check_cross_reference,
    "cross_reference_numeric": check_cross_reference_numeric,
    "cross_reference_volume": check_cross_reference_volume,
    "warning_compliance": check_warning_compliance,
    "age_statement": check_age_statement,
}
