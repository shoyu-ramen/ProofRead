"""Direct rule-engine tests — the riskiest piece in v1."""

import pytest

from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import CheckOutcome, ExtractedField, ExtractionContext


def _ctx(fields: dict[str, ExtractedField], **kw) -> ExtractionContext:
    return ExtractionContext(
        fields=fields,
        beverage_type=kw.get("beverage_type", "beer"),
        container_size_ml=kw.get("container_size_ml", 355),
        is_imported=kw.get("is_imported", False),
        abv_pct=kw.get("abv_pct", None),
    )


def _rule(rule_id: str):
    return next(r for r in load_rules("beer") if r.id == rule_id)


def test_loads_all_eight_v1_rules():
    rule_ids = {r.id for r in load_rules("beer")}
    expected = {
        "beer.brand_name.presence",
        "beer.class_type.presence",
        "beer.alcohol_content.format",
        "beer.net_contents.presence",
        "beer.name_address.presence",
        "beer.country_of_origin.presence_if_imported",
        "beer.health_warning.exact_text",
        "beer.health_warning.size",
    }
    assert expected.issubset(rule_ids), f"Missing: {expected - rule_ids}"


def test_health_warning_passes_on_canonical(canonical_warning):
    engine = RuleEngine([_rule("beer.health_warning.exact_text")])
    ctx = _ctx({"health_warning": ExtractedField(value=canonical_warning)})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.PASS
    assert result.citation == "27 CFR 16.21"


def test_health_warning_fails_on_substituted_character(canonical_warning):
    engine = RuleEngine([_rule("beer.health_warning.exact_text")])
    typo = canonical_warning.replace("Surgeon", "Sergent")
    ctx = _ctx({"health_warning": ExtractedField(value=typo)})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL
    assert result.expected is not None
    assert "edit distance" in (result.finding or "")


def test_health_warning_fails_when_missing():
    engine = RuleEngine([_rule("beer.health_warning.exact_text")])
    ctx = _ctx({})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL
    assert result.expected is not None  # canonical text is shown


def test_health_warning_passes_with_extra_whitespace(canonical_warning):
    """Whitespace normalization should not affect equality."""
    engine = RuleEngine([_rule("beer.health_warning.exact_text")])
    spaced = canonical_warning.replace(" ", "  ").replace("\n", "")
    ctx = _ctx({"health_warning": ExtractedField(value=spaced)})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.PASS


def test_health_warning_size_is_advisory_in_v1():
    """Type-size has no calibration in v1 — must surface as advisory, not fail."""
    engine = RuleEngine([_rule("beer.health_warning.size")])
    ctx = _ctx({})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.ADVISORY


def test_country_of_origin_skipped_when_not_imported():
    engine = RuleEngine([_rule("beer.country_of_origin.presence_if_imported")])
    ctx = _ctx({}, is_imported=False)
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.NA


def test_country_of_origin_required_when_imported():
    engine = RuleEngine([_rule("beer.country_of_origin.presence_if_imported")])
    ctx = _ctx({}, is_imported=True)
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


def test_country_of_origin_passes_when_imported_and_declared():
    engine = RuleEngine([_rule("beer.country_of_origin.presence_if_imported")])
    ctx = _ctx(
        {"country_of_origin": ExtractedField(value="Germany")},
        is_imported=True,
    )
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.PASS


# --- is_imported claim/label divergence inference (SPEC §0.5) ---


_CLAIM_RULE_ID = "claim_consistency.is_imported"


def test_is_imported_inferred_from_label_when_user_claimed_domestic():
    """User claimed domestic but label declares "Product of Germany" — the
    country-of-origin rule must fire (FAIL or PASS based on extraction),
    and an advisory must be emitted to surface the divergence."""
    engine = RuleEngine([_rule("beer.country_of_origin.presence_if_imported")])
    ctx = _ctx(
        {"country_of_origin": ExtractedField(value="Germany")},
        is_imported=False,
    )
    results = engine.evaluate(ctx)
    statuses = {r.rule_id: r.status for r in results}

    # Country rule fired (the field is present, so it passes).
    assert statuses["beer.country_of_origin.presence_if_imported"] == CheckOutcome.PASS
    # Claim-consistency advisory was appended.
    assert _CLAIM_RULE_ID in statuses
    assert statuses[_CLAIM_RULE_ID] == CheckOutcome.ADVISORY
    advisory = next(r for r in results if r.rule_id == _CLAIM_RULE_ID)
    assert "Germany" in (advisory.finding or "")
    assert "claimed domestic" in (advisory.finding or "")


def test_is_imported_unchanged_when_user_claimed_imported():
    """User already said imported; nothing to infer, no advisory."""
    engine = RuleEngine([_rule("beer.country_of_origin.presence_if_imported")])
    ctx = _ctx(
        {"country_of_origin": ExtractedField(value="Germany")},
        is_imported=True,
    )
    results = engine.evaluate(ctx)
    rule_ids = {r.rule_id for r in results}
    assert _CLAIM_RULE_ID not in rule_ids


def test_is_imported_unchanged_when_country_field_empty():
    """No country on the label and user claimed domestic — leave as-is."""
    engine = RuleEngine([_rule("beer.country_of_origin.presence_if_imported")])
    ctx = _ctx({}, is_imported=False)
    results = engine.evaluate(ctx)
    rule_ids = {r.rule_id for r in results}
    assert _CLAIM_RULE_ID not in rule_ids
    # And the country rule itself reports NA (rule did not apply).
    [country] = [r for r in results if r.rule_id.startswith("beer.country_of_origin")]
    assert country.status == CheckOutcome.NA


def test_is_imported_consistent_no_advisory():
    """Claim=imported, label=Germany — consistent, no advisory."""
    engine = RuleEngine([_rule("beer.country_of_origin.presence_if_imported")])
    ctx = _ctx(
        {"country_of_origin": ExtractedField(value="Germany")},
        is_imported=True,
    )
    results = engine.evaluate(ctx)
    advisories = [r for r in results if r.rule_id == _CLAIM_RULE_ID]
    assert advisories == []


def test_is_imported_inference_does_not_mutate_caller_context():
    """The engine must clone the ctx for inference rather than mutating
    the caller's input — `process_scan` and `verify()` build the context
    once and may inspect it after rule evaluation."""
    engine = RuleEngine([_rule("beer.country_of_origin.presence_if_imported")])
    ctx = _ctx(
        {"country_of_origin": ExtractedField(value="Germany")},
        is_imported=False,
    )
    engine.evaluate(ctx)
    assert ctx.is_imported is False, (
        "evaluate() must not mutate the caller's is_imported flag"
    )


def test_is_imported_inference_treats_whitespace_only_country_as_empty():
    """A whitespace-only country value isn't a real declaration; don't flip."""
    engine = RuleEngine([_rule("beer.country_of_origin.presence_if_imported")])
    ctx = _ctx(
        {"country_of_origin": ExtractedField(value="   ")},
        is_imported=False,
    )
    results = engine.evaluate(ctx)
    assert _CLAIM_RULE_ID not in {r.rule_id for r in results}


def test_abv_format_optional_passes_when_absent():
    engine = RuleEngine([_rule("beer.alcohol_content.format")])
    ctx = _ctx({})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.PASS


def test_abv_format_passes_on_varied_ttb_formats():
    engine = RuleEngine([_rule("beer.alcohol_content.format")])
    valid_formats = [
        "4.8% ABV",
        "4.8% ABV.",
        "4.8% ALC./VOL.",
        "4.8% alc/vol",
        "4.8% ALC BY VOL",
        "4.8%",
        "ALC 4.8% BY VOL",
        "ALCOHOL 4.8% BY VOLUME",
    ]
    for fmt in valid_formats:
        ctx = _ctx({"alcohol_content": ExtractedField(value=fmt)})
        [result] = engine.evaluate(ctx)
        assert result.status == CheckOutcome.PASS, f"Failed on: {fmt}"


def test_abv_format_fails_on_garbled_value():
    engine = RuleEngine([_rule("beer.alcohol_content.format")])
    ctx = _ctx({"alcohol_content": ExtractedField(value="five point five")})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


def test_rule_versioning_round_trips():
    engine = RuleEngine([_rule("beer.health_warning.exact_text")])
    ctx = _ctx({})
    [result] = engine.evaluate(ctx)
    assert result.rule_version == 1


# --- safe expression evaluator (replaces eval() in _eval_expr) ---


def test_safe_eval_rejects_lambda_escape():
    """A YAML rule with a lambda-import payload must not execute."""
    engine = RuleEngine([])
    ctx = _ctx({}, is_imported=True)
    payload = "(lambda: __import__('os').system('echo PWNED'))()"
    assert engine._eval_expr(payload, ctx) is False


def test_safe_eval_rejects_subclass_walk():
    """Classic sandbox-escape via object.__subclasses__() must not execute."""
    engine = RuleEngine([])
    ctx = _ctx({})
    payload = "().__class__.__bases__[0].__subclasses__()"
    assert engine._eval_expr(payload, ctx) is False


def test_safe_eval_rejects_attribute_access():
    """No attribute access at all — even on whitelisted names."""
    engine = RuleEngine([])
    ctx = _ctx({}, is_imported=True)
    assert engine._eval_expr("is_imported.__class__", ctx) is False


def test_safe_eval_rejects_function_calls():
    engine = RuleEngine([])
    ctx = _ctx({})
    assert engine._eval_expr("print('hi')", ctx) is False
    assert engine._eval_expr("open('/etc/passwd')", ctx) is False


def test_safe_eval_rejects_unknown_identifier():
    engine = RuleEngine([])
    ctx = _ctx({})
    assert engine._eval_expr("definitely_not_a_real_var", ctx) is False


def test_safe_eval_allows_simple_comparisons():
    engine = RuleEngine([])
    ctx = _ctx(
        {},
        is_imported=True,
        abv_pct=5.5,
        container_size_ml=355,
    )
    assert engine._eval_expr("is_imported == True", ctx) is True
    assert engine._eval_expr("is_imported", ctx) is True
    assert engine._eval_expr("not is_imported", ctx) is False
    assert engine._eval_expr("abv_pct > 5.0", ctx) is True
    assert engine._eval_expr("abv_pct > 10.0", ctx) is False
    assert engine._eval_expr("container_size_ml >= 355", ctx) is True
    assert engine._eval_expr("beverage_type == 'beer'", ctx) is True
    assert engine._eval_expr("is_imported and abv_pct > 5", ctx) is True
    assert engine._eval_expr("is_imported or abv_pct > 100", ctx) is True


def test_safe_eval_chained_comparisons():
    engine = RuleEngine([])
    ctx = _ctx({}, abv_pct=5.5)
    assert engine._eval_expr("0 < abv_pct < 10", ctx) is True
    assert engine._eval_expr("0 < abv_pct < 5", ctx) is False


def test_canonical_health_warning_file_matches_fixture(canonical_warning):
    """Guard against drift between the production canonical text file and the
    test fixture. If they diverge, the rule engine and tests would each be
    correct against their own truth source while production fails on real
    user labels.
    """
    from app.rules.canonical import load_canonical
    from app.rules.checks import _normalize

    file_text = _normalize(load_canonical("health_warning"))
    fixture_text = _normalize(canonical_warning)
    assert file_text == fixture_text, (
        "canonical/health_warning.txt has drifted from the test fixture. "
        "Either the file or the fixture needs updating to match."
    )


# --- beer.net_contents.presence — format check ---


def test_net_contents_passes_with_recognised_units():
    engine = RuleEngine([_rule("beer.net_contents.presence")])
    for value in ("12 FL OZ", "355 mL", "12 fl. oz", "1 L", "16 fluid ounces"):
        ctx = _ctx({"net_contents": ExtractedField(value=value)})
        [result] = engine.evaluate(ctx)
        assert result.status == CheckOutcome.PASS, f"Expected PASS for {value!r}: {result.finding}"


def test_net_contents_fails_on_unitless_number():
    """A bare '12' is unitless and not a TTB-recognised statement."""
    engine = RuleEngine([_rule("beer.net_contents.presence")])
    ctx = _ctx({"net_contents": ExtractedField(value="12")})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


def test_net_contents_fails_on_spelled_out_number():
    """'twelve fluid ounces' has the right unit but no digits."""
    engine = RuleEngine([_rule("beer.net_contents.presence")])
    ctx = _ctx({"net_contents": ExtractedField(value="twelve fluid ounces")})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


def test_net_contents_still_fails_when_missing():
    """Original presence behavior preserved."""
    engine = RuleEngine([_rule("beer.net_contents.presence")])
    ctx = _ctx({})
    [result] = engine.evaluate(ctx)
    assert result.status == CheckOutcome.FAIL


# --- loader fail-fast behavior ---


def test_loader_rejects_unknown_check_type(tmp_path, monkeypatch):
    from app.config import settings
    from app.rules.loader import RuleDefinitionError, reset_cache

    bad_yaml = tmp_path / "bogus.yaml"
    bad_yaml.write_text(
        "- id: beer.bogus.rule\n"
        "  version: 1\n"
        "  beverage_types: [beer]\n"
        "  citation: 27 CFR 7.99\n"
        "  description: Bogus rule with an unknown check type\n"
        "  severity: required\n"
        "  checks:\n"
        "    - type: this_check_type_does_not_exist\n"
        "      params:\n"
        "        field: brand_name\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "rule_definitions_path", str(tmp_path))
    reset_cache()
    try:
        with pytest.raises(RuleDefinitionError) as exc_info:
            load_rules()
        assert "this_check_type_does_not_exist" in str(exc_info.value)
        assert "beer.bogus.rule" in str(exc_info.value)
    finally:
        reset_cache()


def test_loader_rejects_malformed_applies_if(tmp_path, monkeypatch):
    from app.config import settings
    from app.rules.loader import RuleDefinitionError, reset_cache

    bad_yaml = tmp_path / "bad_expr.yaml"
    bad_yaml.write_text(
        "- id: beer.bad_expr.rule\n"
        "  version: 1\n"
        "  beverage_types: [beer]\n"
        "  citation: 27 CFR 7.99\n"
        "  description: Has a syntactically broken applies_if\n"
        "  severity: required\n"
        "  applies_if: 'is_imported ===='\n"
        "  checks:\n"
        "    - type: presence\n"
        "      params:\n"
        "        field: brand_name\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "rule_definitions_path", str(tmp_path))
    reset_cache()
    try:
        with pytest.raises(RuleDefinitionError) as exc_info:
            load_rules()
        assert "applies_if" in str(exc_info.value)
        assert "beer.bad_expr.rule" in str(exc_info.value)
    finally:
        reset_cache()


def test_loader_rejects_duplicate_rule_id(tmp_path, monkeypatch):
    from app.config import settings
    from app.rules.loader import RuleDefinitionError, reset_cache

    dup_yaml = tmp_path / "dup.yaml"
    dup_yaml.write_text(
        "- id: beer.dup.rule\n"
        "  version: 1\n"
        "  beverage_types: [beer]\n"
        "  citation: 27 CFR 7.99\n"
        "  description: First definition\n"
        "  severity: required\n"
        "  checks:\n"
        "    - type: presence\n"
        "      params:\n"
        "        field: brand_name\n"
        "- id: beer.dup.rule\n"
        "  version: 2\n"
        "  beverage_types: [beer]\n"
        "  citation: 27 CFR 7.99\n"
        "  description: Duplicate id (should fail)\n"
        "  severity: required\n"
        "  checks:\n"
        "    - type: presence\n"
        "      params:\n"
        "        field: brand_name\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "rule_definitions_path", str(tmp_path))
    reset_cache()
    try:
        with pytest.raises(RuleDefinitionError) as exc_info:
            load_rules()
        assert "duplicate" in str(exc_info.value).lower()
        assert "beer.dup.rule" in str(exc_info.value)
    finally:
        reset_cache()


def test_loader_accepts_valid_applies_if(tmp_path, monkeypatch):
    """Sanity check: a well-formed expression loads cleanly."""
    from app.config import settings
    from app.rules.loader import reset_cache

    good_yaml = tmp_path / "good.yaml"
    good_yaml.write_text(
        "- id: beer.good.rule\n"
        "  version: 1\n"
        "  beverage_types: [beer]\n"
        "  citation: 27 CFR 7.99\n"
        "  description: Has a valid applies_if\n"
        "  severity: required\n"
        "  applies_if: 'is_imported == True'\n"
        "  checks:\n"
        "    - type: presence\n"
        "      params:\n"
        "        field: brand_name\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "rule_definitions_path", str(tmp_path))
    reset_cache()
    try:
        rules = load_rules()
        assert any(r.id == "beer.good.rule" for r in rules)
    finally:
        reset_cache()
