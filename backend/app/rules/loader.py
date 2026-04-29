import ast
from functools import lru_cache
from pathlib import Path

import yaml

from app.config import settings
from app.rules.checks import CHECK_REGISTRY
from app.rules.types import Check, Rule, Severity


class RuleDefinitionError(ValueError):
    """Raised when a YAML rule definition fails load-time validation.

    Surfacing failures here (rather than at evaluation time) means a typo'd
    check type, expression, or duplicate rule_id breaks the boot — not the
    first scan to hit the affected rule. SPEC §0.5 fail-honestly applies to
    our own configuration, not just the model's output.
    """


@lru_cache(maxsize=8)
def _load_all_rules() -> tuple[Rule, ...]:
    base = Path(settings.rule_definitions_path)
    rules: list[Rule] = []
    seen_ids: set[str] = set()
    for yml in sorted(base.glob("*.yaml")):
        with open(yml, encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        for entry in data:
            rule = _build_rule(entry, source=yml.name)
            if rule.id in seen_ids:
                raise RuleDefinitionError(
                    f"duplicate rule_id {rule.id!r} (second occurrence in "
                    f"{yml.name}); rule_ids must be unique across the rule set."
                )
            seen_ids.add(rule.id)
            rules.append(rule)
    return tuple(rules)


def _build_rule(entry: dict, *, source: str) -> Rule:
    rule_id = entry.get("id") or "<unknown>"

    checks: list[Check] = []
    for c in entry.get("checks", []):
        check_type = c.get("type")
        if check_type not in CHECK_REGISTRY:
            known = ", ".join(sorted(CHECK_REGISTRY))
            raise RuleDefinitionError(
                f"{source}: rule {rule_id!r} references unknown check "
                f"type {check_type!r}. Known check types: {known}."
            )
        checks.append(Check(type=check_type, params=c.get("params", {})))

    applies_if = entry.get("applies_if")
    exempt_if = entry.get("exempt_if")
    for label, expr in (("applies_if", applies_if), ("exempt_if", exempt_if)):
        if expr is None:
            continue
        try:
            ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise RuleDefinitionError(
                f"{source}: rule {rule_id!r} {label} expression "
                f"{expr!r} is not valid Python: {exc.msg}."
            ) from exc

    return Rule(
        id=entry["id"],
        version=int(entry.get("version", 1)),
        beverage_types=entry.get("beverage_types", []),
        citation=entry["citation"],
        description=entry["description"],
        severity=Severity(entry.get("severity", "required")),
        checks=checks,
        fix_suggestion=entry.get("fix_suggestion"),
        applies_if=applies_if,
        exempt_if=exempt_if,
    )


def load_rules(beverage_type: str | None = None) -> list[Rule]:
    rules = list(_load_all_rules())
    if beverage_type is not None:
        rules = [r for r in rules if beverage_type in r.beverage_types]
    return rules


def reset_cache() -> None:
    _load_all_rules.cache_clear()
