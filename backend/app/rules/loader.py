from functools import lru_cache
from pathlib import Path

import yaml

from app.config import settings
from app.rules.types import Check, Rule, Severity


@lru_cache(maxsize=8)
def _load_all_rules() -> tuple[Rule, ...]:
    base = Path(settings.rule_definitions_path)
    rules: list[Rule] = []
    for yml in sorted(base.glob("*.yaml")):
        with open(yml, encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        for entry in data:
            checks = [
                Check(type=c["type"], params=c.get("params", {}))
                for c in entry.get("checks", [])
            ]
            rules.append(
                Rule(
                    id=entry["id"],
                    version=int(entry.get("version", 1)),
                    beverage_types=entry.get("beverage_types", []),
                    citation=entry["citation"],
                    description=entry["description"],
                    severity=Severity(entry.get("severity", "required")),
                    checks=checks,
                    fix_suggestion=entry.get("fix_suggestion"),
                    applies_if=entry.get("applies_if"),
                    exempt_if=entry.get("exempt_if"),
                )
            )
    return tuple(rules)


def load_rules(beverage_type: str | None = None) -> list[Rule]:
    rules = list(_load_all_rules())
    if beverage_type is not None:
        rules = [r for r in rules if beverage_type in r.beverage_types]
    return rules


def reset_cache() -> None:
    _load_all_rules.cache_clear()
