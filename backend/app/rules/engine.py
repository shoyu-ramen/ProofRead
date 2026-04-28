import ast
from typing import Any

from app.rules.checks import CHECK_REGISTRY
from app.rules.types import (
    CheckOutcome,
    ExtractionContext,
    Rule,
    RuleResult,
    Severity,
    worse,
)

# Whitelist of names available to applies_if / exempt_if expressions.
# Keep narrow: just the context attributes that are sensible to gate on.
_EXPR_ALLOWED_KEYS = {
    "is_imported",
    "abv_pct",
    "container_size_ml",
    "beverage_type",
}


# Field-level confidence below this threshold triggers ADVISORY downgrade.
# Matches the Claude vision extractor default. SPEC §0.5: "every check has an
# explicit confidence threshold below which it downgrades from required to
# advisory, with the reason surfaced to the user."
LOW_CONFIDENCE_THRESHOLD = 0.6


class RuleEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = rules

    def evaluate(self, ctx: ExtractionContext) -> list[RuleResult]:
        results: list[RuleResult] = []
        for rule in self.rules:
            if ctx.beverage_type not in rule.beverage_types:
                continue
            if not self._applies(rule, ctx):
                results.append(
                    RuleResult(
                        rule_id=rule.id,
                        rule_version=rule.version,
                        citation=rule.citation,
                        status=CheckOutcome.NA,
                        finding="Rule did not apply to this scan",
                        fix_suggestion=rule.fix_suggestion,
                    )
                )
                continue
            results.append(self._evaluate_rule(rule, ctx))
        return results

    def _applies(self, rule: Rule, ctx: ExtractionContext) -> bool:
        if rule.exempt_if and self._eval_expr(rule.exempt_if, ctx):
            return False
        if rule.applies_if and not self._eval_expr(rule.applies_if, ctx):
            return False
        return True

    def _eval_expr(self, expr: str, ctx: ExtractionContext) -> bool:
        env = {k: getattr(ctx, k, None) for k in _EXPR_ALLOWED_KEYS}
        try:
            return bool(_safe_eval(expr, env))
        except Exception:
            return False

    def _evaluate_rule(self, rule: Rule, ctx: ExtractionContext) -> RuleResult:
        # Confidence-aware degradation (SPEC §0.5): if any field this rule
        # depends on was marked unreadable by the extractor, OR has confidence
        # below the threshold, downgrade required rules to ADVISORY rather
        # than guessing pass/fail. Advisory rules already short-circuit.
        referenced_fields = _referenced_fields(rule)
        unreliable = _unreliable_fields(referenced_fields, ctx)

        if rule.severity == Severity.REQUIRED and unreliable:
            joined = ", ".join(sorted(unreliable))
            return RuleResult(
                rule_id=rule.id,
                rule_version=rule.version,
                citation=rule.citation,
                status=CheckOutcome.ADVISORY,
                finding=(
                    f"Couldn't verify with confidence — image quality "
                    f"prevented reliable reading of: {joined}. "
                    f"Rescan recommended."
                ),
                fix_suggestion=rule.fix_suggestion,
            )

        worst = CheckOutcome.PASS
        finding: str | None = None
        expected: str | None = None
        bbox = None

        for check in rule.checks:
            check_fn = CHECK_REGISTRY.get(check.type)
            if check_fn is None:
                if worse(worst, CheckOutcome.ADVISORY) == CheckOutcome.ADVISORY:
                    worst = CheckOutcome.ADVISORY
                    finding = f"Check type '{check.type}' is not implemented"
                continue

            result = check_fn(check.params, ctx)
            new_worst = worse(worst, result.outcome)
            if new_worst != worst:
                worst = new_worst
                finding = result.finding
                expected = result.expected
                bbox = result.bbox
            if worst == CheckOutcome.FAIL:
                break

        # Advisory rules never fail outright — failures degrade to advisory.
        if rule.severity == Severity.ADVISORY and worst == CheckOutcome.FAIL:
            worst = CheckOutcome.ADVISORY

        return RuleResult(
            rule_id=rule.id,
            rule_version=rule.version,
            citation=rule.citation,
            status=worst,
            finding=finding,
            expected=expected,
            fix_suggestion=rule.fix_suggestion,
            bbox=bbox,
        )


def _referenced_fields(rule: Rule) -> set[str]:
    out: set[str] = set()
    for check in rule.checks:
        f = check.params.get("field")
        if isinstance(f, str):
            out.add(f)
    return out


def _unreliable_fields(referenced: set[str], ctx: ExtractionContext) -> set[str]:
    """A field is unreliable if the extractor marked it unreadable OR its
    confidence is below the threshold. Fields that are simply missing from
    `ctx.fields` are NOT unreliable — presence checks should produce FAIL
    with the proper reason."""
    if not referenced:
        return set()
    unreadable = set(ctx.unreadable_fields) & referenced
    low_conf = {
        name
        for name in referenced
        if (f := ctx.fields.get(name)) is not None
        and f.confidence < LOW_CONFIDENCE_THRESHOLD
    }
    return unreadable | low_conf


# ---------------------------------------------------------------------------
# Safe expression evaluator for `applies_if` / `exempt_if` rule guards.
#
# Walks a parsed AST and only executes a fixed whitelist of node types
# (literals, names from the supplied env, comparisons, boolean ops, unary
# +/-/not). This blocks the lambda/subclass-walk escapes that defeat
# eval(..., {"__builtins__": {}}, env). Anything outside the whitelist
# raises and the caller treats it as a non-applicable rule.
# ---------------------------------------------------------------------------

_CMP_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _safe_eval(expr: str, env: dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    return _walk(tree.body, env)


def _walk(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown identifier {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        v = _walk(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not _walk(v, env):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if _walk(v, env):
                    return True
            return False
        raise ValueError(f"unsupported bool op: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left = _walk(node.left, env)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _walk(comparator, env)
            cmp = _CMP_OPS.get(type(op))
            if cmp is None:
                raise ValueError(f"unsupported comparison: {type(op).__name__}")
            if not cmp(left, right):
                return False
            left = right
        return True
    raise ValueError(f"unsupported node: {type(node).__name__}")
