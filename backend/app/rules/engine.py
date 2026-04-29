import ast
import dataclasses
from typing import Any

from app.config import settings
from app.rules.checks import CHECK_REGISTRY
from app.rules.inference import infer_is_imported
from app.rules.types import (
    CheckOutcome,
    ExtractionContext,
    Rule,
    RuleResult,
    Severity,
    worse,
)

# Synthetic rule_id surfaced when the user's `is_imported` claim
# disagrees with what the label says — i.e. the label declares a
# country of origin but the request claimed domestic. Surfaced as an
# ADVISORY so it never fails a report; the country-of-origin rule
# itself is what enforces compliance. Citing SPEC §0.5 because this is
# the fail-honestly path: rather than silently skip the rule on a stale
# claim, we run it and tell the user we did.
_CLAIM_CONSISTENCY_RULE_ID = "claim_consistency.is_imported"
_CLAIM_CONSISTENCY_RULE_VERSION = 1
_CLAIM_CONSISTENCY_CITATION = "SPEC §0.5"

# Whitelist of names available to applies_if / exempt_if expressions.
# Keep narrow: just the context attributes that are sensible to gate on.
_EXPR_ALLOWED_KEYS = {
    "is_imported",
    "abv_pct",
    "container_size_ml",
    "beverage_type",
}


# Field-level confidence below this threshold triggers ADVISORY downgrade.
# Re-export of `settings.low_confidence_threshold` so callers reading the
# value via `from app.rules.engine import LOW_CONFIDENCE_THRESHOLD` keep
# working — the canonical store is the settings object, this is a
# read-side alias.
LOW_CONFIDENCE_THRESHOLD = settings.low_confidence_threshold


class RuleEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = rules

    def evaluate(self, ctx: ExtractionContext) -> list[RuleResult]:
        # Reconcile the user's `is_imported` claim against the label
        # (SPEC §0.5 fail-honestly). If the label declares a country of
        # origin but the request claimed domestic, flip the flag so the
        # country-of-origin rule fires — and emit an advisory so the
        # user sees why. The original claim is left on the caller's
        # context; we evaluate against a shallow clone with the inferred
        # value so nothing else in the pipeline sees a mutated input.
        effective_imported, divergence = infer_is_imported(
            ctx.fields, ctx.is_imported
        )
        if divergence is not None:
            ctx = dataclasses.replace(ctx, is_imported=effective_imported)

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

        if divergence is not None:
            results.append(_claim_consistency_advisory(divergence, ctx.fields))
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
        surface = _surface_for_rule(referenced_fields, ctx)

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
                surface=surface,
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
            surface=surface,
        )


def _claim_consistency_advisory(
    divergence: str,
    fields: dict[str, Any],
) -> RuleResult:
    """Synthetic ADVISORY surfaced when claim/label disagree on `is_imported`.

    `divergence` is the reason code from `infer_is_imported`. Today only
    `"label_indicates_imported"` is emitted; the helper still switches
    on it so future inference cases land here too.
    """
    coo = fields.get("country_of_origin")
    coo_value = coo.value if coo is not None else None
    if divergence == "label_indicates_imported":
        finding = (
            "Label indicates imported"
            + (f" ({coo_value!r})" if coo_value else "")
            + " but request claimed domestic — country-of-origin rule "
            "applied as a precaution. Update the request's "
            "`is_imported` flag to silence this advisory."
        )
    else:
        finding = (
            "Claim and label disagree on import status; rules evaluated "
            "with the label-derived value as a precaution."
        )
    return RuleResult(
        rule_id=_CLAIM_CONSISTENCY_RULE_ID,
        rule_version=_CLAIM_CONSISTENCY_RULE_VERSION,
        citation=_CLAIM_CONSISTENCY_CITATION,
        status=CheckOutcome.ADVISORY,
        finding=finding,
        expected=None,
        fix_suggestion=(
            "If the label is imported, set `is_imported=true` on the "
            "request. If the country statement on the label is incorrect, "
            "remove it before resubmitting."
        ),
        bbox=coo.bbox if coo is not None else None,
        surface=coo.source_image_id if coo is not None else None,
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
    threshold = settings.low_confidence_threshold
    unreadable = set(ctx.unreadable_fields) & referenced
    low_conf = {
        name
        for name in referenced
        if (f := ctx.fields.get(name)) is not None
        and f.confidence < threshold
    }
    return unreadable | low_conf


def _surface_for_rule(
    referenced: set[str], ctx: ExtractionContext
) -> str | None:
    """Pick the surface to report on a rule_result.

    Mobile uses this to know which captured image to highlight. Picks the
    first referenced field that was extracted with a `source_image_id`
    (sorted for determinism). Returns `None` when the rule isn't tied to
    a specific field, when the field wasn't extracted, or when the
    extractor didn't carry a source_image_id (verify path's seven-field
    extractor only knows "front" today)."""
    if not referenced:
        return None
    for name in sorted(referenced):
        f = ctx.fields.get(name)
        if f is not None and f.source_image_id:
            return f.source_image_id
    return None


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
