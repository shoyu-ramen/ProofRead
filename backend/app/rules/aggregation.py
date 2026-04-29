"""Single source of truth for the per-scan / per-verify "overall" verdict.

Both the multi-image scan pipeline (`services/pipeline.py`) and the
single-shot verify path (`services/verify.py`) used to maintain their
own roll-up function. They diverged in one specific way:

  * `pipeline.overall_status` returned `"pass"` whenever there were no
    fails / warns / advisories — even when the rule list was empty
    (zero rules ran), which it never is on the production paths but
    IS true in some unit-test fixtures.

  * `verify._aggregate_overall` returned `"na"` (the literal string)
    in the empty-rule-list case via the `CheckOutcome.NA` default of
    its `worst` accumulator.

That divergence means the verify response DTO and the scan response
DTO can disagree on a label that produces no results — confusing for
the mobile/web client, which special-cases "pass" / "fail" / "advisory"
and now has to also handle "na" only on one path.

Resolution: prefer the pipeline's interpretation. When no rule fired
fail / warn / advisory, the user-facing meaning is "nothing to flag",
which the API contract calls `"pass"`. Both paths now route through
`overall_status()` below.

The `image_quality == "unreadable"` short-circuit is preserved on both
paths: when the frame is unreadable, the rule list reflects nothing
about the actual label, so any aggregate verdict is a guess. We surface
`"unreadable"` instead.
"""

from __future__ import annotations

from app.rules.types import CheckOutcome, RuleResult


def overall_status(
    results: list[RuleResult],
    *,
    image_quality: str = "good",
    unreadable_fields: list[str] | None = None,
) -> str:
    """Roll per-rule results up to a single user-facing verdict.

    Precedence:

      1. `image_quality == "unreadable"` — the rule list does not
         reflect the label; the only honest verdict is "unreadable".
      2. Empty rule list AND any unreadable fields — same logic as #1.
         The verify path can land here when the foreign-language guard
         or sensor short-circuit fires before the rule engine runs.
      3. Otherwise: FAIL > WARN > ADVISORY > PASS. Empty rule list with
         no unreadable fields rolls up to "pass" (nothing to flag).

    `unreadable_fields` is optional — None and `[]` are equivalent;
    only the verify path's short-circuit branches care about it.
    """
    if image_quality == "unreadable":
        return "unreadable"
    if not results and unreadable_fields:
        return "unreadable"
    if any(r.status == CheckOutcome.FAIL for r in results):
        return "fail"
    if any(r.status == CheckOutcome.WARN for r in results):
        return "warn"
    if any(r.status == CheckOutcome.ADVISORY for r in results):
        return "advisory"
    return "pass"
