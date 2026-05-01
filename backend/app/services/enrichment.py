"""Post-verify enrichment: TTB COLA lookup + AI per-rule explanations.

The /v1/verify orchestrator (and, in time, /v1/scans finalize) calls
this after the rule engine has produced a verdict. Runs the external
lookup and the explanation generation concurrently, persists everything
into the L3 durable cache, and returns the enriched payload for the
response.

Pure-additive contract: any failure here (TTB outage, Anthropic
timeout, DB hiccup) silently drops the corresponding field. The verdict
is never gated on enrichment — the user still gets their pass/fail in
the cold-path response time, plus whatever enrichment succeeded.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.rules.types import CheckOutcome, RuleResult
from app.services.explanation import RuleExplanationInput, explain_rules
from app.services.external import ExternalMatch, get_adapter
from app.services.persisted_cache import PersistedHit, PersistedLabelCache
from app.services.verify import VerifyReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichmentResult:
    """Output of enrich_verdict; consumed by the API layer to fill the
    response and to overwrite the L1 cache slot under the same key.
    """

    external_match: dict[str, Any] | None
    # Mapping of rule_id -> one-sentence explanation, scoped to the
    # rules that failed on THIS verdict. Stale entries from a prior
    # cached run (rules that no longer fail) are filtered out so the UI
    # never renders an explanation for a passing rule.
    explanations: dict[str, str] | None
    # The L3 row id we wrote to (or read from). The API layer doesn't
    # use this directly today; carried through so future enrichers can
    # update the same row without a fresh perceptual lookup.
    persisted_entry_id: uuid.UUID | None


async def enrich_verdict(
    *,
    report: VerifyReport,
    beverage_type: str,
    container_size_ml: int,
    is_imported: bool,
    persisted_cache: PersistedLabelCache | None,
    persisted_hit: PersistedHit | None,
    signature: tuple[int, ...] | None,
    first_frame_signature_hex: str | None = None,
) -> EnrichmentResult:
    """Run TTB COLA lookup + explanation generation concurrently, persist
    to L3, return the enriched payload.

    Reuse rules:
      * `persisted_hit` carries cached external_match + cached
        explanations from L3. We reuse them and only generate fresh
        data for what's missing.
      * `report.image_quality == "unreadable"` short-circuits — the
        verdict already refused the label and we don't have a brand to
        look up.
      * The AI explanation step is per-rule-id, so a label that fails
        a different set of rules across runs (e.g. different container
        size) accumulates explanations into the same L3 row over time.

    Persistence is best-effort: any DB exception is swallowed and
    logged; the response still goes out with whatever enrichment we
    managed to compute.
    """
    if report.image_quality == "unreadable":
        return EnrichmentResult(None, None, None)

    cached_external = persisted_hit.external_match if persisted_hit else None
    cached_explanations = persisted_hit.explanations if persisted_hit else None
    persisted_entry_id = persisted_hit.entry_id if persisted_hit else None

    # Compare against the enum members rather than raw strings: the
    # str-subclass `==` works today, but a future non-`str` enum refactor
    # would silently turn this into a no-op.
    failed_rules: list[RuleResult] = [
        r
        for r in report.rule_results
        if r.status in (CheckOutcome.FAIL, CheckOutcome.ADVISORY)
    ]
    failed_rule_ids = {r.rule_id for r in failed_rules}

    needs_external = (
        cached_external is None
        and settings.ttb_cola_lookup_enabled
        and _extract_brand(report) is not None
    )
    missing_explanation_ids = failed_rule_ids - set(
        (cached_explanations or {}).keys()
    )
    needs_explanations = (
        settings.explanation_enabled and bool(missing_explanation_ids)
    )

    external_task: asyncio.Task[ExternalMatch | None] | None = None
    explanation_task: asyncio.Task[dict[str, str]] | None = None
    if needs_external:
        external_task = asyncio.create_task(
            _run_external_lookup(report, beverage_type)
        )
    if needs_explanations:
        rules_to_explain = [
            r for r in failed_rules if r.rule_id in missing_explanation_ids
        ]
        explanation_inputs = _build_explanation_inputs(
            rules_to_explain, report.extracted
        )
        explanation_task = asyncio.create_task(
            explain_rules(
                explanation_inputs,
                beverage_type=beverage_type,
                container_size_ml=container_size_ml,
                is_imported=is_imported,
                image_quality=report.image_quality,
                timeout_s=settings.explanation_timeout_s,
            )
        )

    new_external_match: ExternalMatch | None = None
    if external_task is not None:
        try:
            new_external_match = await external_task
        except Exception as exc:  # pragma: no cover — adapter swallows by spec
            logger.warning("external lookup raised unexpectedly: %s", exc)

    fresh_explanations: dict[str, str] = {}
    if explanation_task is not None:
        try:
            fresh_explanations = await explanation_task
        except Exception as exc:  # pragma: no cover — service swallows by spec
            logger.warning("explanation generation raised unexpectedly: %s", exc)

    # Merge cached + fresh, then narrow to currently-failing rule_ids so
    # the response never carries stale explanations for rules that pass
    # under the current container size / claim / rule version.
    merged: dict[str, str] = dict(cached_explanations or {})
    merged.update(fresh_explanations)
    response_explanations: dict[str, str] | None = (
        {rid: txt for rid, txt in merged.items() if rid in failed_rule_ids}
        or None
    )

    final_external_match: dict[str, Any] | None = (
        cached_external
        if cached_external is not None
        else (new_external_match.to_dict() if new_external_match else None)
    )

    # L3 write-through. Best-effort; the response goes out either way.
    if (
        persisted_cache is not None
        and signature is not None
        and report.raw_extraction is not None
    ):
        try:
            if persisted_entry_id is None:
                persisted_entry_id = await persisted_cache.upsert(
                    signature=signature,
                    beverage_type=beverage_type,
                    extraction=report.raw_extraction,
                )
            if final_external_match is not None and cached_external is None:
                await persisted_cache.update_external_match(
                    persisted_entry_id, final_external_match
                )
            if fresh_explanations:
                # Persist the FULL merged set (including rules not failing
                # on this run) so a later scan with a different container
                # size finds explanations for those other rules too.
                durable_set = dict(cached_explanations or {})
                durable_set.update(fresh_explanations)
                await persisted_cache.update_explanations(
                    persisted_entry_id, durable_set
                )
            # Idempotent post-upsert stamps for the known-label feature.
            # Both writes are guarded by `IS NULL` inside the cache
            # methods so a second scan never overwrites a value already
            # there. Brand stamp is also a no-op when ``upsert`` already
            # filled it from the same extraction; we issue it
            # unconditionally for the case where the row pre-existed
            # this column (rows from an older deploy).
            brand = _extract_brand(report)
            if brand is not None:
                await persisted_cache.stamp_brand_name_normalized(
                    persisted_entry_id, brand
                )
            if first_frame_signature_hex:
                await persisted_cache.stamp_first_frame_signature(
                    persisted_entry_id, first_frame_signature_hex
                )
        except Exception as exc:
            logger.warning("L3 persistence skipped: %s", exc)

    return EnrichmentResult(
        external_match=final_external_match,
        explanations=response_explanations,
        persisted_entry_id=persisted_entry_id,
    )


async def _run_external_lookup(
    report: VerifyReport,
    beverage_type: str,
) -> ExternalMatch | None:
    adapter = get_adapter("ttb_cola")
    if adapter is None:
        return None
    brand = _extract_brand(report)
    if brand is None:
        return None
    return await adapter.lookup(
        brand=brand,
        beverage_type=beverage_type,
        fanciful_name=_extract_fanciful(report),
        timeout_s=settings.ttb_cola_timeout_s,
    )


def _extract_brand(report: VerifyReport) -> str | None:
    """Pick the brand string out of the extraction summary.

    Field name varies across the rule sets (`brand_name` is the
    primary, `brand` is the legacy alias). Caller can tolerate None.
    """
    for key in ("brand_name", "brand"):
        info = report.extracted.get(key)
        if isinstance(info, dict):
            v = info.get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _extract_fanciful(report: VerifyReport) -> str | None:
    info = report.extracted.get("fanciful_name")
    if isinstance(info, dict):
        v = info.get("value")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _build_explanation_inputs(
    rules: list[RuleResult],
    extracted: dict[str, dict[str, Any]],
) -> list[RuleExplanationInput]:
    """Convert rule_results into the explanation service's input shape.

    Attach every primitive-valued extracted field so the model has
    enough context to ground its sentence in THIS label's actual values
    rather than generic regulatory text.
    """
    field_values: dict[str, str | None] = {}
    for key, info in extracted.items():
        if not isinstance(info, dict):
            continue
        v = info.get("value")
        if v is None:
            field_values[key] = None
        elif isinstance(v, (str, int, float, bool)):
            field_values[key] = str(v)
        # Drop non-primitive values (e.g. nested dicts) — they don't
        # belong in a one-sentence prompt.

    return [
        RuleExplanationInput(
            rule_id=r.rule_id,
            rule_status=r.status if isinstance(r.status, str) else r.status.value,
            citation=r.citation,
            finding=r.finding,
            expected=r.expected,
            fix_suggestion=r.fix_suggestion,
            field_values=dict(field_values),
        )
        for r in rules
    ]
