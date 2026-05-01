"""Unit tests for the post-verify enrichment orchestrator.

Pure logic coverage — the underlying explanation service and TTB COLA
adapter have their own tests. These tests pin the merge / persist /
filter rules that the orchestrator layers on top.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.rules.types import CheckOutcome, RuleResult
from app.services.enrichment import enrich_verdict
from app.services.external.types import ExternalMatch
from app.services.persisted_cache import PersistedHit
from app.services.verify import VerifyReport


def _rule(rule_id: str, status: CheckOutcome, citation: str = "27 CFR 7.22") -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        rule_version=1,
        citation=citation,
        status=status,
        finding="finding text",
        expected="expected text",
        fix_suggestion="static fix",
        bbox=None,
        surface="panorama",
    )


def _report(
    *,
    rule_results: list[RuleResult],
    image_quality: str = "good",
    extracted: dict | None = None,
) -> VerifyReport:
    return VerifyReport(
        overall="fail",
        rule_results=rule_results,
        extracted=extracted
        or {"brand_name": {"value": "ANYTOWN ALE", "confidence": 0.9}},
        unreadable_fields=[],
        image_quality=image_quality,
        image_quality_notes=None,
        elapsed_ms=42,
    )


@pytest.mark.asyncio
async def test_enrich_skips_unreadable_image_quality():
    """An unreadable scan never has reliable extraction, so we don't
    burn tokens on explanations or pollute L3 with a junk row."""
    report = _report(
        rule_results=[_rule("beer.health_warning.exact_text", CheckOutcome.FAIL)],
        image_quality="unreadable",
    )

    result = await enrich_verdict(
        report=report,
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        persisted_cache=None,
        persisted_hit=None,
        signature=None,
    )

    assert result.external_match is None
    assert result.explanations is None
    assert result.persisted_entry_id is None


@pytest.mark.asyncio
async def test_enrich_generates_explanations_for_failed_rules():
    """Cold path with no L3 hit: fresh explanations are generated for
    every failing/advisory rule and returned keyed by rule_id."""
    report = _report(
        rule_results=[
            _rule("beer.brand_name.matches_application", CheckOutcome.PASS),
            _rule("beer.health_warning.exact_text", CheckOutcome.FAIL),
            _rule("beer.alcohol_content.format", CheckOutcome.ADVISORY),
        ],
    )

    fake_explanations = {
        "beer.health_warning.exact_text": "Your warning differs from the statutory text.",
        "beer.alcohol_content.format": "Your label uses 'ABV' but TTB requires 'alc/vol'.",
    }
    with patch(
        "app.services.enrichment.explain_rules",
        AsyncMock(return_value=fake_explanations),
    ):
        result = await enrich_verdict(
            report=report,
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            persisted_cache=None,
            persisted_hit=None,
            signature=None,
        )

    assert result.explanations == fake_explanations
    assert result.external_match is None  # adapter disabled by default


@pytest.mark.asyncio
async def test_enrich_filters_stale_explanations_for_passing_rules():
    """L3 carries explanations for every rule we ever explained for this
    label, including ones that PASS on the current run (different
    container size, claim, etc.). The response must only include
    explanations for currently-failing rule_ids — surfacing a stale
    explanation alongside a green pass would mislead the user."""
    report = _report(
        rule_results=[
            _rule("beer.brand_name.matches_application", CheckOutcome.PASS),
            _rule("beer.health_warning.exact_text", CheckOutcome.FAIL),
        ],
    )

    cached = PersistedHit(
        entry_id="00000000-0000-0000-0000-000000000001",  # ignored in this test
        extraction=None,  # not used since persisted_cache is None below
        external_match=None,
        explanations={
            "beer.brand_name.matches_application": "STALE — was failing earlier",
            "beer.health_warning.exact_text": "Cached explanation for the warning rule",
        },
        min_distance=0,
        signature=(0xABCD,),
    )

    # No fresh generation needed because the failing rule already has a
    # cached explanation.
    with patch(
        "app.services.enrichment.explain_rules",
        AsyncMock(return_value={}),
    ):
        result = await enrich_verdict(
            report=report,
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            persisted_cache=None,
            persisted_hit=cached,
            signature=None,
        )

    assert result.explanations == {
        "beer.health_warning.exact_text": "Cached explanation for the warning rule"
    }
    # The pass rule explanation must be filtered out.
    assert "beer.brand_name.matches_application" not in (result.explanations or {})


@pytest.mark.asyncio
async def test_enrich_only_calls_explain_rules_for_missing_rule_ids():
    """If L3 already has explanations for some failing rules, we only
    pay tokens to generate the missing ones — not the entire set."""
    report = _report(
        rule_results=[
            _rule("rule.a", CheckOutcome.FAIL),
            _rule("rule.b", CheckOutcome.FAIL),
        ],
    )

    cached = PersistedHit(
        entry_id="00000000-0000-0000-0000-000000000001",
        extraction=None,
        external_match=None,
        explanations={"rule.a": "cached"},
        min_distance=0,
        signature=(0,),
    )

    fresh_mock = AsyncMock(return_value={"rule.b": "fresh"})
    with patch("app.services.enrichment.explain_rules", fresh_mock):
        result = await enrich_verdict(
            report=report,
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            persisted_cache=None,
            persisted_hit=cached,
            signature=None,
        )

    # explain_rules called exactly once with only the missing rule.
    assert fresh_mock.call_count == 1
    call_inputs = fresh_mock.call_args.args[0]
    assert [r.rule_id for r in call_inputs] == ["rule.b"]

    assert result.explanations == {"rule.a": "cached", "rule.b": "fresh"}


@pytest.mark.asyncio
async def test_enrich_returns_none_when_no_failed_rules_and_no_cache():
    """A clean pass with explanation_enabled=True still produces nothing
    to explain — return None for explanations rather than an empty dict
    so the response field stays absent."""
    report = _report(
        rule_results=[_rule("rule.a", CheckOutcome.PASS)],
    )

    explain_mock = AsyncMock(return_value={})
    with patch("app.services.enrichment.explain_rules", explain_mock):
        result = await enrich_verdict(
            report=report,
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            persisted_cache=None,
            persisted_hit=None,
            signature=None,
        )

    assert result.explanations is None
    # No failing rules → no API call.
    assert explain_mock.call_count == 0


@pytest.mark.asyncio
async def test_enrich_uses_cached_external_match_when_present():
    """If L3 has the TTB COLA match cached, we use it verbatim and don't
    re-fire the adapter (which would be a network round-trip)."""
    report = _report(
        rule_results=[_rule("rule.a", CheckOutcome.FAIL)],
    )

    cached_match = {"source": "ttb_cola", "source_id": "20-42-001"}
    cached = PersistedHit(
        entry_id="00000000-0000-0000-0000-000000000001",
        extraction=None,
        external_match=cached_match,
        explanations={"rule.a": "cached"},
        min_distance=0,
        signature=(0,),
    )

    # If `_run_external_lookup` got called, this mock would be exercised
    # and we'd see a non-None new match — but we expect it NOT to be
    # called because the cached match short-circuits.
    with (
        patch(
            "app.services.enrichment._run_external_lookup",
            AsyncMock(return_value=ExternalMatch(
                source="ttb_cola",
                source_id="should-not-appear",
                brand=None,
                fanciful_name=None,
                class_type=None,
                approval_date=None,
                label_image_url=None,
                confidence=1.0,
                source_url=None,
            )),
        ) as ext_mock,
        patch("app.services.enrichment.explain_rules", AsyncMock(return_value={})),
    ):
        result = await enrich_verdict(
            report=report,
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            persisted_cache=None,
            persisted_hit=cached,
            signature=None,
        )

    assert result.external_match == cached_match
    assert ext_mock.call_count == 0


@pytest.mark.asyncio
async def test_enrich_swallows_explain_rules_exception():
    """The verdict path is never gated on enrichment. A raise from the
    explanation service must surface as None explanations, not a 500."""
    report = _report(
        rule_results=[_rule("rule.a", CheckOutcome.FAIL)],
    )

    with patch(
        "app.services.enrichment.explain_rules",
        AsyncMock(side_effect=RuntimeError("network down")),
    ):
        result = await enrich_verdict(
            report=report,
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            persisted_cache=None,
            persisted_hit=None,
            signature=None,
        )

    assert result.explanations is None
    assert result.external_match is None


@pytest.mark.asyncio
async def test_enrich_persists_to_l3_on_reverse_lookup_hit():
    """L2 reverse-lookup hits must populate L3 too. Without this, a
    label served by L2 (in-process) would never be persisted across
    process restarts — the L3 row only ever gets created from the
    cold-path success exit.

    Pin: a VerifyReport that came off the reverse-lookup hit branch
    still carries `raw_extraction` (set to `reverse_hit.extraction`),
    `panel_signature`, and `reverse_lookup_hit=True`. The enrichment
    service treats this identically to a cold-path report for L3
    upsert purposes — both have the data needed to write the row.
    """
    from unittest.mock import AsyncMock, MagicMock

    from app.services.vision import VisionExtraction

    # Synthesize a "reverse-lookup hit" report: extraction reused from
    # a prior cold path, panel_signature carried through from that L2
    # entry, raw_extraction set to the reused extraction.
    reused_extraction = VisionExtraction(
        raw_response="reused", fields={}, unreadable=set()
    )
    report = _report(
        rule_results=[_rule("rule.a", CheckOutcome.FAIL)],
    )
    report.reverse_lookup_hit = True
    report.raw_extraction = reused_extraction
    report.panel_signature = (0xCAFE,)

    fake_cache = MagicMock()
    fake_cache.upsert = AsyncMock(return_value="00000000-0000-0000-0000-0000000000aa")
    fake_cache.update_external_match = AsyncMock()
    fake_cache.update_explanations = AsyncMock()

    with patch(
        "app.services.enrichment.explain_rules",
        AsyncMock(return_value={"rule.a": "fresh explanation"}),
    ):
        await enrich_verdict(
            report=report,
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            persisted_cache=fake_cache,
            persisted_hit=None,
            signature=report.panel_signature,
        )

    # The upsert MUST fire on a reverse-hit path so L3 accumulates the
    # corpus rather than silently leaking labels that only ever transit
    # the in-process L2.
    assert fake_cache.upsert.call_count == 1
    call_kwargs = fake_cache.upsert.call_args.kwargs
    assert call_kwargs["signature"] == (0xCAFE,)
    assert call_kwargs["beverage_type"] == "beer"
    assert call_kwargs["extraction"] is reused_extraction
    # And the explanation gets persisted alongside.
    assert fake_cache.update_explanations.call_count == 1


@pytest.mark.asyncio
async def test_enrich_skips_external_when_brand_missing():
    """The TTB COLA adapter needs at least a brand to query. With no
    brand extracted, we skip the lookup entirely rather than firing a
    request that's guaranteed to return None."""
    report = _report(
        rule_results=[_rule("rule.a", CheckOutcome.FAIL)],
        extracted={
            # No brand_name / brand entry. The adapter would have nothing
            # to search on.
            "alcohol_content": {"value": "5.5% ABV", "confidence": 0.85},
        },
    )

    # ttb_cola_lookup_enabled is False by default in tests, but we want
    # to confirm the brand-missing short-circuit is independent of that.
    with (
        patch(
            "app.services.enrichment.settings.ttb_cola_lookup_enabled", True
        ),
        patch(
            "app.services.enrichment._run_external_lookup", AsyncMock(return_value=None)
        ) as ext_mock,
        patch("app.services.enrichment.explain_rules", AsyncMock(return_value={})),
    ):
        await enrich_verdict(
            report=report,
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            persisted_cache=None,
            persisted_hit=None,
            signature=None,
        )

    # _run_external_lookup is never called because the brand pre-check
    # short-circuits in `enrich_verdict` itself.
    assert ext_mock.call_count == 0
