"""Single-shot label verification orchestrator.

The OCR-based pipeline (`pipeline.py`) is multi-image and beverage-type-aware
for the v1 scan flow. `verify` is the lighter path the agent UI uses: one
image, one VLM call, one rule-engine evaluation, one verdict — all returned
synchronously so it fits in the ≤5 s budget Sarah called for.

SPEC §0.5 fail-honestly guarantees apply here too:

  1. The image goes through `sensor_check.assess_capture_quality()` first.
     A frame the pre-check flags as unreadable never reaches the VLM —
     we return `overall="unreadable"` with each rule downgraded to
     ADVISORY so the user is told to rescan rather than served a guess.
  2. Even when the pre-check passes, its verdict is merged with the
     extractor's verdict pessimistically. Whichever is worse wins: if the
     sensor sees obvious motion blur but the model says "good", we surface
     "degraded" so a confident wrong-pass is impossible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import (
    CheckOutcome,
    ExtractionContext,
    RuleResult,
    worse,
)
from app.services.adversarial import (
    detect_foreign_language,
    merge_signals,
    screenshot_signal_from_source,
)
from app.services.health_warning_second_pass import (
    CrossCheckResult,
    HealthWarningExtractor,
    cross_check,
)
from app.services.sensor_check import (
    CaptureQualityReport,
    SurfaceCaptureQuality,
    assess_capture_quality,
)
from app.services.vision import VisionExtractor

# Field-level confidence cap (matches the rule engine threshold). Once a
# field has been pulled out of a degraded surface, its model-supplied
# confidence cannot exceed the surface confidence — the model can't be
# more confident in the text than the frame allows.
_QUALITY_RANK = {"good": 0, "degraded": 1, "unreadable": 2}


@dataclass
class VerifyInput:
    image_bytes: bytes
    media_type: str
    beverage_type: str
    container_size_ml: int
    is_imported: bool
    application: dict[str, Any]


@dataclass
class VerifyReport:
    overall: str  # "pass" | "warn" | "fail" | "advisory" | "unreadable" | "na"
    rule_results: list[RuleResult]
    extracted: dict[str, dict[str, Any]] = field(default_factory=dict)
    unreadable_fields: list[str] = field(default_factory=list)
    image_quality: str = "good"
    image_quality_notes: str | None = None
    health_warning_cross_check: dict[str, Any] | None = None
    elapsed_ms: int = 0


def _aggregate_overall(
    results: list[RuleResult],
    unreadable_fields: list[str],
    image_quality: str,
) -> str:
    """Roll per-rule results up to a single user-facing verdict.

    "unreadable" wins over everything: when the image is unreadable, the
    rule engine's results don't reflect the actual label and any pass/fail
    is a guess. Otherwise: FAIL > WARN > ADVISORY > PASS > NA.
    """
    if image_quality == "unreadable":
        return "unreadable"
    if not results and unreadable_fields:
        return "unreadable"
    worst = CheckOutcome.NA
    for r in results:
        worst = worse(worst, r.status)
    return worst.value


def verify(
    inp: VerifyInput,
    *,
    extractor: VisionExtractor,
    health_warning_reader: HealthWarningExtractor | None = None,
    skip_capture_quality: bool = False,
) -> VerifyReport:
    """Run a single image end-to-end: sensor pre-check → vision → rules.

    `skip_capture_quality` bypasses the sensor pre-flight. Reserved for
    callers that have already vetted (or synthesized) the frame — e.g.
    unit tests of the rule layer that supply sentinel byte strings, or
    a future re-enqueue path that has separately persisted a
    capture-quality verdict.

    `health_warning_reader` is the SPEC §0.5 redundant second pass on
    the Government Warning. When supplied, it runs after the primary
    extraction and the two reads are reconciled with `cross_check()`:
    a disagreement downgrades the warning rule to ADVISORY ("couldn't
    verify"), a confirmed-noncompliant agreement leaves the FAIL alone.
    """
    started = time.monotonic()

    # 1. Sensor pre-check — every byte goes through this first. Cheap, PIL
    #    only, deterministic. The result is forwarded to the VLM and
    #    merged into the final image_quality verdict pessimistically.
    if skip_capture_quality:
        capture = CaptureQualityReport(
            surfaces=[], overall_verdict="good", overall_confidence=1.0
        )
    else:
        capture = _safe_capture_quality(inp.image_bytes)
    front = capture.surfaces[0] if capture.surfaces else None

    rules = load_rules(beverage_type=inp.beverage_type)
    if not rules:
        raise ValueError(
            f"No rules configured for beverage_type={inp.beverage_type!r}. "
            "Add a rule definition file under app/rules/definitions/."
        )

    if capture.overall_verdict == "unreadable":
        # 2. Fatal-degradation short circuit. Don't burn a VLM call on a
        #    frame the sensor module already proved unreadable.
        rule_results = _unreadable_rule_results(rules)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return VerifyReport(
            overall="unreadable",
            rule_results=rule_results,
            extracted={},
            unreadable_fields=[],
            image_quality="unreadable",
            image_quality_notes=_summarize_capture_issues(capture),
            elapsed_ms=elapsed_ms,
        )

    extraction = extractor.extract(inp.image_bytes, media_type=inp.media_type)

    # Cap field confidence at the surface confidence. The model can only be
    # as confident in a reading as the frame it came from supports.
    if front is not None:
        for f in extraction.fields.values():
            f.confidence = round(min(f.confidence, front.confidence), 3)

    # Adversarial-input guards (SPEC §0.5). Foreign-language labels are
    # OUT of scope for v1 — refuse with a clear message rather than
    # serving the user a confusing edit-distance failure on the warning
    # rule. Screenshot uploads are surfaced as a soft advisory note so
    # the user can confirm without being blocked.
    foreign_language = detect_foreign_language(
        *(f.value for f in extraction.fields.values()),
    )
    # Reuse the sensor pre-check's capture-source classification rather than
    # re-decoding the image to inspect EXIF a second time. `front` is None
    # only when `skip_capture_quality=True`, which already implies the caller
    # has vetted (or synthesized) the frame.
    screenshot_signal = (
        screenshot_signal_from_source(front.capture_source) if front else None
    )

    if foreign_language is not None:
        rule_results = _unreadable_rule_results(rules)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return VerifyReport(
            overall="unreadable",
            rule_results=rule_results,
            extracted={},
            unreadable_fields=[],
            image_quality="unreadable",
            image_quality_notes=merge_signals(
                (foreign_language, screenshot_signal),
                existing_notes=_summarize_capture_issues(capture),
            ),
            elapsed_ms=elapsed_ms,
        )

    image_quality = _worse_quality(
        capture.overall_verdict, _quality_from_extraction(extraction)
    )
    image_quality_notes = merge_signals(
        (screenshot_signal,),
        existing_notes=_summarize_capture_issues(capture),
    )

    ctx = ExtractionContext(
        fields=extraction.fields,
        beverage_type=inp.beverage_type,
        container_size_ml=inp.container_size_ml,
        is_imported=inp.is_imported,
        application={
            **inp.application,
            "image_quality": image_quality,
            "image_quality_notes": image_quality_notes,
            "capture_quality": _capture_summary(capture),
        },
        unreadable_fields=list(extraction.unreadable),
    )

    engine = RuleEngine(rules)
    rule_results = engine.evaluate(ctx)

    if image_quality in {"degraded", "unreadable"}:
        rule_results = _downgrade_fails_for_unreadable_surface(
            rule_results, ctx, front
        )

    # 3. Health Warning redundant second-pass (SPEC §0.5). The Government
    #    Warning is the most legally consequential element on the label;
    #    two independent readings raise the trust bar significantly. A
    #    disagreement between primary and second-pass downgrades the
    #    warning rule to advisory.
    cross_check_result: CrossCheckResult | None = None
    if health_warning_reader is not None:
        try:
            second = health_warning_reader.read_warning(
                inp.image_bytes, media_type=inp.media_type
            )
        except Exception:
            second = None
        if second is not None:
            primary_warning = _primary_warning_read(extraction, ctx)
            cross_check_result = cross_check(primary_warning, second)
            rule_results = _apply_warning_cross_check(
                rule_results, cross_check_result, inp.beverage_type
            )

    elapsed_ms = int((time.monotonic() - started) * 1000)

    extracted_summary: dict[str, dict[str, Any]] = {}
    for name, ef in ctx.fields.items():
        extracted_summary[name] = {
            "value": ef.value,
            "confidence": ef.confidence,
            "bbox": list(ef.bbox) if ef.bbox else None,
            "unreadable": name in ctx.unreadable_fields,
        }
    for name in extraction.unreadable:
        extracted_summary[name] = {
            "value": None,
            "confidence": 0.0,
            "bbox": None,
            "unreadable": True,
        }

    return VerifyReport(
        overall=_aggregate_overall(
            rule_results, list(extraction.unreadable), image_quality
        ),
        rule_results=rule_results,
        extracted=extracted_summary,
        unreadable_fields=list(extraction.unreadable),
        image_quality=image_quality,
        image_quality_notes=image_quality_notes,
        health_warning_cross_check=_serialize_cross_check(cross_check_result),
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_capture_quality(image_bytes: bytes) -> CaptureQualityReport:
    """Run the sensor pre-check; never let it crash the verify path.

    A failure inside the pre-check (e.g. unsupported format) shouldn't take
    the whole request down. Surface it as a "degraded" verdict so the rest
    of the path still runs but the report carries an honest unknown.

    Decode-only failures (no surface produced any pixels) are likewise
    downgraded to "degraded" — without any pre-check signal we have no
    basis to short-circuit before the extractor has a chance to run. The
    rule engine remains responsible for flagging genuinely unreadable
    extractions via its own confidence-based downgrades.
    """
    try:
        report = assess_capture_quality({"front": image_bytes})
    except Exception:
        return CaptureQualityReport(
            surfaces=[], overall_verdict="degraded", overall_confidence=0.0
        )
    # All surfaces failed at decode time → no signal, not "unreadable".
    if report.surfaces and all(
        s.verdict == "unreadable" and s.metrics.megapixels == 0.0
        for s in report.surfaces
    ):
        return CaptureQualityReport(
            surfaces=report.surfaces,
            overall_verdict="degraded",
            overall_confidence=0.0,
        )
    return report


def _summarize_capture_issues(capture: CaptureQualityReport) -> str | None:
    if not capture.surfaces:
        return "Capture quality could not be assessed."
    parts: list[str] = []
    for s in capture.surfaces:
        if s.issues:
            parts.append("; ".join(s.issues))
    if not parts:
        return None
    return " | ".join(parts)


def _capture_summary(capture: CaptureQualityReport) -> dict[str, Any]:
    return {
        "overall_verdict": capture.overall_verdict,
        "overall_confidence": capture.overall_confidence,
        "surfaces": [
            {
                "surface": s.surface,
                "verdict": s.verdict,
                "confidence": s.confidence,
                "issues": s.issues,
                "suggestions": s.suggestions,
            }
            for s in capture.surfaces
        ],
    }


def _quality_from_extraction(extraction: Any) -> str:
    """If the extractor returned an explicit image_quality (newer extractors
    might), respect it; otherwise infer one from the unreadable list."""
    explicit = getattr(extraction, "image_quality", None)
    if isinstance(explicit, str) and explicit in _QUALITY_RANK:
        return explicit
    if extraction.unreadable and not extraction.fields:
        return "unreadable"
    if extraction.unreadable:
        return "degraded"
    return "good"


def _worse_quality(a: str, b: str) -> str:
    return a if _QUALITY_RANK.get(a, 0) >= _QUALITY_RANK.get(b, 0) else b


def _unreadable_rule_results(rules: list) -> list[RuleResult]:
    """Per-rule ADVISORY results when the pre-check decided the image is
    unreadable — lets the user see every TTB requirement (citation + fix
    suggestion) while the system is explicit that nothing was verified."""
    out: list[RuleResult] = []
    for r in rules:
        out.append(
            RuleResult(
                rule_id=r.id,
                rule_version=r.version,
                citation=r.citation,
                status=CheckOutcome.ADVISORY,
                finding=(
                    "Couldn't verify with confidence — capture quality is "
                    "unreadable. Reshoot before relying on this verdict."
                ),
                expected=None,
                fix_suggestion=r.fix_suggestion,
            )
        )
    return out


def _primary_warning_read(extraction: Any, ctx: ExtractionContext):
    """Pack the primary extractor's health-warning reading as a WarningRead."""
    from app.services.health_warning_second_pass import WarningRead

    field_obj = ctx.fields.get("health_warning")
    if field_obj is not None and field_obj.value:
        return WarningRead(
            value=field_obj.value,
            found=True,
            confidence=field_obj.confidence,
            source="primary_extractor",
        )
    if "health_warning" in (extraction.unreadable or []):
        return WarningRead(
            value=None,
            found=False,
            confidence=0.0,
            source="primary_extractor",
        )
    return WarningRead(
        value=None,
        found=False,
        confidence=0.0,
        source="primary_extractor",
    )


def _apply_warning_cross_check(
    results: list[RuleResult],
    cc: CrossCheckResult,
    beverage_type: str,
) -> list[RuleResult]:
    """Reconcile the cross-check outcome with the rule engine's verdict on
    the Government Warning rule.

    Outcomes:
      * confirmed_compliant  → leave PASS in place
      * confirmed_noncompliant → leave FAIL in place (both reads agree the
        label's text is wrong; the verdict is well-supported)
      * disagreement → downgrade the warning rule to ADVISORY regardless
        of the engine's call — we couldn't verify, so we don't claim
      * primary_only / no_warning_present → leave the engine's call alone
    """
    target_id_substring = "health_warning"
    if cc.outcome not in {"disagreement"}:
        return results

    out: list[RuleResult] = []
    for r in results:
        if target_id_substring not in r.rule_id:
            out.append(r)
            continue
        if r.status in (CheckOutcome.PASS, CheckOutcome.FAIL, CheckOutcome.WARN):
            out.append(
                RuleResult(
                    rule_id=r.rule_id,
                    rule_version=r.rule_version,
                    citation=r.citation,
                    status=CheckOutcome.ADVISORY,
                    finding=cc.notes,
                    expected=r.expected,
                    fix_suggestion=r.fix_suggestion,
                    bbox=r.bbox,
                )
            )
            continue
        out.append(r)
    return out


def _serialize_cross_check(cc: CrossCheckResult | None) -> dict[str, Any] | None:
    if cc is None:
        return None
    return {
        "outcome": cc.outcome,
        "edit_distance_to_canonical": cc.edit_distance_to_canonical,
        "edit_distance_between_reads": cc.edit_distance_between_reads,
        "notes": cc.notes,
        "primary": _serialize_warning_read(cc.primary),
        "secondary": _serialize_warning_read(cc.secondary),
    }


def _serialize_warning_read(read: Any) -> dict[str, Any] | None:
    if read is None:
        return None
    return {
        "value": read.value,
        "found": read.found,
        "confidence": read.confidence,
        "source": read.source,
    }


def _downgrade_fails_for_unreadable_surface(
    results: list[RuleResult],
    ctx: ExtractionContext,
    surface: SurfaceCaptureQuality | None,
) -> list[RuleResult]:
    """If the only surface is unreadable, FAILs that fired off no extracted
    field are unsupportable — convert them to ADVISORY."""
    if surface is None or surface.verdict != "unreadable":
        return results
    out: list[RuleResult] = []
    for r in results:
        if r.status != CheckOutcome.FAIL:
            out.append(r)
            continue
        out.append(
            RuleResult(
                rule_id=r.rule_id,
                rule_version=r.rule_version,
                citation=r.citation,
                status=CheckOutcome.ADVISORY,
                finding=(
                    (r.finding or "Required element could not be verified")
                    + " · downgraded to advisory because the source frame was"
                    " unreadable; reshoot before relying on this verdict."
                ),
                expected=r.expected,
                fix_suggestion=r.fix_suggestion,
                bbox=r.bbox,
            )
        )
    return out
