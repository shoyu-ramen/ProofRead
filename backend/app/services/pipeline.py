"""Scan pipeline.

Two extractor paths converge on the same rule engine:

  * `vision` (preferred) — Claude Opus 4.7 reads each surface, returns
    field-level confidence and an image_quality verdict. Robust under
    extreme conditions (SPEC §0.5).
  * `ocr` (fallback) — Google Vision (or mock) text detection followed
    by regex/pattern field extraction. Used when no Claude API key is
    configured or the vision path raises.

Either way the rule engine is the deterministic judge. The model never
declares pass/fail; it only reads.

Camera-sensor pre-flight (SPEC §0.5 fail-honestly):
  Before either extractor runs, every surface is run through
  `sensor_check.assess_capture_quality()`. The result is:
    1. Embedded in the report so the user / admin can see what the
       device produced.
    2. Forwarded to the vision extractor as an objective prior so the
       model can be honest about image_quality instead of guessing.
    3. Used to skip OCR + downgrade rule results on surfaces we already
       know are unreadable, saving API spend and avoiding wrong-fail
       verdicts driven by garbage frames.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import CheckOutcome, ExtractionContext, RuleResult
from app.services.extractors.beer import extract_beer_fields
from app.services.extractors.claude_vision import ProducerRecord
from app.services.ocr import OCRBlock, OCRProvider, OCRResult
from app.services.sensor_check import (
    CaptureQualityReport,
    SurfaceCaptureQuality,
    assess_capture_quality,
)


@dataclass
class ScanInput:
    beverage_type: str
    container_size_ml: int
    images: dict[str, bytes]
    is_imported: bool = False
    producer_record: ProducerRecord | None = None


class VisionExtractor(Protocol):
    def extract(
        self,
        *,
        beverage_type: str,
        container_size_ml: int,
        images: dict[str, bytes],
        producer_record: ProducerRecord | None = None,
        is_imported: bool = False,
        capture_quality: CaptureQualityReport | None = None,
    ) -> ExtractionContext: ...


@dataclass
class ScanReport:
    overall: str  # "pass" | "warn" | "fail" | "advisory" | "unreadable"
    rule_results: list[RuleResult]
    image_quality: str = "good"
    image_quality_notes: str | None = None
    extractor: str = "ocr"
    ocr_results: dict[str, OCRResult] = field(default_factory=dict)
    fields_summary: dict[str, Any] = field(default_factory=dict)
    capture_quality: CaptureQualityReport | None = None


def overall_status(results: list[RuleResult], *, image_quality: str = "good") -> str:
    """Thin shim that delegates to the shared aggregator.

    The canonical implementation now lives in `app.rules.aggregation` so
    the verify and scan paths roll up identically. Kept as a module-level
    function so any external caller / test that imported it directly
    keeps working without import-path churn.
    """
    from app.rules.aggregation import overall_status as _shared

    return _shared(results, image_quality=image_quality)


def process_scan(
    scan: ScanInput,
    ocr: OCRProvider | None = None,
    *,
    vision: VisionExtractor | None = None,
    skip_capture_quality: bool = False,
) -> ScanReport:
    """Run a scan through extraction → rule engine → report.

    At least one of `vision` or `ocr` must be supplied. When both are given,
    the vision extractor is preferred; OCR is the fallback if vision raises.

    `skip_capture_quality` bypasses the sensor pre-flight. Reserved for
    callers that have already vetted the frames (e.g. unit tests of the
    rule engine that supply synthetic byte-strings, not real images).
    """
    if vision is None and ocr is None:
        raise ValueError("process_scan requires either vision= or ocr=")
    if scan.beverage_type != "beer":
        raise NotImplementedError(
            f"v1 supports beer only; got beverage_type={scan.beverage_type!r}"
        )

    capture = (
        None if skip_capture_quality else assess_capture_quality(scan.images)
    )
    quality_by_surface: dict[str, SurfaceCaptureQuality] = (
        capture.by_surface() if capture else {}
    )

    extractor_used = "ocr"
    ocr_results: dict[str, OCRResult] = {}
    ctx: ExtractionContext | None = None

    if vision is not None:
        try:
            ctx = vision.extract(
                beverage_type=scan.beverage_type,
                container_size_ml=scan.container_size_ml,
                images=scan.images,
                producer_record=scan.producer_record,
                is_imported=scan.is_imported,
                capture_quality=capture,
            )
            extractor_used = ctx.application.get("model_provider", "vision")
        except Exception:
            if ocr is None:
                raise
            ctx = None  # fall through to OCR

    if ctx is None:
        assert ocr is not None  # guaranteed by entry check
        ocr_results = _run_ocr(scan.images, ocr, quality_by_surface)
        ctx = extract_beer_fields(
            ocr_results,
            container_size_ml=scan.container_size_ml,
            is_imported=scan.is_imported,
        )
        if scan.producer_record is not None:
            ctx.application["producer_record"] = {
                "brand": scan.producer_record.brand,
                "class_type": scan.producer_record.class_type,
                "container_size_ml": scan.producer_record.container_size_ml,
            }
        extractor_used = "ocr"

    if quality_by_surface:
        _propagate_surface_confidence(ctx, quality_by_surface)
        # Pessimistic merge: keep whichever of (sensor verdict, extractor
        # verdict) is worse. The vision agent and the sensor pre-check
        # sometimes disagree — if the sensor sees obvious motion blur but
        # the model says "good", surfacing the sensor verdict avoids a
        # confident wrong "pass" (SPEC §0.5 fail-honestly).
        sensor_verdict = capture.overall_verdict if capture else "good"
        extractor_verdict = ctx.application.get("image_quality", "good")
        ctx.application["image_quality"] = _worse_quality(
            sensor_verdict, extractor_verdict
        )
        if not ctx.application.get("image_quality_notes"):
            ctx.application["image_quality_notes"] = _capture_notes(capture)
        ctx.application.setdefault(
            "capture_quality_summary", _capture_summary(capture) if capture else None
        )

    engine = RuleEngine(load_rules(beverage_type="beer"))
    rule_results = engine.evaluate(ctx)
    if quality_by_surface:
        rule_results = _apply_capture_downgrade(rule_results, ctx, quality_by_surface)

    image_quality = ctx.application.get("image_quality", "good")
    image_quality_notes = ctx.application.get("image_quality_notes")

    return ScanReport(
        overall=overall_status(rule_results, image_quality=image_quality),
        rule_results=rule_results,
        image_quality=image_quality,
        image_quality_notes=image_quality_notes,
        extractor=extractor_used,
        ocr_results=ocr_results,
        fields_summary={
            name: {
                "value": f.value,
                "confidence": f.confidence,
                "bbox": f.bbox,
                "source_image_id": f.source_image_id,
                "unreadable": name in ctx.unreadable_fields,
            }
            for name, f in ctx.fields.items()
        },
        capture_quality=capture,
    )


# ---------------------------------------------------------------------------
# Capture-quality plumbing
# ---------------------------------------------------------------------------


def _run_ocr(
    images: dict[str, bytes],
    ocr: OCRProvider,
    quality_by_surface: dict[str, SurfaceCaptureQuality],
) -> dict[str, OCRResult]:
    """Call the OCR provider per surface, skipping unreadable ones."""
    results: dict[str, OCRResult] = {}
    for surface, image_bytes in images.items():
        sq = quality_by_surface.get(surface)
        if sq and sq.verdict == "unreadable":
            results[surface] = _placeholder_ocr_result(sq)
            continue
        results[surface] = ocr.process(image_bytes, hint=surface)
    return results


def _placeholder_ocr_result(sq: SurfaceCaptureQuality) -> OCRResult:
    note = "; ".join(sq.issues) or "Image deemed unreadable by capture-quality check"
    return OCRResult(
        full_text="",
        blocks=[OCRBlock(text="", bbox=(0, 0, 0, 0), confidence=0.0)],
        provider="capture_quality_skip",
        raw={
            "skipped": True,
            "verdict": sq.verdict,
            "confidence": sq.confidence,
            "issues": sq.issues,
            "note": note,
        },
    )


def _propagate_surface_confidence(
    ctx: ExtractionContext,
    quality_by_surface: dict[str, SurfaceCaptureQuality],
) -> None:
    for fname, f in ctx.fields.items():
        sq = quality_by_surface.get(f.source_image_id) if f.source_image_id else None
        if sq is None:
            continue
        # Cap field confidence at the surface confidence — even a confidently
        # extracted string is only as trustworthy as the frame it came from.
        f.confidence = round(min(f.confidence, sq.confidence), 3)
        if sq.verdict == "unreadable" and fname not in ctx.unreadable_fields:
            ctx.unreadable_fields.append(fname)


def _apply_capture_downgrade(
    results: list[RuleResult],
    ctx: ExtractionContext,
    quality_by_surface: dict[str, SurfaceCaptureQuality],
) -> list[RuleResult]:
    """Downgrade FAIL → ADVISORY when the offending field came from an
    unreadable surface. SPEC §0.5: a wrong "fail" we cannot stand behind
    is the second-worst outcome — refuse it in favor of "couldn't verify".

    Two regimes:

    * **Some surfaces unreadable** — only downgrade rules whose evidence
      lived on an unreadable surface (or, by convention, the
      health-warning rule when the back is the unreadable side).
    * **All surfaces unreadable** — downgrade *every* FAIL. Nothing the
      pipeline returns reflects the actual label, so no FAIL is
      defensible; only ADVISORY is.
    """
    if not quality_by_surface:
        return results

    has_unreadable = any(
        q.verdict == "unreadable" for q in quality_by_surface.values()
    )
    if not has_unreadable:
        return results

    all_unreadable = all(
        q.verdict == "unreadable" for q in quality_by_surface.values()
    )

    out: list[RuleResult] = []
    for r in results:
        if r.status != CheckOutcome.FAIL:
            out.append(r)
            continue

        if all_unreadable:
            out.append(_downgrade_to_advisory(r))
            continue

        field_in_rule = _field_referenced(r.rule_id)
        f = ctx.fields.get(field_in_rule) if field_in_rule else None
        sq = (
            quality_by_surface.get(f.source_image_id)
            if f and f.source_image_id
            else None
        )
        back_unreadable = any(
            "back" in s and q.verdict == "unreadable"
            for s, q in quality_by_surface.items()
        )
        # Blob-level downgrade: even if the surface verdict isn't
        # "unreadable", downgrade when the field's bbox overlaps a glare
        # blob materially. The field couldn't have been read confidently
        # if it was inside a saturated blob; the rule that depends on it
        # shouldn't FAIL.
        blob_occluded = (
            f is not None
            and f.bbox is not None
            and sq is not None
            and _bbox_inside_glare(f.bbox, sq.glare_blobs)
        )
        downgrade = (
            (sq is not None and sq.verdict == "unreadable")
            or (back_unreadable and "health_warning" in r.rule_id)
            or blob_occluded
        )
        if not downgrade:
            out.append(r)
            continue
        out.append(_downgrade_to_advisory(r, blob_occluded=blob_occluded))
    return out


def _bbox_inside_glare(
    field_bbox: tuple[int, int, int, int] | None,
    glare_blobs,
) -> bool:
    """True when ≥30% of the field bbox falls inside any glare blob."""
    if not field_bbox or not glare_blobs:
        return False
    fx, fy, fw, fh = field_bbox
    field_area = max(1, fw * fh)
    for blob in glare_blobs:
        bx, by, bw, bh = blob.bbox.x, blob.bbox.y, blob.bbox.w, blob.bbox.h
        ix = max(fx, bx)
        iy = max(fy, by)
        ax = min(fx + fw, bx + bw)
        ay = min(fy + fh, by + bh)
        iw = max(0, ax - ix)
        ih = max(0, ay - iy)
        if (iw * ih) / field_area >= 0.30:
            return True
    return False


def _downgrade_to_advisory(r: RuleResult, *, blob_occluded: bool = False) -> RuleResult:
    if blob_occluded:
        suffix = (
            " · downgraded to advisory because the field's region overlaps a"
            " specular glare blob; reshoot with the bottle tilted away from"
            " the light source before relying on this verdict."
        )
    else:
        suffix = (
            " · downgraded to advisory because the source frame was"
            " unreadable; reshoot before relying on this verdict."
        )
    return RuleResult(
        rule_id=r.rule_id,
        rule_version=r.rule_version,
        citation=r.citation,
        status=CheckOutcome.ADVISORY,
        finding=((r.finding or "Required element could not be verified") + suffix),
        expected=r.expected,
        fix_suggestion=r.fix_suggestion,
        bbox=r.bbox,
        surface=r.surface,
    )


_QUALITY_RANK = {"good": 0, "degraded": 1, "unreadable": 2}


def _worse_quality(a: str, b: str) -> str:
    """Return whichever image_quality verdict is more pessimistic."""
    return a if _QUALITY_RANK.get(a, 0) >= _QUALITY_RANK.get(b, 0) else b


def _field_referenced(rule_id: str) -> str | None:
    """`beer.health_warning.exact_text` → `health_warning`."""
    parts = rule_id.split(".")
    if len(parts) < 3:
        return None
    return parts[1]


def _capture_notes(capture: CaptureQualityReport | None) -> str | None:
    if capture is None:
        return None
    issues = []
    for s in capture.surfaces:
        if s.issues:
            issues.append(f"{s.surface}: " + "; ".join(s.issues))
    return " | ".join(issues) or None


def _capture_summary(capture: CaptureQualityReport) -> dict[str, Any]:
    """Compact, JSON-friendly summary used in the API + admin debug view.

    Region-aware fields (label bbox + per-region metrics, glare blobs,
    backlight, motion direction, sensor tier) are included so the admin
    UI can render a rich diagnostic — and so the model's reasoning can be
    audited after the fact (which blob the brand bbox overlapped, etc.).
    """
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
                "sensor": s.sensor.describe(),
                "sensor_tier": s.sensor.tier,
                "label_bbox": (
                    s.label_bbox.as_tuple() if s.label_bbox is not None else None
                ),
                "label_verdict": s.label_verdict,
                "metrics": {
                    "sharpness": round(s.metrics.sharpness, 1),
                    "glare_fraction": round(s.metrics.glare_fraction, 3),
                    "brightness_mean": round(s.metrics.brightness_mean, 1),
                    "brightness_stddev": round(s.metrics.brightness_stddev, 1),
                    "color_cast": round(s.metrics.color_cast, 1),
                    "megapixels": s.metrics.megapixels,
                },
                "metrics_label": (
                    {
                        "sharpness": round(s.metrics_label.sharpness, 1),
                        "glare_fraction": round(s.metrics_label.glare_fraction, 3),
                        "brightness_mean": round(s.metrics_label.brightness_mean, 1),
                        "brightness_stddev": round(s.metrics_label.brightness_stddev, 1),
                    }
                    if s.metrics_label is not None
                    else None
                ),
                "glare_blobs": [
                    {
                        "bbox": b.bbox.as_tuple(),
                        "area_fraction_frame": round(b.area_fraction_frame, 3),
                        "area_fraction_label": round(b.area_fraction_label, 3),
                    }
                    for b in s.glare_blobs
                ],
                "backlit": s.backlit,
                "motion_blur_direction": s.motion_blur_direction,
            }
            for s in capture.surfaces
        ],
    }
