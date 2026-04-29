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

Fast path (cache hit): when a `VerifyCache` is supplied and an identical
prior request has been served, the cached `VerifyReport` is restamped
with this run's elapsed_ms and returned directly. The cache key is a
SHA-256 over the image bytes plus everything that can change the
verdict — beverage type, container size, imported flag, claim payload,
rule-set fingerprint — so a hit is genuinely the same request as
before, and a hit's verdict is identical to what the cold path would
have produced. The hit cost is dominated by `hashlib.sha256` and the
rule-set fingerprint (sub-millisecond on a typical frame), keeping the
end-to-end well under the 50 ms budget the iterative-design workflow
calls for.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
from app.services.verify_cache import make_cache_key, restamp_report
from app.services.vision import VisionExtractor

if TYPE_CHECKING:
    # Type-only — avoids the same circular concern that pushed VerifyReport
    # out of `verify_cache`'s runtime imports.
    from app.services.verify_cache import VerifyCache

logger = logging.getLogger(__name__)

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
    # Set by the cache layer on a hit so the API can surface a
    # "verified instantly" affordance and dashboards can compute
    # cold/warm latency separately. False on the cold path so a
    # degraded run is never silently labelled as cached.
    cache_hit: bool = False


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
    cache: VerifyCache | None = None,
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

    `cache`, when supplied, short-circuits the entire pipeline (sensor
    pre-check, VLM call, second-pass reader, rules) for repeats of an
    identical request. The cached verdict is the very same one the
    cold path produced — the cache only ever returns what was once a
    fresh, fail-honestly run.
    """
    started = time.monotonic()

    # Fast path. Compute the cache key as cheaply as possible so a miss
    # adds no measurable cost and a hit returns inside the 50 ms budget
    # the iterative-design workflow demands.
    cache_key: str | None = None
    if cache is not None:
        rules_for_key = load_rules(beverage_type=inp.beverage_type)
        cache_key = make_cache_key(
            image_bytes=inp.image_bytes,
            media_type=inp.media_type,
            beverage_type=inp.beverage_type,
            container_size_ml=inp.container_size_ml,
            is_imported=inp.is_imported,
            application=inp.application,
            rules=rules_for_key,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            hit = restamp_report(cached, elapsed_ms)
            hit.cache_hit = True
            return hit

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
        return _finalize(
            VerifyReport(
                overall="unreadable",
                rule_results=rule_results,
                extracted={},
                unreadable_fields=[],
                image_quality="unreadable",
                image_quality_notes=_summarize_capture_issues(capture),
                elapsed_ms=elapsed_ms,
            ),
            cache=cache,
            cache_key=cache_key,
        )

    # SPEC §0.5: hand the extractor everything we already know — the sensor
    # pre-check, the producer's claim, and the user-supplied beverage
    # type/container/imported flags. The model uses these as priors only;
    # the rule engine still does the actual claim-vs-label cross-check.
    producer_record = inp.application.get("producer_record")
    if not isinstance(producer_record, dict):
        producer_record = None

    # Reduce the bytes we send to the model. Two compounding wins:
    #   1. Crop to the detected label region when one was localized — the
    #      model isn't billed to look at the user's hand or the bar wall.
    #   2. Cap the long edge at TARGET_LONG_EDGE (≈1568 px). Anthropic
    #      auto-resizes anything larger to that limit on its side anyway,
    #      so pre-resizing is free quality-wise and saves wire bytes +
    #      Anthropic's resize step. Re-encoding to JPEG-85 cuts upload
    #      size further with no perceptible OCR loss at this resolution.
    # When normalisation cropped, we suppress the per-region sensor
    # briefing (its coordinates no longer match what the model sees) and
    # translate any returned bboxes back to original-image space.
    normalized = _normalize_for_vision(inp.image_bytes, capture)
    image_bytes_for_model = normalized.bytes
    media_type_for_model = normalized.media_type
    capture_for_briefing = None if normalized.cropped else capture

    # Primary extraction + redundant Government-Warning second-pass run
    # concurrently. Both are blocking HTTP calls to Anthropic, so a thread
    # pool is the simplest way to overlap their wall-clock cost — the
    # second pass becomes effectively free in latency terms (it finishes
    # well before the larger primary call) while still giving us the
    # independent read SPEC §0.5 mandates. A second-pass failure (timeout,
    # rate-limit, malformed JSON) is logged and the verdict falls back to
    # primary-only, identical to the previous serial behaviour.
    extraction, second_warning_read = _run_extractors_concurrently(
        extractor=extractor,
        health_warning_reader=health_warning_reader,
        image_bytes=image_bytes_for_model,
        media_type=media_type_for_model,
        capture=capture_for_briefing,
        producer_record=producer_record,
        beverage_type=inp.beverage_type,
        container_size_ml=inp.container_size_ml,
        is_imported=inp.is_imported,
    )

    if normalized.cropped:
        _translate_extraction_bboxes(extraction, normalized.offset)

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
        return _finalize(
            VerifyReport(
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
            ),
            cache=cache,
            cache_key=cache_key,
        )

    image_quality = _worse_quality(
        capture.overall_verdict, _quality_from_extraction(extraction)
    )
    # The model's own notes ("hand occludes lower text", "specular highlight
    # over right third of label") are exactly the kind of region-specific
    # diagnosis the user message benefits from. Surface them alongside the
    # sensor pre-check's frame-level summary.
    sensor_summary = _summarize_capture_issues(capture)
    model_notes = getattr(extraction, "image_quality_notes", None)
    if model_notes:
        sensor_summary = (
            f"{sensor_summary} | [model] {model_notes}"
            if sensor_summary
            else f"[model] {model_notes}"
        )
    image_quality_notes = merge_signals(
        (screenshot_signal,),
        existing_notes=sensor_summary,
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

    # 3. Reconcile the redundant Government-Warning read (SPEC §0.5). The
    #    second-pass ran in parallel with the primary above; here we
    #    cross-check the two independent reads. Disagreement downgrades
    #    the warning rule to advisory — confident-wrong is the failure
    #    mode we refuse to ship. Agreement leaves the engine's verdict
    #    intact, with both reads supporting it.
    cross_check_result: CrossCheckResult | None = None
    if second_warning_read is not None:
        primary_warning = _primary_warning_read(extraction, ctx)
        cross_check_result = cross_check(primary_warning, second_warning_read)
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

    return _finalize(
        VerifyReport(
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
        ),
        cache=cache,
        cache_key=cache_key,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Anthropic auto-resizes anything beyond ~1568 px on the long edge to that
# size before billing image tokens. Pre-resizing on our side saves the
# wire bytes (faster upload) without changing the model's view.
_VISION_TARGET_LONG_EDGE = 1568
_VISION_JPEG_QUALITY = 85
# Skip cropping when the label already covers most of the frame — re-
# encoding to save <30% of pixels is a wall-clock loss. The bbox detector
# already enforces a "≥10% of frame" floor on the small side, so any
# bbox that reaches us here is a real label region.
_CROP_MIN_GAIN_RATIO = 0.70
_CROP_MARGIN_FRACTION = 0.08
_CROP_MIN_MARGIN_PX = 24
_CROP_MIN_EDGE_PX = 64


@dataclass(frozen=True)
class _NormalizedImage:
    """Bytes the verify path actually sends to the model.

    `cropped` is True when the helper sliced down to the detected label
    region, which is the signal the orchestrator uses to suppress the
    sensor-region briefing (its bbox/glare-blob coordinates no longer
    match what the model is looking at) and to translate model-returned
    bboxes back to original-image coordinates after extraction. `offset`
    is `(dx, dy)` to add back during that translation; `(0, 0)` for the
    no-crop path.
    """

    bytes: bytes
    media_type: str
    cropped: bool
    offset: tuple[int, int]


def _normalize_for_vision(
    image_bytes: bytes, capture: CaptureQualityReport
) -> _NormalizedImage:
    """Crop + downscale + re-encode the upload before the model sees it.

    Three independent levers, applied only when each one nets a wire-byte
    win on this specific image:

      * Crop to the detected label region — only when it covers <70% of
        the frame (re-encoding to save <30% of pixels is a wall-clock
        loss). When this fires it's the biggest win.
      * Downscale to ≤1568 px on the long edge — Anthropic auto-resizes
        anything bigger to that size on its side, so pre-resizing is free
        quality-wise and saves wire bytes. Skipped when the image is
        already at or below the cap.
      * Re-encode as JPEG-85 — only when (a) we modified the pixels above,
        OR (b) the JPEG is actually smaller than the original encoding.
        For small artwork PNGs (flat colors, tight palettes) the original
        often beats JPEG, and uploading the larger output would slow the
        request rather than speed it up.

    Any failure (decode error, no bbox, edge case) returns the original
    bytes unchanged — normalisation is a *speed* optimisation, never a
    correctness path.
    """
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as exc:
        logger.debug("Vision normalisation skipped — decode failed: %s", exc)
        return _passthrough(image_bytes)

    crop_box, offset = _label_crop_box(capture, img.size)
    cropped = False
    if crop_box is not None:
        try:
            img = img.crop(crop_box)
            cropped = True
        except Exception as exc:
            logger.debug("Vision normalisation skipped crop step: %s", exc)
            offset = (0, 0)

    resized = False
    if max(img.size) > _VISION_TARGET_LONG_EDGE:
        scale = _VISION_TARGET_LONG_EDGE / max(img.size)
        new_size = (
            max(1, int(round(img.size[0] * scale))),
            max(1, int(round(img.size[1] * scale))),
        )
        try:
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            resized = True
        except Exception as exc:
            logger.debug("Vision normalisation skipped resize: %s", exc)

    if not cropped and not resized:
        # Nothing changed — re-encoding can only make this slower (JPEG of
        # a small artwork PNG often inflates the byte count). Send the
        # original unchanged.
        return _passthrough(image_bytes)

    if img.mode not in ("RGB", "L"):
        try:
            img = img.convert("RGB")
        except Exception:
            return _passthrough(image_bytes)

    try:
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_VISION_JPEG_QUALITY, optimize=True)
        out = buf.getvalue()
    except Exception as exc:
        logger.debug("Vision normalisation skipped re-encode: %s", exc)
        return _passthrough(image_bytes)

    # If our re-encode somehow ended up bigger than the original (rare on
    # cropped/resized output, but possible for tiny near-blank crops),
    # send the original. We're optimising for wire bytes; a larger upload
    # is the opposite of "speed it up".
    if len(out) >= len(image_bytes) and not cropped:
        return _passthrough(image_bytes)

    return _NormalizedImage(
        bytes=out,
        media_type="image/jpeg",
        cropped=cropped,
        offset=offset if cropped else (0, 0),
    )


def _passthrough(image_bytes: bytes) -> _NormalizedImage:
    return _NormalizedImage(
        bytes=image_bytes,
        media_type=_guess_media_type(image_bytes),
        cropped=False,
        offset=(0, 0),
    )


def _label_crop_box(
    capture: CaptureQualityReport, image_size: tuple[int, int]
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int]]:
    """Return `((x0, y0, x1, y1), (dx, dy))` for the label crop, or
    `(None, (0, 0))` when cropping wouldn't help.
    """
    if not capture.surfaces:
        return None, (0, 0)
    front = capture.surfaces[0]
    bbox = front.label_bbox
    if bbox is None:
        return None, (0, 0)
    frame_w, frame_h = image_size
    if frame_w <= 0 or frame_h <= 0:
        return None, (0, 0)
    bbox_area = bbox.area
    frame_area = frame_w * frame_h
    if bbox_area <= 0 or bbox_area / frame_area > _CROP_MIN_GAIN_RATIO:
        return None, (0, 0)

    margin = max(
        _CROP_MIN_MARGIN_PX,
        int(_CROP_MARGIN_FRACTION * max(bbox.w, bbox.h)),
    )
    x0 = max(0, bbox.x - margin)
    y0 = max(0, bbox.y - margin)
    x1 = min(frame_w, bbox.x + bbox.w + margin)
    y1 = min(frame_h, bbox.y + bbox.h + margin)
    if x1 - x0 < _CROP_MIN_EDGE_PX or y1 - y0 < _CROP_MIN_EDGE_PX:
        return None, (0, 0)
    return (x0, y0, x1, y1), (x0, y0)


def _translate_extraction_bboxes(
    extraction: Any, offset: tuple[int, int]
) -> None:
    """Map model-returned bboxes back to original-image space.

    The model worked from a (cropped, resized) image. Its bboxes come back
    in that frame's pixel coordinates, but downstream consumers (the API
    response, future UI overlays) expect coordinates relative to the
    original upload. Adding the crop offset is approximate after the
    resize step — it's accurate to within the resize ratio, which is more
    than precise enough for highlight rendering. If we ever need exact
    pre-resize coordinates we'll need to also scale by the resize ratio.
    """
    dx, dy = offset
    if dx == 0 and dy == 0:
        return
    for ef in extraction.fields.values():
        if ef.bbox is None:
            continue
        x, y, w, h = ef.bbox
        ef.bbox = (x + dx, y + dy, w, h)


def _guess_media_type(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] == b"GIF8":
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _run_extractors_concurrently(
    *,
    extractor: VisionExtractor,
    health_warning_reader: HealthWarningExtractor | None,
    image_bytes: bytes,
    media_type: str,
    capture: CaptureQualityReport | None,
    producer_record: dict[str, Any] | None,
    beverage_type: str,
    container_size_ml: int,
    is_imported: bool,
):
    """Dispatch the primary extractor and the optional second-pass reader on
    independent threads so their wall-clock cost overlaps.

    Returns `(primary_extraction, second_warning_read | None)`.

    The primary extractor's failure propagates — the verify path cannot
    serve a verdict without it. The second-pass reader's failure is logged
    and swallowed: the cross-check tolerates a missing secondary by leaving
    the engine's verdict alone, which is the same behaviour the previous
    serial path had.
    """
    capture_kwarg = capture if (capture is not None and capture.surfaces) else None

    def _primary():
        return extractor.extract(
            image_bytes,
            media_type=media_type,
            capture_quality=capture_kwarg,
            producer_record=producer_record,
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
        )

    def _secondary():
        if health_warning_reader is None:
            return None
        try:
            return health_warning_reader.read_warning(
                image_bytes, media_type=media_type
            )
        except Exception as exc:
            # Same swallow-and-log behaviour as the prior serial path —
            # the cross-check tolerates a None secondary read.
            logger.warning(
                "Health-warning second-pass failed; falling back to "
                "primary-only verdict: %s",
                exc,
            )
            return None

    # max_workers=2 is the upper bound we ever need (one primary + one
    # secondary). Building the pool inline keeps thread lifetimes scoped
    # to a single request — no shared state survives the call.
    with ThreadPoolExecutor(max_workers=2) as pool:
        primary_future = pool.submit(_primary)
        secondary_future = (
            pool.submit(_secondary) if health_warning_reader is not None else None
        )
        # Resolve the primary first: any extractor exception (including
        # ExtractorUnavailable from the chain) must surface to the caller
        # before we even look at the secondary.
        primary_result = primary_future.result()
        secondary_result = (
            secondary_future.result() if secondary_future is not None else None
        )
    return primary_result, secondary_result


def _finalize(
    report: VerifyReport,
    *,
    cache: VerifyCache | None,
    cache_key: str | None,
) -> VerifyReport:
    """Single seam where cold-path results land in the cache.

    Every cold-path return funnels through here so the three exit points
    (sensor-unreadable short-circuit, foreign-language short-circuit,
    full-success) cache identically. `cache_hit` is left at its dataclass
    default of False — the *next* request that resolves to a hit will
    flip it via `restamp_report` + assignment in the fast path.
    """
    if cache is not None and cache_key is not None:
        cache.put(cache_key, report)
    return report


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
