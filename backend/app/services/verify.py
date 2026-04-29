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
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import (
    CheckOutcome,
    ExtractionContext,
    RuleResult,
)
from app.services.adversarial import (
    detect_foreign_language,
    merge_signals,
    screenshot_signal_from_source,
)
from app.services.health_warning_second_pass import (
    CrossCheckResult,
    HealthWarningExtractor,
    ObstructionSignal,
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

# Surface tag used for the v1 mobile single-shot happy path. The client
# uploads one unrolled-label image; the merge step retags every field's
# `source_image_id` with this constant so the response carries a stable
# identifier the UI can join against the uploaded image. Multi-panel
# callers (web/agent) keep the `panel_N` shape.
PANORAMA_SURFACE = "panorama"

# Process-wide pool reused across requests. Per-request `ThreadPoolExecutor`
# allocation pays ~1–2 ms per worker for thread create + join; on a hot
# verify endpoint that is wasted overhead the singleton avoids. Sized for
# 4 panels × 2 extractors (primary + second-pass) × 4 concurrent requests
# on a small Fly/Railway box. Lazy and lock-protected so two cold requests
# can't each construct their own pool. `shutdown_pool()` is invoked from
# the FastAPI lifespan so the workers exit cleanly on graceful restart.
_POOL_MAX_WORKERS = 32
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=_POOL_MAX_WORKERS,
                thread_name_prefix="verify-extractor",
            )
    return _pool


def shutdown_pool() -> None:
    """Tear down the singleton pool. Called from FastAPI lifespan exit so
    the workers finish in-flight calls before the process exits. Safe to
    call when the pool was never lazily constructed (no-op)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


@dataclass
class Panel:
    """One label face submitted to the verify pipeline.

    A bottle's TTB-relevant text is split across surfaces — brand on the
    front, gov-warning usually on the back, name/address often on a neck
    band — so a single capture can never read every required field on a
    real product. The panel list is the unit the multi-panel verify path
    iterates over: each panel runs through the sensor pre-check, the
    primary extractor, and the redundant Government-Warning second-pass
    independently, then their reads are merged field-by-field with
    highest-confidence-wins (`_merge_panel_extractions`).
    """

    image_bytes: bytes
    media_type: str


@dataclass
class VerifyInput:
    # Legacy single-panel handles. Every request has at least this one
    # face; multi-panel callers pass additional faces via `extra_panels`.
    # Keeping the legacy names rather than collapsing into a `panels`
    # list lets the dozens of existing tests in `tests/test_verify*.py`
    # build inputs unchanged — `inp.panels` (the property below) is the
    # uniform handle the verify pipeline reads.
    image_bytes: bytes
    media_type: str
    beverage_type: str
    container_size_ml: int
    is_imported: bool
    application: dict[str, Any]
    # Additional panels in submission order. Empty for the v1 mobile
    # single-shot path (one unrolled-label panorama). When non-empty,
    # the merged extraction tags each field with `panel_<index>`
    # (`panel_0` is `image_bytes`, `panel_1` is `extra_panels[0]`, …);
    # when empty, every field is tagged with the constant
    # `"panorama"` so the client can highlight the uploaded image.
    extra_panels: list[Panel] = field(default_factory=list)

    @property
    def panels(self) -> list[Panel]:
        return [Panel(self.image_bytes, self.media_type), *self.extra_panels]


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
    """Thin shim that delegates to the shared aggregator.

    Kept as a module-level callable so any external caller / test that
    imported it directly keeps working — the canonical implementation
    now lives in `app.rules.aggregation` so the verify and scan paths
    can never drift on the empty-rule-list interpretation.
    """
    from app.rules.aggregation import overall_status

    return overall_status(
        results,
        image_quality=image_quality,
        unreadable_fields=unreadable_fields,
    )


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
    from app.telemetry import current_trace_id, traced_span

    started = time.monotonic()
    panels = inp.panels

    # Fast path. Compute the cache key as cheaply as possible so a miss
    # adds no measurable cost and a hit returns inside the 50 ms budget
    # the iterative-design workflow demands.
    cache_key: str | None = None
    if cache is not None:
        with traced_span("verify.cache_lookup"):
            rules_for_key = load_rules(beverage_type=inp.beverage_type)
            cache_key = make_cache_key(
                panels=[(p.image_bytes, p.media_type) for p in panels],
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
            from app.services import verify_stats

            verify_stats.record_warm(
                elapsed_ms=elapsed_ms, overall=hit.overall
            )
            logger.info(
                "verify warm path=warm elapsed_ms=%d overall=%s trace_id=%s",
                elapsed_ms,
                hit.overall,
                current_trace_id() or "-",
            )
            return hit

    # 1. Sensor pre-check — every panel goes through this first. The
    #    `assess_capture_quality` API is already multi-surface; we feed
    #    it `{panel_0: bytes, panel_1: bytes, …}` and it returns a
    #    pessimistic overall verdict ("unreadable" if any panel is
    #    unreadable) plus per-surface metrics. The merged verdict drives
    #    the same fail-honestly short-circuit the single-shot path used.
    with traced_span("verify.sensor_check"):
        if skip_capture_quality:
            capture = CaptureQualityReport(
                surfaces=[], overall_verdict="good", overall_confidence=1.0
            )
        else:
            capture = _safe_capture_quality_multi(panels)
        surfaces_by_panel = _surfaces_by_panel_index(capture, len(panels))

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

    # Reduce the bytes we send to the model. Two compounding wins per
    # panel:
    #   1. Crop to the detected label region when one was localized — the
    #      model isn't billed to look at the user's hand or the bar wall.
    #   2. Cap the long edge at TARGET_LONG_EDGE. Anthropic auto-resizes
    #      anything larger to that limit on its side anyway, so pre-
    #      resizing is free quality-wise and saves wire bytes.
    # When normalisation cropped, we suppress the per-region sensor
    # briefing for that panel (its coordinates no longer match what the
    # model sees) and translate any returned bboxes back to that panel's
    # original-image space.
    normalized_panels = [
        _normalize_for_vision(p.image_bytes, capture, surface_index=i)
        for i, p in enumerate(panels)
    ]

    # Per-panel primary extraction + redundant Government-Warning second-
    # pass, all concurrent in one thread pool. With N panels and second-
    # pass enabled, this dispatches up to 2N blocking HTTP calls so wall-
    # clock is bounded by the slowest individual call rather than the
    # serial sum. The second pass becomes effectively free in latency
    # terms (the smaller model finishes well before the primary) while
    # still giving us the independent read SPEC §0.5 mandates.
    with traced_span("verify.extractors", panel_count=len(panels)):
        per_panel_primary, per_panel_secondary = _run_extractors_concurrently(
            extractor=extractor,
            health_warning_reader=health_warning_reader,
            panels=normalized_panels,
            capture=capture,
            producer_record=producer_record,
            beverage_type=inp.beverage_type,
            container_size_ml=inp.container_size_ml,
            is_imported=inp.is_imported,
        )

    # Merge per-panel reads field-by-field, highest-confidence-wins, and
    # tag each merged field's `source_image_id` with the panel it came
    # from so the UI can render "Brand — front" / "Warning — back".
    extraction = _merge_panel_extractions(per_panel_primary)

    # Translate bboxes back to each panel's original-upload coordinates
    # using that panel's crop offset. The merged extraction's
    # source_image_id ("panel_N") tells us which panel's offset to apply.
    _translate_merged_bboxes(extraction, normalized_panels)

    # Cap field confidence at the source panel's surface confidence. The
    # model can only be as confident in a reading as the frame it came
    # from supports — and "the frame" is now per-panel, not a single
    # `front`. Fields whose source panel had no surface (skip_capture_
    # quality path) keep their raw confidence.
    for f in extraction.fields.values():
        idx = _panel_index_from_source(f.source_image_id)
        surface = surfaces_by_panel.get(idx) if idx is not None else None
        if surface is not None:
            f.confidence = round(min(f.confidence, surface.confidence), 3)

    # Pick the strongest second-pass read across all panels. The warning
    # is usually on the back; running second-pass on every panel and
    # picking the highest-confidence found read means we don't have to
    # know in advance which panel carries it.
    second_warning_read = _pick_best_secondary_warning(per_panel_secondary)
    front = surfaces_by_panel.get(0)

    # Adversarial-input guards (SPEC §0.5). Foreign-language labels are
    # OUT of scope for v1 — refuse with a clear message rather than
    # serving the user a confusing edit-distance failure on the warning
    # rule. Screenshot uploads are surfaced as a soft advisory note so
    # the user can confirm without being blocked.
    foreign_language = detect_foreign_language(
        *(f.value for f in extraction.fields.values()),
    )
    # Recall guard for the warning: when the warning is hidden by glare,
    # the primary extractor's other fields are often sparse (label crop
    # focused on the back, fine-print obscured) and the foreign-language
    # heuristic fires on too few English keywords. The second-pass reader
    # is an independent signal — if it saw any warning text, OR even saw
    # the warning region, the label is plausibly English-with-obstruction
    # and we'd rather refuse the foreign-language shortcut so the
    # cross-check + ADVISORY downgrade can do their job. The cost of a
    # false suppress is one extra rule-engine pass on a Spanish label
    # (which the warning rule will FAIL on anyway); the cost of a false
    # foreign-language is a confident "unreadable" verdict on a glared
    # English label, which is exactly the recall miss we are paying off.
    if foreign_language is not None and _warning_signals_english(
        second_warning_read, getattr(extraction, "image_quality_notes", None)
    ):
        logger.debug(
            "Suppressing foreign-language guard: warning signals indicate "
            "English label with obstruction"
        )
        foreign_language = None
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

    with traced_span("verify.rule_engine", n_rules=len(rules)):
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
    #
    #    The obstruction signal comes from the sensor pre-check (any
    #    panel's glare blobs, motion blur, or degraded label region) and
    #    lets the cross-check refuse a confident "missing warning"
    #    verdict whenever the warning could plausibly be hidden — recall
    #    on the warning is load-bearing for SPEC §0.5.
    obstruction = _build_obstruction_signal(capture)
    cross_check_result: CrossCheckResult | None = None
    if second_warning_read is not None:
        primary_warning = _primary_warning_read(extraction, ctx)
        cross_check_result = cross_check(
            primary_warning,
            second_warning_read,
            obstruction_signal=obstruction,
        )
        rule_results = _apply_warning_cross_check(
            rule_results, cross_check_result, inp.beverage_type
        )

    # Backstop for the no-second-pass path (and for any second-pass path
    # whose cross_check left the engine's "missing warning" FAIL alone):
    # when the engine's verdict is "warning missing" but the capture
    # report shows obstruction over the label, downgrade to ADVISORY.
    # The same recall principle that drives the cross-check applies even
    # when only one reader ran — we cannot claim the warning is missing
    # if glare could be hiding it.
    rule_results = _downgrade_missing_warning_under_obstruction(
        rule_results, ctx, obstruction
    )

    # Blob-overlap downgrade: a FAIL is unsupportable when the rule's
    # field bbox sits inside a sensor-pre-check glare blob. The pipeline
    # path has applied this since v1; verify-path parity ensures a
    # `/v1/verify` user gets the same fail-honestly downgrade as a
    # `/v1/scans` user on the same physical capture.
    rule_results = _downgrade_fails_for_glare_blob(
        rule_results, ctx, surfaces_by_panel
    )

    elapsed_ms = int((time.monotonic() - started) * 1000)

    extracted_summary: dict[str, dict[str, Any]] = {}
    for name, ef in ctx.fields.items():
        extracted_summary[name] = {
            "value": ef.value,
            "confidence": ef.confidence,
            "bbox": list(ef.bbox) if ef.bbox else None,
            "unreadable": name in ctx.unreadable_fields,
            "source_image_id": ef.source_image_id,
        }
    for name in extraction.unreadable:
        extracted_summary[name] = {
            "value": None,
            "confidence": 0.0,
            "bbox": None,
            "unreadable": True,
            "source_image_id": None,
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


# Long-edge cap before sending to the vision API. Bumped from 1568 to 2400
# after the wide-angle held-up-bottle failure mode: a 12 MP capture has the
# label at ~5 % of frame, and resizing to 1568 left the label only ~80 px
# wide — below the OCR threshold for the model. 2400 keeps the label at
# ~120 px (readable) while staying inside Anthropic's per-call image budget.
_VISION_TARGET_LONG_EDGE = 2400
_VISION_JPEG_QUALITY = 85
# Crop whenever the detected label leaves ANY meaningful background outside
# it. The previous 0.70 gate skipped the crop on near-full-frame detections,
# but those are exactly the cases where Anthropic's auto-downscale shrinks
# the label below the OCR threshold — even a 20 % crop preserves the
# difference between "readable" and "unreadable" text after downscale. The
# bbox detector enforces a 4 %-of-frame floor on the small side, so any
# bbox that reaches us is still a real label region.
_CROP_MIN_GAIN_RATIO = 0.92
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
    image_bytes: bytes,
    capture: CaptureQualityReport,
    *,
    surface_index: int = 0,
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

    `surface_index` is the position of this panel inside `capture.surfaces`.
    The crop bbox is read from the matching surface so a back-panel call
    uses the back panel's detected label region instead of the front's.

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

    crop_box, offset = _label_crop_box(capture, img.size, surface_index=surface_index)
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
    capture: CaptureQualityReport,
    image_size: tuple[int, int],
    *,
    surface_index: int = 0,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int]]:
    """Return `((x0, y0, x1, y1), (dx, dy))` for the label crop, or
    `(None, (0, 0))` when cropping wouldn't help.
    """
    if not capture.surfaces or surface_index >= len(capture.surfaces):
        return None, (0, 0)
    surface = capture.surfaces[surface_index]
    bbox = surface.label_bbox
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


def _translate_merged_bboxes(
    extraction: Any, normalized_panels: list[_NormalizedImage]
) -> None:
    """Map merged-extraction bboxes back to each source panel's original
    coordinates.

    The merged extraction's `source_image_id` is `panel_N`; we look up
    that panel's normalisation offset (`(dx, dy)` from the crop step)
    and add it to the bbox to put coordinates back in the panel's
    original-upload space. Each panel has its own coordinate system —
    a back-panel bbox is relative to the back image, not the front.
    The UI uses `source_image_id` to know which panel image to overlay.

    Adding the crop offset is approximate after the resize step — it's
    accurate to within the resize ratio, which is more than precise
    enough for highlight rendering. Fields whose `source_image_id` we
    can't resolve (unknown shape, no matching panel) are left alone.
    """
    for ef in extraction.fields.values():
        if ef.bbox is None:
            continue
        idx = _panel_index_from_source(ef.source_image_id)
        if idx is None or idx >= len(normalized_panels):
            continue
        norm = normalized_panels[idx]
        if not norm.cropped:
            continue
        dx, dy = norm.offset
        if dx == 0 and dy == 0:
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
    panels: list[_NormalizedImage],
    capture: CaptureQualityReport | None,
    producer_record: dict[str, Any] | None,
    beverage_type: str,
    container_size_ml: int,
    is_imported: bool,
):
    """Dispatch primary + optional second-pass for every panel concurrently.

    Returns `(primary_extractions_per_panel, secondary_reads_per_panel)`,
    each list aligned to `panels` order. With N panels and second-pass
    enabled, this dispatches up to 2N HTTP calls in one thread pool so
    wall-clock is bounded by the slowest call rather than the serial sum.

    The primary extractor's failure propagates — the verify path cannot
    serve a verdict without it. The second-pass reader's failure is logged
    and swallowed: the cross-check tolerates a missing secondary by leaving
    the engine's verdict alone, which is the same behaviour the previous
    serial path had. Per-panel briefing suppression: when normalisation
    cropped a panel, the sensor briefing for *that panel* would point at
    the wrong region, so we strip the briefing for cropped panels and
    keep it for uncropped ones.
    """

    def _primary_for_panel(panel_idx: int):
        norm = panels[panel_idx]
        # Cropped panels must drop the briefing — its label-bbox and
        # glare-blob coordinates are in the original frame, not the crop.
        # Uncropped panels keep the briefing so the model still gets
        # the sensor signal it can act on.
        capture_kwarg = (
            capture if (capture is not None and capture.surfaces and not norm.cropped) else None
        )
        return extractor.extract(
            norm.bytes,
            media_type=norm.media_type,
            capture_quality=capture_kwarg,
            producer_record=producer_record,
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
        )

    def _secondary_for_panel(panel_idx: int, abort: threading.Event):
        if health_warning_reader is None:
            return None
        # Fast-cancel hatch: if the primary already failed, the request is
        # toast — don't burn the secondary's HTTP budget on a verdict the
        # caller will never see. Checked at function entry; the in-flight
        # call itself can't be cooperatively interrupted, so we accept
        # that calls already past this gate run to completion.
        if abort.is_set():
            return None
        from app.services import verify_stats

        norm = panels[panel_idx]
        try:
            read = health_warning_reader.read_warning(
                norm.bytes, media_type=norm.media_type
            )
        except Exception as exc:
            outcome = verify_stats.classify_second_pass_exception(exc)
            verify_stats.record_second_pass(outcome)
            logger.warning(
                "Health-warning second-pass failed on panel %d (%s/%s); "
                "falling back to primary-only on this panel: %s",
                panel_idx,
                outcome,
                type(exc).__name__,
                exc,
            )
            # Surface to Sentry so a rate-limit / connection-error spike
            # is visible in the dashboard. No-op when Sentry is not
            # initialized; tags scope the breadcrumb to second-pass calls.
            from app.telemetry import capture_exception

            capture_exception(
                exc, outcome=outcome, component="health_warning_second_pass"
            )
            return None
        verify_stats.record_second_pass("success")
        return read

    n = len(panels)
    pool = _get_pool()

    # Submit all primaries first so they start in parallel. Secondaries are
    # gated on a shared abort event — when any primary raises we set the
    # event so still-queued secondaries return None immediately rather than
    # making a doomed HTTP call.
    abort = threading.Event()
    primary_futures = [pool.submit(_primary_for_panel, i) for i in range(n)]
    secondary_futures: list = (
        [pool.submit(_secondary_for_panel, i, abort) for i in range(n)]
        if health_warning_reader is not None
        else [None] * n
    )

    # FIRST_EXCEPTION semantics: as soon as any primary raises, bail. This
    # prevents a fast-failing primary (rate-limit, network) from being
    # blocked on the slower secondary's 8 s timeout.
    done, _pending = wait(primary_futures, return_when=FIRST_EXCEPTION)
    failed = next((f for f in done if f.exception() is not None), None)
    if failed is not None:
        # Cancel still-queued primaries (already-running ones can't be
        # interrupted; their results are discarded). Set the abort event
        # so still-queued secondaries return None immediately.
        for f in primary_futures:
            f.cancel()
        for f in secondary_futures:
            if f is not None:
                f.cancel()
        abort.set()
        raise failed.exception()  # type: ignore[misc]

    # All primaries succeeded — drain the secondaries the same way the
    # serial path would have.
    primary_results = [f.result() for f in primary_futures]
    secondary_results = [
        (f.result() if f is not None else None) for f in secondary_futures
    ]
    return primary_results, secondary_results


def _surfaces_by_panel_index(
    capture: CaptureQualityReport, panel_count: int
) -> dict[int, SurfaceCaptureQuality]:
    """Map panel index → the sensor surface for that panel.

    The pre-check is fed `{panel_0: ..., panel_1: ...}` so each surface's
    `surface` attribute is the panel id. Tolerates the skip-capture-
    quality path (empty surfaces list) by returning an empty mapping —
    callers default-on-miss when they look up a missing index.
    """
    out: dict[int, SurfaceCaptureQuality] = {}
    for s in capture.surfaces:
        idx = _panel_index_from_source(s.surface)
        if idx is not None and 0 <= idx < panel_count:
            out[idx] = s
    return out


def _panel_index_from_source(source: str | None) -> int | None:
    """Extract the panel index from a `source_image_id`.

    Recognised shapes:
      * `"panel_N"` → `N` (multi-panel orchestrator id)
      * `"panorama"` → `0` (v1 mobile single-shot happy path; the
        panorama is always the first panel by construction)

    Returns None for anything else (including the pre-multi-panel ids
    "front"/"back" and extractor-set custom ids), so those fields just
    don't get surface-confidence capping or bbox translation.
    """
    if not source:
        return None
    if source == PANORAMA_SURFACE:
        return 0
    if not source.startswith("panel_"):
        return None
    try:
        return int(source[len("panel_"):])
    except ValueError:
        return None


def _safe_capture_quality_multi(panels: list[Panel]) -> CaptureQualityReport:
    """Run the multi-surface sensor pre-check; tolerate failures the same
    way the single-image path did.

    Surface ids are `panel_0`, `panel_1`, … so downstream code can map
    a surface back to its panel by parsing the id. A failure inside the
    pre-check is downgraded to "degraded" rather than killing the
    request — the rule engine's own confidence-aware degradation
    catches genuinely unreadable extractions.
    """
    if not panels:
        return CaptureQualityReport(
            surfaces=[], overall_verdict="degraded", overall_confidence=0.0
        )
    images = {f"panel_{i}": p.image_bytes for i, p in enumerate(panels)}
    try:
        report = assess_capture_quality(images)
    except Exception:
        return CaptureQualityReport(
            surfaces=[], overall_verdict="degraded", overall_confidence=0.0
        )
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


def _merge_panel_extractions(
    per_panel: list[Any],
) -> Any:
    """Merge per-panel `VisionExtraction`s into a single extraction.

    Field-by-field: highest-confidence wins. The winning field's
    `source_image_id` is rewritten so the response can label which
    image each value came from. Single-panel inputs (the v1 mobile
    happy path) are tagged `"panorama"`; multi-panel inputs use
    `"panel_<index>"`.

    `unreadable` reports per-field whether the field was not found on
    *any* panel — a field that any panel read successfully is no longer
    "unreadable" overall, since we have a value for it. Otherwise the
    intersection of unreadable lists shows which fields nothing on the
    bottle covered.

    Top-level signals (`image_quality`, `image_quality_notes`,
    `beverage_type_observed`) are aggregated:
      * image_quality → worst across panels (pessimistic, same rationale
        as `_worse_quality`).
      * image_quality_notes → concatenated with panel labels so the user
        sees per-panel diagnostics.
      * beverage_type_observed → the strongest non-`unknown` consensus,
        else `unknown`. Disagreement is recorded in image_quality_notes
        upstream by the model itself.
    """
    from app.services.vision import VisionExtraction
    from dataclasses import replace as dc_replace

    if not per_panel:
        return VisionExtraction(
            fields={}, unreadable=[], raw_response="", image_quality=None
        )
    if len(per_panel) == 1:
        # Single-panel call (the v1 mobile path): retag every field with
        # the synthetic `"panorama"` surface so the API response carries
        # a stable identifier the client can join against the uploaded
        # image. Multi-panel callers fall through to the per-index path.
        single = per_panel[0]
        retagged = {
            name: dc_replace(f, source_image_id=PANORAMA_SURFACE)
            for name, f in single.fields.items()
        }
        return VisionExtraction(
            fields=retagged,
            unreadable=list(single.unreadable),
            raw_response=single.raw_response,
            image_quality=single.image_quality,
            image_quality_notes=single.image_quality_notes,
            beverage_type_observed=single.beverage_type_observed,
        )

    merged_fields: dict[str, Any] = {}
    unreadable_per_field: dict[str, int] = {}
    for idx, ex in enumerate(per_panel):
        for name, f in ex.fields.items():
            tagged = dc_replace(f, source_image_id=f"panel_{idx}")
            existing = merged_fields.get(name)
            if existing is None or tagged.confidence > existing.confidence:
                merged_fields[name] = tagged
        for name in ex.unreadable:
            unreadable_per_field[name] = unreadable_per_field.get(name, 0) + 1

    # Only fields that NO panel produced a value for stay unreadable.
    final_unreadable = [
        name for name in unreadable_per_field if name not in merged_fields
    ]

    # Worst image_quality across panels.
    qualities = [e.image_quality for e in per_panel if e.image_quality]
    if qualities:
        worst = qualities[0]
        for q in qualities[1:]:
            worst = _worse_quality(worst, q)
        merged_quality: str | None = worst
    else:
        merged_quality = None

    # Per-panel notes get a "[panel N] " prefix so the user can tell which
    # diagnosis applies to which face. Empty notes drop out.
    note_parts: list[str] = []
    for idx, ex in enumerate(per_panel):
        if ex.image_quality_notes:
            note_parts.append(f"[panel {idx}] {ex.image_quality_notes}")
    merged_notes = " | ".join(note_parts) if note_parts else None

    # Bev consensus: pick the most common non-"unknown" reading.
    seen: dict[str, int] = {}
    for ex in per_panel:
        bev = ex.beverage_type_observed
        if bev and bev != "unknown":
            seen[bev] = seen.get(bev, 0) + 1
    if seen:
        merged_bev = max(seen.items(), key=lambda kv: kv[1])[0]
    else:
        merged_bev = "unknown"

    return VisionExtraction(
        fields=merged_fields,
        unreadable=final_unreadable,
        raw_response="",  # per-panel raw payloads are large; debug via logs
        image_quality=merged_quality,
        image_quality_notes=merged_notes,
        beverage_type_observed=merged_bev,
    )


def _pick_best_secondary_warning(
    per_panel_secondary: list[Any],
) -> Any:
    """Select the strongest second-pass warning read across panels.

    Preference order:
      1. A read with `found=True` and the highest confidence — this is
         the best we have on what the warning actually says.
      2. A read with `found=False` (the model looked, didn't see one) —
         indicates the warning isn't on any panel, which the cross-check
         can use to leave the engine's "missing warning" verdict alone.
      3. `None` — every second-pass call failed or was disabled.

    Single-panel: returns that panel's read unchanged. Multi-panel: lets
    us run the second-pass on every panel concurrently without having
    to know in advance which one carries the warning.
    """
    if not per_panel_secondary:
        return None
    found_reads = [r for r in per_panel_secondary if r is not None and getattr(r, "found", False)]
    if found_reads:
        return max(found_reads, key=lambda r: getattr(r, "confidence", 0.0))
    not_found_reads = [r for r in per_panel_secondary if r is not None]
    if not_found_reads:
        return not_found_reads[0]
    return None


def _finalize(
    report: VerifyReport,
    *,
    cache: VerifyCache | None,
    cache_key: str | None,
    extractor_label: str = "primary",
) -> VerifyReport:
    """Single seam where cold-path results land in the cache + observability.

    Every cold-path return funnels through here so the three exit points
    (sensor-unreadable short-circuit, foreign-language short-circuit,
    full-success) cache identically AND emit one structured log line +
    one stats bump. `cache_hit` is left at its dataclass default of False
    — the *next* request that resolves to a hit will flip it via
    `restamp_report` + assignment in the fast path.
    """
    if cache is not None and cache_key is not None:
        cache.put(cache_key, report)

    # Stats + structured log. Kept inside _finalize so every cold exit
    # (full-success, foreign-language refusal, sensor-unreadable
    # short-circuit) records identically without forcing every caller
    # to remember to bump.
    from app.services import verify_stats
    from app.telemetry import current_trace_id

    verify_stats.record_cold(elapsed_ms=report.elapsed_ms, overall=report.overall)
    cross_check_outcome = (
        report.health_warning_cross_check.get("outcome")
        if isinstance(report.health_warning_cross_check, dict)
        else None
    )
    logger.info(
        "verify cold path=cold elapsed_ms=%d overall=%s image_quality=%s "
        "cross_check=%s extractor=%s n_rules=%d n_unreadable=%d trace_id=%s",
        report.elapsed_ms,
        report.overall,
        report.image_quality,
        cross_check_outcome,
        extractor_label,
        len(report.rule_results),
        len(report.unreadable_fields),
        current_trace_id() or "-",
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
    """Pack the primary extractor's health-warning reading as a WarningRead.

    Infers `region_visible` from the model's image_quality_notes: when the
    notes mention a glared/obscured warning region, we treat that as a
    region_visible=true signal even though the field came back unreadable.
    The primary extractor doesn't have a structured `region_visible`
    field (its schema covers seven fields and complicating it costs
    latency on every label), so we lift the signal out of free text.
    Anything matching `warning` near `glar(e|ed)`, `obscur`, `washed`,
    `cover`, or `hidden` counts.
    """
    from app.services.health_warning_second_pass import WarningRead

    field_obj = ctx.fields.get("health_warning")
    if field_obj is not None and field_obj.value:
        return WarningRead(
            value=field_obj.value,
            found=True,
            confidence=field_obj.confidence,
            source="primary_extractor",
            region_visible=True,
        )
    region_visible = _primary_notes_indicate_warning_obstruction(
        getattr(extraction, "image_quality_notes", None)
    )
    return WarningRead(
        value=None,
        found=False,
        confidence=0.0,
        source="primary_extractor",
        region_visible=region_visible,
    )


def _warning_signals_english(
    second_warning_read: Any | None, image_quality_notes: str | None
) -> bool:
    """Independent evidence that the label is English-with-obstruction
    rather than foreign-language.

    The foreign-language guard fires when the primary's extracted text
    is too sparse to confirm English vocabulary — but a glared warning
    is exactly the case where the primary's text WILL be sparse (other
    fields readable, the warning paragraph blocked by glare). Without
    this suppression, every glared English label gets flagged as
    foreign-language and the cross-check never runs.

    Strong-evidence signals (any one suppresses the guard):
      * second-pass found warning text (anything from a fragment up to
        the full canonical text) — recognizable English compliance
        vocabulary by definition
      * second-pass set region_visible=true — the model recognized a
        warning-shaped block on the label, which is uniquely English
        (Spanish/French/German labels have their own native warnings)
      * primary's image_quality_notes mention a glared/obscured warning
        region — same signal, lifted from free text
    """
    if second_warning_read is not None:
        if getattr(second_warning_read, "found", False):
            return True
        if getattr(second_warning_read, "region_visible", False):
            return True
    if _primary_notes_indicate_warning_obstruction(image_quality_notes):
        return True
    return False


def _primary_notes_indicate_warning_obstruction(notes: str | None) -> bool:
    """True when the primary extractor's free-text notes describe an
    obstructed warning region.

    Conservative pattern: a sentence/clause must mention a warning concept
    AND an obstruction concept. This avoids the false-positive where the
    notes say "warning region clear; brand glared" — that's the brand
    that's glared, not the warning.
    """
    if not notes:
        return False
    import re as _re
    lowered = notes.lower()
    if "warning" not in lowered and "fine print" not in lowered:
        return False
    obstruction_terms = (
        r"glar(e|ed|ing)|obscur|washed|cover(?:ed|ing)?|"
        r"hidden|specular|saturat|smudg|illegible|unreadab"
    )
    # Look for an obstruction term within 60 characters of "warning" /
    # "fine print" — same clause, in the noise-prose of a one-line note.
    for m in _re.finditer(r"warning|fine print", lowered):
        window = lowered[max(0, m.start() - 60) : m.end() + 60]
        if _re.search(obstruction_terms, window):
            return True
    return False


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
      * unverifiable_obstructed → downgrade to ADVISORY: the warning may
        be hidden under glare/blur/occlusion, so a "missing warning" or
        edit-distance FAIL would be a confident wrong-fail
      * primary_only / no_warning_present → leave the engine's call alone
    """
    target_id_substring = "health_warning"
    if cc.outcome not in {"disagreement", "unverifiable_obstructed"}:
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
                    surface=r.surface,
                )
            )
            continue
        out.append(r)
    return out


def _build_obstruction_signal(
    capture: CaptureQualityReport,
) -> ObstructionSignal:
    """Translate the sensor pre-check into the cross-check's obstruction signal.

    The warning is usually on the back panel and lives in fine-print at the
    bottom or side of the label. Any of the following on ANY panel makes the
    no-warning conclusion suspect — recall on the warning is load-bearing,
    so we err on the side of "could be obstructed":

      * label_verdict is degraded or unreadable on any panel
      * glare blobs whose total area covers ≥ 5 % of the label region
      * label-region glare fraction over the GLARE_DEGRADED threshold
      * motion_blur_direction is set on a panel whose label is degraded

    Each branch produces a human-readable reason that flows into the
    cross-check's finding string so the agent UI can tell the user why
    the rule is advisory rather than the dread "couldn't verify" stub.
    """
    from app.services.sensor_check import GLARE_DEGRADED

    reasons: list[str] = []
    for s in capture.surfaces:
        panel_label = s.surface or "panel"
        if s.label_verdict in {"degraded", "unreadable"}:
            reasons.append(
                f"{panel_label} label region grades '{s.label_verdict}'"
            )
        elif s.verdict in {"degraded", "unreadable"}:
            # Frame-level degradation without a label-region verdict
            # (no label localized) — still relevant: the glare/blur is
            # SOMEWHERE in the frame and we couldn't isolate the label,
            # so we cannot rule out obstruction over the warning.
            reasons.append(
                f"{panel_label} frame grades '{s.verdict}'"
            )
        if s.glare_blobs:
            label_glare = sum(b.area_fraction_label for b in s.glare_blobs)
            if label_glare >= 0.05:
                reasons.append(
                    f"{panel_label}: glare blobs cover "
                    f"{label_glare * 100:.0f}% of the label region"
                )
        metrics_label = s.metrics_label
        if (
            metrics_label is not None
            and metrics_label.glare_fraction > GLARE_DEGRADED
        ):
            reasons.append(
                f"{panel_label} label region is "
                f"{metrics_label.glare_fraction * 100:.0f}% saturated"
            )
        if s.motion_blur_direction and s.label_verdict in {"degraded", "unreadable"}:
            reasons.append(
                f"{panel_label}: {s.motion_blur_direction} motion blur over "
                "the label"
            )
        if s.backlit:
            reasons.append(f"{panel_label} is backlit (label silhouetted)")

    if not reasons:
        return ObstructionSignal.clear()
    return ObstructionSignal(
        is_obstructed=True,
        # Deduplicate while preserving order — the same panel can produce
        # multiple reasons (e.g. degraded AND glare AND backlit) and the
        # agent UI doesn't need to see them stacked verbatim.
        reason="; ".join(dict.fromkeys(reasons)),
    )


def _downgrade_missing_warning_under_obstruction(
    results: list[RuleResult],
    ctx: ExtractionContext,
    obstruction: ObstructionSignal,
) -> list[RuleResult]:
    """Refuse to FAIL on a missing Government Warning when the capture
    report shows obstruction over the label.

    Backstops two paths the cross-check doesn't cover:

      1. The no-second-pass path: when `verify()` is called without a
         `health_warning_reader`, there is no cross-check at all — but the
         engine can still FAIL on a missing warning if the primary
         extractor produced no value, and that FAIL would be a confident
         wrong-fail under glare just as surely.
      2. Cross-check returned `no_warning_present` because both reads
         said no-warning AND neither set region_visible AND there was
         no obstruction signal at the cross-check level — but a frame-
         level obstruction we computed AFTER the cross-check still
         applies. (Defense in depth: the cross-check's obstruction-
         signal path is the primary defense; this is the safety net.)

    Recognizes a "warning missing" FAIL by:
      * rule_id contains "health_warning"
      * status is FAIL
      * the health_warning field was unreadable / absent (i.e. the FAIL
        is on missing text rather than on edit-distance against text we
        actually read)

    An edit-distance FAIL on actual text is NOT downgraded here: if we
    read characters successfully and they don't match canonical, that's
    a substantive non-compliance we want to surface. Only the "I see
    nothing" branch becomes advisory under obstruction.
    """
    if not obstruction.is_obstructed:
        return results
    health_warning_field = ctx.fields.get("health_warning")
    has_text = (
        health_warning_field is not None
        and health_warning_field.value
        and health_warning_field.value.strip()
    )
    if has_text:
        # The reader saw real text; an edit-distance FAIL on that text is
        # still a substantive verdict. Don't downgrade.
        return results

    out: list[RuleResult] = []
    for r in results:
        if "health_warning" not in r.rule_id or r.status != CheckOutcome.FAIL:
            out.append(r)
            continue
        out.append(
            RuleResult(
                rule_id=r.rule_id,
                rule_version=r.rule_version,
                citation=r.citation,
                status=CheckOutcome.ADVISORY,
                finding=(
                    "Couldn't verify the Government Warning — the warning "
                    "field came back empty, but the capture report shows "
                    f"obstruction over the label ({obstruction.reason}). "
                    "Reshoot before relying on this verdict."
                ),
                expected=r.expected,
                fix_suggestion=r.fix_suggestion,
                bbox=r.bbox,
                surface=r.surface,
            )
        )
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
        "region_visible": getattr(read, "region_visible", False),
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
                surface=r.surface,
            )
        )
    return out


def _downgrade_fails_for_glare_blob(
    results: list[RuleResult],
    ctx: ExtractionContext,
    surfaces_by_panel: dict[int, SurfaceCaptureQuality],
) -> list[RuleResult]:
    """FAIL → ADVISORY when the rule's field bbox sits inside a glare blob.

    Mirrors `pipeline._apply_capture_downgrade`'s `blob_occluded` branch
    (`backend/app/services/pipeline.py:317-322`) so verify and scans
    apply the same fail-honestly recall guarantee on a saturated label
    region. Without this, a brewer's `/v1/verify` request can FAIL on a
    rule whose evidence sat inside a specular highlight — exactly the
    confident-wrong outcome SPEC §0.5 forbids.

    The bbox the field carried (now in original-image coordinates after
    `_translate_merged_bboxes`) is compared against the source panel's
    glare blobs. Tolerates fields with no `source_image_id` (no-op),
    fields that didn't extract a bbox (no-op), and surfaces with no
    blobs (no-op).
    """
    if not surfaces_by_panel:
        return results
    out: list[RuleResult] = []
    for r in results:
        if r.status != CheckOutcome.FAIL:
            out.append(r)
            continue
        field_name = _field_name_for_rule(r.rule_id)
        f = ctx.fields.get(field_name) if field_name else None
        if f is None or f.bbox is None:
            out.append(r)
            continue
        idx = _panel_index_from_source(f.source_image_id)
        sq = surfaces_by_panel.get(idx) if idx is not None else None
        if sq is None or not sq.glare_blobs:
            out.append(r)
            continue
        if not _bbox_inside_glare(f.bbox, sq.glare_blobs):
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
                    + " · downgraded to advisory because the field's region"
                    " overlaps a specular glare blob; reshoot with the bottle"
                    " tilted away from the light source before relying on"
                    " this verdict."
                ),
                expected=r.expected,
                fix_suggestion=r.fix_suggestion,
                bbox=r.bbox,
                surface=r.surface,
            )
        )
    return out


def _field_name_for_rule(rule_id: str) -> str | None:
    """`spirits.health_warning.exact_text` → `health_warning`.

    Mirrors `pipeline._field_referenced` so the verify-path blob
    downgrade keys off the same field name as the scan path.
    """
    parts = rule_id.split(".")
    if len(parts) < 3:
        return None
    return parts[1]


def _bbox_inside_glare(
    field_bbox: tuple[int, int, int, int] | None,
    glare_blobs,
) -> bool:
    """True when ≥30 % of the field bbox falls inside any glare blob.

    Mirrors `pipeline._bbox_inside_glare` exactly (same threshold, same
    geometry) so the verify and scan paths agree on what counts as
    "occluded by glare".
    """
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
