"""POST /v1/verify — single-shot label verification used by the agent UI.

Multipart request: image + producer record (JSON) + beverage type. Synchronous
response: per-rule verdict, extracted fields, overall status.

v1 mobile happy path uploads a single unrolled-label panorama image
(`images=[panorama.jpg]`); the response tags every rule result and field
with `surface="panorama"` / `source_image_id="panorama"`. The web/agent
UI may still post multiple panels via `images=`; those go through the
multi-panel orchestrator and report per-panel `panel_N` ids.
"""

from __future__ import annotations

import asyncio
import json
import threading

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.services.anthropic_client import ExtractorUnavailable
from app.services.health_warning_second_pass import (
    ClaudeHealthWarningExtractor,
    HealthWarningExtractor,
)
from app.services.reverse_lookup import ReverseLookupCache
from app.services.verify import Panel, VerifyInput, verify
from app.services.verify_cache import VerifyCache
from app.services.vision import VisionExtractor, get_default_extractor

router = APIRouter(prefix="/verify", tags=["verify"])

# 503 payload shape returned when every configured vision extractor is
# unreachable (or none is configured). The mobile/web UI maps the `code`
# to a friendly message rather than dumping `detail` verbatim, so the
# detail is allowed to be developer-facing.
_SERVICE_UNAVAILABLE_CODE = "vision_unavailable"
# 422 payload code for beverage_type values the rule engine does not yet
# cover — same `{code, message}` shape as the 503 above so the client only
# has one error envelope to parse. SPEC §1.2 lists wine + spirits as v2;
# we ship spirits in v1 and gate wine here until the wine rule set lands.
_UNSUPPORTED_BEVERAGE_CODE = "beverage_type_unsupported"
# 413 payload code for uploads larger than `settings.max_image_bytes`. Same
# `{code, message}` envelope so a single client handler covers every error
# mode — UI maps the code to a friendly "image too large" affordance.
_IMAGE_TOO_LARGE_CODE = "image_too_large"
# 504 payload code for requests that exceed `settings.verify_request_timeout_s`.
# Distinct from 503 (vision_unavailable) because the failure mode is
# different — a timeout means the upstream is slow, not absent.
_VERIFY_TIMEOUT_CODE = "verify_timeout"


# Beverage types accepted by /v1/verify. Wine is intentionally omitted —
# the rule engine has no wine.yaml in v1, so accepting wine here would
# 500 inside `verify()` (`load_rules` returns []). SPEC §1.2 schedules
# wine for v2; the verify path mirrors the scans-API gate at
# `app/api/scans.py:create_scan`.
_BEVERAGE_TYPES = {"beer", "spirits"}


class RuleResultDTO(BaseModel):
    rule_id: str
    rule_version: int
    citation: str
    status: str
    finding: str | None = None
    expected: str | None = None
    fix_suggestion: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    # Which submitted image this rule's evidence was read from. Matches
    # the `source_image_id` shape on `FieldExtractionDTO`. The v1 mobile
    # single-shot happy path reports `"panorama"`; the multi-panel path
    # (web UI / agent) reports `"panel_0"`, `"panel_1"`, … in submission
    # order. `None` when the rule isn't tied to a specific field or when
    # the field had no source_image_id. Mobile uses this to know which
    # captured image to highlight when the user taps a result.
    surface: str | None = None


class FieldExtractionDTO(BaseModel):
    value: str | None = None
    confidence: float = 0.0
    bbox: list[int] | None = None
    unreadable: bool = False
    # Which submitted image this field's value came from. The v1 mobile
    # single-shot happy path reports `"panorama"` (one unrolled label
    # image). The multi-panel path reports `"panel_0"`, `"panel_1"`, …
    # in submission order. The UI uses this to know which captured
    # image the bbox is relative to and (multi-panel) to render
    # per-source labels alongside the extracted text.
    source_image_id: str | None = None


class VerifyResponse(BaseModel):
    overall: str
    rule_results: list[RuleResultDTO]
    extracted: dict[str, FieldExtractionDTO]
    unreadable_fields: list[str]
    image_quality: str = "good"
    image_quality_notes: str | None = None
    elapsed_ms: int
    # True when this verdict came from the in-process cache rather than a
    # fresh VLM call. The UI uses it to render an "instant" affordance
    # during the iterative-design workflow (re-submitting the same
    # artwork export) and to suppress the spinner that the cold path needs.
    cache_hit: bool = False
    # SPEC §0.5: outcome of the redundant Government-Warning second pass.
    # `None` when the second-pass reader is disabled or unconfigured;
    # otherwise a payload like:
    #   {"outcome": "confirmed_compliant" | "confirmed_noncompliant" |
    #               "disagreement" | "primary_only" | "no_warning_present",
    #    "edit_distance_to_canonical": int | None,
    #    "edit_distance_between_reads": int | None,
    #    "notes": str,
    #    "primary": {"value": str|None, "found": bool, "confidence": float, "source": str} | None,
    #    "secondary": {...} | None}
    # The UI surfaces `outcome` and `notes` to give the operator visibility
    # into why a warning rule was downgraded to advisory; dashboards key off
    # `outcome` to track inter-pass agreement rate over time.
    health_warning_cross_check: dict | None = None


# Module-level extractor cache so we don't recreate the Anthropic client per
# request. Set by `get_extractor()`; tests can override via dependency_overrides
# or by patching this attribute directly.
_extractor_cache: VisionExtractor | None = None

# Same idea for the redundant Government-Warning second-pass reader. The
# Anthropic SDK pools HTTP connections under the hood, so reusing one
# client across requests keeps TLS/keep-alive warm — that matters because
# the second pass runs concurrently with the primary on every cold call.
_health_warning_extractor_cache: HealthWarningExtractor | None = None

# Process-wide verify-result cache. Lazily constructed so importing this
# module doesn't allocate the LRU on cold start. Tests reset it via the
# autouse fixture in `tests/test_verify_api.py`. The lock guards the
# check-then-set against concurrent cold requests racing into the
# factory and each constructing their own cache (one would survive,
# the others' puts would silently disappear).
_verify_cache: VerifyCache | None = None
_verify_cache_lock = threading.Lock()

# Process-wide reverse-image lookup cache. Same lazy + lock pattern as
# the byte-exact verify cache above. Sized larger (4096 entries × ~5 KB
# extraction snapshots ≈ 20 MB worst-case) than the byte-exact cache
# because perceptual entries pay off across more workflows: a brewery
# with 200 SKUs and a year of iterative artwork churn can plausibly
# accumulate thousands of historically-verified labels worth keeping
# resident, where the byte-exact cache only buys you the most recent
# few-dozen exports per active SKU.
_reverse_lookup_cache: ReverseLookupCache | None = None
_reverse_lookup_cache_lock = threading.Lock()


def get_extractor() -> VisionExtractor:
    """Lazily construct (and cache) the configured vision chain.

    `ExtractorUnavailable` is intentionally NOT cached — that lets a missing
    env var be fixed and recovered without restarting the process. A
    `RuntimeError` (config-bug like an unknown extractor name) IS effectively
    cached because this function will keep raising it on every call.
    """
    global _extractor_cache
    if _extractor_cache is None:
        _extractor_cache = get_default_extractor()
    return _extractor_cache


def _override_extractor(extractor: VisionExtractor | None) -> None:
    """Test hook: install (or clear with None) a custom extractor."""
    global _extractor_cache
    _extractor_cache = extractor


def get_health_warning_extractor() -> HealthWarningExtractor | None:
    """Lazily construct the redundant Government-Warning reader.

    Returns None when the feature flag is off or the API key is missing —
    `verify()` already tolerates a None reader by falling back to the
    primary-only verdict, so an absent second pass degrades gracefully
    rather than failing the request.
    """
    global _health_warning_extractor_cache
    if not settings.enable_health_warning_second_pass:
        return None
    if not settings.anthropic_api_key:
        # No key → no second pass; primary path still serves verdicts.
        return None
    if _health_warning_extractor_cache is None:
        try:
            _health_warning_extractor_cache = ClaudeHealthWarningExtractor()
        except ExtractorUnavailable:
            # Same handling as the missing-key branch above: degrade to
            # primary-only rather than failing the verify request.
            return None
    return _health_warning_extractor_cache


def _override_health_warning_extractor(
    extractor: HealthWarningExtractor | None,
) -> None:
    """Test hook: install (or clear with None) a custom second-pass reader."""
    global _health_warning_extractor_cache
    _health_warning_extractor_cache = extractor


def get_verify_cache() -> VerifyCache | None:
    """Lazily build the per-process verify cache.

    Sized for a typical iterative session (a brewer re-submitting the
    same Illustrator export 10–50 times across an hour) and several
    such sessions in parallel. 1024 entries × ~100 KB each is well
    inside the per-machine RSS budget. Returns None when
    `verify_cache_max_entries` is 0 — the operator's escape hatch when
    they need to debug a stuck cache without redeploying.

    Construction is double-checked under a lock so two concurrent cold
    requests can't each create their own cache (the second would
    overwrite the first's puts).
    """
    global _verify_cache
    if settings.verify_cache_max_entries <= 0:
        return None
    if _verify_cache is not None:
        return _verify_cache
    with _verify_cache_lock:
        if _verify_cache is None:
            _verify_cache = VerifyCache(
                max_entries=settings.verify_cache_max_entries
            )
    return _verify_cache


def _reset_verify_cache() -> None:
    """Test hook: drop the process-wide verify cache.

    Tests assert behavior on cold→warm transitions and stale-rule
    invalidation, which means each test must start with an empty
    cache. Production code never calls this.
    """
    global _verify_cache
    with _verify_cache_lock:
        _verify_cache = None


def get_reverse_lookup_cache() -> ReverseLookupCache | None:
    """Lazily build the per-process reverse-image lookup cache.

    Returns None when `reverse_lookup_max_entries` is 0 — the operator's
    escape hatch for disabling the perceptual layer without redeploying
    (e.g. while triaging a suspected false-positive verdict reuse).
    Construction is double-checked under a lock for the same reason
    `get_verify_cache()` is — two cold requests racing into the factory
    would each construct their own instance and one's promotions would
    silently disappear.
    """
    global _reverse_lookup_cache
    if settings.reverse_lookup_max_entries <= 0:
        return None
    if _reverse_lookup_cache is not None:
        return _reverse_lookup_cache
    with _reverse_lookup_cache_lock:
        if _reverse_lookup_cache is None:
            _reverse_lookup_cache = ReverseLookupCache(
                max_entries=settings.reverse_lookup_max_entries,
                hamming_threshold=settings.reverse_lookup_hamming_threshold,
            )
    return _reverse_lookup_cache


def _reset_reverse_lookup_cache() -> None:
    """Test hook: drop the process-wide reverse-lookup cache."""
    global _reverse_lookup_cache
    with _reverse_lookup_cache_lock:
        _reverse_lookup_cache = None


@router.post("", response_model=VerifyResponse)
async def verify_label(
    image: UploadFile | None = File(None),
    images: list[UploadFile] | None = File(None),
    beverage_type: str = Form(...),
    container_size_ml: int = Form(...),
    is_imported: bool = Form(False),
    application: str = Form(...),
) -> VerifyResponse:
    if beverage_type == "wine":
        # Distinct 422 (rather than the generic 400 below) so the UI can
        # surface a friendly "wine support arrives in v2" affordance
        # instead of treating this as a malformed request.
        raise HTTPException(
            status_code=422,
            detail={
                "code": _UNSUPPORTED_BEVERAGE_CODE,
                "message": (
                    "Wine support arrives in v2; submit a beer or spirits "
                    "label today."
                ),
            },
        )
    if beverage_type not in _BEVERAGE_TYPES:
        raise HTTPException(400, f"beverage_type must be one of {sorted(_BEVERAGE_TYPES)}")
    if container_size_ml <= 0 or container_size_ml > 10_000:
        raise HTTPException(400, "container_size_ml must be between 1 and 10000")

    try:
        application_obj = json.loads(application)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"application must be valid JSON: {exc}") from exc
    if not isinstance(application_obj, dict):
        raise HTTPException(400, "application must be a JSON object")

    # Multi-panel input: `images=` (list) takes precedence; the legacy
    # `image=` (single) still works for callers that only have one
    # face. At least one is required. We cap the panel count to keep
    # a malicious or buggy client from blowing up the thread pool —
    # 4 is enough for front + back + neck + base on the worst-shaped
    # bottle, and beyond that the UX stops being "snap two photos".
    submitted: list[UploadFile] = []
    if images:
        submitted = [u for u in images if u is not None]
    if not submitted and image is not None:
        submitted = [image]
    if not submitted:
        raise HTTPException(
            400,
            "At least one image is required: provide `image` (single) "
            "or `images` (multipart list).",
        )
    if len(submitted) > 4:
        raise HTTPException(
            400,
            f"Too many panels: at most 4 supported, got {len(submitted)}.",
        )

    # Per-upload size cap. Reject *before* `read()` materializes the bytes
    # so a 50 MB POST can't blow the worker's RSS on a 256 MB Fly/Railway
    # machine. Uvicorn already buffers the upload to memory once `read()`
    # is called; checking `upload.size` first lets us 413 the request
    # without ever touching that buffer.
    max_bytes = settings.max_image_bytes
    panels: list[Panel] = []
    for idx, upload in enumerate(submitted):
        # `upload.size` is populated by Starlette from the multipart parser;
        # `None` means streaming-mode (rare for our flow, but defensive
        # branch reads + counts).
        declared = getattr(upload, "size", None)
        if declared is not None and declared > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": _IMAGE_TOO_LARGE_CODE,
                    "message": (
                        f"Image upload {idx} is {declared} bytes; the maximum "
                        f"is {max_bytes} bytes. Resize before submitting."
                    ),
                },
            )
        panel_bytes = await upload.read()
        if len(panel_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": _IMAGE_TOO_LARGE_CODE,
                    "message": (
                        f"Image upload {idx} is {len(panel_bytes)} bytes; the "
                        f"maximum is {max_bytes} bytes. Resize before submitting."
                    ),
                },
            )
        if not panel_bytes:
            raise HTTPException(
                400, f"image upload {idx} is empty"
            )
        panel_media = upload.content_type or "image/png"
        if not panel_media.startswith("image/"):
            raise HTTPException(
                400,
                f"image upload {idx} content_type must be image/*, got "
                f"{panel_media!r}",
            )
        panels.append(Panel(image_bytes=panel_bytes, media_type=panel_media))

    try:
        extractor = get_extractor()
    except ExtractorUnavailable as exc:
        # No backend is currently configurable / reachable. 503 (not 500) so
        # the client treats this as transient and the UI can show a clean
        # retry affordance rather than an ominous 500-with-stack-trace.
        raise HTTPException(
            status_code=503,
            detail={
                "code": _SERVICE_UNAVAILABLE_CODE,
                "message": str(exc),
            },
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    inp = VerifyInput(
        image_bytes=panels[0].image_bytes,
        media_type=panels[0].media_type,
        beverage_type=beverage_type,
        container_size_ml=container_size_ml,
        is_imported=is_imported,
        application=application_obj,
        extra_panels=panels[1:],
    )

    # `verify()` is a synchronous orchestrator that issues blocking HTTP
    # calls (Anthropic SDK + the second-pass reader). Run it off the event
    # loop so a single in-flight verification doesn't park every other
    # request on this worker — the cold path can take 2–4 s, and FastAPI
    # serves async handlers on the loop directly. Inside `verify()` the
    # primary extractor and the second-pass reader run concurrently on
    # their own thread pool.
    second_pass = get_health_warning_extractor()
    try:
        # `asyncio.wait_for` enforces a wall-clock cap on the whole verify
        # call so a flaky upstream cannot chain SDK retries together (each
        # SDK call is 20 s × 2 retries → 60 s+ worst case) and exhaust
        # uvicorn's worker pool. SPEC §0 caps the success path at 25 s p95;
        # this is the safety net for the worst-case, not the success budget.
        report = await asyncio.wait_for(
            asyncio.to_thread(
                verify,
                inp,
                extractor=extractor,
                health_warning_reader=second_pass,
                cache=get_verify_cache(),
                reverse_cache=get_reverse_lookup_cache(),
            ),
            timeout=settings.verify_request_timeout_s,
        )
    except asyncio.TimeoutError as exc:
        # 504 (Gateway Timeout) so the client distinguishes "vision backend
        # was slow" from "vision backend was missing" (503). Same envelope
        # shape so a single error handler covers both.
        raise HTTPException(
            status_code=504,
            detail={
                "code": _VERIFY_TIMEOUT_CODE,
                "message": (
                    f"Verification exceeded the "
                    f"{settings.verify_request_timeout_s:.0f}s wall-clock cap. "
                    f"The vision backend may be degraded; retry shortly."
                ),
            },
        ) from exc
    except ExtractorUnavailable as exc:
        # Every backend in the chain failed at request time (e.g. Anthropic
        # rate-limit + Qwen socket error). Same 503 shape as the construction
        # failure above so the client only handles one transient mode.
        raise HTTPException(
            status_code=503,
            detail={
                "code": _SERVICE_UNAVAILABLE_CODE,
                "message": (
                    "Every vision backend is currently unreachable. "
                    f"Last error: {exc}"
                ),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover — defensive fallback
        raise HTTPException(500, f"verification failed: {exc}") from exc

    return VerifyResponse(
        overall=report.overall,
        rule_results=[
            RuleResultDTO(
                rule_id=r.rule_id,
                rule_version=r.rule_version,
                citation=r.citation,
                status=r.status.value,
                finding=r.finding,
                expected=r.expected,
                fix_suggestion=r.fix_suggestion,
                bbox=r.bbox,
                surface=r.surface,
            )
            for r in report.rule_results
        ],
        extracted={
            name: FieldExtractionDTO(
                value=info.get("value"),
                confidence=info.get("confidence", 0.0),
                bbox=info.get("bbox"),
                unreadable=info.get("unreadable", False),
                source_image_id=info.get("source_image_id"),
            )
            for name, info in report.extracted.items()
        },
        unreadable_fields=report.unreadable_fields,
        image_quality=report.image_quality,
        image_quality_notes=report.image_quality_notes,
        elapsed_ms=report.elapsed_ms,
        cache_hit=report.cache_hit,
        health_warning_cross_check=report.health_warning_cross_check,
    )


class VerifyStatsResponse(BaseModel):
    """Snapshot of in-process verify-pipeline counters.

    Cheap to compute (counters live in memory) and safe to poll. Intended
    for an admin dashboard / on-call quick-check; not a replacement for
    OpenTelemetry → Honeycomb (SPEC §0). Counters reset on process restart;
    `cold_count + warm_count + reverse_lookup_hits` is the number of
    /v1/verify calls served by THIS instance since boot.
    """

    cold_count: int
    warm_count: int
    cold_elapsed_ms_recent: list[int]
    warm_elapsed_ms_recent: list[int]
    second_pass_outcomes: dict[str, int]
    overall_verdicts: dict[str, int]
    cache: dict[str, int] | None = None
    reverse_lookup: dict[str, int] | None = None
    reverse_lookup_elapsed_ms_recent: list[int] = Field(default_factory=list)


@router.get("/_stats", response_model=VerifyStatsResponse)
async def verify_stats_endpoint() -> VerifyStatsResponse:
    """Operator-facing peek at this instance's verify counters.

    Not gated yet — tag for admin auth when the auth middleware lands. The
    payload is read-only (snapshots only; never mutates) so the worst a
    leaked URL can do is leak rough rate-of-traffic and verdict mix.
    """
    from app.services.verify_stats import snapshot

    snap = snapshot()
    cache_payload: dict[str, int] | None = None
    cache = get_verify_cache()
    if cache is not None:
        cs = cache.stats()
        cache_payload = {
            "hits": cs.hits,
            "misses": cs.misses,
            "size": cs.size,
            "max_entries": cs.max_entries,
        }
    reverse_payload: dict[str, int] | None = None
    rcache = get_reverse_lookup_cache()
    if rcache is not None:
        rs = rcache.stats()
        # Two views of the same data: the cache's own internal counters
        # (`hits`/`misses` from `ReverseLookupCache.stats()`) plus the
        # verify-orchestrator's view from `verify_stats` (which excludes
        # misses caused by failed dhash). Surfacing both lets operators
        # diagnose phash failures separately from real lookup misses.
        reverse_payload = {
            "hits": rs.hits,
            "misses": rs.misses,
            "size": rs.size,
            "max_entries": rs.max_entries,
            "hamming_threshold": rs.hamming_threshold,
            "orchestrator_hits": snap.reverse_lookup_hits,
            "orchestrator_misses": snap.reverse_lookup_misses,
        }
    return VerifyStatsResponse(
        cold_count=snap.cold_count,
        warm_count=snap.warm_count,
        cold_elapsed_ms_recent=snap.cold_elapsed_ms_recent,
        warm_elapsed_ms_recent=snap.warm_elapsed_ms_recent,
        second_pass_outcomes=snap.second_pass_outcomes,
        overall_verdicts=snap.overall_verdicts,
        cache=cache_payload,
        reverse_lookup=reverse_payload,
        reverse_lookup_elapsed_ms_recent=snap.reverse_lookup_elapsed_ms_recent,
    )
