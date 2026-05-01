"""POST /v1/detect-container — pre-capture beverage-container gate.

Single-image multipart upload. Returns a `DetectContainerResponse`
payload that the mobile / web UI uses to decide whether to start the
cylindrical-scan flow. The point of this route is to refuse to start a
10.8 s scan when the camera is pointed at a face, a hand, or a wall —
the user feels every second of that scan, so a cheap pre-flight check
pays for itself many times over.

The route also runs the known-label probe: if the user's first camera
frame matches a label we've previously verified (by brand text or by
first-frame perceptual hash), the response carries a `known_label`
payload that lets the mobile UI skip the panorama capture entirely and
jump straight to the report. Recognition is best-effort — both lookups
are wrapped in try/except so a DB outage never breaks detect-container.

Mirrors the multipart shape and error envelope of /v1/verify so the
client only has one error handler for both endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import CheckOutcome, ExtractionContext
from app.services.anthropic_client import ExtractorUnavailable
from app.services.container_check import (
    ContainerDetection,
    detect_container,
)
from app.services.persisted_cache import (
    PersistedHit,
    PersistedLabelCache,
    signature_to_hex,
)
from app.services.reverse_lookup import compute_dhash_bytes
from app.services.shadow_model import (
    _emit_shadow_telemetry,
    _timed_shadow_predict,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detect-container", tags=["detect"])


# 503 payload code — same shape as the verify route's `vision_unavailable`
# so the client maps a single transient failure mode for both endpoints.
_SERVICE_UNAVAILABLE_CODE = "vision_unavailable"
# 413 payload code for oversize uploads. Same envelope shape as verify.
_IMAGE_TOO_LARGE_CODE = "image_too_large"


class KnownLabelVerdictRule(BaseModel):
    rule_id: str
    rule_version: int
    citation: str
    status: str
    finding: str | None = None
    expected: str | None = None
    fix_suggestion: str | None = None
    explanation: str | None = None


class KnownLabelVerdictSummary(BaseModel):
    overall: str
    rule_results: list[KnownLabelVerdictRule]
    extracted: dict[str, dict[str, Any]]
    image_quality: str


class KnownLabelPayload(BaseModel):
    entry_id: str
    beverage_type: str
    # Derived from the cached extraction's net_contents (with a fallback
    # to the beverage-type's most common size). The mobile UI POSTs this
    # back through `/v1/scans/from-cache` so the from-cache route knows
    # to re-run the rule engine with the same dimensions the recognition
    # verdict was built off.
    container_size_ml: int
    is_imported: bool
    brand_name: str | None = None
    fanciful_name: str | None = None
    verdict_summary: KnownLabelVerdictSummary
    source: Literal["brand", "first_frame", "both"]


class DetectContainerResponse(BaseModel):
    detected: bool
    container_type: Literal["bottle", "can", "box"] | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    reason: str | None = None
    brand_name: str | None = None
    net_contents: str | None = None
    # Single-panel dhash of the upload bytes, surfaced as a 16-char
    # lowercase hex string (matches `signature_to_hex` encoding for one
    # panel). Mobile stores this on the scan draft and forwards it on
    # `/v1/scans/{id}/finalize` so `enrich_verdict` can stamp it onto
    # the L3 row. Null when dhash computation fails (corrupt bytes).
    image_dhash: str | None = None
    # Recognition payload populated when the brand-text or first-frame
    # lookup hits a previously-verified label. Null on miss or when
    # detected=False.
    known_label: KnownLabelPayload | None = None


# Common net-contents → mL conversions for beer / wine / spirits. Keeps
# the parser tight to the printed forms the L3 corpus carries; anything
# we can't parse falls back to the per-beverage default below.
_NET_CONTENTS_DIRECT_ML = {
    # US fluid ounces — the strings the model returns most often.
    "12 fl oz": 355,
    "16 fl oz": 473,
    "22 fl oz": 650,
    "24 fl oz": 710,
    "32 fl oz": 946,
    "40 fl oz": 1183,
    # Common mL declarations.
    "330 ml": 330,
    "355 ml": 355,
    "375 ml": 375,
    "473 ml": 473,
    "500 ml": 500,
    "750 ml": 750,
    "1 l": 1000,
    "1 liter": 1000,
}

# Per-beverage defaults when net_contents is absent or unparseable. Beer
# defaults to 355 mL (12 oz can/bottle); wine to 750 mL; spirits to
# 750 mL. The from-cache flow always re-runs the rule engine with the
# user's actual size, so these fallbacks just keep the recognition
# verdict_summary readable; they don't gate any compliance decision.
_DEFAULT_SIZE_BY_BEVERAGE = {
    "beer": 355,
    "wine": 750,
    "spirits": 750,
}


def _parse_container_size_ml(net_contents: str | None) -> int | None:
    """Best-effort parse of a printed net-contents string to mL.

    Returns ``None`` when the string is missing or doesn't match a known
    form — the caller layers in beverage-type-specific fallbacks. Match
    is case-insensitive against a small lookup table; we deliberately do
    NOT regex-extract numbers because a string like "12 OZ" without
    "FL" could be a dry weight on a bitters bottle and we'd rather
    decline than guess.
    """
    if not net_contents:
        return None
    needle = net_contents.strip().lower()
    if needle in _NET_CONTENTS_DIRECT_ML:
        return _NET_CONTENTS_DIRECT_ML[needle]
    # Try common normalizations: "12 FL. OZ.", "12 fl. oz", "12fl oz".
    cleaned = (
        needle.replace(".", "")
        .replace(",", "")
        .replace("(", " ")
        .replace(")", " ")
    )
    cleaned = " ".join(cleaned.split())
    if cleaned in _NET_CONTENTS_DIRECT_ML:
        return _NET_CONTENTS_DIRECT_ML[cleaned]
    # Allow the printed form to carry both US and metric ("12 FL OZ
    # (355 mL)" → match the leading half).
    for prefix in (cleaned.split(" ")[:3], cleaned.split(" ")[:2]):
        candidate = " ".join(prefix)
        if candidate in _NET_CONTENTS_DIRECT_ML:
            return _NET_CONTENTS_DIRECT_ML[candidate]
    return None


def _is_imported_from_extraction_fields(
    fields: dict[str, Any],
) -> bool:
    """Same logic as the verify-path name_address rule.

    `is_imported` is True when the cached extraction recorded a
    country_of_origin that isn't blank and isn't one of the US synonyms.
    Used to seed the from-cache rule re-eval and to surface the value
    in the recognition payload so mobile can pre-fill the user's claim.
    """
    info = fields.get("country_of_origin")
    if not isinstance(info, dict):
        return False
    value = info.get("value")
    if not isinstance(value, str):
        return False
    cleaned = value.strip().lower()
    if not cleaned:
        return False
    return cleaned not in {"usa", "u.s.a.", "us", "u.s.", "united states", "united states of america"}


def _extract_field_value(fields: dict[str, Any], name: str) -> str | None:
    info = fields.get(name)
    if isinstance(info, dict):
        v = info.get("value")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _build_known_label_payload(
    *,
    hit: PersistedHit,
    source: Literal["brand", "first_frame", "both"],
    detect_net_contents: str | None,
    requested_beverage_type: str | None,
) -> KnownLabelPayload | None:
    """Re-run the rule engine on the cached extraction and assemble the
    payload the mobile UI consumes.

    SPEC §0.5: never serve a frozen verdict. The cached row stores the
    *extraction*; the verdict is recomputed here from the user's actual
    container_size + import flag (derived for the recognition surface,
    overridden by the user's input on the from-cache call). Returns
    None when we can't even pick a beverage_type or sensible size from
    the cached row — better to show no recognition than a wrong one.
    """
    extraction = hit.extraction
    beverage_type = (
        extraction.beverage_type_observed
        or requested_beverage_type
        or "beer"
    )
    is_imported = _is_imported_from_extraction_fields(
        {
            name: {
                "value": fe.value,
                "confidence": fe.confidence,
            }
            for name, fe in extraction.fields.items()
        }
    )

    # 1: detect-container's own net_contents read.
    # 2: cached extraction's net_contents.
    # 3: beverage-type default.
    cached_net_contents = _extract_field_value(
        {
            name: {"value": fe.value, "confidence": fe.confidence}
            for name, fe in extraction.fields.items()
        },
        "net_contents",
    )
    container_size_ml = (
        _parse_container_size_ml(detect_net_contents)
        or _parse_container_size_ml(cached_net_contents)
        or _DEFAULT_SIZE_BY_BEVERAGE.get(beverage_type, 355)
    )

    rules = load_rules(beverage_type=beverage_type)
    if not rules:
        return None
    engine = RuleEngine(rules)
    ctx = ExtractionContext(
        fields=dict(extraction.fields),
        beverage_type=beverage_type,
        container_size_ml=container_size_ml,
        is_imported=is_imported,
        unreadable_fields=list(extraction.unreadable),
    )
    rule_results = engine.evaluate(ctx)

    explanations = hit.explanations or {}
    rules_payload: list[KnownLabelVerdictRule] = []
    for r in rule_results:
        status_str = r.status.value if isinstance(r.status, CheckOutcome) else str(r.status)
        rules_payload.append(
            KnownLabelVerdictRule(
                rule_id=r.rule_id,
                rule_version=r.rule_version,
                citation=r.citation,
                status=status_str,
                finding=r.finding,
                expected=r.expected,
                fix_suggestion=r.fix_suggestion,
                explanation=explanations.get(r.rule_id),
            )
        )

    from app.rules.aggregation import overall_status

    overall = overall_status(
        rule_results,
        image_quality=extraction.image_quality or "good",
        unreadable_fields=list(extraction.unreadable),
    )

    extracted_summary: dict[str, dict[str, Any]] = {
        name: {"value": fe.value, "confidence": fe.confidence}
        for name, fe in extraction.fields.items()
    }

    return KnownLabelPayload(
        entry_id=str(hit.entry_id),
        beverage_type=beverage_type,
        container_size_ml=container_size_ml,
        is_imported=is_imported,
        brand_name=_extract_field_value(extracted_summary, "brand_name"),
        fanciful_name=_extract_field_value(extracted_summary, "fanciful_name"),
        verdict_summary=KnownLabelVerdictSummary(
            overall=overall,
            rule_results=rules_payload,
            extracted=extracted_summary,
            image_quality=extraction.image_quality or "good",
        ),
        source=source,
    )


async def _safe_lookup_by_brand(
    cache: PersistedLabelCache,
    *,
    brand_name: str,
    beverage_type: str | None,
) -> PersistedHit | None:
    try:
        return await cache.lookup_by_brand(beverage_type, brand_name)
    except Exception as exc:
        logger.warning("known-label brand lookup failed: %s", exc)
        return None


async def _safe_lookup_by_first_frame(
    cache: PersistedLabelCache,
    *,
    signature_hex: str,
    beverage_type: str | None,
) -> PersistedHit | None:
    try:
        return await cache.lookup_by_first_frame(signature_hex, beverage_type)
    except Exception as exc:
        logger.warning("known-label first-frame lookup failed: %s", exc)
        return None


@router.post("", response_model=DetectContainerResponse)
async def detect_container_endpoint(
    image: UploadFile = File(...),
) -> DetectContainerResponse:
    """Inspect a single image and decide whether a beverage container is
    the primary subject.

    Multipart shape mirrors `/v1/verify`'s single-image legacy field but
    with no producer-record / beverage-type / container-size form
    parameters — this is a pre-flight gate, not a compliance check.
    """
    declared = getattr(image, "size", None)
    max_bytes = settings.max_image_bytes
    if declared is not None and declared > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "code": _IMAGE_TOO_LARGE_CODE,
                "message": (
                    f"Image upload is {declared} bytes; the maximum is "
                    f"{max_bytes} bytes. Resize before submitting."
                ),
            },
        )
    image_bytes = await image.read()
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "code": _IMAGE_TOO_LARGE_CODE,
                "message": (
                    f"Image upload is {len(image_bytes)} bytes; the maximum "
                    f"is {max_bytes} bytes. Resize before submitting."
                ),
            },
        )
    if not image_bytes:
        raise HTTPException(400, "image upload is empty")
    media_type = image.content_type or "image/jpeg"
    if not media_type.startswith("image/"):
        raise HTTPException(
            400,
            f"image content_type must be image/*, got {media_type!r}",
        )

    # `detect_container` is synchronous (issues a blocking Anthropic SDK
    # call). Run it off the event loop so a single in-flight detection
    # doesn't park every other request on this worker.
    try:
        result = await asyncio.to_thread(
            detect_container, image_bytes, media_type
        )
    except ExtractorUnavailable as exc:
        # Pre-capture gate is best-effort. 503 (not 500) so the client
        # treats this as a transient/retryable condition; the UI can
        # then either prompt the user to retry or fall through to the
        # scan path with a "couldn't pre-check" warning.
        raise HTTPException(
            status_code=503,
            detail={
                "code": _SERVICE_UNAVAILABLE_CODE,
                "message": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    # Compute the single-panel dhash up-front so we can (a) feed the
    # known-label first-frame lookup and (b) hand the hex back to the
    # client to forward through `/v1/scans/{id}/finalize`. Best-effort:
    # a None hash just means we skip the first-frame probe and the
    # client doesn't get a hex to forward (the L3 row will simply never
    # acquire a first-frame stamp from this scan).
    image_dhash_int = await asyncio.to_thread(compute_dhash_bytes, image_bytes)
    image_dhash_hex: str | None = None
    if image_dhash_int is not None:
        image_dhash_hex = signature_to_hex((int(image_dhash_int),))

    known_label_payload: KnownLabelPayload | None = None
    if result.detected:
        # Resolve the L3 cache lazily through the same singleton the
        # verify path uses so detect-container, verify, and scans share
        # one client (and one set of locks). Best-effort: a missing
        # singleton (cache disabled by config) just skips recognition.
        try:
            from app.api.verify import get_persisted_label_cache

            cache = get_persisted_label_cache()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("persisted cache lookup failed to resolve: %s", exc)
            cache = None

        if cache is not None:
            brand_hit: PersistedHit | None = None
            first_frame_hit: PersistedHit | None = None
            if result.brand_name:
                brand_hit = await _safe_lookup_by_brand(
                    cache,
                    brand_name=result.brand_name,
                    beverage_type=None,
                )
            if image_dhash_hex is not None:
                first_frame_hit = await _safe_lookup_by_first_frame(
                    cache,
                    signature_hex=image_dhash_hex,
                    beverage_type=None,
                )

            chosen_hit: PersistedHit | None = None
            source: Literal["brand", "first_frame", "both"] | None = None
            if brand_hit is not None and first_frame_hit is not None:
                # Same row, both lookups agree → strongest signal.
                if brand_hit.entry_id == first_frame_hit.entry_id:
                    chosen_hit = brand_hit
                    source = "both"
                else:
                    # Disagreement: prefer first_frame (perceptual match
                    # is more specific than brand-text alone — two
                    # different SKUs can share a brand).
                    chosen_hit = first_frame_hit
                    source = "first_frame"
            elif brand_hit is not None:
                chosen_hit = brand_hit
                source = "brand"
            elif first_frame_hit is not None:
                chosen_hit = first_frame_hit
                source = "first_frame"

            if chosen_hit is not None and source is not None:
                try:
                    known_label_payload = _build_known_label_payload(
                        hit=chosen_hit,
                        source=source,
                        detect_net_contents=result.net_contents,
                        requested_beverage_type=None,
                    )
                except Exception as exc:
                    logger.warning(
                        "known-label payload build failed: %s", exc
                    )

    response = DetectContainerResponse(
        detected=result.detected,
        container_type=result.container_type,
        bbox=result.bbox,
        confidence=result.confidence,
        reason=result.reason,
        brand_name=result.brand_name,
        net_contents=result.net_contents,
        image_dhash=image_dhash_hex,
        known_label=known_label_payload,
    )

    # Shadow-mode inference seam (MODEL_INTEGRATION_PLAN §3.2). Fire-and-
    # forget on a daemon thread so a slow or hung future model never
    # widens detect-container latency. The prediction is logged ONLY —
    # never reflected in the response. v1 ships a no-op so this is a
    # zero-cost forward-compat hook.
    _spawn_shadow_predict(image_bytes, response.brand_name)

    return response


def _spawn_shadow_predict(
    first_frame_bytes: bytes, vlm_brand: str | None
) -> None:
    """Run `shadow_predict` on a daemon thread and emit telemetry.

    Module-level so tests can patch it. Daemon-thread launch keeps the
    response path latency-clean (MODEL_INTEGRATION_PLAN §3.2's "fire-
    and-forget" contract). Any exception inside the thread is caught
    by `_timed_shadow_predict`'s defensive try/except — the thread
    never unwinds with an unhandled error.
    """

    def _run() -> None:
        prediction = _timed_shadow_predict(first_frame_bytes, None)
        _emit_shadow_telemetry(prediction, vlm_brand)

    threading.Thread(target=_run, daemon=True).start()
