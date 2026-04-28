"""POST /v1/verify — single-shot label verification used by the agent UI.

Multipart request: image + producer record (JSON) + beverage type. Synchronous
response: per-rule verdict, extracted fields, overall status.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.anthropic_client import ExtractorUnavailable
from app.services.verify import VerifyInput, verify
from app.services.vision import VisionExtractor, get_default_extractor

router = APIRouter(prefix="/verify", tags=["verify"])

# 503 payload shape returned when every configured vision extractor is
# unreachable (or none is configured). The mobile/web UI maps the `code`
# to a friendly message rather than dumping `detail` verbatim, so the
# detail is allowed to be developer-facing.
_SERVICE_UNAVAILABLE_CODE = "vision_unavailable"


_BEVERAGE_TYPES = {"beer", "wine", "spirits"}


class RuleResultDTO(BaseModel):
    rule_id: str
    rule_version: int
    citation: str
    status: str
    finding: str | None = None
    expected: str | None = None
    fix_suggestion: str | None = None
    bbox: tuple[int, int, int, int] | None = None


class FieldExtractionDTO(BaseModel):
    value: str | None = None
    confidence: float = 0.0
    bbox: list[int] | None = None
    unreadable: bool = False


class VerifyResponse(BaseModel):
    overall: str
    rule_results: list[RuleResultDTO]
    extracted: dict[str, FieldExtractionDTO]
    unreadable_fields: list[str]
    image_quality: str = "good"
    image_quality_notes: str | None = None
    elapsed_ms: int


# Module-level extractor cache so we don't recreate the Anthropic client per
# request. Set by `get_extractor()`; tests can override via dependency_overrides
# or by patching this attribute directly.
_extractor_cache: VisionExtractor | None = None


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


@router.post("", response_model=VerifyResponse)
async def verify_label(
    image: UploadFile = File(...),
    beverage_type: str = Form(...),
    container_size_ml: int = Form(...),
    is_imported: bool = Form(False),
    application: str = Form(...),
) -> VerifyResponse:
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

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "image upload is empty")

    media_type = image.content_type or "image/png"
    if not media_type.startswith("image/"):
        raise HTTPException(400, f"image content_type must be image/*, got {media_type!r}")

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
        image_bytes=image_bytes,
        media_type=media_type,
        beverage_type=beverage_type,
        container_size_ml=container_size_ml,
        is_imported=is_imported,
        application=application_obj,
    )

    try:
        report = verify(inp, extractor=extractor)
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
            )
            for r in report.rule_results
        ],
        extracted={
            name: FieldExtractionDTO(
                value=info.get("value"),
                confidence=info.get("confidence", 0.0),
                bbox=info.get("bbox"),
                unreadable=info.get("unreadable", False),
            )
            for name, info in report.extracted.items()
        },
        unreadable_fields=report.unreadable_fields,
        image_quality=report.image_quality,
        image_quality_notes=report.image_quality_notes,
        elapsed_ms=report.elapsed_ms,
    )
