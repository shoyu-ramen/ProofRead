"""POST /v1/scans + lifecycle endpoints.

Backed by Postgres-compatible SQLAlchemy. Scans, scan images, OCR
results, extracted fields, reports, and rule results are persisted at
the appropriate lifecycle steps:

* ``POST /v1/scans`` creates a ``scans`` row.
* ``PUT  /v1/scans/{id}/upload/{surface}`` writes the body to the
  storage backend and inserts/updates a ``scan_images`` row.
* ``POST /v1/scans/{id}/finalize`` runs the OCR + extractor + rule
  engine, then persists ``ocr_results``, ``extracted_fields``,
  ``reports``, and ``rule_results``.
* ``GET  /v1/scans/{id}`` and ``GET /v1/scans/{id}/report`` read from
  the DB.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, get_current_user
from app.config import settings
from app.db import get_session
from app.models import (
    ExtractedFieldRow,
    OCRResultRow,
    Report,
    RuleResultRow,
    Scan,
    ScanImage,
)
from app.services.extractors.claude_vision import (
    ClaudeVisionExtractor,
)
from app.services.ocr import OCRProvider, OCRResult, get_default_provider
from app.services.pipeline import ScanInput, VisionExtractor, process_scan
from app.services.storage import StorageBackend, get_default_storage, scan_image_key

router = APIRouter(prefix="/scans", tags=["scans"])

_SUPPORTED_SURFACES = ("front", "back")


class ProducerRecordDTO(BaseModel):
    """Optional producer-side metadata used for case-only / substantive
    cross-checks. The Stone's Throw vs STONE'S THROW gin scenario in the
    prototype is the canonical motivating example."""

    brand: str | None = None
    class_type: str | None = None
    container_size_ml: int | None = None


class CreateScanRequest(BaseModel):
    beverage_type: str = Field(pattern="^(beer|wine|spirits)$")
    container_size_ml: int = Field(gt=0, le=10_000)
    is_imported: bool = False
    producer_record: ProducerRecordDTO | None = None


class UploadURL(BaseModel):
    surface: str
    signed_url: str


class CreateScanResponse(BaseModel):
    scan_id: str
    upload_urls: list[UploadURL]


class ScanStatusResponse(BaseModel):
    scan_id: str
    status: str
    overall: str | None = None
    image_quality: str | None = None


class RuleResultDTO(BaseModel):
    rule_id: str
    rule_version: int
    citation: str
    status: str
    finding: str | None
    expected: str | None
    fix_suggestion: str | None
    bbox: tuple[int, int, int, int] | None


class ReportResponse(BaseModel):
    scan_id: str
    overall: str
    image_quality: str
    image_quality_notes: str | None
    extractor: str
    rule_results: list[RuleResultDTO]
    fields_summary: dict


def get_ocr_provider() -> OCRProvider:
    return get_default_provider()


def get_storage() -> StorageBackend:
    return get_default_storage()


def get_vision_extractor() -> VisionExtractor | None:
    """Construct the scan-path vision extractor chain when configured.

    Returns whichever backends are constructable, in preference order:

      1. Claude — when `vision_extractor=claude` AND `ANTHROPIC_API_KEY` is set.
      2. Qwen3-VL — when `enable_qwen_fallback=True` AND `qwen_vl_base_url` is set.

    The pipeline accepts vision=None gracefully and falls back to OCR, so
    when nothing is configurable we return None rather than raising — that
    keeps the test suite running without an Anthropic key and the OCR
    backstop remains the final tier even when the chain is exhausted at
    request time.
    """
    if settings.vision_extractor not in {"claude", "mock"}:
        return None

    extractors: list[VisionExtractor] = []

    if settings.vision_extractor == "claude" and settings.anthropic_api_key:
        try:
            extractors.append(ClaudeVisionExtractor(model=settings.anthropic_model))
        except Exception:
            # Claude misconfigured — keep going so a healthy Qwen can still serve.
            pass

    if settings.enable_qwen_fallback and settings.qwen_vl_base_url:
        from app.services.extractors.qwen_vl import QwenVLExtractor as QwenScan

        try:
            extractors.append(QwenScan())
        except Exception:
            pass

    if not extractors:
        return None
    if len(extractors) == 1:
        return extractors[0]

    from app.services.vision_chain import ChainedScanExtractor

    return ChainedScanExtractor(extractors)


@router.post("", response_model=CreateScanResponse, status_code=status.HTTP_201_CREATED)
async def create_scan(
    req: CreateScanRequest,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    storage: StorageBackend = Depends(get_storage),
) -> CreateScanResponse:
    if req.beverage_type != "beer":
        raise HTTPException(
            status_code=400,
            detail="v1 supports beer only; wine and spirits land in v2.",
        )

    scan = Scan(
        id=uuid.uuid4(),
        user_id=user.id,
        beverage_type=req.beverage_type,
        container_size_ml=req.container_size_ml,
        is_imported=req.is_imported,
        status="uploading",
        container_size_source="user",
    )
    session.add(scan)
    await session.commit()
    await session.refresh(scan)

    base = str(request.base_url).rstrip("/")
    upload_urls: list[UploadURL] = []
    for surface in _SUPPORTED_SURFACES:
        backend_url = await storage.generate_signed_url(
            scan_image_key(str(scan.id), surface), surface=surface, method="PUT"
        )
        # For LocalFsStorage we route uploads through our own PUT endpoint.
        # For S3 we'd return the presigned URL directly. The mobile client
        # uses whichever shape comes back without caring.
        if backend_url.startswith("local://"):
            backend_url = f"{base}/v1/scans/{scan.id}/upload/{surface}"
        upload_urls.append(UploadURL(surface=surface, signed_url=backend_url))

    return CreateScanResponse(scan_id=str(scan.id), upload_urls=upload_urls)


@router.put("/{scan_id}/upload/{surface}", status_code=status.HTTP_204_NO_CONTENT)
async def upload_image(
    scan_id: str,
    surface: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    storage: StorageBackend = Depends(get_storage),
) -> None:
    scan = await _load_scan(session, scan_id)
    if surface not in _SUPPORTED_SURFACES:
        raise HTTPException(400, f"unknown surface {surface!r}")
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")

    key = scan_image_key(str(scan.id), surface)
    await storage.put(key, body)

    # Upsert the scan_images row (UniqueConstraint on (scan_id, surface)).
    existing = await session.scalar(
        select(ScanImage).where(
            ScanImage.scan_id == scan.id, ScanImage.surface == surface
        )
    )
    if existing is None:
        session.add(
            ScanImage(
                id=uuid.uuid4(),
                scan_id=scan.id,
                surface=surface,
                s3_key=key,
            )
        )
    else:
        existing.s3_key = key
    await session.commit()


@router.post("/{scan_id}/finalize", response_model=ScanStatusResponse)
async def finalize_scan(
    scan_id: str,
    session: AsyncSession = Depends(get_session),
    ocr: OCRProvider = Depends(get_ocr_provider),
    vision: VisionExtractor | None = Depends(get_vision_extractor),
    storage: StorageBackend = Depends(get_storage),
) -> ScanStatusResponse:
    scan = await _load_scan(session, scan_id)
    if scan.status not in {"uploading", "failed"}:
        raise HTTPException(409, f"scan is {scan.status}")

    images = (
        await session.scalars(select(ScanImage).where(ScanImage.scan_id == scan.id))
    ).all()
    surfaces_present = {img.surface for img in images}
    missing = set(_SUPPORTED_SURFACES) - surfaces_present
    if missing:
        raise HTTPException(400, f"missing surfaces: {sorted(missing)}")

    # Pull image bytes out of storage to feed the pipeline.
    image_bytes: dict[str, bytes] = {}
    for img in images:
        image_bytes[img.surface] = await storage.get(img.s3_key)

    scan.status = "processing"
    await session.commit()

    try:
        report = process_scan(
            ScanInput(
                beverage_type=scan.beverage_type,
                container_size_ml=scan.container_size_ml,
                images=image_bytes,
                is_imported=scan.is_imported,
            ),
            ocr=ocr,
            vision=vision,
        )
    except Exception as e:
        scan.status = "failed"
        await session.commit()
        raise HTTPException(500, f"processing failed: {e}") from e

    # Persist OCR results, extracted fields, the report, and per-rule rows.
    images_by_surface = {img.surface: img for img in images}
    for surface, ocr_result in report.ocr_results.items():
        scan_image = images_by_surface.get(surface)
        if scan_image is None:
            continue
        session.add(_ocr_row_for(scan_image.id, ocr_result))

    for field_id, summary in report.fields_summary.items():
        bbox = summary.get("bbox") if isinstance(summary, dict) else None
        session.add(
            ExtractedFieldRow(
                id=uuid.uuid4(),
                scan_id=scan.id,
                field_id=field_id,
                value=(summary.get("value") if isinstance(summary, dict) else None),
                bbox=list(bbox) if bbox is not None else None,
                confidence=(
                    summary.get("confidence", 0.0) if isinstance(summary, dict) else 0.0
                ),
                source_image_id=None,
            )
        )

    rule_versions = sorted({str(r.rule_version) for r in report.rule_results})
    rule_version_str = ",".join(rule_versions) if rule_versions else "1"

    report_row = Report(
        id=uuid.uuid4(),
        scan_id=scan.id,
        overall=report.overall,
        rule_version=rule_version_str,
        image_quality=report.image_quality,
        image_quality_notes=report.image_quality_notes,
        extractor=report.extractor,
    )
    session.add(report_row)
    await session.flush()

    for r in report.rule_results:
        session.add(
            RuleResultRow(
                id=uuid.uuid4(),
                report_id=report_row.id,
                rule_id=r.rule_id,
                rule_version=r.rule_version,
                status=r.status.value,
                finding=r.finding,
                expected=r.expected,
                citation=r.citation,
                fix_suggestion=r.fix_suggestion,
                bbox=list(r.bbox) if r.bbox is not None else None,
                image_id=None,
            )
        )

    scan.status = "complete"
    scan.completed_at = datetime.now(UTC).replace(tzinfo=None)
    await session.commit()

    return ScanStatusResponse(
        scan_id=str(scan.id),
        status=scan.status,
        overall=report.overall,
        image_quality=report.image_quality,
    )


@router.get("/{scan_id}", response_model=ScanStatusResponse)
async def get_scan(
    scan_id: str,
    session: AsyncSession = Depends(get_session),
) -> ScanStatusResponse:
    scan = await _load_scan(session, scan_id)
    overall: str | None = None
    image_quality: str | None = None
    if scan.status == "complete":
        report = await session.scalar(
            select(Report).where(Report.scan_id == scan.id)
        )
        if report is not None:
            overall = report.overall
            image_quality = report.image_quality
    return ScanStatusResponse(
        scan_id=str(scan.id),
        status=scan.status,
        overall=overall,
        image_quality=image_quality,
    )


@router.get("/{scan_id}/report", response_model=ReportResponse)
async def get_report(
    scan_id: str,
    session: AsyncSession = Depends(get_session),
) -> ReportResponse:
    scan = await _load_scan(session, scan_id)
    report = await session.scalar(select(Report).where(Report.scan_id == scan.id))
    if report is None:
        raise HTTPException(409, f"scan is {scan.status}; report not ready")

    rule_rows = (
        await session.scalars(
            select(RuleResultRow).where(RuleResultRow.report_id == report.id)
        )
    ).all()
    rule_results = [
        RuleResultDTO(
            rule_id=r.rule_id,
            rule_version=r.rule_version,
            citation=r.citation,
            status=r.status,
            finding=r.finding,
            expected=r.expected,
            fix_suggestion=r.fix_suggestion,
            bbox=tuple(r.bbox) if r.bbox is not None else None,
        )
        for r in rule_rows
    ]

    field_rows = (
        await session.scalars(
            select(ExtractedFieldRow).where(ExtractedFieldRow.scan_id == scan.id)
        )
    ).all()
    fields_summary: dict = {}
    for f in field_rows:
        fields_summary[f.field_id] = {
            "value": f.value,
            "confidence": f.confidence,
            "bbox": tuple(f.bbox) if f.bbox is not None else None,
        }

    return ReportResponse(
        scan_id=str(scan.id),
        overall=report.overall,
        image_quality=report.image_quality,
        image_quality_notes=report.image_quality_notes,
        extractor=report.extractor,
        rule_results=rule_results,
        fields_summary=fields_summary,
    )


# --- helpers ---------------------------------------------------------------


async def _load_scan(session: AsyncSession, scan_id: str) -> Scan:
    try:
        scan_uuid = uuid.UUID(scan_id)
    except (TypeError, ValueError):
        raise HTTPException(404, "scan not found") from None
    scan = await session.scalar(select(Scan).where(Scan.id == scan_uuid))
    if scan is None:
        raise HTTPException(404, "scan not found")
    return scan


def _ocr_row_for(scan_image_id: uuid.UUID, ocr_result: OCRResult) -> OCRResultRow:
    raw = _jsonable(ocr_result.raw)
    return OCRResultRow(
        id=uuid.uuid4(),
        scan_image_id=scan_image_id,
        provider=ocr_result.provider,
        raw_json=raw if isinstance(raw, dict) else {"raw": raw},
        text=ocr_result.full_text,
        confidence=_avg_confidence(ocr_result),
        ms=int(time.monotonic() * 1000) % 100000,
    )


def _avg_confidence(ocr_result: OCRResult) -> float:
    if not ocr_result.blocks:
        return 0.0
    return sum(b.confidence for b in ocr_result.blocks) / len(ocr_result.blocks)


def _jsonable(value):
    """Best-effort conversion of nested dataclasses/tuples to JSON-friendly types."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        return asdict(value)
    except TypeError:
        return value
