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

v1 mobile uploads a single unrolled-label panorama image. ``POST
/v1/scans`` returns one entry in ``upload_urls`` with ``surface =
"panorama"``; finalize expects exactly that one upload to be present.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, get_current_user
from app.config import settings
from app.db import get_session
from app.models import (
    ExtractedFieldRow,
    LabelCacheEntry,
    OCRResultRow,
    Report,
    RuleResultRow,
    Scan,
    ScanImage,
)
from app.rules.aggregation import overall_status as aggregate_overall_status
from app.rules.engine import RuleEngine
from app.rules.loader import load_rules
from app.rules.types import CheckOutcome, ExtractionContext
from app.services.enrichment import enrich_verdict
from app.services.extractors.claude_vision import (
    ClaudeVisionExtractor,
)
from app.services.ocr import OCRProvider, OCRResult, get_default_provider
from app.services.persisted_cache import extraction_from_dict
from app.services.pipeline import ScanInput, VisionExtractor, process_scan
from app.services.reverse_lookup import compute_dhash_bytes
from app.services.storage import StorageBackend, get_default_storage, scan_image_key
from app.services.verify import VerifyReport

router = APIRouter(prefix="/scans", tags=["scans"])

# v1 mobile sends a single unrolled-label panorama. The pipeline still
# accepts a per-surface dict, so adding a new surface here is the only
# change required to extend (e.g. neck band) later.
_SUPPORTED_SURFACES = ("panorama",)


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
    # Which scan_image surface the rule's evidence was read from. v1
    # mobile uploads only `"panorama"`; older rows in the DB may carry
    # `"front"`/`"back"` from the pre-panorama flow. Mirrors the
    # verify-path field so the mobile report screen can highlight the
    # captured image when the user taps a result. `None` when the rule
    # isn't tied to a specific extracted field.
    surface: str | None = None
    # AI-generated one-sentence plain-language explanation for fail /
    # advisory rules. Mirrors the `/v1/verify` field; populated by the
    # finalize hook and persisted to `RuleResultRow.explanation`. Null
    # for passing rules and when the explanation service is disabled
    # or fails open.
    explanation: str | None = None


class ReportResponse(BaseModel):
    scan_id: str
    overall: str
    image_quality: str
    image_quality_notes: str | None
    extractor: str
    rule_results: list[RuleResultDTO]
    fields_summary: dict
    # External-source match (TTB COLA approval, etc.). Mirrors the
    # `/v1/verify` field; persisted to `Report.external_match_json`.
    # Null when the lookup is disabled, no match, or below confidence
    # threshold. Shape is the dict form of `ExternalMatch`.
    external_match: dict | None = None


class HistoryItem(BaseModel):
    scan_id: str
    label: str
    overall: str
    scanned_at: datetime


class HistoryResponse(BaseModel):
    items: list[HistoryItem]


class FlagRuleResultRequest(BaseModel):
    comment: str


class FromCacheRequest(BaseModel):
    """Body for `POST /v1/scans/from-cache`.

    The mobile UI takes the `KnownLabelPayload` returned by
    `/v1/detect-container`, hands it to the user as a "we recognized
    this" sheet, and POSTs back here on confirm. The user's actual
    container_size_ml + is_imported come from the same payload (the
    detect-container route derived them off the cached row), but the
    mobile UI is allowed to override them — e.g. a 12-pack vs single can
    of the same SKU.
    """

    entry_id: str
    beverage_type: str = Field(pattern="^(beer|wine|spirits)$")
    container_size_ml: int = Field(gt=0, le=10_000)
    is_imported: bool = False


class FromCacheResponse(BaseModel):
    scan_id: str
    status: str
    overall: str
    image_quality: str


def get_ocr_provider() -> OCRProvider:
    return get_default_provider()


def get_storage() -> StorageBackend:
    return get_default_storage()


def get_persisted_label_cache_for_scans():
    """Singleton bridge to the L3 cache used by both API surfaces.

    Both /v1/verify and /v1/scans share the same `label_cache` table
    and the same `PersistedLabelCache` settings; routing both API
    layers through the lazy factory in `api.verify` keeps the
    gating + lock invariants centralised. Returns None when the
    operator hasn't enabled the durable tier.
    """
    from app.api.verify import get_persisted_label_cache

    return get_persisted_label_cache()


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


@router.get("", response_model=HistoryResponse)
async def get_history(
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> HistoryResponse:
    """GET /v1/scans (paginated history).

    Returns all scans for the current user, joined with their latest report
    overall status and brand name (if extracted).
    """
    # Join Scan with Report (latest) and optionally the brand_name field.
    # For a prototype we keep it simple: just the scans for this user.
    query = (
        select(Scan, Report.overall, ExtractedFieldRow.value)
        .outerjoin(Report, Report.scan_id == Scan.id)
        .outerjoin(
            ExtractedFieldRow,
            (ExtractedFieldRow.scan_id == Scan.id)
            & (ExtractedFieldRow.field_id == "brand_name"),
        )
        .where(Scan.user_id == user.id)
        .order_by(Scan.created_at.desc())
        .limit(50)
    )

    rows = (await session.execute(query)).all()
    items = []
    for scan, overall, brand_name in rows:
        items.append(
            HistoryItem(
                scan_id=str(scan.id),
                label=brand_name or f"{scan.beverage_type.title()} Label",
                overall=overall or "pending",
                scanned_at=scan.created_at,
            )
        )
    return HistoryResponse(items=items)


@router.post("", response_model=CreateScanResponse, status_code=status.HTTP_201_CREATED)
async def create_scan(
    req: CreateScanRequest,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    storage: StorageBackend = Depends(get_storage),
) -> CreateScanResponse:
    if req.beverage_type != "beer":
        # Same `{code, message}` envelope as /v1/verify so the mobile
        # client only needs one error-shape parser. SPEC §1.2 lists wine
        # and spirits as v1 non-goals for the multi-image scan flow;
        # spirits is unlocked on the single-shot /v1/verify path.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "beverage_type_unsupported",
                "message": (
                    "v1 scans support beer only; wine and spirits land in v2."
                ),
            },
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


@router.post(
    "/from-cache",
    response_model=FromCacheResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_scan_from_cache(
    req: FromCacheRequest,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> FromCacheResponse:
    """Materialize a `Scan` from a previously-verified L3 cache entry.

    Flow:
      1. Resolve the `entry_id` to a `LabelCacheEntry` row (404 if
         missing or its `extraction_json` is null/empty).
      2. Re-run the rule engine FRESH with `req.container_size_ml` +
         `req.is_imported` — we never serve the explanations cached on
         the row as a verdict, only as supplemental context. The rule
         engine is the deterministic judge.
      3. Persist a `Scan` (`container_size_source="cache"`),
         `ExtractedFieldRow`s from the cached extraction, a `Report` +
         `RuleResultRow`s from the fresh rule eval, and the cache row's
         enrichment fields (external_match + per-rule explanations).
      4. Return the same shape `/v1/scans/{id}/finalize` does so the
         mobile client can navigate to the report screen identically.

    No `ScanImage` rows are created — the UI's "View results" path
    skips the panorama capture entirely and the report screen tolerates
    missing image rows.
    """
    try:
        entry_uuid = uuid.UUID(req.entry_id)
    except (TypeError, ValueError):
        raise HTTPException(404, "label cache entry not found") from None

    entry = await session.scalar(
        select(LabelCacheEntry).where(LabelCacheEntry.id == entry_uuid)
    )
    if entry is None or not entry.extraction_json:
        raise HTTPException(404, "label cache entry not found")

    # Rebuild the cached extraction in its dataclass form so we can feed
    # it through the rule engine and persist its fields.
    extraction = extraction_from_dict(entry.extraction_json)

    rules = load_rules(beverage_type=req.beverage_type)
    if not rules:
        raise HTTPException(
            422,
            f"no rules configured for beverage_type={req.beverage_type!r}",
        )
    engine = RuleEngine(rules)
    ctx = ExtractionContext(
        fields=dict(extraction.fields),
        beverage_type=req.beverage_type,
        container_size_ml=req.container_size_ml,
        is_imported=req.is_imported,
        unreadable_fields=list(extraction.unreadable),
    )
    rule_results = engine.evaluate(ctx)

    image_quality = extraction.image_quality or "good"
    image_quality_notes = extraction.image_quality_notes
    overall = aggregate_overall_status(
        rule_results,
        image_quality=image_quality,
        unreadable_fields=list(extraction.unreadable),
    )

    rule_versions = sorted({str(r.rule_version) for r in rule_results})
    rule_version_str = ",".join(rule_versions) if rule_versions else "1"

    now = datetime.now(UTC).replace(tzinfo=None)
    scan = Scan(
        id=uuid.uuid4(),
        user_id=user.id,
        beverage_type=req.beverage_type,
        container_size_ml=req.container_size_ml,
        is_imported=req.is_imported,
        status="complete",
        container_size_source="cache",
        completed_at=now,
    )
    session.add(scan)
    await session.flush()

    for name, fe in extraction.fields.items():
        bbox_value: Any = list(fe.bbox) if fe.bbox is not None else None
        session.add(
            ExtractedFieldRow(
                id=uuid.uuid4(),
                scan_id=scan.id,
                field_id=name,
                value=fe.value,
                bbox=bbox_value,
                confidence=float(fe.confidence),
                source_image_id=None,
            )
        )

    report_row = Report(
        id=uuid.uuid4(),
        scan_id=scan.id,
        overall=overall,
        rule_version=rule_version_str,
        image_quality=image_quality,
        image_quality_notes=image_quality_notes,
        extractor="cache",
        external_match_json=(
            dict(entry.external_match_json)
            if entry.external_match_json is not None
            else None
        ),
    )
    session.add(report_row)
    await session.flush()

    explanations = (
        dict(entry.explanations_json) if entry.explanations_json else {}
    )
    for r in rule_results:
        status_str = (
            r.status.value if isinstance(r.status, CheckOutcome) else str(r.status)
        )
        session.add(
            RuleResultRow(
                id=uuid.uuid4(),
                report_id=report_row.id,
                rule_id=r.rule_id,
                rule_version=r.rule_version,
                status=status_str,
                finding=r.finding,
                expected=r.expected,
                citation=r.citation,
                fix_suggestion=r.fix_suggestion,
                bbox=list(r.bbox) if r.bbox is not None else None,
                image_id=None,
                surface=r.surface,
                explanation=explanations.get(r.rule_id),
            )
        )

    await session.commit()

    return FromCacheResponse(
        scan_id=str(scan.id),
        status="complete",
        overall=overall,
        image_quality=image_quality,
    )


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
    # Reject oversize uploads before reading the body. `Content-Length` is
    # advisory (clients can lie), so we also re-check the materialized size
    # below — the early reject saves a buffer allocation in the common case.
    max_bytes = settings.max_image_bytes
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"upload is {declared} bytes; the maximum is "
                        f"{max_bytes} bytes. Resize before submitting."
                    ),
                )
        except ValueError:
            pass
    body = await request.body()
    if len(body) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"upload is {len(body)} bytes; the maximum is {max_bytes} "
                f"bytes. Resize before submitting."
            ),
        )
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
    first_frame_signature_hex: str | None = Form(default=None),
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

    # Persist the bare report + rule rows FIRST. SPEC §0.5 forbids
    # gating the verdict on enrichment, and a slow Anthropic call (up
    # to `explanation_timeout_s` = 6 s) or a TTB COLA outage cannot
    # leave the scan stuck in "processing" or block the user from
    # seeing their pass/fail. Enrichment writes are issued as UPDATEs
    # below; if those fail, the report still goes out cleanly with
    # `explanation=None` / `external_match=None`.
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

    rule_rows: list[RuleResultRow] = []
    for r in report.rule_results:
        row = RuleResultRow(
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
            surface=r.surface,
        )
        session.add(row)
        rule_rows.append(row)

    scan.status = "complete"
    scan.completed_at = datetime.now(UTC).replace(tzinfo=None)
    await session.commit()

    # Enrichment runs after the response-shape data is durable. Pure-
    # additive: any failure here logs and drops the corresponding field.
    # The L3 perceptual cache is consulted (and updated) using a
    # signature derived from raw upload bytes. This is a different key
    # space from the verify-path's normalized phash, so an L3 entry
    # written by /v1/verify won't be found by /v1/scans (and vice
    # versa). That's acceptable for v1: the two paths each accumulate
    # their own corpus, and a future iteration can unify the keying by
    # lifting normalization out of verify.
    try:
        signatures = []
        for _surface, raw_bytes in image_bytes.items():
            try:
                signatures.append(compute_dhash_bytes(raw_bytes))
            except Exception:
                signatures.append(None)
        scan_signature: tuple[int, ...] | None = (
            tuple(int(s) for s in signatures if s is not None)
            if signatures and all(s is not None for s in signatures)
            else None
        )

        persisted_cache = get_persisted_label_cache_for_scans()
        persisted_hit = None
        if persisted_cache is not None and scan_signature is not None:
            try:
                persisted_hit = await persisted_cache.lookup(
                    signature=scan_signature,
                    beverage_type=scan.beverage_type,
                )
            except Exception:
                persisted_hit = None

        adapter_report = VerifyReport(
            overall=report.overall,
            rule_results=list(report.rule_results),
            extracted=dict(report.fields_summary),
            unreadable_fields=[],
            image_quality=report.image_quality,
            image_quality_notes=report.image_quality_notes,
            elapsed_ms=0,
            raw_extraction=report.raw_extraction,
        )
        enrichment = await enrich_verdict(
            report=adapter_report,
            beverage_type=scan.beverage_type,
            container_size_ml=scan.container_size_ml,
            is_imported=scan.is_imported,
            persisted_cache=persisted_cache,
            persisted_hit=persisted_hit,
            signature=scan_signature,
            first_frame_signature_hex=first_frame_signature_hex,
        )
        if enrichment.external_match is not None:
            report_row.external_match_json = enrichment.external_match
        if enrichment.explanations:
            by_id = {r.rule_id: r for r in rule_rows}
            for rule_id, text in enrichment.explanations.items():
                row = by_id.get(rule_id)
                if row is not None:
                    row.explanation = text
        await session.commit()
    except Exception:
        # Best-effort. Roll back the enrichment-only state but leave
        # the verdict commit alone (already durable). The next
        # /v1/scans/{id}/report read returns the un-enriched shape,
        # which the UI tolerates.
        await session.rollback()

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
            surface=r.surface,
            explanation=r.explanation,
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
        external_match=report.external_match_json,
    )


@router.post("/{scan_id}/rule-results/{rule_id}/flag", status_code=status.HTTP_204_NO_CONTENT)
async def flag_rule_result(
    scan_id: str,
    rule_id: str,
    req: FlagRuleResultRequest,
    session: AsyncSession = Depends(get_session),
) -> None:
    scan = await _load_scan(session, scan_id)
    report = await session.scalar(select(Report).where(Report.scan_id == scan.id))
    if report is None:
        raise HTTPException(404, "Report not found")

    rule_result = await session.scalar(
        select(RuleResultRow).where(
            RuleResultRow.report_id == report.id, RuleResultRow.rule_id == rule_id
        )
    )
    if rule_result is None:
        raise HTTPException(404, "Rule result not found")

    rule_result.is_flagged = True
    rule_result.flag_comment = req.comment
    await session.commit()


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
