"""Export training data for the image-quality classifier (2.d).

Run from the backend root::

    python -m scripts.export_training_data --output-dir training_exports/

Reads ``scans`` × ``scan_images`` × ``extracted_fields`` × ``label_cache``
joined on ``scan_id``, computes derived training columns (``quality_label``
binary, 12-d ``sensor_feature_vector`` recomputed from stored bytes,
deterministic train/val/test split), and writes per-split files plus a
``manifest.json`` per output directory. Layout follows
MODEL_INTEGRATION_PLAN §2.3::

    training_exports/
      v{YYYYMMDD}/
        quality_classifier/{train,val,test}.parquet  (or .jsonl fallback)
        quality_classifier/manifest.json
        first_frame_signals/samples.parquet
        first_frame_signals/manifest.json
      watermark.json

PII handling (MODEL_INTEGRATION_PLAN §2.4):

  * ``user_id`` is NEVER written.
  * Raw extraction values (``extracted_fields.value``,
    ``label_cache.extraction_json`` / ``external_match_json`` /
    ``explanations_json``) are NEVER written.
  * ``captured_at`` is truncated to date precision.

Idempotency: a ``training_exports/watermark.json`` records
``last_exported_at`` so re-runs only process new scans. The split is a
deterministic hash of ``scan_id`` so existing rows never move splits
across runs.

Parquet vs JSONL: parquet is preferred (columnar, native PyTorch
support); the script writes parquet when ``pyarrow`` is available and
falls back to JSONL when it is not. The manifest's ``format`` field
lets the trainer pick the right loader.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger("export_training_data")


# Sensor feature vector ordering (MODEL_INTEGRATION_PLAN §2.2). Encode
# string-typed enums as small ints for ML-friendly storage. Keep these
# tables next to the export code rather than re-importing from
# sensor_check — drift here is more obvious than a silent enum reorder.
_MOTION_DIR_ENCODE = {None: 0, "horizontal": 1, "vertical": 2, "diagonal": 3}
_CAPTURE_SOURCE_ENCODE = {
    "photo": 0,
    "screenshot": 1,
    "uncertain": 2,
    "artwork": 3,
}
_SENSOR_TIER_ENCODE = {
    "modern_flagship": 0,
    "modern_midrange": 1,
    "older_flagship": 2,
    "older_midrange": 3,
    "low_end": 4,
    "unknown": 5,
}

# Per MODEL_INTEGRATION_PLAN §2.2 quality label thresholds. Centralized
# so the values are visible to the trainer team without grepping.
_MEAN_CONF_THRESHOLD = 0.85
_HEALTH_WARN_CONF_THRESHOLD = 0.80


@dataclass
class _SplitAssignment:
    train: int = 0
    val: int = 0
    test: int = 0


def _split_for_scan(scan_id: str) -> str:
    """Deterministic train/val/test assignment.

    Per MODEL_INTEGRATION_PLAN §2.2: ``sha256(scan_id)[-1] < 0x20`` →
    test (12.5%); ``< 0x40`` → val (12.5%); else train (75%). Hash on
    the string form of the UUID so SQLite (TEXT) and Postgres (UUID)
    runs land identically.
    """
    h = hashlib.sha256(scan_id.encode("ascii")).digest()
    last = h[-1]
    if last < 0x20:
        return "test"
    if last < 0x40:
        return "val"
    return "train"


def _encode_sensor_features(surface_quality: Any) -> list[float]:
    """Translate a ``SurfaceCaptureQuality`` into the 12-d feature vector.

    Order per MODEL_INTEGRATION_PLAN §2.2::

      [sharpness, glare_fraction, brightness_mean, brightness_stddev,
       color_cast, megapixels, backlit(0/1), motion_blur_direction(0-3),
       smudge_likely(0/1), wet_likely(0/1), capture_source(0-3),
       sensor_tier(0-5)]
    """
    m = surface_quality.metrics
    sensor = surface_quality.sensor
    return [
        float(m.sharpness),
        float(m.glare_fraction),
        float(m.brightness_mean),
        float(m.brightness_stddev),
        float(m.color_cast),
        float(m.megapixels),
        1.0 if surface_quality.backlit else 0.0,
        float(_MOTION_DIR_ENCODE.get(surface_quality.motion_blur_direction, 0)),
        1.0 if surface_quality.lens_smudge_likely else 0.0,
        1.0 if surface_quality.wet_bottle_likely else 0.0,
        float(_CAPTURE_SOURCE_ENCODE.get(surface_quality.capture_source, 2)),
        float(_SENSOR_TIER_ENCODE.get(sensor.tier, 5)),
    ]


def _zero_feature_vector() -> list[float]:
    """Fallback vector for rows where the sensor recompute failed.

    The trainer can drop these rows or treat them as "unknown sensor";
    we still emit the row so PII / quality_label coverage stays
    consistent across batches.
    """
    return [0.0] * 12


def _quality_label(
    field_confidences: list[tuple[str, float]],
) -> int:
    """Binary quality label per MODEL_INTEGRATION_PLAN §2.2.

    label = 1 iff
        mean(field confidences) > 0.85
        AND health_warning confidence > 0.80
    """
    if not field_confidences:
        return 0
    mean_conf = sum(c for _, c in field_confidences) / len(field_confidences)
    if mean_conf <= _MEAN_CONF_THRESHOLD:
        return 0
    health_conf = next(
        (c for fid, c in field_confidences if fid == "health_warning"), None
    )
    if health_conf is None or health_conf <= _HEALTH_WARN_CONF_THRESHOLD:
        return 0
    return 1


@dataclass
class ExportRow:
    """One materialized row destined for the quality_classifier export.

    Strictly excludes PII per §2.4: no ``user_id``, no raw OCR text,
    no extraction / external-match / explanation JSON.
    """

    scan_id: str
    beverage_type: str
    container_size_ml: int | None
    is_imported: bool
    status: str
    created_at_date: str  # date-precision ISO string (no time)
    completed_at_date: str | None
    surface: str | None
    s3_key: str | None
    width: int | None
    height: int | None
    captured_at_date: str | None  # date-precision per §2.4
    quality_label: int
    sensor_feature_vector: list[float]
    brand_name_normalized: str | None  # safe per §2.1 — already public label text
    first_frame_signature_hex: str | None
    panel_count: int | None
    split: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "beverage_type": self.beverage_type,
            "container_size_ml": self.container_size_ml,
            "is_imported": self.is_imported,
            "status": self.status,
            "created_at_date": self.created_at_date,
            "completed_at_date": self.completed_at_date,
            "surface": self.surface,
            "s3_key": self.s3_key,
            "width": self.width,
            "height": self.height,
            "captured_at_date": self.captured_at_date,
            "quality_label": self.quality_label,
            "sensor_feature_vector": self.sensor_feature_vector,
            "brand_name_normalized": self.brand_name_normalized,
            "first_frame_signature_hex": self.first_frame_signature_hex,
            "panel_count": self.panel_count,
            "split": self.split,
        }


@dataclass
class FirstFrameRow:
    """One row for the first_frame_signals/ subdirectory.

    Only the brand + first-frame perceptual hash + beverage type +
    panel count. No PII; matches the §2.1 column inclusion list for
    ``label_cache``.
    """

    brand_name_normalized: str
    first_frame_signature_hex: str
    beverage_type: str
    panel_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "brand_name_normalized": self.brand_name_normalized,
            "first_frame_signature_hex": self.first_frame_signature_hex,
            "beverage_type": self.beverage_type,
            "panel_count": self.panel_count,
        }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _try_import_pyarrow() -> Any | None:
    """Return the ``pyarrow`` module if importable, else ``None``.

    The trainer environment will have pyarrow; the test/CI environment
    may not. The fallback writes JSONL — the manifest's ``format``
    field tells the loader which path to take.
    """
    try:
        import pyarrow as pa  # type: ignore[import-not-found]

        return pa
    except ImportError:
        return None


def _write_rows(
    path: Path, rows: list[dict[str, Any]], format_hint: str
) -> str:
    """Write rows to ``path`` in the preferred format.

    Returns the format actually used: ``"parquet"`` if pyarrow was
    available and we wrote parquet; ``"jsonl"`` otherwise. The path's
    suffix is rewritten to match the chosen format so downstream loaders
    don't have to second-guess.
    """
    pa = _try_import_pyarrow()
    if format_hint == "parquet" and pa is not None:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]

            table = pa.Table.from_pylist(rows)
            target = path.with_suffix(".parquet")
            pq.write_table(table, str(target))
            return "parquet"
        except Exception as exc:
            logger.warning(
                "parquet write failed (%s); falling back to JSONL", exc
            )
    target = path.with_suffix(".jsonl")
    with target.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r) + "\n")
    return "jsonl"


# ---------------------------------------------------------------------------
# DB query
# ---------------------------------------------------------------------------


async def _query_rows(
    *, since: datetime | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the join query and return raw row dicts.

    Splits into two outputs:
      * ``quality_rows`` — one row per (scan, scan_image) pair carrying
        every field the export needs except the recomputed sensor
        vector + quality label (those are derived per-row downstream).
      * ``label_cache_rows`` — one row per ``label_cache`` entry with a
        non-null first-frame signature, for the first_frame_signals/
        export.

    Querying the two outputs separately keeps the join shape simple
    (no Cartesian explosion when a scan has 4 panels and 5 extracted
    fields) and lets each output filter independently.
    """
    from sqlalchemy import select

    from app.db import get_session_factory
    from app.models import (
        ExtractedFieldRow,
        LabelCacheEntry,
        Scan,
        ScanImage,
    )

    factory = get_session_factory()
    async with factory() as session:
        # 1. label_cache rows for the first_frame_signals/ output. We
        #    SELECT only the safe columns (§2.1 excludes the JSON blobs).
        #    Run this first so a label_cache-only run still produces an
        #    output (e.g. cache populated by warm hits without ever
        #    creating a Scan row).
        lc_q = (
            select(
                LabelCacheEntry.brand_name_normalized,
                LabelCacheEntry.first_frame_signature_hex,
                LabelCacheEntry.beverage_type,
                LabelCacheEntry.panel_count,
            )
            .where(LabelCacheEntry.brand_name_normalized.is_not(None))
            .where(LabelCacheEntry.first_frame_signature_hex.is_not(None))
        )
        lc_result = await session.execute(lc_q)
        label_cache_rows = [
            {
                "brand_name_normalized": brand,
                "first_frame_signature_hex": sig,
                "beverage_type": bev,
                "panel_count": panel_count,
            }
            for brand, sig, bev, panel_count in lc_result.all()
        ]

        # 2. Base envelope: scans + images + extracted-field confidences.
        scan_q = select(Scan)
        if since is not None:
            # SQLite + Postgres both store the server-default `created_at`
            # as a tz-naive UTC value. Strip any tz on the watermark so
            # the comparison doesn't trip on a mixed-aware/naive error.
            since_naive = since.replace(tzinfo=None) if since.tzinfo else since
            scan_q = scan_q.where(Scan.created_at > since_naive)
        scan_q = scan_q.order_by(Scan.created_at)
        scan_result = await session.execute(scan_q)
        scans = list(scan_result.scalars().all())

        if not scans:
            return [], label_cache_rows

        scan_ids = [s.id for s in scans]
        # Images keyed by scan_id for the join.
        img_q = select(ScanImage).where(ScanImage.scan_id.in_(scan_ids))
        img_result = await session.execute(img_q)
        images_by_scan: dict[UUID, list[ScanImage]] = {}
        for img in img_result.scalars().all():
            images_by_scan.setdefault(img.scan_id, []).append(img)

        # Extracted fields keyed by scan_id for the label-derivation step.
        # NB: we read ``confidence`` (not ``value``) — see §2.4.
        ef_q = select(
            ExtractedFieldRow.scan_id,
            ExtractedFieldRow.field_id,
            ExtractedFieldRow.confidence,
        ).where(ExtractedFieldRow.scan_id.in_(scan_ids))
        ef_result = await session.execute(ef_q)
        confidences_by_scan: dict[UUID, list[tuple[str, float]]] = {}
        for scan_id, field_id, conf in ef_result.all():
            confidences_by_scan.setdefault(scan_id, []).append(
                (field_id, float(conf))
            )

        # Look up the first label_cache hit per scan for brand-name
        # decoration on the quality_classifier rows. The cache key is
        # not joinable to scans on a single column (signatures are
        # per-panel), so this is a best-effort match: take any row
        # with the same beverage type + panel count, prefer one with
        # the same brand. v1 keeps it simple — a scan that isn't
        # tagged just gets ``brand_name_normalized=None`` in the export.
        # Trainer does not depend on this column for the quality label.
        # We could refine this when first-frame signatures are stamped
        # on the Scan row directly.

    # Build the quality_classifier row stream as plain dicts; sensor
    # vector + quality_label are derived in `_materialize_export_rows`.
    quality_rows: list[dict[str, Any]] = []
    for scan in scans:
        imgs = images_by_scan.get(scan.id, [])
        confs = confidences_by_scan.get(scan.id, [])
        if not imgs:
            # A scan without any image isn't useful as a training row
            # (nothing to recompute features against); skip it.
            continue
        for img in imgs:
            quality_rows.append(
                {
                    "scan_id": str(scan.id),
                    "beverage_type": scan.beverage_type,
                    "container_size_ml": scan.container_size_ml,
                    "is_imported": bool(scan.is_imported),
                    "status": scan.status,
                    "created_at": scan.created_at,
                    "completed_at": scan.completed_at,
                    "surface": img.surface,
                    "s3_key": img.s3_key,
                    "width": img.width,
                    "height": img.height,
                    "captured_at": img.captured_at,
                    "field_confidences": confs,
                }
            )
    return quality_rows, label_cache_rows


# ---------------------------------------------------------------------------
# Sensor recomputation
# ---------------------------------------------------------------------------


async def _recompute_sensor_features(
    rows: list[dict[str, Any]],
    *,
    image_loader: Any,
    batch_size: int = 50,
    per_image_timeout_s: float = 15.0,
) -> None:
    """Mutate ``rows`` in place, adding ``sensor_feature_vector`` to each.

    Reads the image bytes via ``image_loader`` (a callable returning
    bytes for an ``s3_key``), runs ``sensor_check.assess_capture_quality``
    on each, and encodes the resulting surface into the 12-d vector.
    Failures (decode error, timeout, missing object) get a zero vector
    so the row is still written — the trainer can drop zero vectors if
    needed.

    Batched to limit memory use; the per-image timeout (§2.2) caps the
    worst case where a corrupt blob blocks the whole export.
    """
    from app.services.sensor_check import assess_capture_quality

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        for row in batch:
            s3_key = row.get("s3_key")
            if not s3_key:
                row["sensor_feature_vector"] = _zero_feature_vector()
                continue
            try:
                image_bytes = await asyncio.wait_for(
                    image_loader(s3_key), timeout=per_image_timeout_s
                )
            except Exception as exc:
                logger.warning(
                    "image fetch failed for %s: %s", s3_key, exc
                )
                row["sensor_feature_vector"] = _zero_feature_vector()
                continue
            try:
                report = await asyncio.wait_for(
                    asyncio.to_thread(
                        assess_capture_quality, {"frame": image_bytes}
                    ),
                    timeout=per_image_timeout_s,
                )
            except Exception as exc:
                logger.warning(
                    "sensor_check failed for %s: %s", s3_key, exc
                )
                row["sensor_feature_vector"] = _zero_feature_vector()
                continue
            if report.surfaces:
                row["sensor_feature_vector"] = _encode_sensor_features(
                    report.surfaces[0]
                )
            else:
                row["sensor_feature_vector"] = _zero_feature_vector()


def _materialize_export_rows(
    quality_rows: list[dict[str, Any]],
) -> list[ExportRow]:
    """Apply derivations (quality_label, split, date truncation) and
    return one ExportRow per source dict.
    """
    out: list[ExportRow] = []
    for r in quality_rows:
        scan_id = r["scan_id"]
        created_at = r.get("created_at")
        completed_at = r.get("completed_at")
        captured_at = r.get("captured_at")
        out.append(
            ExportRow(
                scan_id=scan_id,
                beverage_type=r["beverage_type"],
                container_size_ml=r.get("container_size_ml"),
                is_imported=r.get("is_imported", False),
                status=r["status"],
                created_at_date=_to_date_str(created_at) or "",
                completed_at_date=_to_date_str(completed_at),
                surface=r.get("surface"),
                s3_key=r.get("s3_key"),
                width=r.get("width"),
                height=r.get("height"),
                captured_at_date=_to_date_str(captured_at),
                quality_label=_quality_label(r.get("field_confidences", [])),
                sensor_feature_vector=r.get(
                    "sensor_feature_vector", _zero_feature_vector()
                ),
                brand_name_normalized=None,
                first_frame_signature_hex=None,
                panel_count=None,
                split=_split_for_scan(scan_id),
            )
        )
    return out


def _to_date_str(dt: datetime | None) -> str | None:
    """Truncate ``dt`` to date-precision ISO string per §2.4.

    ``None`` passes through. Microseconds, hour/minute/second are all
    dropped — this is the §2.4 requirement to limit re-identification
    via timing patterns.
    """
    if dt is None:
        return None
    return dt.date().isoformat()


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------


def _read_watermark(path: Path) -> datetime | None:
    """Read the high-water mark from ``training_exports/watermark.json``.

    Returns ``None`` when the file doesn't exist (first run) or is
    malformed (treated as first run rather than crashing the export).
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("watermark unreadable, treating as first run: %s", exc)
        return None
    iso = data.get("last_exported_at")
    if not isinstance(iso, str):
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def _write_watermark(path: Path, *, last_exported_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_exported_at": last_exported_at.isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Storage shim
# ---------------------------------------------------------------------------


async def _default_image_loader(s3_key: str) -> bytes:
    """Read image bytes via the configured storage backend.

    Used outside tests; tests pass a custom loader so they don't need
    the storage singleton.
    """
    from app.services.storage import get_default_storage

    storage = get_default_storage()
    return await storage.get(s3_key)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_export(
    *,
    output_dir: Path,
    image_loader: Any | None = None,
    now: datetime | None = None,
    skip_sensor_recompute: bool = False,
) -> dict[str, Any]:
    """Top-level export run.

    ``image_loader`` is injectable so tests can fake the storage layer.
    ``now`` is injectable for deterministic version directory naming
    and watermark recording in tests. ``skip_sensor_recompute=True``
    leaves zero feature vectors; useful for CI / smoke runs that don't
    have real image bytes to recompute against.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if image_loader is None:
        image_loader = _default_image_loader

    output_dir.mkdir(parents=True, exist_ok=True)
    watermark_path = output_dir / "watermark.json"
    since = _read_watermark(watermark_path)

    started = time.monotonic()
    quality_rows, label_cache_rows = await _query_rows(since=since)
    elapsed_query_ms = int((time.monotonic() - started) * 1000)

    if not quality_rows and not label_cache_rows:
        logger.info(
            "no new rows since %s; nothing to export",
            since.isoformat() if since else "epoch",
        )
        # Idempotency: don't bump the watermark if there was nothing to
        # export — the next run should still see the same window.
        return {
            "status": "noop",
            "quality_rows": 0,
            "first_frame_rows": 0,
            "elapsed_query_ms": elapsed_query_ms,
            "since": since.isoformat() if since else None,
        }

    # Recompute sensor features (the §2.2 derived column).
    if not skip_sensor_recompute:
        await _recompute_sensor_features(
            quality_rows, image_loader=image_loader
        )
    else:
        for r in quality_rows:
            r["sensor_feature_vector"] = _zero_feature_vector()

    # Materialize ExportRows.
    export_rows = _materialize_export_rows(quality_rows)

    # Group by split.
    splits = {"train": [], "val": [], "test": []}
    for er in export_rows:
        splits[er.split].append(er.as_dict())

    # Write output directory.
    version_dir = output_dir / f"v{now.strftime('%Y%m%d')}"
    qc_dir = version_dir / "quality_classifier"
    qc_dir.mkdir(parents=True, exist_ok=True)
    written_format = None
    split_counts = _SplitAssignment()
    for split, rows in splits.items():
        target = qc_dir / split
        fmt = _write_rows(target, rows, "parquet")
        written_format = fmt
        if split == "train":
            split_counts.train = len(rows)
        elif split == "val":
            split_counts.val = len(rows)
        else:
            split_counts.test = len(rows)

    label_balance = {
        "positive": sum(1 for r in export_rows if r.quality_label == 1),
        "negative": sum(1 for r in export_rows if r.quality_label == 0),
    }
    qc_manifest = {
        "license": "INTERNAL-USE-ONLY",
        "schema_version": 1,
        "exported_at": now.isoformat(),
        "row_counts": {
            "train": split_counts.train,
            "val": split_counts.val,
            "test": split_counts.test,
            "total": len(export_rows),
        },
        "label_balance": label_balance,
        "format": written_format or "parquet",
        "feature_vector_dim": 12,
        "feature_vector_order": [
            "sharpness",
            "glare_fraction",
            "brightness_mean",
            "brightness_stddev",
            "color_cast",
            "megapixels",
            "backlit",
            "motion_blur_direction",
            "lens_smudge_likely",
            "wet_bottle_likely",
            "capture_source",
            "sensor_tier",
        ],
    }
    (qc_dir / "manifest.json").write_text(json.dumps(qc_manifest, indent=2))

    # First-frame signals.
    ff_dir = version_dir / "first_frame_signals"
    ff_dir.mkdir(parents=True, exist_ok=True)
    ff_format = _write_rows(
        ff_dir / "samples", label_cache_rows, "parquet"
    )
    ff_manifest = {
        "license": "INTERNAL-USE-ONLY",
        "schema_version": 1,
        "exported_at": now.isoformat(),
        "row_count": len(label_cache_rows),
        "format": ff_format,
    }
    (ff_dir / "manifest.json").write_text(json.dumps(ff_manifest, indent=2))

    # Bump watermark only after both writes succeeded.
    _write_watermark(watermark_path, last_exported_at=now)

    return {
        "status": "ok",
        "version_dir": str(version_dir),
        "quality_rows": len(export_rows),
        "first_frame_rows": len(label_cache_rows),
        "label_balance": label_balance,
        "split_counts": {
            "train": split_counts.train,
            "val": split_counts.val,
            "test": split_counts.test,
        },
        "format": written_format,
        "since": since.isoformat() if since else None,
        "elapsed_query_ms": elapsed_query_ms,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="export_training_data",
        description=(
            "Export training-data parquet files for the image-quality "
            "classifier (MODEL_INTEGRATION_PLAN §2)."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_exports"),
        help=(
            "Directory to write under (default: training_exports/). "
            "A versioned subdirectory v{YYYYMMDD}/ is created on each run."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the sensor-vector recompute (which requires reading "
            "every image's bytes). Writes zero feature vectors. Useful "
            "for CI and smoke checks."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging to stderr.",
    )
    return p


async def _amain(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    summary = await run_export(
        output_dir=args.output_dir,
        skip_sensor_recompute=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
