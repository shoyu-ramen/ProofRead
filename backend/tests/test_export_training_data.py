"""Tests for backend/scripts/export_training_data.py.

Per MODEL_INTEGRATION_PLAN §2 the export must:

  * Materialize quality_classifier rows from the scans + scan_images +
    extracted_fields join, with a deterministic 75/12.5/12.5 split.
  * Recompute the 12-d sensor feature vector from stored image bytes.
  * NEVER write user_id, raw extracted-field values, or
    extraction/external/explanation JSON.
  * Truncate `captured_at` (and `created_at` etc.) to date precision.
  * Be idempotent — re-running with no new scans is a no-op.

These tests run against an in-memory SQLite (the existing conftest
`db_setup` fixture) so the join logic, watermark behavior, and PII
exclusion can all be exercised without standing up Postgres.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.db import get_session_factory
from app.models import (
    ExtractedFieldRow,
    LabelCacheEntry,
    Scan,
    ScanImage,
)


@pytest.fixture
def output_dir(tmp_path) -> Path:
    return tmp_path / "training_exports"


# Plain bytes for the image_loader stub; the export only forwards them
# to sensor_check, which a separate test stubs out.
_FAKE_IMAGE_BYTES = b"\x89PNG\r\n\x1a\n" + b"x" * 32


async def _stub_image_loader(_s3_key: str) -> bytes:
    return _FAKE_IMAGE_BYTES


async def _seed_scan(
    *,
    user_id: uuid.UUID,
    beverage_type: str = "beer",
    container_size_ml: int = 355,
    is_imported: bool = False,
    status: str = "complete",
    created_at: datetime | None = None,
    field_confidences: list[tuple[str, float]] | None = None,
    surface: str = "front",
    s3_key_suffix: str = "front",
) -> uuid.UUID:
    """Insert a Scan + ScanImage + ExtractedFieldRow set for testing.

    Tests build their fixture rows by calling this helper; it returns
    the scan_id so the test can assert split assignment / row counts.
    """
    factory = get_session_factory()
    scan_id = uuid.uuid4()
    async with factory() as session:
        scan = Scan(
            id=scan_id,
            user_id=user_id,
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
            status=status,
        )
        if created_at is not None:
            scan.created_at = created_at
        session.add(scan)
        await session.flush()

        img = ScanImage(
            scan_id=scan_id,
            surface=surface,
            s3_key=f"scans/{scan_id}/{s3_key_suffix}",
            width=1200,
            height=1800,
        )
        session.add(img)

        for field_id, conf in field_confidences or []:
            session.add(
                ExtractedFieldRow(
                    scan_id=scan_id,
                    field_id=field_id,
                    value=f"raw-text-for-{field_id}",  # MUST NOT appear in export
                    confidence=conf,
                )
            )
        await session.commit()
    return scan_id


@pytest.fixture
def patched_sensor_check(monkeypatch):
    """Skip the real sensor_check call (which needs a real PNG); install
    a stub that returns a known SurfaceCaptureQuality so the encode
    step has something to work with."""
    from app.services import sensor_check
    from app.services.sensor_check import (
        CaptureQualityReport,
        ImageQualityMetrics,
        SensorMetadata,
        SurfaceCaptureQuality,
    )

    def _fake_assess(images: dict[str, bytes]) -> CaptureQualityReport:
        surfaces = []
        for surface_name in images:
            surfaces.append(
                SurfaceCaptureQuality(
                    surface=surface_name,
                    sensor=SensorMetadata(tier="modern_flagship"),
                    metrics=ImageQualityMetrics(
                        sharpness=180.0,
                        glare_fraction=0.05,
                        brightness_mean=140.0,
                        brightness_stddev=42.0,
                        color_cast=0.10,
                        megapixels=2.16,
                        width_px=1200,
                        height_px=1800,
                    ),
                    verdict="good",
                    confidence=0.95,
                    motion_blur_direction=None,
                    backlit=False,
                    lens_smudge_likely=False,
                    wet_bottle_likely=False,
                    capture_source="photo",
                )
            )
        return CaptureQualityReport(
            surfaces=surfaces, overall_verdict="good", overall_confidence=0.95
        )

    monkeypatch.setattr(sensor_check, "assess_capture_quality", _fake_assess)
    yield


def _read_jsonl(path: Path) -> list[dict]:
    """Read either a parquet or jsonl file as a list of dicts.

    Tests must work in environments without pyarrow; the export
    script's fallback writes JSONL when pyarrow is missing. We try
    parquet first, then jsonl with the same stem.
    """
    pq = path.with_suffix(".parquet")
    if pq.exists():
        try:
            import pyarrow.parquet as _pq

            tbl = _pq.read_table(str(pq))
            return tbl.to_pylist()
        except Exception:
            pass
    jl = path.with_suffix(".jsonl")
    if not jl.exists():
        return []
    return [json.loads(line) for line in jl.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_split_for_scan_is_deterministic():
    """The split is hash-based so a re-run never moves an existing row.
    Run the same scan_id through twice and confirm identical assignment.
    """
    from scripts.export_training_data import _split_for_scan

    sid = "11111111-1111-4111-8111-111111111111"
    assert _split_for_scan(sid) == _split_for_scan(sid)


def test_split_for_scan_distribution_is_roughly_75_12_12():
    """1000 random UUIDs → ~75/12.5/12.5. Assert loose bounds so the
    deterministic assignment is provably distributing."""
    from scripts.export_training_data import _split_for_scan

    counts = {"train": 0, "val": 0, "test": 0}
    for _ in range(1000):
        counts[_split_for_scan(str(uuid.uuid4()))] += 1
    assert 700 < counts["train"] < 800
    assert 80 < counts["val"] < 180
    assert 80 < counts["test"] < 180


def test_quality_label_one_when_high_conf_with_warning():
    from scripts.export_training_data import _quality_label

    confs = [
        ("brand_name", 0.92),
        ("class_type", 0.88),
        ("alcohol_content", 0.90),
        ("health_warning", 0.91),
    ]
    assert _quality_label(confs) == 1


def test_quality_label_zero_without_health_warning():
    """Per §2.2, the label is 1 ONLY if the warning's confidence is
    above the second threshold. A run with high mean but missing
    warning is a negative training example."""
    from scripts.export_training_data import _quality_label

    confs = [
        ("brand_name", 0.95),
        ("class_type", 0.93),
        ("alcohol_content", 0.91),
    ]
    assert _quality_label(confs) == 0


def test_quality_label_zero_when_low_mean():
    from scripts.export_training_data import _quality_label

    confs = [
        ("brand_name", 0.50),
        ("class_type", 0.60),
        ("health_warning", 0.55),
    ]
    assert _quality_label(confs) == 0


def test_quality_label_zero_when_warning_below_threshold():
    """High mean but warning below 0.80 — Claude wasn't confident in
    the warning, so the panorama is a negative example for §1.a's
    purpose."""
    from scripts.export_training_data import _quality_label

    confs = [
        ("brand_name", 0.95),
        ("class_type", 0.93),
        ("alcohol_content", 0.91),
        ("health_warning", 0.65),
    ]
    assert _quality_label(confs) == 0


def test_quality_label_zero_on_empty_inputs():
    from scripts.export_training_data import _quality_label

    assert _quality_label([]) == 0


def test_encode_sensor_features_yields_12_dims():
    """Feature-vector contract: exactly 12 floats, in the documented
    order. A change in length is a breaking change for the trainer."""
    from app.services.sensor_check import (
        ImageQualityMetrics,
        SensorMetadata,
        SurfaceCaptureQuality,
    )
    from scripts.export_training_data import _encode_sensor_features

    sq = SurfaceCaptureQuality(
        surface="front",
        sensor=SensorMetadata(tier="older_midrange"),
        metrics=ImageQualityMetrics(
            sharpness=120.5,
            glare_fraction=0.12,
            brightness_mean=132.0,
            brightness_stddev=38.0,
            color_cast=0.08,
            megapixels=1.5,
            width_px=1200,
            height_px=1800,
        ),
        verdict="good",
        confidence=0.8,
        backlit=True,
        motion_blur_direction="diagonal",
        lens_smudge_likely=False,
        wet_bottle_likely=True,
        capture_source="screenshot",
    )
    vec = _encode_sensor_features(sq)
    assert len(vec) == 12
    assert vec[0] == 120.5  # sharpness
    assert vec[1] == 0.12  # glare_fraction
    assert vec[2] == 132.0  # brightness_mean
    assert vec[3] == 38.0  # brightness_stddev
    assert vec[4] == 0.08  # color_cast
    assert vec[5] == 1.5  # megapixels
    assert vec[6] == 1.0  # backlit
    assert vec[7] == 3.0  # motion_blur_direction (diagonal=3)
    assert vec[8] == 0.0  # lens_smudge_likely (False)
    assert vec[9] == 1.0  # wet_bottle_likely (True)
    assert vec[10] == 1.0  # capture_source (screenshot=1)
    assert vec[11] == 3.0  # sensor_tier (older_midrange=3)


@pytest.mark.asyncio
async def test_happy_path_writes_files_and_manifest(
    db_setup, patched_sensor_check, output_dir
):
    """Seed two scans → run export → confirm split files + manifest +
    watermark exist with sane content."""
    from app.auth import _TEST_USER
    from scripts.export_training_data import run_export

    # Seed two scans with different conf profiles → one positive, one negative.
    await _seed_scan(
        user_id=_TEST_USER.id,
        field_confidences=[
            ("brand_name", 0.95),
            ("class_type", 0.92),
            ("health_warning", 0.91),
        ],
    )
    await _seed_scan(
        user_id=_TEST_USER.id,
        field_confidences=[("brand_name", 0.30)],
    )

    summary = await run_export(
        output_dir=output_dir,
        image_loader=_stub_image_loader,
        now=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert summary["status"] == "ok"
    assert summary["quality_rows"] == 2
    assert summary["label_balance"]["positive"] >= 1
    assert summary["label_balance"]["negative"] >= 1

    version_dir = output_dir / "v20260501"
    assert version_dir.exists()
    qc_dir = version_dir / "quality_classifier"
    assert (qc_dir / "manifest.json").exists()
    manifest = json.loads((qc_dir / "manifest.json").read_text())
    assert manifest["license"] == "INTERNAL-USE-ONLY"
    assert manifest["feature_vector_dim"] == 12
    assert manifest["row_counts"]["total"] == 2
    assert sum(manifest["row_counts"][s] for s in ("train", "val", "test")) == 2
    assert (output_dir / "watermark.json").exists()


@pytest.mark.asyncio
async def test_pii_exclusion_user_id_never_written(
    db_setup, patched_sensor_check, output_dir
):
    """SPEC §2.4: user_id MUST NOT appear in any output file. Iterate
    every written line and confirm the seeded user_id is never there.
    Same for raw extracted-field values (`extracted_fields.value`)."""
    from app.auth import _TEST_USER
    from scripts.export_training_data import run_export

    await _seed_scan(
        user_id=_TEST_USER.id,
        field_confidences=[
            ("brand_name", 0.92),
            ("health_warning", 0.91),
        ],
    )

    await run_export(
        output_dir=output_dir,
        image_loader=_stub_image_loader,
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    user_id_str = str(_TEST_USER.id)
    qc_dir = output_dir / "v20260501" / "quality_classifier"
    for split in ("train", "val", "test"):
        rows = _read_jsonl(qc_dir / split)
        for row in rows:
            payload = json.dumps(row)
            assert user_id_str not in payload, (
                f"user_id leaked into {split}: {row}"
            )
            assert "raw-text-for-" not in payload, (
                f"raw extracted-field value leaked into {split}: {row}"
            )
            assert "user_id" not in row.keys()


@pytest.mark.asyncio
async def test_idempotency_second_run_is_noop(
    db_setup, patched_sensor_check, output_dir
):
    """Run twice with no new data between → second run reports `noop`
    and writes no new version directory.

    Scan is seeded with explicit `created_at` BEFORE the first watermark
    so the second run's `Scan.created_at > watermark` filter correctly
    excludes it.
    """
    from app.auth import _TEST_USER
    from scripts.export_training_data import run_export

    await _seed_scan(
        user_id=_TEST_USER.id,
        field_confidences=[("brand_name", 0.95), ("health_warning", 0.91)],
        created_at=datetime(2026, 4, 1, 12, 0, 0),
    )

    first = await run_export(
        output_dir=output_dir,
        image_loader=_stub_image_loader,
        now=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert first["status"] == "ok"
    assert first["quality_rows"] == 1

    # Second run, no new scans seeded. Bump `now` past the first
    # watermark so any incremental query window is non-empty in time
    # but still finds zero new rows.
    second = await run_export(
        output_dir=output_dir,
        image_loader=_stub_image_loader,
        now=datetime(2026, 5, 2, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert second["status"] == "noop"
    assert second["quality_rows"] == 0
    # Watermark should still reference the FIRST run's `now` so a third
    # run after a new scan is added will see the new row.
    assert not (output_dir / "v20260502").exists()


@pytest.mark.asyncio
async def test_deterministic_split_assignment(
    db_setup, patched_sensor_check, output_dir
):
    """The same scan_id MUST land in the same split across runs. Seed
    a known scan_id, run, read the split file it landed in, and check
    that `_split_for_scan` agrees."""
    from app.auth import _TEST_USER
    from scripts.export_training_data import _split_for_scan, run_export

    scan_id = await _seed_scan(
        user_id=_TEST_USER.id,
        field_confidences=[("brand_name", 0.95), ("health_warning", 0.91)],
    )
    expected_split = _split_for_scan(str(scan_id))

    await run_export(
        output_dir=output_dir,
        image_loader=_stub_image_loader,
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    qc_dir = output_dir / "v20260501" / "quality_classifier"
    found_in: list[str] = []
    for split in ("train", "val", "test"):
        rows = _read_jsonl(qc_dir / split)
        for row in rows:
            if row.get("scan_id") == str(scan_id):
                found_in.append(split)

    assert found_in == [expected_split], (
        f"scan landed in {found_in}, expected [{expected_split}]"
    )


@pytest.mark.asyncio
async def test_dates_are_truncated_to_day_precision(
    db_setup, patched_sensor_check, output_dir
):
    """SPEC §2.4: captured_at + created_at MUST be date-precision (no
    hour/minute/second). Confirm the written ISO strings have length
    exactly 10 (YYYY-MM-DD)."""
    from app.auth import _TEST_USER
    from scripts.export_training_data import run_export

    await _seed_scan(
        user_id=_TEST_USER.id,
        field_confidences=[("brand_name", 0.95), ("health_warning", 0.91)],
        created_at=datetime(2026, 4, 1, 14, 33, 7, tzinfo=timezone.utc),
    )

    await run_export(
        output_dir=output_dir,
        image_loader=_stub_image_loader,
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    qc_dir = output_dir / "v20260501" / "quality_classifier"
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        rows.extend(_read_jsonl(qc_dir / split))
    assert len(rows) == 1
    row = rows[0]
    assert row["created_at_date"] == "2026-04-01"
    # Must NOT contain time components.
    assert "T" not in row["created_at_date"]
    assert ":" not in row["created_at_date"]


@pytest.mark.asyncio
async def test_first_frame_signals_export(
    db_setup, patched_sensor_check, output_dir
):
    """The first_frame_signals/ subdirectory must include only safe
    label_cache columns: brand_name_normalized, first_frame_signature_hex,
    beverage_type, panel_count. NO extraction_json / external_match_json /
    explanations_json."""
    from scripts.export_training_data import run_export

    factory = get_session_factory()
    async with factory() as session:
        session.add(
            LabelCacheEntry(
                beverage_type="beer",
                panel_count=2,
                signature_hex="abc123,def456",
                brand_name_normalized="anytown ale",
                first_frame_signature_hex="cafebabe12345678",
                extraction_json={"raw": "MUST NOT EXPORT"},
                external_match_json={"sensitive": "MUST NOT EXPORT"},
                explanations_json={"hidden": "MUST NOT EXPORT"},
            )
        )
        await session.commit()

    await run_export(
        output_dir=output_dir,
        image_loader=_stub_image_loader,
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    ff_dir = output_dir / "v20260501" / "first_frame_signals"
    assert (ff_dir / "manifest.json").exists()
    rows = _read_jsonl(ff_dir / "samples")
    assert len(rows) == 1
    row = rows[0]
    assert row["brand_name_normalized"] == "anytown ale"
    assert row["first_frame_signature_hex"] == "cafebabe12345678"
    assert row["beverage_type"] == "beer"
    assert row["panel_count"] == 2
    # PII guard: every JSON-blob column must be absent.
    payload = json.dumps(row)
    assert "MUST NOT EXPORT" not in payload
    assert "extraction_json" not in row
    assert "external_match_json" not in row
    assert "explanations_json" not in row


@pytest.mark.asyncio
async def test_dry_run_skips_sensor_recompute(
    db_setup, patched_sensor_check, output_dir
):
    """The CLI `--dry-run` flag MUST skip the expensive sensor_check
    recompute. Each row's feature vector should be the zero vector."""
    from app.auth import _TEST_USER
    from scripts.export_training_data import run_export

    await _seed_scan(
        user_id=_TEST_USER.id,
        field_confidences=[("brand_name", 0.95), ("health_warning", 0.91)],
    )

    await run_export(
        output_dir=output_dir,
        image_loader=_stub_image_loader,
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
        skip_sensor_recompute=True,
    )

    qc_dir = output_dir / "v20260501" / "quality_classifier"
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        rows.extend(_read_jsonl(qc_dir / split))
    assert len(rows) == 1
    assert rows[0]["sensor_feature_vector"] == [0.0] * 12


@pytest.mark.asyncio
async def test_no_data_run_returns_noop_without_writing(
    db_setup, patched_sensor_check, output_dir
):
    """Empty database → no-op, no version directory written, no
    watermark bumped (so a future run with new data will see it)."""
    from scripts.export_training_data import run_export

    summary = await run_export(
        output_dir=output_dir,
        image_loader=_stub_image_loader,
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    assert summary["status"] == "noop"
    assert not (output_dir / "v20260501").exists()
    assert not (output_dir / "watermark.json").exists()


def test_cli_runs_via_module_entry(monkeypatch, tmp_path, db_setup, capsys):
    """Smoke test the `python -m scripts.export_training_data` invocation.
    Patches asyncio.run so the script doesn't actually fire — we only
    care that argparse + main() wire up correctly."""
    import scripts.export_training_data as mod

    captured: dict = {}

    async def _fake_run_export(**kwargs):
        captured.update(kwargs)
        return {"status": "noop"}

    monkeypatch.setattr(mod, "run_export", _fake_run_export)
    out_dir = tmp_path / "exports"
    rc = mod.main(["--output-dir", str(out_dir), "--dry-run"])
    assert rc == 0
    assert captured["output_dir"] == out_dir
    assert captured["skip_sensor_recompute"] is True
