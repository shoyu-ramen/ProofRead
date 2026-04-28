"""Stress-test runner: degrade every label × condition × severity, run
each variant through the existing sensor pre-check, optionally probe the
Claude vision extractor, and emit a structured result list the report
module turns into a markdown matrix.

Design notes:

  * The harness is read-only with respect to ``app/``. We import
    ``assess_capture_quality`` and the vision extractor by their public
    paths but never monkey-patch internals.
  * Variants are saved to a temp dir (one PNG per (label, condition,
    severity)) so they can be inspected after the run. The temp dir is
    `tempfile.TemporaryDirectory()`-managed so it disappears when the
    process exits unless a caller passes ``keep_artifacts=True``.
  * Vision-extractor sampling is gated on ``ANTHROPIC_API_KEY``. To keep
    cost predictable we run ONE label × ONE mid-severity per condition,
    so the per-run spend is bounded by `len(DEGRADATIONS)` calls.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.services.sensor_check import (
    CaptureQualityReport,
    assess_capture_quality,
)
from validation.stress_test.degradations import (
    DEGRADATIONS,
    SEVERITIES,
    Severity,
    apply,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class StressResult:
    """One (label × condition × severity) outcome."""

    label: str
    condition: str
    severity: Severity
    sensor_verdict: str  # "good" | "degraded" | "unreadable" | "error"
    sensor_confidence: float
    issues: list[str] = field(default_factory=list)
    artifact_path: str | None = None
    error: str | None = None


@dataclass
class VisionSample:
    """One Claude-vision-extractor probe (mid-severity, one label).

    ``ok`` is True when the extractor returned at least one parseable
    field; the harness doesn't assess CORRECTNESS of the extraction (that
    needs ground truth and is out of scope) — we're measuring whether
    the call survives the degradation, full stop.
    """

    label: str
    condition: str
    severity: Severity
    ok: bool
    fields_returned: int
    unreadable_fields: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass
class StressMatrix:
    results: list[StressResult]
    vision_samples: list[VisionSample]
    labels: list[str]
    conditions: list[str]
    severities: list[Severity]
    artifact_dir: str | None = None


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------


def _baseline_verdict(image_bytes: bytes) -> str:
    report = assess_capture_quality({"front": image_bytes})
    return report.overall_verdict


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_stress_matrix(
    label_paths: list[Path],
    *,
    artifact_dir: Path | None = None,
    keep_artifacts: bool = False,
    sample_vision: bool = True,
    vision_label_index: int = 0,
) -> StressMatrix:
    """Run every (label × condition × severity) through sensor_check.

    Args:
        label_paths: paths to source label PNGs.
        artifact_dir: where to write degraded PNGs. Created if missing.
            If None, uses a `TemporaryDirectory` that's cleaned up at the
            end of the function unless ``keep_artifacts`` is True.
        keep_artifacts: leave the artifact dir on disk.
        sample_vision: gate the Claude-vision sampling step. If
            ``ANTHROPIC_API_KEY`` is not set we skip even when True.
        vision_label_index: which label to probe (0..3). Defaults to the
            first (clean pass label) — keeps cost bounded.

    Returns:
        A populated :class:`StressMatrix`.
    """
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        owned_tmp = None
        out_dir = artifact_dir
    else:
        owned_tmp = tempfile.TemporaryDirectory(prefix="proofread_stress_")
        out_dir = Path(owned_tmp.name)

    label_names = [p.stem for p in label_paths]
    conditions = list(DEGRADATIONS.keys())
    results: list[StressResult] = []

    for label_path in label_paths:
        label_name = label_path.stem
        with Image.open(label_path) as src:
            src.load()
            base = src.copy()

        baseline_bytes = _to_png_bytes(base)
        baseline = _baseline_verdict(baseline_bytes)
        logger.info("baseline verdict for %s: %s", label_name, baseline)

        for condition in conditions:
            for severity in SEVERITIES:
                tag = f"{label_name}__{condition}__{severity}"
                artifact_path: str | None = None
                try:
                    degraded = apply(condition, base, severity)
                    bytes_ = _to_png_bytes(degraded)
                    artifact_path = str(out_dir / f"{tag}.png")
                    with open(artifact_path, "wb") as fh:
                        fh.write(bytes_)
                    report: CaptureQualityReport = assess_capture_quality(
                        {"front": bytes_}
                    )
                    surface = report.surfaces[0] if report.surfaces else None
                    issues = list(surface.issues) if surface else []
                    confidence = float(surface.confidence) if surface else 0.0
                    results.append(
                        StressResult(
                            label=label_name,
                            condition=condition,
                            severity=severity,
                            sensor_verdict=report.overall_verdict,
                            sensor_confidence=confidence,
                            issues=issues,
                            artifact_path=artifact_path,
                        )
                    )
                except Exception as exc:  # pragma: no cover — logged + recorded
                    logger.exception("degradation failed: %s", tag)
                    results.append(
                        StressResult(
                            label=label_name,
                            condition=condition,
                            severity=severity,
                            sensor_verdict="error",
                            sensor_confidence=0.0,
                            issues=[],
                            artifact_path=artifact_path,
                            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                        )
                    )

    vision_samples = _run_vision_samples(
        label_paths=label_paths,
        sample_vision=sample_vision,
        vision_label_index=vision_label_index,
        out_dir=out_dir,
    )

    matrix = StressMatrix(
        results=results,
        vision_samples=vision_samples,
        labels=label_names,
        conditions=conditions,
        severities=list(SEVERITIES),
        artifact_dir=str(out_dir) if (keep_artifacts or artifact_dir) else None,
    )

    if owned_tmp is not None and not keep_artifacts:
        owned_tmp.cleanup()

    return matrix


# ---------------------------------------------------------------------------
# Vision sampling
# ---------------------------------------------------------------------------


def _run_vision_samples(
    *,
    label_paths: list[Path],
    sample_vision: bool,
    vision_label_index: int,
    out_dir: Path,
) -> list[VisionSample]:
    """Probe Claude vision once per condition, mid-severity.

    Returns an empty list when the API key is missing or sampling is
    explicitly disabled. Failures are recorded, not raised — a single
    network blip shouldn't sink the whole stress run.
    """
    if not sample_vision:
        return []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("ANTHROPIC_API_KEY not set; skipping vision sampling")
        return []

    try:
        from app.services.vision import ClaudeVisionExtractor
    except Exception as exc:  # pragma: no cover
        logger.warning("ClaudeVisionExtractor unavailable: %s", exc)
        return []

    label_path = label_paths[vision_label_index]
    label_name = label_path.stem
    with Image.open(label_path) as src:
        src.load()
        base = src.copy()

    extractor = ClaudeVisionExtractor()
    samples: list[VisionSample] = []
    severity: Severity = "medium"
    for condition in DEGRADATIONS:
        sample_start = time.perf_counter()
        try:
            degraded = apply(condition, base, severity)
            bytes_ = _to_png_bytes(degraded)
            extraction = extractor.extract(bytes_, media_type="image/png")
            elapsed = time.perf_counter() - sample_start
            samples.append(
                VisionSample(
                    label=label_name,
                    condition=condition,
                    severity=severity,
                    ok=bool(extraction.fields),
                    fields_returned=len(extraction.fields),
                    unreadable_fields=list(extraction.unreadable),
                    elapsed_s=elapsed,
                )
            )
            logger.info(
                "vision sample %s/%s: fields=%d unreadable=%d",
                condition,
                severity,
                len(extraction.fields),
                len(extraction.unreadable),
            )
        except Exception as exc:  # pragma: no cover — network/api dependent
            elapsed = time.perf_counter() - sample_start
            samples.append(
                VisionSample(
                    label=label_name,
                    condition=condition,
                    severity=severity,
                    ok=False,
                    fields_returned=0,
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_s=elapsed,
                )
            )
            logger.warning("vision sample %s/%s failed: %s", condition, severity, exc)

    return samples
