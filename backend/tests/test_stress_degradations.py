"""Behavioral tests for the stress-test harness.

We assert the harness mechanics — every degradation runs without raising
on every label at every severity, returns a real PIL image with non-zero
dimensions, and the resulting bytes still flow through
``assess_capture_quality``. We do NOT assert specific verdicts: the
verdict is what the stress run is measuring; pinning it here would make
the calibration unmaintainable.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.services.sensor_check import (
    CaptureQualityReport,
    assess_capture_quality,
)
from validation.stress_test.degradations import DEGRADATIONS, SEVERITIES

LABELS_DIR = Path(__file__).resolve().parents[2] / "artwork" / "labels"
LABEL_FILES = (
    "01_pass_old_tom_distillery.png",
    "02_warn_stones_throw_gin.png",
    "03_fail_mountain_crest_ipa.png",
    "04_unreadable_heritage_vineyards.png",
)


@pytest.fixture(scope="module")
def label_images() -> list[Image.Image]:
    """Cache one decoded copy of each label for the whole module."""
    images = []
    for name in LABEL_FILES:
        path = LABELS_DIR / name
        if not path.exists():
            pytest.skip(f"label fixture missing: {path}")
        with Image.open(path) as img:
            img.load()
            images.append(img.copy())
    return images


@pytest.mark.parametrize("condition", sorted(DEGRADATIONS.keys()))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_degradation_runs_on_every_label(condition, severity, label_images):
    """Each degradation must produce a valid PIL image for every label
    at every severity."""
    fn = DEGRADATIONS[condition]
    for img in label_images:
        out = fn(img, severity)
        assert isinstance(out, Image.Image), (
            f"{condition}/{severity} returned {type(out).__name__}, not PIL.Image"
        )
        w, h = out.size
        assert w > 0 and h > 0, f"{condition}/{severity} produced empty image"


@pytest.mark.parametrize("condition", sorted(DEGRADATIONS.keys()))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_sensor_check_accepts_degraded_bytes(condition, severity, label_images):
    """`assess_capture_quality` must return a real CaptureQualityReport
    on the degraded bytes — never None, never raise. Behavior, not the
    verdict, is the contract here."""
    fn = DEGRADATIONS[condition]
    img = label_images[0]  # one label is enough for the behavioral check
    out = fn(img, severity)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    report = assess_capture_quality({"front": buf.getvalue()})
    assert report is not None
    assert isinstance(report, CaptureQualityReport)
    assert report.surfaces, "report must contain at least one surface"
    assert report.surfaces[0].surface == "front"
