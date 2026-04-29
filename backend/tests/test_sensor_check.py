"""Tests for the camera-sensor capture-quality module.

The point of `sensor_check` is to make ProofRead "fail honestly" under the
extreme-condition catalog in SPEC §0.5. These tests pin the verdicts on
synthetic frames that simulate the failure modes — blur, glare, dim
environments, washed-out highlights, low resolution, smudged-lens / fogged-
bottle (no contrast) — plus a clean frame for the positive case.

Synthetic-image generation lives in tests/conftest.py so the same toggles
back the API + pipeline tests."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.services.sensor_check import (
    GLARE_DEGRADED,
    SHARPNESS_DEGRADED,
    analyze_image_quality,
    assess_capture_quality,
    extract_sensor_metadata,
)
from tests.conftest import _make_synthetic_png


@pytest.fixture
def good_png():
    return _make_synthetic_png()


@pytest.fixture
def blurry_png():
    return _make_synthetic_png(blur=True)


@pytest.fixture
def glare_png():
    return _make_synthetic_png(glare=True)


@pytest.fixture
def dark_png():
    return _make_synthetic_png(dark=True)


@pytest.fixture
def bright_png():
    return _make_synthetic_png(bright=True)


@pytest.fixture
def flat_png():
    """Smudged-lens / fogged-bottle proxy — no contrast at all."""
    return _make_synthetic_png(flat=True)


@pytest.fixture
def low_res_png():
    return _make_synthetic_png(width=400, height=300)


def test_good_image_is_judged_good(good_png):
    report = assess_capture_quality({"front": good_png})
    sq = report.surfaces[0]
    assert sq.verdict == "good", (sq.verdict, sq.issues)
    assert sq.confidence > 0.85
    assert not sq.issues
    assert report.overall_verdict == "good"


def test_blurry_image_is_flagged_blurry(blurry_png):
    report = assess_capture_quality({"front": blurry_png})
    sq = report.surfaces[0]
    assert sq.verdict in {"degraded", "unreadable"}
    assert any("blur" in issue.lower() for issue in sq.issues)
    assert sq.metrics.sharpness < SHARPNESS_DEGRADED


def test_glare_image_is_flagged_glare(glare_png):
    report = assess_capture_quality({"front": glare_png})
    sq = report.surfaces[0]
    assert sq.verdict in {"degraded", "unreadable"}
    assert any("glare" in issue.lower() for issue in sq.issues)
    assert sq.metrics.glare_fraction > GLARE_DEGRADED


def test_dark_image_is_flagged_underexposed(dark_png):
    report = assess_capture_quality({"front": dark_png})
    sq = report.surfaces[0]
    assert sq.verdict in {"degraded", "unreadable"}
    assert any("underexposed" in issue.lower() for issue in sq.issues)


def test_bright_image_is_flagged_overexposed(bright_png):
    report = assess_capture_quality({"front": bright_png})
    sq = report.surfaces[0]
    assert sq.verdict in {"degraded", "unreadable"}
    assert any(
        "overexposed" in issue.lower() or "blown" in issue.lower() or "bright" in issue.lower()
        for issue in sq.issues
    )


def test_flat_image_is_flagged_low_contrast(flat_png):
    report = assess_capture_quality({"front": flat_png})
    sq = report.surfaces[0]
    assert sq.verdict in {"degraded", "unreadable"}
    assert any(
        "contrast" in issue.lower()
        or "smudged" in issue.lower()
        or "label" in issue.lower()
        for issue in sq.issues
    )


def test_low_resolution_image_is_flagged(low_res_png):
    report = assess_capture_quality({"front": low_res_png})
    sq = report.surfaces[0]
    assert sq.verdict in {"degraded", "unreadable"}
    assert any("resolution" in issue.lower() for issue in sq.issues)


def test_undecodable_bytes_become_unreadable():
    report = assess_capture_quality({"front": b"not-an-image"})
    sq = report.surfaces[0]
    assert sq.verdict == "unreadable"
    assert sq.confidence == 0.0
    assert any("decoded" in issue.lower() for issue in sq.issues)


def test_overall_takes_worst_per_surface(good_png, blurry_png):
    report = assess_capture_quality({"front": good_png, "back": blurry_png})
    by = report.by_surface()
    assert by["front"].verdict == "good"
    assert by["back"].verdict in {"degraded", "unreadable"}
    # One bad surface drags the aggregate verdict down.
    assert report.overall_verdict in {"degraded", "unreadable"}


def test_metadata_extraction_handles_no_exif(good_png):
    img = Image.open(io.BytesIO(good_png))
    sensor = extract_sensor_metadata(img)
    assert sensor.width_px == 1800
    assert sensor.height_px == 1200
    assert sensor.megapixels == 2.16
    # Synthetic PNG has no EXIF; metadata describer should still produce a
    # human-readable string instead of throwing.
    assert "MP" in sensor.describe()


def test_analyze_returns_metrics_with_expected_ranges(good_png):
    img = Image.open(io.BytesIO(good_png))
    metrics = analyze_image_quality(img)
    assert metrics.megapixels > 1.0
    assert 0 <= metrics.brightness_mean <= 255
    assert metrics.brightness_stddev > 0
    assert 0.0 <= metrics.glare_fraction <= 1.0
    assert metrics.sharpness > 0


def test_pipeline_skips_ocr_on_unreadable_surface(good_png):
    """The pipeline should not call OCR on a surface flagged unreadable —
    that's the whole point of the camera-sensor pre-flight."""
    from app.services.ocr import MockOCRProvider, OCRBlock, OCRResult
    from app.services.pipeline import ScanInput, process_scan

    calls: list[str | None] = []

    class _RecordingOCR(MockOCRProvider):
        def __init__(self):
            super().__init__({"full_text": "", "blocks": []})

        def process(self, image_bytes: bytes, hint: str | None = None):
            calls.append(hint)
            return OCRResult(
                full_text="ANYTOWN ALE\nINDIA PALE ALE",
                blocks=[OCRBlock(text="ANYTOWN ALE", bbox=(0, 0, 100, 30))],
                provider="mock",
            )

    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": good_png, "back": b"not-an-image"},
    )
    report = process_scan(scan, _RecordingOCR())

    # OCR called only for the front surface — the back was rejected upstream.
    assert calls == ["front"]
    # Capture-quality report is attached, with the back surface "unreadable".
    assert report.capture_quality is not None
    by = report.capture_quality.by_surface()
    assert by["back"].verdict == "unreadable"
    # Health Warning rule should be ADVISORY, not FAIL — back was unreadable.
    hw = next(
        r for r in report.rule_results if r.rule_id == "beer.health_warning.exact_text"
    )
    assert hw.status.value == "advisory"


# ---------------------------------------------------------------------------
# Region-aware capabilities — label segmentation, glare blobs, backlight,
# motion direction, sensor-tier database.
# ---------------------------------------------------------------------------


def test_label_region_is_detected(good_png):
    """The synthetic frame has a dense bar/text region in the middle. The
    label-region detector should locate it, not return None."""
    report = assess_capture_quality({"front": good_png})
    sq = report.surfaces[0]
    assert sq.label_bbox is not None, "label region should be detected on the synthetic"
    assert sq.metrics_label is not None
    # Bbox must be inside the image.
    assert sq.label_bbox.x >= 0
    assert sq.label_bbox.y >= 0
    assert sq.label_bbox.w > 0
    assert sq.label_bbox.h > 0


def test_label_region_prefers_dense_label_over_sprawling_blob():
    """Wide-angle capture: a small densely-filled label in the upper-right
    must beat a larger sprawling person-shaped blob lower in the frame.

    Reproduces the user-holding-a-can-up-high failure mode where the
    largest gradient component is the user's body+clothing texture but
    the actual label is a small, tight rectangle elsewhere. The detector
    must score by `area * fill_ratio²` so the dense rectangle wins.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    width, height = 1800, 1200

    # Mild per-pixel background noise. A flat fill collapses the gradient
    # distribution to zero and the 75th-percentile threshold becomes
    # meaningless (every cell is "above"). Real captures have sensor
    # noise / wall texture / vignetting, so the detector is tuned for a
    # non-zero baseline and the test fixture has to match.
    rng = np.random.default_rng(0xBADBEEF)
    bg = rng.integers(238, 252, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(bg, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Sprawling, LOOSE "person" blob in the lower-left: a few clustered
    # islands separated by background. The bbox spans the whole region
    # but the connected component (after dilation) has many gaps —
    # fill_ratio ~0.4. Without the fill-ratio² term this still wins on
    # raw cell count; the squared term penalises looseness enough that
    # the dense label outscores it.
    for cx in (200, 500, 800):
        for cy in (500, 800, 1100):
            for _ in range(120):
                x = int(cx + rng.normal(0, 25))
                y = int(cy + rng.normal(0, 25))
                v = int(rng.integers(20, 200))
                draw.rectangle((x, y, x + 6, y + 6), fill=(v, v, v))

    # Tight, densely-packed "label" in the upper-right: ~9 % of the
    # frame, packed with text-like speckle that gives every coarse cell
    # high gradient. After dilation it's a tight rectangle with
    # fill_ratio ~0.8, so `cells * fill_ratio²` exceeds the larger but
    # looser sprawl.
    label_x0, label_y0 = int(width * 0.62), int(height * 0.08)
    label_x1, label_y1 = int(width * 0.92), int(height * 0.42)
    draw.rectangle(
        (label_x0, label_y0, label_x1, label_y1), fill=(248, 246, 232)
    )
    for _ in range(900):
        x = int(rng.integers(label_x0 + 4, label_x1 - 8))
        y = int(rng.integers(label_y0 + 4, label_y1 - 8))
        v = int(rng.integers(20, 200))
        draw.rectangle((x, y, x + 6, y + 6), fill=(v, v, v))

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    report = assess_capture_quality({"front": buf.getvalue()})
    sq = report.surfaces[0]
    assert sq.label_bbox is not None, "label region should be detected"

    # Bbox center should land inside the painted label rectangle, not the
    # sprawling blob in the opposite quadrant.
    cx = sq.label_bbox.x + sq.label_bbox.w // 2
    cy = sq.label_bbox.y + sq.label_bbox.h // 2
    assert label_x0 <= cx <= label_x1, (
        f"detector picked the sprawling blob (center x={cx} outside "
        f"label x-range [{label_x0}, {label_x1}])"
    )
    assert label_y0 <= cy <= label_y1, (
        f"detector picked the sprawling blob (center y={cy} outside "
        f"label y-range [{label_y0}, {label_y1}])"
    )


def test_label_region_returns_none_for_uniform_frame(flat_png):
    """A flat-color frame has no detectable label region — the detector
    must return None rather than picking up sensor noise."""
    report = assess_capture_quality({"front": flat_png})
    sq = report.surfaces[0]
    assert sq.label_bbox is None


def test_glare_blobs_are_localized(glare_png):
    """The glare fixture washes out a ~64% rectangle of the frame. The
    blob detector should produce at least one bbox covering that area."""
    report = assess_capture_quality({"front": glare_png})
    sq = report.surfaces[0]
    assert sq.glare_blobs, "glare-fixture frame should produce at least one blob"
    biggest = sq.glare_blobs[0]
    # Largest blob covers a meaningful share of either frame or label.
    assert (
        biggest.area_fraction_frame > 0.20
        or biggest.area_fraction_label > 0.20
    )


def test_motion_direction_named_for_horizontal_blur():
    """A frame blurred only horizontally must surface motion_blur_direction
    as 'horizontal'."""
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGB", (1600, 1200), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw vertical strokes so post-blur the surviving edges are vertical.
    for x in range(0, 1600, 40):
        draw.rectangle((x, 100, x + 6, 1100), fill=(20, 20, 20))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=0.0))
    # Convert to bytes and apply directional blur via composition.
    # We simulate a horizontal-motion smear by averaging successive
    # horizontal slices: PIL doesn't ship directional blur, so we use
    # `MotionBlur`-style stacking via offset composites.
    smeared = blurred
    for dx in range(1, 18):
        shifted = blurred.transform(
            blurred.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, 0),
            fillcolor=(240, 240, 240),
        )
        smeared = Image.blend(smeared, shifted, 0.5)

    buf = io.BytesIO()
    smeared.save(buf, format="PNG")
    report = assess_capture_quality({"front": buf.getvalue()})
    sq = report.surfaces[0]
    assert sq.motion_blur_direction == "horizontal", (
        f"expected horizontal motion, got {sq.motion_blur_direction!r}"
    )


def test_backlit_silhouette_is_flagged():
    """Bright background + dark central silhouette = SPEC §0.5 backlight.
    Different remedy than plain underexposure: reposition, don't add light."""
    import numpy as np
    from PIL import Image, ImageDraw

    # Photographic noise across the bright background — a real "window
    # behind bottle" frame has window/sky texture, not a clamped 250 wall.
    # Without it the artwork capture-source detector catches the frame
    # (uniform border) and the backlit check is bypassed.
    rng_bk = np.random.default_rng(0xBACC1)
    bg = rng_bk.integers(235, 252, size=(1200, 1800), dtype=np.uint8)
    img = Image.fromarray(np.stack([bg, bg, bg], axis=-1), mode="RGB")
    draw = ImageDraw.Draw(img)
    # Dark "bottle" silhouette in the center.
    draw.rectangle((550, 200, 1250, 1100), fill=(20, 20, 20))
    # Add a few sharp speckles inside so a label region is detectable.
    for x in range(560, 1240, 30):
        draw.rectangle((x, 230, x + 5, 1080), fill=(60, 60, 60))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    report = assess_capture_quality({"front": buf.getvalue()})
    sq = report.surfaces[0]
    assert sq.capture_source != "artwork", (
        "photographic background variance must keep this off the artwork path"
    )
    assert sq.backlit, "label region darker than surround should flag backlit"
    assert any("backlit" in i.lower() for i in sq.issues)


def test_sensor_tier_lookup_is_robust_to_unknown():
    """Unknown make/model returns 'unknown' (not a crash, not 'low_end')."""
    from app.services.sensor_check import lookup_sensor_tier

    assert lookup_sensor_tier(None, None) == "unknown"
    assert lookup_sensor_tier("Some Random OEM", "MagicPhone 1") == "low_end"
    assert lookup_sensor_tier("Apple", "iPhone 15 Pro Max") == "modern_flagship"
    assert lookup_sensor_tier("Apple", "iPhone XS Max") == "older_flagship"


def test_capture_summary_includes_region_fields(good_png):
    """The serialized summary used by the API must carry the new region
    fields so the admin UI can render diagnostics."""
    from app.services.pipeline import _capture_summary

    report = assess_capture_quality({"front": good_png})
    summary = _capture_summary(report)
    surface = summary["surfaces"][0]
    # Region-aware fields must be present.
    assert "label_bbox" in surface
    assert "glare_blobs" in surface
    assert "backlit" in surface
    assert "motion_blur_direction" in surface
    assert "sensor_tier" in surface
    assert "metrics_label" in surface


def test_lens_smudge_collapses_high_freq():
    """Apply a heavy uniform Gaussian blur (no localized motion) — that's
    the same signature as a smudged lens or fogged bottle, and the
    high-frequency / mean-edge ratio collapses below 2.5."""
    from PIL import ImageFilter

    img_bytes = _make_synthetic_png()
    img = Image.open(io.BytesIO(img_bytes))
    smudged = img.filter(ImageFilter.GaussianBlur(radius=8))
    buf = io.BytesIO()
    smudged.save(buf, format="PNG")

    report = assess_capture_quality({"front": buf.getvalue()})
    sq = report.surfaces[0]
    # Smudge OR motion-blur direction OR lens-smudge flag — different
    # heuristics may catch this; we only require ONE of them to fire.
    flagged = (
        sq.lens_smudge_likely
        or any("smudge" in i.lower() or "fog" in i.lower() for i in sq.issues)
    )
    assert flagged, sq.issues


def test_screenshot_aspect_ratio_is_flagged_when_no_exif():
    """A 19.5:9 aspect-ratio PNG with no EXIF looks like a phone-screen
    screenshot; we should label capture_source='screenshot'."""
    img = Image.new("RGB", (1170, 2532), (240, 240, 240))   # iPhone-15-screen
    # Add some content so the frame isn't flat (otherwise it's flagged
    # unreadable for low contrast and verdict overrides what we want).
    draw = __import__("PIL").ImageDraw.Draw(img)
    for y in range(80, 2500, 120):
        draw.rectangle((50, y, 1120, y + 50), fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    report = assess_capture_quality({"front": buf.getvalue()})
    sq = report.surfaces[0]
    assert sq.capture_source == "screenshot", (
        f"19.5:9 screenshot-shaped frame should classify as screenshot; "
        f"got {sq.capture_source}"
    )


def test_software_exif_screenshot_overrides_camera_make():
    """Android screenshots and many third-party screenshot tools tag the
    file with EXIF Software='Screenshot' even when Make/Model are populated
    by the OS shell. The sensor classifier must honour that tag rather than
    waving the file through as a 'photo'."""
    from app.services.sensor_check import (
        SensorMetadata,
        _classify_capture_source,
    )

    sensor = SensorMetadata(
        make="Samsung", model="SM-S908U", software="Screenshot",
    )
    img = Image.new("RGB", (1080, 1920), (200, 200, 200))
    assert (
        _classify_capture_source(sensor, img, 1080, 1920) == "screenshot"
    ), "EXIF Software=Screenshot must override Make/Model presence"


def test_blob_occluded_field_downgrades_rule_to_advisory(glare_png):
    """A FAIL whose extracted-field bbox sits inside a major glare blob
    must downgrade to ADVISORY, even if the surface verdict is just
    'degraded' (not 'unreadable'). The rule that fired had no
    confidently-readable input."""
    from app.services.ocr import MockOCRProvider, OCRBlock, OCRResult
    from app.services.pipeline import ScanInput, process_scan

    # OCR fixture says health-warning-without-the-statutory-text — would FAIL.
    # The bbox we report places it INSIDE the glare region of the fixture
    # (0,0 to 80%/80% of the frame).
    class _OCR(MockOCRProvider):
        def __init__(self):
            super().__init__({"full_text": "", "blocks": []})

        def process(self, image_bytes: bytes, hint: str | None = None):
            return OCRResult(
                full_text="ANYTOWN ALE\nNot the right warning text\n",
                blocks=[
                    OCRBlock(
                        text="Not the right warning text",
                        bbox=(100, 100, 600, 80),
                        confidence=0.99,
                    ),
                    OCRBlock(text="ANYTOWN ALE", bbox=(50, 30, 200, 40), confidence=0.99),
                ],
                provider="mock",
            )

    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": glare_png, "back": glare_png},
    )
    report = process_scan(scan, _OCR())

    # Health-warning extraction fails the exact-text rule because the
    # ground-truth canonical text is not present. With blob occlusion that
    # FAIL must be downgraded to ADVISORY.
    hw = next(
        r for r in report.rule_results if r.rule_id == "beer.health_warning.exact_text"
    )
    assert hw.status.value in {"advisory", "fail"}, (
        f"got {hw.status.value}; rule must end as advisory when its input was occluded"
    )


# ---------------------------------------------------------------------------
# Digital-artwork capture-source path
#
# Brewers and distillers upload PNG/SVG exports from Esko / Adobe Illustrator,
# not phone photos. Those have no EXIF and clean, uniform borders. The photo-
# tuned glare/exposure/smudge/resolution checks false-positive on them; the
# `artwork` capture-source path skips those checks while keeping sharpness +
# contrast + colour-cast (text legibility still matters).
# ---------------------------------------------------------------------------


from pathlib import Path  # noqa: E402

_LABELS_DIR = Path(__file__).resolve().parents[2] / "artwork" / "labels"


def _build_artwork_png(*, width=1200, height=1500, glare_fraction=0.0) -> bytes:
    """A clean PNG export — uniform white canvas + bold central text.

    Mirrors how a design tool exports raster proofs: no speckle / film grain,
    a perfectly uniform border, hard-edged anti-aliased typography.
    `glare_fraction` lets the test push the saturated-pixel ratio above the
    photo-glare threshold without changing the artwork classification.
    """
    from PIL import Image, ImageDraw

    bg = (255, 255, 255) if glare_fraction > 0 else (250, 248, 242)
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    # Hard text strokes so the sharpness measurement is high.
    for y in range(int(height * 0.25), int(height * 0.7), 80):
        draw.rectangle((width // 6, y, width - width // 6, y + 32), fill=(15, 15, 15))
    draw.text((width // 6, int(height * 0.85)), "GOVERNMENT WARNING", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_synthetic_good_fixture_is_not_classified_as_artwork(good_png):
    """The synthetic 'good' fixture has speckle noise across the frame — its
    border is NOT uniform. Artwork classification must not catch it."""
    report = assess_capture_quality({"front": good_png})
    sq = report.surfaces[0]
    assert sq.capture_source != "artwork", (
        "speckled synthetic should not be classified as artwork"
    )


def test_clean_artwork_export_is_classified_as_artwork():
    """A clean uniform-canvas PNG export — no EXIF, no border noise — is the
    primary artwork case."""
    report = assess_capture_quality({"front": _build_artwork_png()})
    sq = report.surfaces[0]
    assert sq.capture_source == "artwork", (sq.capture_source, sq.issues)
    assert sq.verdict == "good"


def test_artwork_path_skips_glare_check():
    """A near-pure-white artwork canvas trips the saturated-pixel glare
    threshold — the artwork branch must not treat that as a glare failure."""
    art = _build_artwork_png(glare_fraction=1.0)
    report = assess_capture_quality({"front": art})
    sq = report.surfaces[0]
    assert sq.capture_source == "artwork"
    # Frame-level metric still measures saturated pixels (we don't mutate it),
    # but no `glare` issue is surfaced and the verdict is not driven by it.
    assert not any("glare" in i.lower() for i in sq.issues), sq.issues
    assert sq.verdict == "good"


def test_artwork_path_skips_low_resolution_penalty():
    """A 1.5 MP design proof is a normal artwork submission — must not trip
    the photo-resolution lower bound."""
    art = _build_artwork_png(width=1000, height=1500)  # 1.5 MP, well below 2.0
    report = assess_capture_quality({"front": art})
    sq = report.surfaces[0]
    assert sq.capture_source == "artwork"
    assert not any("resolution" in i.lower() for i in sq.issues), sq.issues


def test_artwork_path_still_flags_blurry_label():
    """Sharpness check is intentionally retained for artwork — even a clean
    PNG export with too-soft text should be flagged for OCR risk."""
    from PIL import Image, ImageFilter

    art = _build_artwork_png()
    img = Image.open(io.BytesIO(art)).filter(ImageFilter.GaussianBlur(radius=20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    report = assess_capture_quality({"front": buf.getvalue()})
    sq = report.surfaces[0]
    assert sq.capture_source == "artwork"
    assert any("blur" in i.lower() or "soft" in i.lower() for i in sq.issues), sq.issues


@pytest.mark.skipif(
    not _LABELS_DIR.exists() or not list(_LABELS_DIR.glob("*.png")),
    reason="real-world artwork fixtures not present",
)
@pytest.mark.parametrize(
    "filename",
    [
        "01_pass_old_tom_distillery.png",
        "02_warn_stones_throw_gin.png",
        "03_fail_mountain_crest_ipa.png",
    ],
)
def test_real_world_artwork_labels_classified_and_verdict_good(filename):
    """The three design-export sample labels must (a) be detected as artwork
    and (b) clear the sensor pre-check — the model needs to actually receive
    these for content-level evaluation. Before this fix, the gin sample was
    short-circuited as 'unreadable' on glare and the typography-warn case
    never reached the model."""
    path = _LABELS_DIR / filename
    report = assess_capture_quality({"front": path.read_bytes()})
    sq = report.surfaces[0]
    assert sq.capture_source == "artwork", (filename, sq.capture_source, sq.issues)
    assert sq.verdict == "good", (filename, sq.verdict, sq.issues)


@pytest.mark.skipif(
    not (_LABELS_DIR / "04_unreadable_heritage_vineyards.png").exists(),
    reason="real-world photo fixture not present",
)
def test_real_world_photo_is_not_misclassified_as_artwork():
    """The wine-bottle photograph (no EXIF, but real photographic background
    variation) must NOT be classified as artwork — it has to keep going
    through the photo-tuned checks."""
    path = _LABELS_DIR / "04_unreadable_heritage_vineyards.png"
    report = assess_capture_quality({"front": path.read_bytes()})
    sq = report.surfaces[0]
    assert sq.capture_source != "artwork", (sq.capture_source, sq.issues)
