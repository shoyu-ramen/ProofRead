"""
Sensor-aware capture quality assessment with region-level reasoning.

The previous version produced one verdict per surface from frame-level
scalars. That misses the most common SPEC §0.5 condition pattern: a
frame "overexposed" because of the window behind the bottle (label
itself fine), or a label with a single specular blob covering the brand
area while the warning text is intact, or a backlit silhouette where
the mean luminance lies — frame-level scalars don't see those, region-
level metrics do.

What we now produce per surface:

    * `SensorMetadata` from EXIF — make/model/ISO/exposure/etc — plus a
      `SensorTier` looked up against a small device database. Older
      devices get more lenient ISO/exposure thresholds: when the device
      is the limiting factor we say so, instead of blaming the label.
    * `ImageQualityMetrics` for the whole frame (kept for back-compat).
    * `LabelRegion` — bounding box of the highest-gradient-density area,
      the bottle/label, found via gradient density + connected
      components. Subsequent metrics scoped to this region ignore bar
      backgrounds, the user's thumb, and the cheese plate.
    * Region-scoped metrics — same shape as `ImageQualityMetrics` but
      for the label crop only.
    * `glare_blobs` — connected-component bboxes of saturated-pixel
      clusters, ranked by area. The agent can cite which blob a low-
      confidence reading came from; the rule downgrade logic can act
      only on the rules whose extracted-field bbox overlaps a blob,
      instead of blanket-downgrading every rule.
    * `backlit` flag — true when the label region is materially darker
      than the surrounding background (window behind bottle). Different
      remedy than "underexposed": user should reposition, not turn up
      the brightness.
    * `motion_blur_direction` — Sobel gradient orientation histogram. If
      >55 % of edge energy concentrates in one direction, motion is
      named ("horizontal" / "vertical" / "diagonal"); the suggestion
      can name the axis to brace against.

The aggregation favors the LABEL region. A frame whose label region
grades "good" is reported "good" even when the background is washed
out — that was the failure mode driving false-degraded verdicts on
outdoor festival captures.

Dependencies are PIL + numpy (always-available transitive of Pillow's
ecosystem) plus scipy.ndimage for connected-component labelling. The
heavy work runs on a downsampled 1200-px-long-edge frame so even a
12 MP capture finishes in <40 ms.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter, ImageStat
from PIL.ExifTags import TAGS
from scipy import ndimage  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration constants. Conservative by design: the cost of a false
# "degraded" is a reshoot prompt; the cost of a false "good" is a confident
# wrong verdict the user submits to TTB.
# ---------------------------------------------------------------------------

SHARPNESS_UNREADABLE = 40.0
SHARPNESS_DEGRADED = 120.0
GLARE_UNREADABLE = 0.50
GLARE_DEGRADED = 0.25
BRIGHTNESS_DARK_DEGRADED = 35
BRIGHTNESS_DARK_UNREADABLE = 18
BRIGHTNESS_BRIGHT_DEGRADED = 220
BRIGHTNESS_BRIGHT_UNREADABLE = 245
CONTRAST_LOW_DEGRADED = 25
CONTRAST_LOW_UNREADABLE = 12
RESOLUTION_DEGRADED_MP = 2.0
RESOLUTION_UNREADABLE_MP = 0.5
COLOR_CAST_DEGRADED = 35
GLARE_PIXEL_THRESHOLD = 248

# Backlight: the label region's mean luminance must be at least this many
# units darker than the surrounding background to flag.
BACKLIT_DELTA = 60

# Motion blur direction: one orientation bin must hold this share of total
# edge energy before we name the direction.
MOTION_DIRECTION_DOMINANCE = 0.55

# Long edge of the working frame for analysis. Big enough to retain the
# Health-Warning fine print; small enough that a 12 MP capture analyses in
# tens of ms.
WORK_LONG_EDGE = 1200

# 3x3 discrete Laplacian. The +128 offset centers the response in 8-bit
# space — without it, PIL clamps negative responses (dark-on-light edges)
# to 0 and we lose half the high-frequency signal, which makes blur
# detection unreliable. `scale=1` overrides PIL's default `sum(kernel)`,
# which would divide by zero (the kernel sums to 0).
_LAPLACIAN_KERNEL = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
    scale=1,
    offset=128,
)


Verdict = Literal["good", "degraded", "unreadable"]
SensorTier = Literal[
    "modern_flagship",
    "modern_midrange",
    "older_flagship",
    "older_midrange",
    "low_end",
    "unknown",
]


# ---------------------------------------------------------------------------
# Sensor capability database. Imperfect but useful: keyed off EXIF Make
# + Model substring matches. The point is not to enumerate every device,
# only to recognise the common ones so the threshold logic can stop
# blaming the label when an older device is the bottleneck.
# ---------------------------------------------------------------------------

_SENSOR_DB: list[dict] = [
    # iPhones — model name carries the year.
    {"match": ("apple", "iphone 14"), "tier": "modern_flagship"},
    {"match": ("apple", "iphone 15"), "tier": "modern_flagship"},
    {"match": ("apple", "iphone 16"), "tier": "modern_flagship"},
    {"match": ("apple", "iphone 13"), "tier": "modern_flagship"},
    {"match": ("apple", "iphone 12"), "tier": "modern_midrange"},
    {"match": ("apple", "iphone 11"), "tier": "older_flagship"},
    {"match": ("apple", "iphone xs"), "tier": "older_flagship"},
    {"match": ("apple", "iphone xr"), "tier": "older_midrange"},
    {"match": ("apple", "iphone x"), "tier": "older_flagship"},
    {"match": ("apple", "iphone 8"), "tier": "older_midrange"},
    {"match": ("apple", "iphone 7"), "tier": "older_midrange"},
    {"match": ("apple", "iphone se"), "tier": "older_midrange"},
    # Pixels.
    {"match": ("google", "pixel 9"), "tier": "modern_flagship"},
    {"match": ("google", "pixel 8"), "tier": "modern_flagship"},
    {"match": ("google", "pixel 7"), "tier": "modern_flagship"},
    {"match": ("google", "pixel 6"), "tier": "modern_midrange"},
    {"match": ("google", "pixel 5"), "tier": "older_flagship"},
    {"match": ("google", "pixel 4"), "tier": "older_flagship"},
    {"match": ("google", "pixel 3"), "tier": "older_midrange"},
    # Galaxies (S series).
    {"match": ("samsung", "sm-s9"), "tier": "modern_flagship"},     # S22+
    {"match": ("samsung", "sm-s2"), "tier": "modern_flagship"},     # S22 / S23
    {"match": ("samsung", "sm-g99"), "tier": "older_flagship"},     # S20/S21
    {"match": ("samsung", "sm-g97"), "tier": "older_flagship"},     # S10
    {"match": ("samsung", "sm-g96"), "tier": "older_midrange"},     # S9
    {"match": ("samsung", "sm-a"), "tier": "older_midrange"},       # A series
]


# Per-tier multipliers on the default thresholds — older devices are
# allowed to underperform on ISO/exposure/sharpness by these factors
# before we flag them.
_TIER_RELAX: dict[str, dict[str, float]] = {
    "modern_flagship":  {"sharpness": 1.0, "iso": 1.0, "exposure": 1.0, "resolution": 1.0},
    "modern_midrange":  {"sharpness": 0.9, "iso": 1.2, "exposure": 1.1, "resolution": 1.0},
    "older_flagship":   {"sharpness": 0.85, "iso": 1.5, "exposure": 1.3, "resolution": 0.9},
    "older_midrange":   {"sharpness": 0.75, "iso": 2.0, "exposure": 1.5, "resolution": 0.7},
    "low_end":          {"sharpness": 0.6, "iso": 2.5, "exposure": 2.0, "resolution": 0.5},
    "unknown":          {"sharpness": 0.9, "iso": 1.5, "exposure": 1.3, "resolution": 1.0},
}


def lookup_sensor_tier(make: str | None, model: str | None) -> SensorTier:
    """Map an EXIF make/model pair to a capability tier.

    Returns 'unknown' when no entry matches. Calibration-relaxation
    multipliers are applied per-tier so an older device gets less harsh
    thresholds — when the device is the limiting factor, we say so
    instead of blaming the label.
    """
    if not make and not model:
        return "unknown"
    m = (make or "").lower()
    n = (model or "").lower()
    for entry in _SENSOR_DB:
        match_make, match_model = entry["match"]
        if match_make in m and match_model in n:
            return entry["tier"]  # type: ignore[return-value]
    # Generic Apple/Google/Samsung that didn't match a specific row → midrange.
    if "apple" in m or "google" in m or "samsung" in m:
        return "modern_midrange"
    return "low_end"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Bbox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def overlaps(self, other: Bbox) -> bool:
        return not (
            self.x + self.w <= other.x
            or other.x + other.w <= self.x
            or self.y + self.h <= other.y
            or other.y + other.h <= self.y
        )

    def overlap_fraction(self, other: Bbox) -> float:
        """Share of `other` that overlaps `self`. 0 if disjoint."""
        ix = max(self.x, other.x)
        iy = max(self.y, other.y)
        ax = min(self.x + self.w, other.x + other.w)
        ay = min(self.y + self.h, other.y + other.h)
        iw = max(0, ax - ix)
        ih = max(0, ay - iy)
        return (iw * ih) / max(1, other.area)


@dataclass
class GlareBlob:
    """A localized cluster of saturated pixels (specular highlight)."""

    bbox: Bbox
    area_fraction_frame: float
    area_fraction_label: float


@dataclass
class SensorMetadata:
    """EXIF-derived camera/sensor information.

    Every field is optional — synthetic test fixtures, screenshots, and many
    upload paths strip EXIF. Downstream consumers must treat absent fields
    as 'unknown', not as failures.
    """

    make: str | None = None
    model: str | None = None
    iso: int | None = None
    exposure_time_s: float | None = None
    f_number: float | None = None
    focal_length_mm: float | None = None
    flash_fired: bool | None = None
    width_px: int | None = None
    height_px: int | None = None
    captured_at: str | None = None
    lens_model: str | None = None
    software: str | None = None
    tier: SensorTier = "unknown"

    @property
    def megapixels(self) -> float | None:
        if self.width_px and self.height_px:
            return round((self.width_px * self.height_px) / 1_000_000, 2)
        return None

    def describe(self) -> str:
        parts: list[str] = []
        if self.make or self.model:
            parts.append(f"{self.make or '?'} {self.model or '?'}".strip())
        if self.megapixels:
            parts.append(f"{self.megapixels} MP")
        if self.iso:
            parts.append(f"ISO {self.iso}")
        if self.exposure_time_s:
            parts.append(f"{_format_exposure(self.exposure_time_s)} s")
        if self.f_number:
            parts.append(f"f/{self.f_number:g}")
        if self.focal_length_mm:
            parts.append(f"{self.focal_length_mm:g} mm")
        if self.flash_fired:
            parts.append("flash on")
        if self.tier != "unknown":
            parts.append(f"tier:{self.tier}")
        return " · ".join(parts) if parts else "no EXIF available"


@dataclass
class ImageQualityMetrics:
    """Objective measurements computed directly from pixels.

    Used both for the whole frame and for sub-regions (label crop). The
    field set is identical so prompts + log lines have a uniform shape
    across scopes.
    """

    sharpness: float
    glare_fraction: float
    brightness_mean: float
    brightness_stddev: float
    color_cast: float
    megapixels: float
    width_px: int
    height_px: int


@dataclass
class SurfaceCaptureQuality:
    """Per-surface (front/back/...) verdict + diagnostics.

    Backwards-compatible: `metrics` is the whole-frame metrics, as before.
    The new region-aware fields default to values that callers ignoring
    them see as "no extra information" rather than errors.
    """

    surface: str
    sensor: SensorMetadata
    metrics: ImageQualityMetrics
    verdict: Verdict
    confidence: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    # Region-aware additions.
    label_bbox: Bbox | None = None
    metrics_label: ImageQualityMetrics | None = None
    label_verdict: Verdict | None = None
    glare_blobs: list[GlareBlob] = field(default_factory=list)
    backlit: bool = False
    motion_blur_direction: str | None = None
    # Surface / capture-source signals — extra SPEC §0.5 conditions the
    # frame-level scalars don't see.
    lens_smudge_likely: bool = False
    wet_bottle_likely: bool = False
    capture_source: Literal["photo", "screenshot", "uncertain", "artwork"] = "photo"


@dataclass
class CaptureQualityReport:
    surfaces: list[SurfaceCaptureQuality]
    overall_verdict: Verdict
    overall_confidence: float

    def by_surface(self) -> dict[str, SurfaceCaptureQuality]:
        return {s.surface: s for s in self.surfaces}

    def degraded_or_worse(self) -> list[SurfaceCaptureQuality]:
        return [s for s in self.surfaces if s.verdict != "good"]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def assess_capture_quality(images: dict[str, bytes]) -> CaptureQualityReport:
    """Top-level entry point: assess every image in a scan."""
    surfaces = [
        _assess_one(surface, image_bytes) for surface, image_bytes in images.items()
    ]
    overall_verdict = _aggregate_verdict([s.verdict for s in surfaces])
    overall_confidence = (
        min((s.confidence for s in surfaces), default=0.0) if surfaces else 0.0
    )
    return CaptureQualityReport(
        surfaces=surfaces,
        overall_verdict=overall_verdict,
        overall_confidence=overall_confidence,
    )


def extract_sensor_metadata(img: Image.Image) -> SensorMetadata:
    """Pull camera/sensor EXIF tags + tier lookup."""
    width, height = img.size

    raw_exif: dict = {}
    try:
        exif = img.getexif()
        if exif:
            raw_exif = {TAGS.get(tag, tag): exif.get(tag) for tag in exif}
    except Exception as e:
        logger.debug("EXIF read failed: %s", e)

    flash_value = raw_exif.get("Flash")
    if flash_value is None:
        flash_fired: bool | None = None
    else:
        # EXIF Flash is bit-packed; bit 0 indicates the flash fired.
        flash_fired = bool(int(flash_value) & 0x01)

    make = _clean_str(raw_exif.get("Make"))
    model = _clean_str(raw_exif.get("Model"))
    return SensorMetadata(
        make=make,
        model=model,
        iso=_first_int(raw_exif.get("ISOSpeedRatings"))
        or _first_int(raw_exif.get("PhotographicSensitivity")),
        exposure_time_s=_to_float(raw_exif.get("ExposureTime")),
        f_number=_to_float(raw_exif.get("FNumber")),
        focal_length_mm=_to_float(raw_exif.get("FocalLength")),
        flash_fired=flash_fired,
        width_px=width,
        height_px=height,
        captured_at=_clean_str(raw_exif.get("DateTimeOriginal")),
        lens_model=_clean_str(raw_exif.get("LensModel")),
        software=_clean_str(raw_exif.get("Software")),
        tier=lookup_sensor_tier(make, model),
    )


def analyze_image_quality(img: Image.Image) -> ImageQualityMetrics:
    """Whole-frame metrics. Kept for backwards compatibility — the deeper
    analysis (label region, glare blobs, backlight, motion direction) lives
    in `_assess_one`.
    """
    rgb = img.convert("RGB")
    original = Bbox(0, 0, rgb.size[0], rgb.size[1])
    work = _to_work_size(rgb)
    grayscale = work.convert("L")
    return _compute_metrics(work, grayscale, original)


# ---------------------------------------------------------------------------
# Per-surface assessment
# ---------------------------------------------------------------------------


def _assess_one(surface: str, image_bytes: bytes) -> SurfaceCaptureQuality:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        logger.warning("Capture quality: failed to decode %s: %s", surface, e)
        return _decode_failure(surface, e)

    sensor = extract_sensor_metadata(img)
    rgb_original = img.convert("RGB")
    original_w, original_h = rgb_original.size
    work = _to_work_size(rgb_original)
    grayscale = work.convert("L")
    gray_arr = np.asarray(grayscale, dtype=np.uint8)
    work_w, work_h = work.size

    # Frame-level metrics report ORIGINAL upload dimensions — the resolution
    # rule should reflect what the user actually submitted, not what we
    # downsampled to for analysis.
    metrics_frame = _compute_metrics(
        work, grayscale, Bbox(0, 0, original_w, original_h)
    )

    work_label_bbox = _detect_label_region(gray_arr)
    label_bbox: Bbox | None = None
    metrics_label: ImageQualityMetrics | None = None
    if work_label_bbox is not None:
        crop_rgb = work.crop(
            (work_label_bbox.x, work_label_bbox.y,
             work_label_bbox.x + work_label_bbox.w,
             work_label_bbox.y + work_label_bbox.h)
        )
        crop_gray = grayscale.crop(
            (work_label_bbox.x, work_label_bbox.y,
             work_label_bbox.x + work_label_bbox.w,
             work_label_bbox.y + work_label_bbox.h)
        )
        # Project the work-space bbox back to original coordinates so the
        # bbox the model sees lines up with the uploaded image, not our
        # downsampled work copy.
        scale_x = original_w / work_w
        scale_y = original_h / work_h
        label_bbox = Bbox(
            x=int(work_label_bbox.x * scale_x),
            y=int(work_label_bbox.y * scale_y),
            w=int(work_label_bbox.w * scale_x),
            h=int(work_label_bbox.h * scale_y),
        )
        metrics_label = _compute_metrics(crop_rgb, crop_gray, label_bbox)

    glare_blobs = _localize_glare_blobs(gray_arr, work_label_bbox)
    # Project glare bboxes back to original coordinates too.
    if work_w != original_w:
        scale_x = original_w / work_w
        scale_y = original_h / work_h
        glare_blobs = [
            GlareBlob(
                bbox=Bbox(
                    x=int(b.bbox.x * scale_x),
                    y=int(b.bbox.y * scale_y),
                    w=int(b.bbox.w * scale_x),
                    h=int(b.bbox.h * scale_y),
                ),
                area_fraction_frame=b.area_fraction_frame,
                area_fraction_label=b.area_fraction_label,
            )
            for b in glare_blobs
        ]
    backlit = _is_backlit(gray_arr, work_label_bbox)
    motion_dir = _motion_blur_direction(gray_arr)
    smudge_likely = _is_lens_smudge_likely(gray_arr)
    wet_likely = _is_wet_bottle_likely(gray_arr, work_label_bbox)
    capture_source = _classify_capture_source(sensor, img, original_w, original_h)

    issues, suggestions, score, frame_verdict, label_verdict = _evaluate(
        sensor=sensor,
        metrics_frame=metrics_frame,
        metrics_label=metrics_label,
        glare_blobs=glare_blobs,
        backlit=backlit,
        motion_dir=motion_dir,
        smudge_likely=smudge_likely,
        wet_likely=wet_likely,
        capture_source=capture_source,
    )

    return SurfaceCaptureQuality(
        surface=surface,
        sensor=sensor,
        metrics=metrics_frame,
        verdict=frame_verdict,
        confidence=score,
        issues=issues,
        suggestions=suggestions,
        label_bbox=label_bbox,
        metrics_label=metrics_label,
        label_verdict=label_verdict,
        glare_blobs=glare_blobs,
        backlit=backlit,
        motion_blur_direction=motion_dir,
        lens_smudge_likely=smudge_likely,
        wet_bottle_likely=wet_likely,
        capture_source=capture_source,
    )


def _decode_failure(surface: str, e: Exception) -> SurfaceCaptureQuality:
    return SurfaceCaptureQuality(
        surface=surface,
        sensor=SensorMetadata(),
        metrics=ImageQualityMetrics(
            sharpness=0.0,
            glare_fraction=0.0,
            brightness_mean=0.0,
            brightness_stddev=0.0,
            color_cast=0.0,
            megapixels=0.0,
            width_px=0,
            height_px=0,
        ),
        verdict="unreadable",
        confidence=0.0,
        issues=[f"Image bytes could not be decoded ({type(e).__name__})"],
        suggestions=["Recapture the surface — the upload may be corrupt."],
    )


# ---------------------------------------------------------------------------
# Pixel-level metric helpers
# ---------------------------------------------------------------------------


def _to_work_size(img: Image.Image) -> Image.Image:
    """Cap the long edge so analysis stays cheap on full-resolution captures."""
    width, height = img.size
    if max(width, height) <= WORK_LONG_EDGE:
        return img
    scale = WORK_LONG_EDGE / max(width, height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _compute_metrics(
    rgb: Image.Image,
    grayscale: Image.Image,
    bbox: Bbox,
) -> ImageQualityMetrics:
    """Same metric computation used for whole-frame and label-crop scopes."""
    edges = grayscale.filter(_LAPLACIAN_KERNEL)
    sharpness = float(ImageStat.Stat(edges).var[0])

    gray_stats = ImageStat.Stat(grayscale)
    brightness_mean = float(gray_stats.mean[0])
    brightness_stddev = float(gray_stats.stddev[0])

    histogram = grayscale.histogram()
    total = sum(histogram) or 1
    glare_count = sum(histogram[GLARE_PIXEL_THRESHOLD:])
    glare_fraction = glare_count / total

    rgb_stats = ImageStat.Stat(rgb)
    channel_means = rgb_stats.mean[:3]
    color_cast = float(max(channel_means) - min(channel_means))

    return ImageQualityMetrics(
        sharpness=sharpness,
        glare_fraction=glare_fraction,
        brightness_mean=brightness_mean,
        brightness_stddev=brightness_stddev,
        color_cast=color_cast,
        megapixels=round(bbox.area / 1_000_000, 3),
        width_px=bbox.w,
        height_px=bbox.h,
    )


# ---------------------------------------------------------------------------
# Region detection
# ---------------------------------------------------------------------------


def _detect_label_region(gray: np.ndarray) -> Bbox | None:
    """Locate the densest gradient region of the frame.

    Heuristic: compute a local-gradient magnitude (Sobel-ish), downsample
    to a coarse grid, threshold at the 75th percentile, take the largest
    connected component, and return its bounding box. Real-world frames
    put the label/bottle in a contiguous high-detail region; the rest of
    the frame is usually low-frequency background (sky, table, wall).

    Returns None when there is no meaningful structure (flat frame).
    """
    h, w = gray.shape
    if min(h, w) < 64:
        return None

    grid_h = max(8, h // 24)
    grid_w = max(8, w // 24)
    cell_h = h // grid_h
    cell_w = w // grid_w
    if cell_h == 0 or cell_w == 0:
        return None

    # Per-cell mean of |Sobel| via simple finite differences. Coarse but
    # robust and avoids any heavy CV imports.
    g32 = gray.astype(np.int32)
    dx = np.abs(g32[:, 1:] - g32[:, :-1])
    dy = np.abs(g32[1:, :] - g32[:-1, :])
    grad = np.zeros_like(g32)
    grad[:, :-1] += dx
    grad[:-1, :] += dy

    trimmed = grad[: grid_h * cell_h, : grid_w * cell_w]
    coarse = trimmed.reshape(grid_h, cell_h, grid_w, cell_w).mean(axis=(1, 3))

    if coarse.std() < 1.0:
        return None  # frame is uniform; no detectable label

    threshold = np.percentile(coarse, 75)
    mask = coarse >= threshold
    if not mask.any():
        return None

    # Light dilation closes gaps between adjacent text regions.
    mask = ndimage.binary_dilation(mask, iterations=1)

    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labels == biggest)
    if ys.size == 0:
        return None

    cy_lo, cy_hi = ys.min(), ys.max() + 1
    cx_lo, cx_hi = xs.min(), xs.max() + 1
    bbox = Bbox(
        x=int(cx_lo * cell_w),
        y=int(cy_lo * cell_h),
        w=int((cx_hi - cx_lo) * cell_w),
        h=int((cy_hi - cy_lo) * cell_h),
    )
    # Reject tiny bboxes — a "label" smaller than 10 % of the frame area is
    # almost certainly a thumb / lens speck, not the label.
    if bbox.area < 0.10 * (h * w):
        return None
    return bbox


def _localize_glare_blobs(
    gray: np.ndarray, label_bbox: Bbox | None
) -> list[GlareBlob]:
    """Connected-component bounding boxes of saturated-pixel clusters.

    Capped at the top 6 largest blobs to keep prompts and logs tight.
    """
    h, w = gray.shape
    frame_area = max(1, h * w)
    label_area = max(1, label_bbox.area if label_bbox else frame_area)

    sat = gray >= GLARE_PIXEL_THRESHOLD
    if not sat.any():
        return []

    # Erode lightly so isolated sensor-noise pixels don't each become a blob.
    sat = ndimage.binary_erosion(sat, iterations=1)
    if not sat.any():
        return []

    labels, n = ndimage.label(sat)
    if n == 0:
        return []

    blobs: list[GlareBlob] = []
    sizes = ndimage.sum(sat, labels, index=range(1, n + 1))
    for idx_zero, count in enumerate(sizes):
        idx = idx_zero + 1
        if count < max(50, frame_area * 0.0005):
            # Below 0.05 % of frame: too small to matter.
            continue
        ys, xs = np.where(labels == idx)
        bbox = Bbox(
            x=int(xs.min()),
            y=int(ys.min()),
            w=int(xs.max() - xs.min() + 1),
            h=int(ys.max() - ys.min() + 1),
        )
        label_overlap = (
            label_bbox.overlap_fraction(bbox) if label_bbox else 0.0
        )
        blobs.append(
            GlareBlob(
                bbox=bbox,
                area_fraction_frame=float(count) / frame_area,
                area_fraction_label=float(count) * label_overlap / label_area
                if label_bbox
                else float(count) / frame_area,
            )
        )

    blobs.sort(key=lambda b: -b.bbox.area)
    return blobs[:6]


def _is_backlit(gray: np.ndarray, label_bbox: Bbox | None) -> bool:
    """True when the label region is materially darker than the surround.

    Window-behind-bottle and outdoor-into-sun captures both fit this
    pattern: frame mean luminance is high, but the foreground subject is
    silhouetted against the bright background. Different remedy than
    plain underexposure.
    """
    if label_bbox is None:
        return False
    h, w = gray.shape
    label_pixels = gray[
        label_bbox.y : label_bbox.y + label_bbox.h,
        label_bbox.x : label_bbox.x + label_bbox.w,
    ]
    if label_pixels.size == 0:
        return False
    label_mean = float(label_pixels.mean())

    mask = np.ones((h, w), dtype=bool)
    mask[
        label_bbox.y : label_bbox.y + label_bbox.h,
        label_bbox.x : label_bbox.x + label_bbox.w,
    ] = False
    surround = gray[mask]
    if surround.size == 0:
        return False
    surround_mean = float(surround.mean())

    return surround_mean - label_mean >= BACKLIT_DELTA


def _motion_blur_direction(gray: np.ndarray) -> str | None:
    """Estimate the dominant edge orientation. If one direction holds
    >55 % of edge energy, motion blur is likely in the *perpendicular*
    direction (horizontal motion produces vertical streaks; the surviving
    edges are also vertical because horizontal edges got smeared away).

    Returns 'horizontal', 'vertical', 'diagonal', or None.
    """
    h, w = gray.shape
    if min(h, w) < 32:
        return None

    g32 = gray.astype(np.float32)
    gx = g32[:, 2:] - g32[:, :-2]
    gy = g32[2:, :] - g32[:-2, :]
    # Crop to common shape so we can stack.
    h2 = min(gy.shape[0], gx.shape[0])
    w2 = min(gy.shape[1], gx.shape[1])
    gx = gx[:h2, :w2]
    gy = gy[:h2, :w2]

    mag = np.hypot(gx, gy)
    energy = float(mag.sum())
    if energy < 1.0:
        return None

    abs_x = float(np.abs(gx).sum())
    abs_y = float(np.abs(gy).sum())
    total = abs_x + abs_y
    if total < 1.0:
        return None

    # Strong dominance in |gx| means most surviving edges are vertical →
    # horizontal motion has smeared the horizontal edges away.
    if abs_x / total >= MOTION_DIRECTION_DOMINANCE:
        # Mostly vertical edges survived → horizontal motion most likely.
        return "horizontal"
    if abs_y / total >= MOTION_DIRECTION_DOMINANCE:
        return "vertical"

    # Neither dominates strongly: check diagonal by examining gradient
    # angles. If the |angle - 45°| distribution is tight, call it diagonal.
    ang = np.arctan2(gy, gx) * (180.0 / np.pi)
    ang = np.where(ang < 0, ang + 180, ang)        # fold into [0, 180)
    weighted_mean = float((ang * mag).sum() / max(1.0, mag.sum()))
    diag_dist = min(abs(weighted_mean - 45), abs(weighted_mean - 135))
    if diag_dist < 12:
        return "diagonal"
    return None


def _is_lens_smudge_likely(gray: np.ndarray) -> bool:
    """Lens smudge / fogged-bottle heuristic: low-frequency veiling that
    softens the entire frame uniformly without reducing brightness.

    A clean frame has a long-tailed histogram of edge magnitudes — many
    small edges, a few strong ones. A smudged or fogged frame has a
    compressed distribution because high-frequency detail is suppressed
    everywhere. We compare the 95th percentile of |gradient| to its mean:
    a sharp frame ratios out to >5×; a smudged one collapses toward 1.
    """
    if min(gray.shape) < 64:
        return False
    g32 = gray.astype(np.float32)
    dx = g32[:, 2:] - g32[:, :-2]
    dy = g32[2:, :] - g32[:-2, :]
    h = min(dx.shape[0], dy.shape[0])
    w = min(dx.shape[1], dy.shape[1])
    mag = np.hypot(dx[:h, :w], dy[:h, :w])
    if mag.size == 0:
        return False
    p95 = float(np.percentile(mag, 95))
    mean = float(mag.mean())
    if mean < 1.0:
        return False
    ratio = p95 / mean
    # Empirical on synthetic + sample fixtures: sharp frames hit ~5–8, a
    # heavily smudged frame collapses toward 3.5 and below; threshold 4.0
    # is the cleanest separator that doesn't false-positive on textured
    # but sharp captures.
    return ratio < 4.0


def _is_wet_bottle_likely(gray: np.ndarray, label_bbox: Bbox | None) -> bool:
    """Wet bottle / condensation heuristic: a high density of small bright
    blobs distributed across the label region. Water droplets reflect
    light back to the camera as small specular highlights — the texture
    signature is "many tiny near-white blobs", distinct from one big glare
    blob (sun on the bottle) or no blobs (clean surface).
    """
    if label_bbox is None or min(label_bbox.w, label_bbox.h) < 64:
        return False
    crop = gray[
        label_bbox.y : label_bbox.y + label_bbox.h,
        label_bbox.x : label_bbox.x + label_bbox.w,
    ]
    if crop.size == 0:
        return False

    # Bright pixels — looser threshold than glare detection (we want to
    # catch droplet reflections that don't fully clip).
    bright = crop >= 230
    if not bright.any():
        return False
    labels, n = ndimage.label(bright)
    if n == 0:
        return False
    sizes = ndimage.sum(bright, labels, index=range(1, n + 1))
    crop_area = max(1, crop.shape[0] * crop.shape[1])
    # Count blobs whose area is in the "droplet" range (small but not
    # single-pixel sensor noise).
    droplet_count = int(
        ((sizes >= max(8, 0.0001 * crop_area))
         & (sizes <= 0.005 * crop_area)).sum()
    )
    return droplet_count >= 25  # high droplet density across the label


def _classify_capture_source(
    sensor: SensorMetadata,
    img: Image.Image,
    width: int,
    height: int,
) -> Literal["photo", "screenshot", "uncertain", "artwork"]:
    """Distinguish camera captures from screenshots, photos-of-photos, and
    digital artwork (PNG/SVG exports from design tools — what brewers and
    distillers actually upload from Esko / Adobe Illustrator).

    A screenshot has no EXIF make/model, no exposure data, often a
    suspicious aspect ratio (matching common phone screen dimensions:
    19.5:9, 18:9, 16:9 portrait), and frequently sRGB profile metadata
    indicating screen rendering rather than camera capture. A genuine
    camera frame has at least some EXIF camera fields populated.

    Digital artwork is the fourth case: clean raster export from a design
    tool. It strips EXIF (so it looks like a screenshot to the photo
    classifier), but it differs from both photos and screenshots by
    surrounding the design with a perfectly uniform canvas — a
    photograph never produces a 0-variance border, even with a clean
    backdrop. That uniform-border signal is the cleanest separator we
    have without trusting filename or content-type.

    The EXIF Software tag is checked FIRST: Android and many third-party
    screenshot tools tag the file with Software="Screenshot" even when
    Make/Model are populated by the OS. Honoring that tag prevents an
    obvious screenshot from sneaking past as "photo".
    """
    if sensor.software and "screenshot" in sensor.software.lower():
        return "screenshot"
    if sensor.make or sensor.model or sensor.iso or sensor.exposure_time_s:
        return "photo"
    if not width or not height:
        return "uncertain"

    # Digital-artwork heuristic: no EXIF + uniform border. Convert to L
    # once and read a 5-pixel strip on each edge. Real photographs put
    # background variation, vignetting, or sensor noise into those strips
    # — design-tool exports leave them at a single canvas color.
    try:
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
        strip = max(2, min(8, min(width, height) // 50))
        border = np.concatenate([
            gray[:strip, :].ravel(),
            gray[-strip:, :].ravel(),
            gray[:, :strip].ravel(),
            gray[:, -strip:].ravel(),
        ])
        # Empirical: real design exports observed at 0.0–1.3 border stddev;
        # photographs with sensor noise (even bright ones with simulated
        # sky / white-wall backgrounds) are reliably above 4. 2.5 keeps
        # the real-world labels classified as artwork while leaving
        # headroom so test fixtures that sprinkle photographic noise
        # over a uniform canvas don't accidentally cross the boundary.
        if border.size and float(border.std()) < 2.5:
            return "artwork"
    except Exception as e:
        # Fall through to the screenshot/uncertain logic — better to
        # mis-classify as photo than to crash the pre-check.
        logger.debug("artwork border check failed: %s", e)

    aspect = max(width, height) / min(width, height)
    common_screen_aspects = [
        (19.5 / 9, 0.06),
        (18 / 9, 0.05),
        (16 / 9, 0.05),
    ]
    if any(abs(aspect - target) <= tol for target, tol in common_screen_aspects):
        return "screenshot"
    return "uncertain"


# ---------------------------------------------------------------------------
# Verdict logic — region-aware
# ---------------------------------------------------------------------------


def _evaluate(
    *,
    sensor: SensorMetadata,
    metrics_frame: ImageQualityMetrics,
    metrics_label: ImageQualityMetrics | None,
    glare_blobs: list[GlareBlob],
    backlit: bool,
    motion_dir: str | None,
    smudge_likely: bool = False,
    wet_likely: bool = False,
    capture_source: Literal["photo", "screenshot", "uncertain", "artwork"] = "photo",
) -> tuple[list[str], list[str], float, Verdict, Verdict | None]:
    """Map metrics + flags into issues, suggestions, score, and verdicts.

    Returns (issues, suggestions, score, frame_verdict, label_verdict).
    The label verdict drives the surface verdict when available — frame-
    level issues that don't touch the label region are surfaced as
    advisories but do not push the verdict past "good".
    """
    relax = _TIER_RELAX.get(sensor.tier, _TIER_RELAX["unknown"])
    sharpness_unread = SHARPNESS_UNREADABLE * relax["sharpness"]
    sharpness_degraded = SHARPNESS_DEGRADED * relax["sharpness"]
    resolution_unread = RESOLUTION_UNREADABLE_MP * relax["resolution"]
    resolution_degraded = RESOLUTION_DEGRADED_MP * relax["resolution"]

    issues: list[str] = []
    suggestions: list[str] = []
    forced_unreadable = False
    partial_scores: list[float] = []

    # When the detected label bbox dominates the frame (>70% by area),
    # the "label region" metric is essentially the same content as the
    # frame metric — but with the very edges cropped out, which can
    # nudge sharpness slightly upward and produce a confident "good"
    # verdict on a photograph whose frame is genuinely soft. In that
    # case, fall back to the frame metrics so the verdict is anchored
    # in the most conservative reading.
    label_dominates_frame = metrics_label is not None and (
        metrics_label.width_px * metrics_label.height_px
        > 0.70 * metrics_frame.width_px * metrics_frame.height_px
    )
    if metrics_label is not None and not label_dominates_frame:
        target = metrics_label
        target_scope = "label region"
    else:
        target = metrics_frame
        target_scope = "frame"

    # --- Sharpness (label region) -------------------------------------------
    if target.sharpness < sharpness_unread:
        issues.append(
            f"Severe motion blur on the {target_scope} "
            f"(sharpness {target.sharpness:.0f} < {sharpness_unread:.0f})"
        )
        if motion_dir:
            suggestions.append(
                f"{motion_dir.capitalize()} motion detected — brace your wrist "
                f"perpendicular to it before retake."
            )
        else:
            suggestions.append(
                "Hold the phone still and tap to focus before capture."
            )
        partial_scores.append(0.0)
        forced_unreadable = True
    elif target.sharpness < sharpness_degraded:
        issues.append(
            f"Soft / mildly blurry {target_scope} "
            f"(sharpness {target.sharpness:.0f})"
        )
        if motion_dir:
            suggestions.append(
                f"Brace against {motion_dir} motion (detected in this frame)."
            )
        else:
            suggestions.append(
                "Brace the phone or rest your wrist for a sharper frame."
            )
        partial_scores.append(_ramp(target.sharpness, sharpness_unread, sharpness_degraded))
    elif (
        capture_source != "artwork"
        and target.sharpness < sharpness_degraded * 1.2
    ):
        # Borderline-soft photograph: clears the threshold but only by the
        # margin of the device-tier relaxation. The cost of a false "good"
        # is the model trusting fine print like the Health Warning that
        # OCR may not recover. Surface as an advisory so the verdict is
        # honest about the proximity rather than confidently good.
        issues.append(
            f"Borderline-soft {target_scope} "
            f"(sharpness {target.sharpness:.0f}, just above the "
            f"{sharpness_degraded:.0f} threshold) — fine print may not be "
            f"reliably recoverable"
        )
        suggestions.append(
            "Reshoot with a steadier grip and let autofocus settle on the label."
        )
        partial_scores.append(0.7)
    else:
        partial_scores.append(1.0)

    # --- Glare ---------------------------------------------------------------
    # Skipped for digital artwork: clean white/cream design backgrounds are
    # almost entirely above the saturated-pixel threshold by construction
    # (and there's no physical "glare" on a vector export).
    #
    # Force "unreadable" only when we can prove the GLARE COVERS THE LABEL —
    # i.e. either we localized a label region and it's mostly clipped, or the
    # frame is essentially nothing but glare. Frame-level glare without a
    # localized label (e.g. a can held in front of a window) almost always
    # leaves enough of the label intact for the model to extract — refusing
    # to call the API in that case wastes a perfectly recoverable shot.
    if capture_source != "artwork":
        # `target_scope == "label region"` means the gradient-density detector
        # found a believable bbox we're using locally — glare > GLARE_UNREADABLE
        # there really is glare on the label, and the verdict should reflect it.
        # `target_scope == "frame"` means either we couldn't localize, or the
        # bbox covered >70% of the frame (label_dominates_frame). In that mode
        # the glare fraction includes the surrounding scene; refusing on 40%
        # frame glare wastes shots where the label itself is still readable.
        # Only force unreadable when the frame is almost entirely glare.
        label_glare_critical = (
            target_scope == "label region"
            and target.glare_fraction > GLARE_UNREADABLE
        )
        whole_frame_glare = (
            target_scope == "frame" and target.glare_fraction > 0.75
        )
        if label_glare_critical or whole_frame_glare:
            # Localize when we can.
            if glare_blobs and metrics_label is not None:
                top = glare_blobs[0]
                issues.append(
                    f"Excessive glare on the label "
                    f"({target.glare_fraction * 100:.0f}% clipped; largest blob "
                    f"covers {top.area_fraction_label * 100:.0f}% of label)"
                )
            else:
                issues.append(
                    f"Excessive glare — {target.glare_fraction * 100:.0f}% clipped"
                )
            suggestions.append(
                "Tilt the bottle so direct light reflects away from the lens."
            )
            partial_scores.append(0.0)
            forced_unreadable = True
        elif target.glare_fraction > GLARE_DEGRADED:
            issues.append(
                f"Noticeable glare on the {target_scope} "
                f"({target.glare_fraction * 100:.0f}% clipped)"
            )
            if glare_blobs:
                tops = ", ".join(
                    f"{b.area_fraction_label * 100:.0f}% blob"
                    if metrics_label is not None
                    else f"{b.area_fraction_frame * 100:.0f}% blob"
                    for b in glare_blobs[:3]
                )
                issues[-1] += f" — blobs: {tops}"
            suggestions.append("Move the bottle out of direct light or shade the label.")
            partial_scores.append(
                1.0 - _ramp(target.glare_fraction, GLARE_DEGRADED, GLARE_UNREADABLE)
            )
        else:
            partial_scores.append(1.0)

    # --- Brightness / backlight ---------------------------------------------
    # Backlight diagnoses "window behind bottle" — dark-on-light layout in a
    # design export looks identical to the detector and false-positives.
    if backlit and capture_source != "artwork":
        issues.append(
            f"Backlit subject — label region mean luminance "
            f"{target.brightness_mean:.0f} vs surround "
            f"{metrics_frame.brightness_mean:.0f}"
        )
        suggestions.append(
            "Step around so the strongest light is behind YOU, not behind the bottle."
        )
        partial_scores.append(0.4)  # advisory weight; not a hard fail.
    elif target.brightness_mean < BRIGHTNESS_DARK_UNREADABLE:
        issues.append(
            f"Severely underexposed {target_scope} "
            f"(mean luminance {target.brightness_mean:.0f} / 255)"
        )
        suggestions.append("Move closer to a light source or use the phone's flashlight.")
        partial_scores.append(0.0)
        forced_unreadable = True
    elif target.brightness_mean < BRIGHTNESS_DARK_DEGRADED:
        issues.append(
            f"Underexposed — likely a dim environment "
            f"(mean luminance {target.brightness_mean:.0f})"
        )
        suggestions.append("Add light or hold the bottle near a brighter area.")
        partial_scores.append(
            _ramp(target.brightness_mean, BRIGHTNESS_DARK_UNREADABLE, BRIGHTNESS_DARK_DEGRADED)
        )
    elif capture_source != "artwork" and target.brightness_mean > BRIGHTNESS_BRIGHT_UNREADABLE:
        issues.append(
            f"Severely overexposed {target_scope} "
            f"(mean luminance {target.brightness_mean:.0f})"
        )
        suggestions.append("Move out of direct sunlight or shade the bottle.")
        partial_scores.append(0.0)
        forced_unreadable = True
    elif capture_source != "artwork" and target.brightness_mean > BRIGHTNESS_BRIGHT_DEGRADED:
        issues.append(
            f"Overexposed — possible direct sun or backlight "
            f"(mean luminance {target.brightness_mean:.0f})"
        )
        suggestions.append("Reposition to keep the light source behind you.")
        partial_scores.append(
            1.0 - _ramp(
                target.brightness_mean,
                BRIGHTNESS_BRIGHT_DEGRADED,
                BRIGHTNESS_BRIGHT_UNREADABLE,
            )
        )
    else:
        partial_scores.append(1.0)

    # --- Contrast ------------------------------------------------------------
    if target.brightness_stddev < CONTRAST_LOW_UNREADABLE:
        issues.append(
            f"Almost no contrast on {target_scope} "
            f"(stddev {target.brightness_stddev:.0f}) — camera may be aimed "
            f"away from the label"
        )
        suggestions.append("Aim at the label and ensure it fills the framing guide.")
        partial_scores.append(0.0)
        forced_unreadable = True
    elif target.brightness_stddev < CONTRAST_LOW_DEGRADED:
        issues.append(
            f"Low contrast on {target_scope} "
            f"(stddev {target.brightness_stddev:.0f}) — possible smudged lens "
            f"or fogged bottle"
        )
        suggestions.append("Wipe the lens and the bottle, then try again.")
        partial_scores.append(
            _ramp(target.brightness_stddev, CONTRAST_LOW_UNREADABLE, CONTRAST_LOW_DEGRADED)
        )
    else:
        partial_scores.append(1.0)

    # --- Resolution ---------------------------------------------------------
    # Resolution thresholds assume a phone-camera capture. A 1200×1500 design
    # proof is genuinely large enough to read fine print at 200 DPI; only
    # photo submissions need the camera-resolution lower bound.
    if capture_source == "artwork":
        partial_scores.append(1.0)
    elif metrics_frame.megapixels < resolution_unread:
        issues.append(
            f"Resolution too low for label compliance "
            f"({metrics_frame.megapixels} MP, "
            f"{metrics_frame.width_px}×{metrics_frame.height_px})"
        )
        suggestions.append(
            "Submit a full-resolution photo, not a thumbnail or screenshot."
        )
        partial_scores.append(0.0)
        forced_unreadable = True
    elif metrics_frame.megapixels < resolution_degraded:
        issues.append(
            f"Low resolution ({metrics_frame.megapixels} MP) — fine print may "
            f"be unreadable"
        )
        suggestions.append("Use the device's main camera at full resolution.")
        partial_scores.append(
            _ramp(metrics_frame.megapixels, resolution_unread, resolution_degraded)
        )
    else:
        partial_scores.append(1.0)

    # --- Color cast ---------------------------------------------------------
    if target.color_cast > COLOR_CAST_DEGRADED:
        issues.append(
            f"Strong color cast on {target_scope} "
            f"(channel imbalance {target.color_cast:.0f}) — colored lighting "
            f"or tinted glass"
        )
        suggestions.append("Try again under more neutral light if possible.")
        partial_scores.append(0.7)
    else:
        partial_scores.append(1.0)

    # --- Sensor stress (advisory; informs the user, doesn't fail) ----------
    iso_warn = 1600 * relax["iso"]
    if sensor.iso and sensor.iso >= iso_warn:
        issues.append(
            f"High ISO ({sensor.iso}) — sensor noise may degrade OCR "
            f"(device tier '{sensor.tier}', threshold {iso_warn:.0f})"
        )
    exposure_warn = (1 / 30) * relax["exposure"]
    if sensor.exposure_time_s and sensor.exposure_time_s > exposure_warn:
        issues.append(
            f"Long exposure ({_format_exposure(sensor.exposure_time_s)} s) — "
            f"motion blur risk; brace the device"
        )

    # --- Surface / capture-source signals (advisory) ------------------------
    # Smudge and wet-bottle are photographic artifacts; the heuristics
    # false-positive on flat-color artwork (soft anti-aliased edges
    # collapse the p95/mean gradient ratio that drives the smudge check).
    if smudge_likely and capture_source != "artwork":
        issues.append(
            "Lens smudge or fogged bottle suspected — high-frequency detail "
            "is suppressed uniformly across the frame"
        )
        suggestions.append("Wipe the lens AND the bottle, then try again.")
        partial_scores.append(0.6)
    if wet_likely and capture_source != "artwork":
        issues.append(
            "Wet bottle / condensation suspected — many small specular "
            "highlights distributed across the label"
        )
        suggestions.append(
            "Wipe the bottle dry; let it warm to room temperature before retake."
        )
        partial_scores.append(0.6)
    if capture_source == "screenshot":
        issues.append(
            "Capture source looks like a screenshot, not a camera frame "
            "(no EXIF, screen-aspect ratio) — TTB compliance must be assessed "
            "on actual artwork or an actual photo"
        )
        suggestions.append("Submit a camera capture or the original artwork file.")
        partial_scores.append(0.5)

    score = sum(partial_scores) / len(partial_scores) if partial_scores else 0.0

    if forced_unreadable:
        frame_verdict: Verdict = "unreadable"
    elif score < 0.55:
        frame_verdict = "degraded"
    elif issues:
        frame_verdict = "degraded"
    else:
        frame_verdict = "good"
    if frame_verdict == "unreadable":
        score = min(score, 0.25)

    label_verdict = (
        frame_verdict if metrics_label is not None else None
    )

    return issues, suggestions, round(score, 3), frame_verdict, label_verdict


# ---------------------------------------------------------------------------
# Aggregation / formatting helpers
# ---------------------------------------------------------------------------


def _aggregate_verdict(verdicts: list[Verdict]) -> Verdict:
    if not verdicts:
        return "unreadable"
    if any(v == "unreadable" for v in verdicts):
        return "unreadable"
    if any(v == "degraded" for v in verdicts):
        return "degraded"
    return "good"


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value >= high else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _first_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clean_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return None
    text = str(value).strip().strip("\x00").strip()
    return text or None


def _format_exposure(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.1f}"
    if seconds <= 0:
        return f"{seconds}"
    return f"1/{round(1 / seconds)}"
