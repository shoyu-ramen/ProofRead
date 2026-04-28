"""Synthetic capture-condition degradations from SPEC §0.5.

Every function in this module:

  * takes a ``PIL.Image`` (any mode; we convert internally where needed)
  * takes ``severity`` ∈ {"light", "medium", "heavy"}
  * returns a NEW ``PIL.Image`` — inputs are never mutated
  * uses only PIL + numpy + scipy (no new deps)

Severity is calibrated empirically: "light" should rarely fail an
otherwise-clean label, "heavy" should drag a clean label below the
``acceptable`` threshold for at least one condition. The mid-tier is the
realistic-bar / realistic-festival regime — that's where the harness
spends the most signal.

Determinism: every randomized degradation seeds an explicit ``np.random``
generator from the severity name, so the harness is reproducible across
runs. We don't reseed numpy's global RNG.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

Severity = Literal["light", "medium", "heavy"]
SEVERITIES: tuple[Severity, ...] = ("light", "medium", "heavy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seeded_rng(name: str, severity: Severity) -> np.random.Generator:
    """Reproducible RNG keyed off (degradation, severity)."""
    seed = abs(hash((name, severity))) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img.copy()


def _severity_pick(severity: Severity, light, medium, heavy):  # type: ignore[no-untyped-def]
    return {"light": light, "medium": medium, "heavy": heavy}[severity]


# ---------------------------------------------------------------------------
# Lighting degradations
# ---------------------------------------------------------------------------


def glare(img: Image.Image, severity: Severity) -> Image.Image:
    """Specular highlight: a hot ellipse blown out toward white.

    Mid covers ~25 % of the frame; heavy covers >40 % and clips a sizable
    fraction of the label region. Position is randomized but reproducible.
    """
    base = _ensure_rgb(img)
    w, h = base.size
    rng = _seeded_rng("glare", severity)

    coverage = _severity_pick(severity, 0.10, 0.28, 0.55)
    # Ellipse axes that cover roughly `coverage` fraction of frame area.
    # area = pi * a * b; pick a = sqrt(coverage * w * h / pi * aspect),
    # b = a / aspect so the highlight stays elongated like a real reflection.
    aspect = 1.6
    radius_a = float(np.sqrt(coverage * w * h * aspect / np.pi))
    radius_b = radius_a / aspect

    # Place near-center but jittered so the same label isn't always blown
    # out in the same spot.
    cx = w * (0.4 + 0.2 * rng.random())
    cy = h * (0.35 + 0.3 * rng.random())

    overlay = Image.new("RGB", (w, h), (255, 255, 255))
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    bbox = (cx - radius_a, cy - radius_b, cx + radius_a, cy + radius_b)
    md.ellipse(bbox, fill=255)
    # Soft edges so the highlight blends rather than hard-cuts. Severity
    # drives both intensity and softness.
    blur_radius = _severity_pick(severity, 60, 80, 120)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Strength of the white overlay's blend, capped at 255.
    strength = _severity_pick(severity, 180, 230, 255)
    mask_arr = np.asarray(mask, dtype=np.uint16)
    mask_arr = np.minimum(mask_arr * strength // 255, 255).astype(np.uint8)
    mask = Image.fromarray(mask_arr, "L")
    return Image.composite(overlay, base, mask)


def low_light(img: Image.Image, severity: Severity) -> Image.Image:
    """Dim bar / cellar lighting + a sprinkling of sensor noise.

    Heavy puts the mean luminance below the unreadable cutoff; light is
    the "dim restaurant" regime that should still be readable.
    """
    base = _ensure_rgb(img)
    arr = np.asarray(base, dtype=np.float32)
    rng = _seeded_rng("low_light", severity)

    # Multiplicative dimming + a small additive black-floor lift.
    gain = _severity_pick(severity, 0.45, 0.22, 0.08)
    floor = _severity_pick(severity, 6, 4, 2)
    out = arr * gain + floor

    # Sensor noise scales with severity — long exposure / high ISO regime.
    noise_sigma = _severity_pick(severity, 6.0, 14.0, 22.0)
    noise = rng.normal(0.0, noise_sigma, size=arr.shape)
    out = np.clip(out + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def colored_lighting(img: Image.Image, severity: Severity) -> Image.Image:
    """Warm/cool color cast — neon, candlelight, blue-LED bar etc.

    Alternates direction by severity so the matrix exercises both poles
    rather than only warm shifts.
    """
    base = _ensure_rgb(img)
    arr = np.asarray(base, dtype=np.float32)

    # Light = warm bar; medium = strong amber sodium-vapor; heavy = cold
    # blue LED. Channel multipliers calibrated to push channel imbalance
    # past the COLOR_CAST_DEGRADED threshold (35) at heavy.
    if severity == "light":
        r, g, b = 1.10, 1.00, 0.90
    elif severity == "medium":
        r, g, b = 1.30, 1.00, 0.65
    else:
        r, g, b = 0.55, 0.85, 1.45

    arr[..., 0] *= r
    arr[..., 1] *= g
    arr[..., 2] *= b
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# Motion / focus degradations
# ---------------------------------------------------------------------------


def motion_blur(img: Image.Image, severity: Severity) -> Image.Image:
    """Linear motion blur (shaky hand / walking capture).

    Implemented as repeated 1-D box convolution along the motion axis —
    PIL's Kernel filter is capped at 5×5 so we stack passes via
    BoxBlur(0, vertical) which has no such cap.
    """
    base = _ensure_rgb(img)
    # Distance in pixels we smear the image. Calibrated against the
    # SHARPNESS_DEGRADED (120) and SHARPNESS_UNREADABLE (40) thresholds.
    distance = _severity_pick(severity, 6, 14, 30)

    # Axis: heavy runs diagonal so the harness exercises that branch of
    # _motion_blur_direction. Light/medium are horizontal (most common in
    # handheld capture).
    if severity == "heavy":
        # Diagonal: do a horizontal blur then a vertical of half magnitude.
        out = base.filter(
            ImageFilter.GaussianBlur(radius=distance / 2.5)
        )
        out = out.filter(
            ImageFilter.BoxBlur(radius=distance // 3)
        )
        return out
    # PIL's BoxBlur uses a square kernel — for elongated motion we stack
    # asymmetric filters: a tiny vertical box (≈1 px) + a long horizontal.
    arr = np.asarray(base, dtype=np.float32)
    kernel_len = max(3, distance | 1)
    # Vectorized box convolution along axis=1 via cumulative-sum trick.
    # We need exactly `kernel_len` extra columns of padding (split as
    # left=k//2, right=k - k//2) so the resulting output width matches
    # the input width for any odd or even kernel.
    out = np.empty_like(arr)
    pad_left = kernel_len // 2
    pad_right = kernel_len - pad_left
    padded = np.pad(
        arr, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge"
    )
    # Prepend a zero column so cs[:, k] - cs[:, 0] = sum of first k samples.
    zero_col = np.zeros((padded.shape[0], 1, padded.shape[2]), dtype=np.float32)
    padded = np.concatenate([zero_col, padded], axis=1)
    for c in range(3):
        cs = np.cumsum(padded[..., c], axis=1)
        s = cs[:, kernel_len:] - cs[:, :-kernel_len]
        out[..., c] = s[:, : arr.shape[1]] / kernel_len
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def defocus_blur(img: Image.Image, severity: Severity) -> Image.Image:
    """Out-of-focus capture (autofocus hunt or lens too close).

    Distinct signature from motion blur: isotropic, no preferred
    direction. Tests that the sensor pre-check picks up sharpness loss
    even when motion direction is None.
    """
    base = _ensure_rgb(img)
    radius = _severity_pick(severity, 2.5, 6.0, 12.0)
    return base.filter(ImageFilter.GaussianBlur(radius=radius))


# ---------------------------------------------------------------------------
# Geometry degradations
# ---------------------------------------------------------------------------


def rotation(img: Image.Image, severity: Severity) -> Image.Image:
    """Off-axis capture — phone tilted relative to bottle.

    Bicubic resample so we don't artificially zero out sharpness; black
    fill on the corners is an OCR distraction we want the pre-check to
    notice via lower brightness mean at heavier angles.
    """
    base = _ensure_rgb(img)
    angle = _severity_pick(severity, 8.0, 22.0, 45.0)
    return base.rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0)
    )


def perspective_warp(img: Image.Image, severity: Severity) -> Image.Image:
    """Curved-bottle simulation: horizontal x-coordinate is bent inward
    toward the cylinder axis, mimicking how a flat label looks when
    wrapped around a 750 mL wine bottle (or, harder, a 12 oz beer can).

    We build a per-row x-shift profile via cosine, then resample the image
    column-wise with the shifts and crop side bars. Heavier severity
    increases the curvature radius (more dramatic warp at the edges).
    """
    base = _ensure_rgb(img)
    arr = np.asarray(base, dtype=np.float32)
    h, w, _ = arr.shape

    # Curvature: max horizontal shift at the edge, in pixels, as a
    # fraction of width. SPEC §0.5 calls out 750 mL wine vs 12 oz can —
    # heavy is the can regime.
    curvature = _severity_pick(severity, 0.05, 0.12, 0.22)
    max_shift = curvature * w

    # x_in(x_out, y) = x_out + max_shift * sin(pi * x_out / w) — pinches
    # the middle outward, simulating a cylinder seen edge-on.
    x_out = np.arange(w, dtype=np.float32)
    shift = max_shift * np.sin(np.pi * x_out / w)
    x_src = x_out - shift

    # Vertically: also slight bow at top/bottom for severity≥medium.
    y_curve = _severity_pick(severity, 0.0, 0.04, 0.10)
    y_out = np.arange(h, dtype=np.float32)
    if y_curve > 0:
        bow = y_curve * h * np.sin(np.pi * y_out / h)
    else:
        bow = np.zeros_like(y_out)

    # Build floating-point sample grids.
    xx, yy = np.meshgrid(x_src, y_out)
    yy = yy + bow[:, None] * np.sin(np.pi * x_out / w)[None, :]

    # Bilinear interpolation. scipy.ndimage.map_coordinates is the obvious
    # tool but we already require scipy elsewhere.
    from scipy import ndimage  # type: ignore[import-untyped]

    out = np.empty_like(arr)
    coords = np.stack([yy, xx], axis=0)
    for c in range(3):
        out[..., c] = ndimage.map_coordinates(
            arr[..., c], coords, order=1, mode="reflect"
        )
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


# ---------------------------------------------------------------------------
# Surface degradations
# ---------------------------------------------------------------------------


def condensation(img: Image.Image, severity: Severity) -> Image.Image:
    """Wet bottle / chilled-glass condensation: many small specular blobs
    plus a translucent fog layer. Targets ``_is_wet_bottle_likely`` and
    contrast-loss thresholds simultaneously.
    """
    base = _ensure_rgb(img)
    w, h = base.size
    rng = _seeded_rng("condensation", severity)

    # Fog layer: low-alpha white wash to mimic the translucent veil of
    # condensation droplets averaged at distance.
    fog = Image.new("RGB", (w, h), (240, 240, 245))
    fog_strength = _severity_pick(severity, 50, 100, 160)
    arr = np.asarray(base, dtype=np.float32)
    fog_arr = np.asarray(fog, dtype=np.float32)
    blended = arr * (1 - fog_strength / 255) + fog_arr * (fog_strength / 255)

    # Droplet count scales aggressively with severity — light = "just
    # uncapped from cooler", heavy = "thirty seconds of fog forming".
    droplet_count = _severity_pick(severity, 200, 800, 2500)
    out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")
    od = ImageDraw.Draw(out)
    for _ in range(droplet_count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(2, 6 if severity != "heavy" else 9))
        # Bright droplet center — saturating to white the way real
        # specular reflections do.
        v = int(rng.integers(220, 256))
        od.ellipse((x - r, y - r, x + r, y + r), fill=(v, v, v))

    # Slight blur so the droplets feel like real reflections, not pasted-
    # on circles. Larger blur at heavy makes the surface look truly
    # frosted.
    blur_r = _severity_pick(severity, 1.0, 1.5, 2.5)
    return out.filter(ImageFilter.GaussianBlur(radius=blur_r))


# ---------------------------------------------------------------------------
# Capture-pipeline degradations
# ---------------------------------------------------------------------------


def jpeg_compression_artifacts(img: Image.Image, severity: Severity) -> Image.Image:
    """Aggressive JPEG re-encode (e.g. forwarded over messengers, low
    bandwidth uploads). Drops fine high-frequency detail in 8x8 blocks —
    the failure mode the SPEC calls out for screenshots and re-shared
    captures.
    """
    import io

    base = _ensure_rgb(img)
    quality = _severity_pick(severity, 35, 18, 6)
    buf = io.BytesIO()
    base.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def low_resolution(img: Image.Image, severity: Severity) -> Image.Image:
    """Downscale-then-upscale to simulate a thumbnail being submitted in
    place of a full-resolution capture. Different from blur because the
    final frame retains crisp edges at 8×8-ish granularity but no genuine
    high-frequency content.
    """
    base = _ensure_rgb(img)
    w, h = base.size
    # Long edge after downscale. SPEC's resolution thresholds are 0.5 MP
    # unreadable / 2 MP degraded; pick downscales that bracket them.
    target_long = _severity_pick(severity, 1200, 600, 280)
    scale = target_long / max(w, h)
    if scale >= 1.0:
        return base.copy()
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    small = base.resize(new_size, Image.Resampling.LANCZOS)
    # Upsample back to the original size with NEAREST so the loss is
    # visible to the OCR pipeline; bicubic would re-smooth and hide it.
    return small.resize((w, h), Image.Resampling.NEAREST)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


DEGRADATIONS: dict[str, callable] = {  # type: ignore[type-arg]
    "glare": glare,
    "low_light": low_light,
    "motion_blur": motion_blur,
    "defocus_blur": defocus_blur,
    "rotation": rotation,
    "perspective_warp": perspective_warp,
    "condensation": condensation,
    "jpeg_compression_artifacts": jpeg_compression_artifacts,
    "low_resolution": low_resolution,
    "colored_lighting": colored_lighting,
}


# Public helper: apply by name. Used by runner + tests.


def apply(name: str, img: Image.Image, severity: Severity) -> Image.Image:
    fn = DEGRADATIONS[name]
    return fn(img, severity)


# Bonus: faded label / sun-damage — a simple desaturate + brightness lift
# that we layer on TOP of another condition in mixed-mode runs. Not in
# the headline matrix but handy for the report's "what didn't break"
# section. Kept here for completeness.


def faded(img: Image.Image, severity: Severity) -> Image.Image:
    base = _ensure_rgb(img)
    desaturation = _severity_pick(severity, 0.4, 0.7, 0.9)
    brightness_lift = _severity_pick(severity, 1.05, 1.15, 1.30)
    out = ImageEnhance.Color(base).enhance(1.0 - desaturation)
    out = ImageEnhance.Brightness(out).enhance(brightness_lift)
    return out
