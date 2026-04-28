"""Shared formatter that turns a `CaptureQualityReport` into a human-readable
briefing block for vision-extractor system/user prompts.

Both the scan-path extractor (`extractors/claude_vision.py`) and the verify-
path extractor (`services/vision.py`) need the same briefing so the model
gets a consistent, region-aware prior on every label it sees: label bbox,
glare blobs, backlight, motion direction, sensor tier, etc. Keeping the
formatter in one place avoids drift between the two paths.
"""

from __future__ import annotations

from typing import Any


def format_capture_quality(capture: Any | None) -> str:
    """Brief the model with the sensor pre-check verdict.

    The pre-check is an *objective* prior — if it flagged severe motion
    blur, the model shouldn't 'see through' it and report high confidence.
    Region-aware fields (label bbox, glare blobs, backlight, motion
    direction, sensor tier) let the model reason locally: "I read brand
    from glare blob #1 area, so confidence is medium" rather than "I read
    everything fine because the average frame looked OK".
    """
    if capture is None:
        return ""
    surfaces = getattr(capture, "surfaces", None)
    if not surfaces:
        return ""

    lines = [
        "Sensor pre-check (objective metrics, NOT your output — use as a prior):",
        f"  overall_verdict    = {getattr(capture, 'overall_verdict', 'unknown')}",
        f"  overall_confidence = {getattr(capture, 'overall_confidence', 0.0):.2f}",
    ]
    for s in surfaces:
        sensor = getattr(s, "sensor", None)
        sensor_desc = sensor.describe() if sensor is not None else "no EXIF"
        m = getattr(s, "metrics", None)
        m_label = getattr(s, "metrics_label", None)
        bbox = getattr(s, "label_bbox", None)
        blobs = getattr(s, "glare_blobs", []) or []
        backlit = getattr(s, "backlit", False)
        motion = getattr(s, "motion_blur_direction", None)

        issues = "; ".join(s.issues) if s.issues else "no issues detected"
        lines.append(
            f"  - {s.surface}: verdict={s.verdict} "
            f"confidence={s.confidence:.2f}  ({sensor_desc})"
        )
        if m is not None:
            lines.append(
                f"      frame metrics: sharpness={m.sharpness:.0f} "
                f"glare={m.glare_fraction*100:.0f}% "
                f"brightness={m.brightness_mean:.0f} "
                f"contrast={m.brightness_stddev:.0f} "
                f"resolution={m.megapixels} MP"
            )
        if bbox is not None and m_label is not None:
            lines.append(
                f"      label region [x={bbox.x},y={bbox.y},"
                f"w={bbox.w},h={bbox.h}]: sharpness={m_label.sharpness:.0f} "
                f"glare={m_label.glare_fraction*100:.0f}% "
                f"brightness={m_label.brightness_mean:.0f}"
            )
        if blobs:
            blob_lines = []
            for i, b in enumerate(blobs[:4]):
                bb = b.bbox
                blob_lines.append(
                    f"#{i+1} [x={bb.x},y={bb.y},w={bb.w},h={bb.h}] "
                    f"({b.area_fraction_label*100:.0f}% of label)"
                )
            lines.append("      glare blobs: " + " · ".join(blob_lines))
        flags: list[str] = []
        if backlit:
            flags.append("backlit (label darker than surround)")
        if motion:
            flags.append(f"motion blur ({motion})")
        if getattr(s, "lens_smudge_likely", False):
            flags.append("lens-smudge / fog suspected")
        if getattr(s, "wet_bottle_likely", False):
            flags.append("wet bottle / condensation suspected")
        source = getattr(s, "capture_source", "photo")
        if source == "screenshot":
            flags.append("screenshot, not a camera frame")
        elif source == "artwork":
            flags.append(
                "digital artwork (no EXIF, uniform border) — photo-quality "
                "checks (glare, exposure, smudge, low-resolution) skipped"
            )
        if flags:
            lines.append("      flags: " + ", ".join(flags))
        lines.append(f"      issues: {issues}")
    lines.append(
        "If your bbox for an extracted field falls inside a glare blob, "
        "that field's confidence MUST reflect the occlusion."
    )
    return "\n".join(lines) + "\n"
