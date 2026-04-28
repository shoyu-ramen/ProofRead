#!/usr/bin/env python3
"""
Prototype: run the four sample labels through Claude Opus 4.7 and produce a
TTB-compliance report for each.

Why Opus 4.7:
  - First Claude with high-resolution image support (up to 2576px on the long
    edge, ~3x more image tokens than 4.6) — the right fit for fine-print
    elements like the Health Warning Statement.
  - Adaptive thinking lets the model decide how much reasoning each label needs
    (a clean bourbon label gets a fast pass; the rotated/glared wine bottle
    gets more thinking before declaring "unreadable").

Camera-sensor pre-flight (SPEC §0.5):
  Before each image is sent to the model, `sensor_check` extracts EXIF
  metadata and computes objective image-quality metrics (Laplacian-variance
  sharpness, glare percentage, brightness, contrast, megapixels). The
  CaptureQualityReport is:
    - Printed alongside the model's verdict so the operator can compare.
    - Embedded in the user message as an objective prior so the model
      grounds its `image_quality` assessment in measurements rather than
      eyeballing.
    - Used to short-circuit obvious unreadables — no API call when the
      frame is already objectively unrecoverable.

The TTB rule set is sent as a cached system prompt so we pay the prefix tokens
once and read them back on labels 2-4.

Usage:
    export ANTHROPIC_API_KEY=...
    python prototype_review.py                  # all four labels
    python prototype_review.py --label PATH     # single label
    python prototype_review.py --no-call        # sensor pre-check only,
                                                  no Claude API request
"""

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field

# Bring the backend's sensor_check module into scope when the prototype is
# run from the repo root (no install, no PYTHONPATH adjustment by the user).
_BACKEND_DIR = Path(__file__).parent / "backend"
if _BACKEND_DIR.exists() and str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.sensor_check import (  # noqa: E402  (path tweak above)
    CaptureQualityReport,
    assess_capture_quality,
)

LABELS_DIR = Path(__file__).parent / "artwork" / "labels"
MODEL = "claude-opus-4-7"

# Producer-side metadata. In the real app this comes from the COLA submission
# or the user's container-size selection (SPEC §F1.4). Hard-coded here so the
# model can flag label-vs-record mismatches like the case-difference in the
# gin sample.
EXPECTED: dict[str, dict] = {
    "01_pass_old_tom_distillery.png": {
        "brand": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "container_size_ml": 750,
    },
    "02_warn_stones_throw_gin.png": {
        "brand": "Stone's Throw",  # title case in the producer record
        "class_type": "London Dry Gin",
        "container_size_ml": 750,
    },
    "03_fail_mountain_crest_ipa.png": {
        "brand": "Mountain Crest Brewing",
        "class_type": "West Coast IPA",
        "container_size_ml": 473,
    },
    "04_unreadable_heritage_vineyards.png": {
        "brand": "Heritage Vineyards",
        "class_type": "Cabernet Sauvignon",
        "container_size_ml": 750,
    },
}

TTB_SYSTEM_PROMPT = """You are a TTB (Alcohol and Tobacco Tax and Trade Bureau) \
compliance auditor. You review alcoholic-beverage labels against the regulations \
in 27 CFR Part 4 (wine), Part 5 (distilled spirits), Part 7 (malt beverages / \
beer), and Part 16 (Health Warning Statement).

For each label, your job is to:
  1. Determine the beverage type (beer, wine, distilled spirits, or unknown).
  2. Assess whether the photo is good enough to evaluate. A sensor pre-check
     is run before you see the image and the user message includes its
     verdict + objective metrics (sharpness, glare, brightness, contrast,
     megapixels, EXIF). Treat that as a prior, not a ceiling — agree with
     it when it matches what you see, override it if you can read elements
     it called unreliable, and explain in image_quality_notes either way.
     If lighting, glare, blur, rotation, or occlusion prevent confident
     reading of any required element, mark image_quality "degraded" or
     "unreadable" and downgrade the affected rules to "advisory" rather
     than guessing.
  3. Extract every TTB-mandated label element you can read.
  4. Apply the relevant rule set and return a per-rule verdict.
  5. Cross-check the label against the producer's claimed brand, class/type,
     and container size. Substantive mismatches are FAIL; trivial typography
     differences (e.g. brand on label is "STONE'S THROW" while record says
     "Stone's Throw") are WARN.

Rules (use these rule_id strings exactly):

BEER (27 CFR 7):
  - beer.brand_name.presence            (7.22)
  - beer.class_type.presence            (7.24)
  - beer.alcohol_content.format         (7.71)  [if declared]
  - beer.net_contents.presence          (7.27)
  - beer.name_address.presence          (7.25)
  - beer.health_warning.exact_text      (16.21)

WINE (27 CFR 4):
  - wine.brand_name.presence
  - wine.class_type.presence
  - wine.appellation.presence_if_claimed
  - wine.vintage.format_if_claimed
  - wine.alcohol_content.presence       (mandatory above 7% ABV)
  - wine.net_contents.standard_fill     (50/100/187/375/500/750/1000/1500/3000 mL)
  - wine.bottler_address.presence
  - wine.sulfite_declaration.presence
  - wine.health_warning.exact_text      (16.21)

DISTILLED SPIRITS (27 CFR 5):
  - spirits.brand_name.presence
  - spirits.class_type.presence
  - spirits.alcohol_content.presence    (mandatory) + format
  - spirits.proof.presence_optional     + format
  - spirits.net_contents.standard_fill  (50/100/200/375/750/1000/1750 mL)
  - spirits.name_address.presence
  - spirits.age_statement.format_if_claimed
  - spirits.health_warning.exact_text   (16.21)

The HEALTH WARNING STATEMENT must appear verbatim:

  GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT
  DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH
  DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO
  DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS.

Edit-distance tolerance is ZERO. Paraphrasing, case changes ("Government
Warning" instead of "GOVERNMENT WARNING"), or substituted/dropped words count
as FAIL — quote the offending text in your finding.

Verdict semantics (overall_verdict):
  - "pass"        : every required rule passes.
  - "warn"        : non-substantive variation worth flagging (typography,
                    case-only mismatches against the producer record,
                    advisory-only failures).
  - "fail"        : at least one required rule fails.
  - "unreadable"  : image quality prevents a confident verdict on required rules.

Be honest. A wrong "pass" is the worst outcome. When you cannot verify with
confidence, downgrade to "advisory" and say so."""


class ExtractedFields(BaseModel):
    brand_name: Optional[str] = None
    class_type: Optional[str] = None
    alcohol_content: Optional[str] = None
    net_contents: Optional[str] = None
    name_address: Optional[str] = None
    health_warning_text: Optional[str] = Field(
        None, description="Verbatim Health Warning Statement text as observed."
    )
    other: Optional[str] = Field(
        None, description="Vintage, age statement, sulfites, proof, etc."
    )


class RuleResult(BaseModel):
    rule_id: str
    citation: str
    status: Literal["pass", "fail", "advisory"]
    finding: str
    expected: Optional[str] = None
    fix_suggestion: Optional[str] = None


class LabelReport(BaseModel):
    beverage_type: Literal["beer", "wine", "spirits", "unknown"]
    image_quality: Literal["good", "degraded", "unreadable"]
    image_quality_notes: str
    overall_verdict: Literal["pass", "warn", "fail", "unreadable"]
    extracted_fields: ExtractedFields
    rule_results: list[RuleResult]
    summary: str


def encode_image(path: Path) -> tuple[str, str, bytes]:
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    raw = path.read_bytes()
    return base64.standard_b64encode(raw).decode("ascii"), media, raw


def _format_capture_block(capture: CaptureQualityReport) -> str:
    """Render the sensor pre-check as a model-readable text block.

    Includes the new region-aware fields — label bbox, per-region metrics,
    glare-blob bboxes, backlight + motion-blur flags, sensor tier — so the
    model can reason locally instead of treating the frame as a single
    monolithic blob."""
    lines = ["Sensor pre-check (objective measurements; treat as a prior):"]
    for s in capture.surfaces:
        sensor = s.sensor.describe()
        m = s.metrics
        metrics = (
            f"sharpness={m.sharpness:.0f}, "
            f"glare={m.glare_fraction * 100:.0f}%, "
            f"brightness={m.brightness_mean:.0f}, "
            f"contrast={m.brightness_stddev:.0f}, "
            f"resolution={m.megapixels} MP"
        )
        issues = "; ".join(s.issues) if s.issues else "no issues detected"
        lines.append(
            f"  {s.surface}: verdict={s.verdict} confidence={s.confidence:.2f} "
            f"({sensor})\n"
            f"    frame metrics  : {metrics}"
        )
        if s.label_bbox is not None and s.metrics_label is not None:
            lm = s.metrics_label
            bb = s.label_bbox
            lines.append(
                f"    label region   : bbox=[x={bb.x},y={bb.y},w={bb.w},h={bb.h}]"
                f"  sharpness={lm.sharpness:.0f}  "
                f"glare={lm.glare_fraction * 100:.0f}%  "
                f"brightness={lm.brightness_mean:.0f}"
            )
        if s.glare_blobs:
            blob_descs = [
                f"#{i+1} [{b.bbox.x},{b.bbox.y},{b.bbox.w}x{b.bbox.h}] "
                f"={b.area_fraction_label*100:.0f}% of label"
                for i, b in enumerate(s.glare_blobs[:4])
            ]
            lines.append("    glare blobs    : " + " · ".join(blob_descs))
        flags = []
        if s.backlit:
            flags.append("backlit")
        if s.motion_blur_direction:
            flags.append(f"motion={s.motion_blur_direction}")
        source = getattr(s, "capture_source", "photo")
        if source == "artwork":
            flags.append(
                "digital artwork (no EXIF, uniform border) — photo-quality "
                "checks (glare, exposure, smudge, low-resolution) skipped"
            )
        elif source == "screenshot":
            flags.append("screenshot, not a camera frame")
        if flags:
            lines.append("    flags          : " + ", ".join(flags))
        lines.append(f"    issues         : {issues}")
    lines.append(
        f"  overall: verdict={capture.overall_verdict} "
        f"confidence={capture.overall_confidence:.2f}"
    )
    lines.append(
        "If a bbox you cite for an extracted field falls inside a glare blob, "
        "that field's confidence MUST reflect the occlusion."
    )
    return "\n".join(lines)


def review_label(
    client: anthropic.Anthropic, path: Path
) -> tuple[LabelReport, object, CaptureQualityReport]:
    b64_data, media_type, raw_bytes = encode_image(path)

    capture = assess_capture_quality({"front": raw_bytes})

    expected = EXPECTED.get(path.name)
    expected_block = (
        f"Producer record: brand={expected['brand']!r}, "
        f"class/type={expected['class_type']!r}, "
        f"container_size_ml={expected['container_size_ml']}."
        if expected
        else "No producer record was provided — audit the label on its own merits."
    )

    capture_block = _format_capture_block(capture)

    response = client.messages.parse(
        model=MODEL,
        max_tokens=8192,
        thinking={"type": "adaptive", "display": "summarized"},
        system=[
            {
                "type": "text",
                "text": TTB_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Audit this label (filename: {path.name}).\n"
                            f"{expected_block}\n\n"
                            f"{capture_block}\n\n"
                            "Return your structured TTB-compliance report."
                        ),
                    },
                ],
            }
        ],
        output_format=LabelReport,
    )
    return response.parsed_output, response.usage, capture


VERDICT_COLOR = {
    "pass": "\033[92m",
    "warn": "\033[93m",
    "fail": "\033[91m",
    "unreadable": "\033[95m",
    "good": "\033[92m",
    "degraded": "\033[93m",
}
STATUS_COLOR = {"pass": "\033[92m", "fail": "\033[91m", "advisory": "\033[93m"}
RESET = "\033[0m"


def print_capture(capture: CaptureQualityReport) -> None:
    """Print sensor-check verdict + per-surface metrics + region detail."""
    overall_color = VERDICT_COLOR.get(capture.overall_verdict, "")
    print(
        f"  Sensor pre-check: {overall_color}{capture.overall_verdict.upper()}{RESET} "
        f"(confidence {capture.overall_confidence:.2f})"
    )
    for s in capture.surfaces:
        v_color = VERDICT_COLOR.get(s.verdict, "")
        m = s.metrics
        print(
            f"    {s.surface:5s} {v_color}{s.verdict:9s}{RESET} "
            f"conf={s.confidence:.2f}  "
            f"sharpness={m.sharpness:6.0f}  "
            f"glare={m.glare_fraction * 100:4.0f}%  "
            f"brightness={m.brightness_mean:5.0f}  "
            f"contrast={m.brightness_stddev:5.0f}  "
            f"{m.megapixels} MP"
        )
        sensor_desc = s.sensor.describe()
        if sensor_desc and sensor_desc != "no EXIF available":
            print(f"           sensor : {sensor_desc}")
        if s.label_bbox is not None and s.metrics_label is not None:
            lb = s.label_bbox
            lm = s.metrics_label
            print(
                f"           label  : bbox=[{lb.x},{lb.y},{lb.w}x{lb.h}]  "
                f"sharpness={lm.sharpness:.0f}  "
                f"glare={lm.glare_fraction * 100:.0f}%  "
                f"brightness={lm.brightness_mean:.0f}"
            )
        if s.glare_blobs:
            blobs = " · ".join(
                f"#{i+1}[{b.bbox.x},{b.bbox.y},{b.bbox.w}x{b.bbox.h}] "
                f"={b.area_fraction_label * 100:.0f}%"
                for i, b in enumerate(s.glare_blobs[:4])
            )
            print(f"           glare  : {blobs}")
        flags = []
        if s.backlit:
            flags.append("backlit")
        if s.motion_blur_direction:
            flags.append(f"motion:{s.motion_blur_direction}")
        if getattr(s, "lens_smudge_likely", False):
            flags.append("smudge/fog")
        if getattr(s, "wet_bottle_likely", False):
            flags.append("wet")
        source = getattr(s, "capture_source", "photo")
        if source == "screenshot":
            flags.append("screenshot")
        elif source == "artwork":
            flags.append("artwork")
        if flags:
            print(f"           flags  : {', '.join(flags)}")
        for issue in s.issues:
            print(f"           ! {issue}")
        for sug in s.suggestions:
            print(f"           → {sug}")


def print_report(
    path: Path,
    report: LabelReport,
    usage,
    capture: CaptureQualityReport | None = None,
) -> None:
    bar = "─" * 78
    print(f"\n{bar}")
    print(f"  {path.name}")
    print(bar)
    if capture is not None:
        print_capture(capture)
        print()
    color = VERDICT_COLOR.get(report.overall_verdict, "")
    print(f"  Verdict       : {color}{report.overall_verdict.upper()}{RESET}")
    print(f"  Beverage type : {report.beverage_type}")
    print(f"  Image quality : {report.image_quality} — {report.image_quality_notes}")
    print()
    print("  Extracted fields:")
    for field, value in report.extracted_fields.model_dump().items():
        if value:
            shown = str(value)
            if len(shown) > 90:
                shown = shown[:87] + "..."
            print(f"    • {field:22s} {shown}")
    print()
    print("  Rule results:")
    for r in report.rule_results:
        c = STATUS_COLOR.get(r.status, "")
        print(f"    {c}[{r.status.upper():8s}]{RESET} {r.rule_id}  ({r.citation})")
        print(f"               {r.finding}")
        if r.status != "pass" and r.fix_suggestion:
            print(f"               → fix: {r.fix_suggestion}")
    print()
    print(f"  Summary: {report.summary}")
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    print(
        f"\n  [tokens] in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_read={cache_read} cache_create={cache_create}"
    )


def print_capture_only(path: Path, capture: CaptureQualityReport) -> None:
    """Output for `--no-call`: pre-check verdict only, no model spend."""
    bar = "─" * 78
    print(f"\n{bar}")
    print(f"  {path.name}")
    print(bar)
    print_capture(capture)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit sample labels with Claude Opus 4.7."
    )
    parser.add_argument("--label", type=Path, help="Audit a single label.")
    parser.add_argument(
        "--no-call",
        action="store_true",
        help=(
            "Run the camera-sensor pre-check only and print its verdict, "
            "without sending the image to the Claude API. Useful for "
            "validating capture-quality behaviour offline."
        ),
    )
    args = parser.parse_args()

    paths = [args.label] if args.label else sorted(LABELS_DIR.glob("*.png"))
    if not paths:
        print(f"error: no labels found in {LABELS_DIR}", file=sys.stderr)
        return 2

    if args.no_call:
        exit_code = 0
        for path in paths:
            if not path.exists():
                print(f"error: {path} does not exist", file=sys.stderr)
                exit_code = 1
                continue
            capture = assess_capture_quality({"front": path.read_bytes()})
            print_capture_only(path, capture)
        return exit_code

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print(
            "error: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for an OAuth bearer token), "
            "or pass --no-call to run the sensor pre-check without an API request.",
            file=sys.stderr,
        )
        return 2

    client = anthropic.Anthropic()
    exit_code = 0
    for path in paths:
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            exit_code = 1
            continue

        # Skip the model call when the pre-check has already declared the
        # frame unrecoverable. Saves spend and surfaces a clear retry hint.
        local_capture = assess_capture_quality({"front": path.read_bytes()})
        if local_capture.overall_verdict == "unreadable":
            print()
            print(f"  [{path.name}] sensor pre-check rejected — skipping API call")
            print_capture_only(path, local_capture)
            continue

        try:
            report, usage, capture = review_label(client, path)
        except anthropic.APIError as e:
            print(f"\n[{path.name}] API error: {e}", file=sys.stderr)
            exit_code = 1
            continue
        print_report(path, report, usage, capture)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
