"""Synthetic beer-label PNG generator.

`synthesize_label(spec)` renders a plausible front + back beer-label image
from a `LabelSpec` dict and returns the bytes plus ground-truth metadata.

The renderer is deliberately simple Pillow — no fancy graphics, just
plausibly-laid-out text on a colored background. The point of the harness
is to exercise the rule engine end-to-end with controlled inputs; pixel
realism is a non-goal.

Two outputs matter:

1. The PNG bytes — what an OCR / vision provider sees.
2. The `ground_truth_text` per surface — what the *perfect-mock* OCR
   returns. By construction it is a verbatim copy of every text fragment
   the synthesizer rendered on that surface, in the same order, joined by
   newlines. That parity is the trust contract: if the harness produces a
   wrong rule outcome under perfect mock OCR, the bug is in the
   synthesizer's specification, not in OCR.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# Canonical Health Warning — single source of truth for the synthesizer.
# Backend's canonical file is the authoritative one for the rule engine;
# we read it at import time so the synthesizer can never drift.
_CANONICAL_HW_PATH = (
    Path(__file__).resolve().parent.parent / "app" / "canonical" / "health_warning.txt"
)
CANONICAL_HEALTH_WARNING = _CANONICAL_HW_PATH.read_text(encoding="utf-8").strip()


# Class/type values must be in app.services.extractors.beer.BEER_CLASSES
# for the extractor to recognize them. We mirror a subset here as the
# allowed input vocabulary; if the extractor list changes, the test
# corpus may stop matching and that's the right signal to revisit this.
SAFE_BEER_CLASSES = [
    "India Pale Ale",
    "Pale Ale",
    "Stout",
    "Lager",
    "Pilsner",
    "Porter",
    "Hefeweizen",
    "Saison",
    "Amber Ale",
    "Brown Ale",
]


@dataclass
class LabelSpec:
    """Specification for one synthetic label.

    Use `from_dict` for a loose dict-style construction. Fields set to
    None are intentionally omitted from the rendered label and from the
    ground-truth text.
    """

    brand: str | None
    class_type: str | None
    abv: str | None  # e.g. "5.5% ABV"
    net_contents: str | None  # e.g. "12 FL OZ"
    name_address: str | None  # e.g. "Brewed and bottled by Foo Co., Bar, ST 00000"
    health_warning_text: str | None  # full warning text or None to omit
    country: str | None = None  # e.g. "Germany"
    is_imported: bool = False
    container_size_ml: int = 355
    # Free-form metadata for harness bookkeeping; not rendered.
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelSpec:
        return cls(
            brand=d.get("brand"),
            class_type=d.get("class_type"),
            abv=d.get("abv"),
            net_contents=d.get("net_contents"),
            name_address=d.get("name_address"),
            health_warning_text=d.get("health_warning_text"),
            country=d.get("country"),
            is_imported=d.get("is_imported", False),
            container_size_ml=d.get("container_size_ml", 355),
            metadata=d.get("metadata", {}),
        )


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

_LABEL_W, _LABEL_H = 800, 1000
_BG_COLORS = [
    (245, 230, 200),  # cream
    (220, 210, 180),  # oat
    (200, 190, 170),  # tan
    (255, 245, 220),  # ivory
]


def _font(size: int) -> ImageFont.ImageFont:
    """Best-effort font loader.

    Tries a small sequence of common system fonts; falls back to Pillow's
    default bitmap font if nothing else is available. The default font is
    quite small, so size hints are advisory under the fallback.
    """
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered(draw: ImageDraw.ImageDraw, y: int, text: str, font, fill=(20, 20, 20)) -> int:
    """Draw `text` centered horizontally at vertical position y.

    Returns the y-coordinate after the line (for stacking).
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (_LABEL_W - w) // 2
    draw.text((x, y), text, font=font, fill=fill)
    return y + h + 12


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Greedy word-wrap to fit within `max_width` pixels.

    Whitespace is preserved as single spaces; this matches the rule
    engine's whitespace normalization and keeps the canonical Health
    Warning intact for the rule check.
    """
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip() if cur else w
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_front(spec: LabelSpec, seed: int) -> tuple[bytes, str]:
    """Render the front surface and return (png_bytes, ground_truth_text).

    The ground-truth text is the OCR-equivalent string a perfect OCR
    would return for this image: every drawn text fragment, in render
    order, joined by newlines. Order matters because some extractors use
    'first hit' semantics across surfaces; rendering brand first matches
    the visual convention.
    """
    rng = random.Random(seed)
    bg = rng.choice(_BG_COLORS)
    img = Image.new("RGB", (_LABEL_W, _LABEL_H), bg)
    draw = ImageDraw.Draw(img)

    rendered: list[str] = []
    y = 60

    # Brand at the top, very large — must be the largest text block on
    # the front so the extractor's "largest block on front" heuristic
    # picks it up.
    if spec.brand is not None:
        f = _font(96)
        y = _draw_centered(draw, y, spec.brand, f) + 30
        rendered.append(spec.brand)

    # Decorative rule line.
    draw.rectangle([(120, y), (_LABEL_W - 120, y + 4)], fill=(80, 60, 30))
    y += 30

    # Class/type below brand, medium size.
    if spec.class_type is not None:
        f = _font(56)
        y = _draw_centered(draw, y, spec.class_type.upper(), f) + 12
        # Recorded with original case so the extractor's regex (case-
        # insensitive against `BEER_CLASSES`) matches.
        rendered.append(spec.class_type.upper())

    y += 80

    # ABV and net contents, side by side.
    f_small = _font(36)
    if spec.abv is not None:
        draw.text((140, y), spec.abv, font=f_small, fill=(20, 20, 20))
        rendered.append(spec.abv)
    if spec.net_contents is not None:
        bbox = draw.textbbox((0, 0), spec.net_contents, font=f_small)
        nw = bbox[2] - bbox[0]
        draw.text((_LABEL_W - 140 - nw, y), spec.net_contents, font=f_small, fill=(20, 20, 20))
        rendered.append(spec.net_contents)

    return _png_bytes(img), "\n".join(rendered)


def _render_back(spec: LabelSpec, seed: int) -> tuple[bytes, str]:
    """Render the back surface and return (png_bytes, ground_truth_text)."""
    rng = random.Random(seed + 1)
    bg = rng.choice(_BG_COLORS)
    img = Image.new("RGB", (_LABEL_W, _LABEL_H), bg)
    draw = ImageDraw.Draw(img)

    rendered: list[str] = []
    y = 60

    # Net contents repeated on back is normal on real labels.
    f_small = _font(28)
    if spec.net_contents is not None:
        y = _draw_centered(draw, y, spec.net_contents, f_small) + 8
        rendered.append(spec.net_contents)

    # Name + address.
    if spec.name_address is not None:
        f_addr = _font(22)
        for line in _wrap_text(draw, spec.name_address, f_addr, _LABEL_W - 120):
            y = _draw_centered(draw, y, line, f_addr) + 0
        rendered.append(spec.name_address)
        y += 20

    # Country of origin (only when imported and declared).
    if spec.is_imported and spec.country is not None:
        f_coo = _font(24)
        coo_line = f"Product of {spec.country}"
        y = _draw_centered(draw, y, coo_line, f_coo) + 16
        rendered.append(coo_line)

    # Health Warning, wrapped.
    if spec.health_warning_text is not None:
        f_hw = _font(22)
        for line in _wrap_text(draw, spec.health_warning_text, f_hw, _LABEL_W - 80):
            draw.text((40, y), line, font=f_hw, fill=(20, 20, 20))
            y += 28
        # Critical: the *exact* health-warning string (not the wrapped
        # multi-line render) is what we record as ground truth, because
        # the rule engine normalizes whitespace before comparison.
        rendered.append(spec.health_warning_text)

    return _png_bytes(img), "\n".join(rendered)


# ----------------------------------------------------------------------------
# Ground-truth derivation
# ----------------------------------------------------------------------------


def derive_ground_truth(spec: LabelSpec) -> dict[str, str]:
    """Map every v1 beer rule_id to its expected status given the spec.

    The mapping mirrors the rule engine's logic AND the v1 beer
    extractor's known behaviour (`app/services/extractors/beer.py`).
    Synthesizer/extractor parity is enforced by `test_precision_targets`.

    Rule ID set is the v1 beer set (per SPEC.md v1.11):

        beer.brand_name.presence            -> pass | fail
        beer.class_type.presence            -> pass | fail
        beer.alcohol_content.format         -> pass | fail (optional → pass)
        beer.net_contents.presence          -> pass | fail
        beer.name_address.presence          -> pass | fail
        beer.country_of_origin...           -> pass | fail | na
        beer.health_warning.exact_text      -> pass | fail
        beer.health_warning.size            -> advisory (always, in v1)

    Methodology note — `brand_name.presence`:

        SPEC v1.10 prescribes the v1 brand extractor as "largest text in
        upper region of front." `_extract_brand_name` in beer.py
        implements this as "pick the biggest block on the front, fall
        back to any surface." So the v1 brand_name presence check passes
        whenever the front has *any* text — even if no human-recognizable
        brand is rendered. We mirror that here: the rule passes whenever
        the front surface is non-empty, regardless of whether `spec.brand`
        was explicitly set. This is an acknowledged v1 extractor
        limitation (a brand-recognition model, not a presence heuristic,
        is out of scope for v1) and the right ground-truth reflection
        of what the extractor actually does. See README "Methodology
        decisions" for the full rationale.
    """
    expected: dict[str, str] = {}

    # Front surface is non-empty as long as the synthesizer renders any of
    # brand / class / abv / net_contents on the front. brand-only being
    # empty does not fail the rule because the v1 extractor falls through
    # to whatever is the largest block on the front.
    front_has_text = any(
        v is not None and str(v).strip()
        for v in (spec.brand, spec.class_type, spec.abv, spec.net_contents)
    )
    expected["beer.brand_name.presence"] = "pass" if front_has_text else "fail"

    if spec.class_type and spec.class_type.strip():
        expected["beer.class_type.presence"] = "pass"
    else:
        expected["beer.class_type.presence"] = "fail"

    # ABV is optional — absent value passes; present value must match the
    # canonical ABV format. The synthesizer only ever generates well-formed
    # ABV strings, so we don't model the malformed case here. (If a future
    # corpus needs malformed ABV, extend the spec with a flag.)
    expected["beer.alcohol_content.format"] = "pass"

    expected["beer.net_contents.presence"] = (
        "pass" if (spec.net_contents and spec.net_contents.strip()) else "fail"
    )

    expected["beer.name_address.presence"] = (
        "pass" if (spec.name_address and spec.name_address.strip()) else "fail"
    )

    if not spec.is_imported:
        expected["beer.country_of_origin.presence_if_imported"] = "na"
    elif spec.country and spec.country.strip():
        expected["beer.country_of_origin.presence_if_imported"] = "pass"
    else:
        expected["beer.country_of_origin.presence_if_imported"] = "fail"

    if spec.health_warning_text is None or not spec.health_warning_text.strip():
        expected["beer.health_warning.exact_text"] = "fail"
    else:
        # The rule normalizes whitespace then computes Levenshtein distance
        # against the canonical. We approximate by exact equality after
        # whitespace normalization.
        normalized_spec = " ".join(spec.health_warning_text.split())
        normalized_canon = " ".join(CANONICAL_HEALTH_WARNING.split())
        expected["beer.health_warning.exact_text"] = (
            "pass" if normalized_spec == normalized_canon else "fail"
        )

    # Size is advisory in v1 regardless of inputs (per SPEC v1.11).
    expected["beer.health_warning.size"] = "advisory"

    return expected


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def synthesize_label(
    spec: LabelSpec | dict[str, Any],
    seed: int = 0,
) -> tuple[bytes, bytes, dict[str, str], dict[str, str]]:
    """Render a label and return (front_png, back_png, ground_truth, ocr_text).

    Args:
        spec: a `LabelSpec` or a dict that `LabelSpec.from_dict` accepts.
        seed: deterministic seed for background-color choice etc.

    Returns:
        front_png_bytes: PNG bytes for the front surface.
        back_png_bytes: PNG bytes for the back surface.
        ground_truth: dict of rule_id → expected status.
        ocr_text: dict of {"front": str, "back": str} — the exact text
            the *perfect mock OCR* should return for each surface. By
            construction this is what the synthesizer drew on each surface,
            so a perfect-OCR run reproduces the spec exactly.
    """
    if isinstance(spec, dict):
        spec = LabelSpec.from_dict(spec)

    front_png, front_text = _render_front(spec, seed=seed)
    back_png, back_text = _render_back(spec, seed=seed)
    ground_truth = derive_ground_truth(spec)
    ocr_text = {"front": front_text, "back": back_text}

    return front_png, back_png, ground_truth, ocr_text
