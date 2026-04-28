#!/usr/bin/env python3
"""
Generate sample alcohol beverage labels for Proofread testing.

Produces four labels covering the verdict states the app must distinguish:

  01_pass           — Old Tom Distillery (Kentucky Straight Bourbon).
                      All TTB-required fields correct and verbatim.

  02_warn           — Stone's Throw Dry Gin.
                      Brand on label is ALL CAPS ("STONE'S THROW") while the
                      application metadata uses title case ("Stone's Throw").
                      Trivial delta — should surface as WARN, not auto-PASS.

  03_fail           — Mountain Crest Brewing IPA.
                      Government Warning is paraphrased and rendered in title
                      case rather than ALL CAPS — a substantive violation that
                      Jenny Park flagged in the discovery interviews.

  04_unreadable     — Heritage Vineyards Cabernet, photographed badly:
                      rotated, overexposed glare, soft focus on one side.
                      Tests the app's "fail loud rather than guess" path.
"""

from __future__ import annotations

import os
import math
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops, ImageEnhance

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels")
os.makedirs(OUT, exist_ok=True)

# ---------- palette ----------
CREAM = (247, 240, 220)
PARCHMENT = (232, 216, 188)
INK = (28, 24, 18)
INK_SOFT = (60, 52, 42)
AMBER = (170, 120, 50)
DEEP_RED = (124, 36, 36)
FOREST = (44, 78, 56)
NAVY = (14, 42, 74)
BURGUNDY = (88, 22, 30)
WHITE = (255, 255, 255)

# ---------- TTB-mandated warning text ----------
WARNING_VERBATIM = (
    "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD "
    "NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF "
    "BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR "
    "ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH "
    "PROBLEMS."
)

WARNING_PARAPHRASED_TITLECASE = (
    "Government Warning: According to the Surgeon General, pregnant women "
    "should avoid alcoholic beverages because of the risk of birth defects. "
    "Drinking impairs your ability to drive and may cause health problems."
)

# ---------- font paths ----------
FONTS = {
    "times":         "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "times_bold":    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "times_italic":  "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
    "arial":         "/System/Library/Fonts/Supplemental/Arial.ttf",
    "arial_bold":    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "arial_narrow":  "/System/Library/Fonts/Supplemental/Arial Narrow.ttf",
    "copperplate":   "/System/Library/Fonts/Supplemental/Copperplate.ttc",
    "didot":         "/System/Library/Fonts/Supplemental/Didot.ttc",
    "baskerville":   "/System/Library/Fonts/Supplemental/Baskerville.ttc",
    "helvetica":     "/System/Library/Fonts/Helvetica.ttc",
}


def f(key: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    """Load a TrueType font. .ttc collections accept an index."""
    return ImageFont.truetype(FONTS[key], size, index=index)


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    """Greedy word wrap to fit max_w in pixels."""
    words = text.split()
    out, cur = [], []
    for w in words:
        candidate = " ".join(cur + [w])
        if draw.textlength(candidate, font=font) <= max_w:
            cur.append(w)
        else:
            if cur:
                out.append(" ".join(cur))
            cur = [w]
    if cur:
        out.append(" ".join(cur))
    return out


def draw_centered(draw, cx: int, y: int, text: str, font, fill) -> int:
    w = draw.textlength(text, font=font)
    draw.text((cx - w / 2, y), text, font=font, fill=fill)
    return y + font.size


def draw_paragraph(draw, x, y, w, text, font, fill, leading_extra=3) -> int:
    lines = wrap(draw, text, font, w)
    line_h = font.size + leading_extra
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_h), line, font=font, fill=fill)
    return y + len(lines) * line_h


def draw_diamond(d: ImageDraw.ImageDraw, cx: int, cy: int, size: int, fill) -> None:
    """Filled diamond centered at (cx, cy)."""
    d.polygon(
        [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)],
        fill=fill,
    )


def draw_top_ornament(d: ImageDraw.ImageDraw, cx: int, cy: int, color) -> None:
    """Small art-deco ornament: line — diamond — line — diamond — line."""
    d.line([(cx - 90, cy), (cx - 22, cy)], fill=color, width=2)
    d.line([(cx + 22, cy), (cx + 90, cy)], fill=color, width=2)
    draw_diamond(d, cx - 12, cy, 5, color)
    draw_diamond(d, cx + 12, cy, 5, color)
    draw_diamond(d, cx, cy, 7, color)


def parchment_texture(img: Image.Image) -> Image.Image:
    """Add a subtle warm vignette and noise so labels don't look sterile."""
    w, h = img.size
    overlay = Image.new("RGB", (w, h), (0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # radial darkening at edges
    for i in range(80):
        a = int(2 + i * 0.6)
        od.rectangle([i, i, w - i, h - i], outline=(a, a, a))
    overlay = overlay.filter(ImageFilter.GaussianBlur(40))
    out = ImageChops.subtract(img, overlay)
    return out


# ============================================================
# LABEL 1 — PASS: Old Tom Distillery
# ============================================================
def make_pass() -> None:
    W, H = 1200, 1500
    img = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(img)
    cx = W // 2

    # Borders
    d.rectangle([60, 60, W - 60, H - 60], outline=INK, width=4)
    d.rectangle([78, 78, W - 78, H - 78], outline=INK, width=1)

    # Top ornament
    draw_top_ornament(d, cx, 130, AMBER)

    # Brand
    draw_centered(d, cx, 180, "OLD TOM", f("times_bold", 130), INK)
    draw_centered(d, cx, 320, "DISTILLERY", f("times_bold", 110), INK)

    # Decorative rule with center diamond
    d.line([(180, 480), (cx - 28, 480)], fill=INK, width=2)
    d.line([(cx + 28, 480), (W - 180, 480)], fill=INK, width=2)
    draw_diamond(d, cx, 480, 12, AMBER)

    # Class / type
    draw_centered(d, cx, 510, "Kentucky Straight Bourbon Whiskey",
                  f("times_italic", 56), INK)

    # Established
    draw_centered(d, cx, 600, "EST. 1887", f("copperplate", 30), AMBER)

    # Age medallion
    my = 690
    d.ellipse([cx - 140, my, cx + 140, my + 280], outline=AMBER, width=4)
    d.ellipse([cx - 130, my + 10, cx + 130, my + 270], outline=AMBER, width=1)
    draw_centered(d, cx, my + 50, "AGED", f("copperplate", 30), INK)
    draw_centered(d, cx, my + 100, "8", f("times_bold", 110), AMBER)
    draw_centered(d, cx, my + 220, "YEARS", f("copperplate", 30), INK)

    # Lower decorative rule
    d.line([(180, 1020), (W - 180, 1020)], fill=INK, width=1)

    # ABV / proof / contents
    draw_centered(d, cx, 1050, "45% Alc./Vol. (90 Proof)",
                  f("times_bold", 44), INK)
    draw_centered(d, cx, 1115, "750 mL", f("times", 38), INK)

    # Producer
    draw_centered(d, cx, 1185, "Bottled by Old Tom Distilling Co.",
                  f("times_italic", 30), INK_SOFT)
    draw_centered(d, cx, 1225, "Bardstown, Kentucky  ·  USA",
                  f("times_italic", 26), INK_SOFT)

    # Government warning (verbatim, ALL CAPS, bold)
    draw_paragraph(d, 110, 1300, W - 220, WARNING_VERBATIM,
                   f("arial_bold", 18), INK, leading_extra=5)

    img = parchment_texture(img)
    img.save(os.path.join(OUT, "01_pass_old_tom_distillery.png"), optimize=True)


# ============================================================
# LABEL 2 — WARN: Stone's Throw Dry Gin (case mismatch)
# ============================================================
def make_warn() -> None:
    W, H = 1200, 1500
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    cx = W // 2

    # Botanical border (top + bottom thin rules)
    d.line([(80, 110), (W - 80, 110)], fill=FOREST, width=3)
    d.line([(80, 130), (W - 80, 130)], fill=FOREST, width=1)
    d.line([(80, H - 130), (W - 80, H - 130)], fill=FOREST, width=1)
    d.line([(80, H - 110), (W - 80, H - 110)], fill=FOREST, width=3)

    draw_centered(d, cx, 170, "SMALL BATCH ·  CRAFT DISTILLED",
                  f("copperplate", 24), FOREST)

    # Brand: ALL CAPS on the label (the application-side record is "Stone's Throw")
    draw_centered(d, cx, 240, "STONE'S", f("didot", 150, index=1), FOREST)
    draw_centered(d, cx, 410, "THROW", f("didot", 150, index=1), FOREST)

    draw_centered(d, cx, 600, "—  London Dry Gin  —",
                  f("baskerville", 50, index=2), INK)

    # Botanical hint (simple geometric flourish)
    fy = 700
    for i in range(7):
        ang = math.pi * (0.1 + i * 0.13)
        x1 = cx + int(math.cos(ang) * 90)
        y1 = fy + int(math.sin(ang) * 30)
        x2 = cx + int(math.cos(ang) * 130)
        y2 = fy + int(math.sin(ang) * 50)
        d.line([(x1, y1), (x2, y2)], fill=FOREST, width=2)
    for i in range(7):
        ang = math.pi * (1.1 + i * 0.13)
        x1 = cx + int(math.cos(ang) * 90)
        y1 = fy + int(math.sin(ang) * 30)
        x2 = cx + int(math.cos(ang) * 130)
        y2 = fy + int(math.sin(ang) * 50)
        d.line([(x1, y1), (x2, y2)], fill=FOREST, width=2)
    d.ellipse([cx - 12, fy - 12, cx + 12, fy + 12], fill=FOREST)

    # Tasting notes-style copy
    draw_centered(d, cx, 800, "Distilled with juniper, coriander, angelica root,",
                  f("baskerville", 30, index=0), INK_SOFT)
    draw_centered(d, cx, 845, "and a whisper of cardamom from Kerala.",
                  f("baskerville", 30, index=0), INK_SOFT)

    # Proof and net contents
    draw_centered(d, cx, 970, "41.5% Alc./Vol.  ·  83 Proof",
                  f("arial_bold", 38), INK)
    draw_centered(d, cx, 1030, "750 mL",
                  f("arial", 34), INK)

    # Producer
    draw_centered(d, cx, 1110, "Distilled & bottled by",
                  f("baskerville", 22, index=2), INK_SOFT)
    draw_centered(d, cx, 1140, "Stone's Throw Distilling Co., Portland, OR",
                  f("baskerville", 28, index=2), INK)

    # Warning (verbatim — only the brand differs)
    draw_paragraph(d, 110, 1200, W - 220, WARNING_VERBATIM,
                   f("arial_bold", 18), INK, leading_extra=5)

    img.save(os.path.join(OUT, "02_warn_stones_throw_gin.png"), optimize=True)


# ============================================================
# LABEL 3 — FAIL: Mountain Crest Brewing (paraphrased warning)
# ============================================================
def make_fail() -> None:
    W, H = 1200, 1500
    img = Image.new("RGB", (W, H), (240, 230, 210))
    d = ImageDraw.Draw(img)
    cx = W // 2

    # Mountain silhouette (top decoration)
    pts = [(80, 280), (260, 130), (380, 220), (560, 100), (720, 240),
           (880, 140), (1040, 220), (W - 80, 180), (W - 80, 320), (80, 320)]
    d.polygon(pts, fill=NAVY)
    d.polygon([(80, 280), (260, 130), (380, 220), (440, 195), (380, 320),
               (80, 320)], fill=BURGUNDY)

    # Banner background
    d.rectangle([60, 360, W - 60, 460], fill=NAVY)
    draw_centered(d, cx, 380, "MOUNTAIN CREST",
                  f("copperplate", 60, index=2), CREAM)

    draw_centered(d, cx, 500, "BREWING COMPANY",
                  f("copperplate", 40, index=0), NAVY)

    # Beer name
    draw_centered(d, cx, 600, "Summit Trail",
                  f("baskerville", 110, index=2), BURGUNDY)
    draw_centered(d, cx, 740, "WEST COAST IPA",
                  f("arial_bold", 50), NAVY)

    # Hop emblem (simple stacked circles)
    hx, hy = cx, 850
    for i, r in enumerate([28, 22, 18, 14]):
        d.ellipse([hx - r, hy - 30 + i * 16, hx + r, hy + 8 + i * 16],
                  outline=FOREST, width=3)

    # Stats
    draw_centered(d, cx, 950, "6.8% ALC./VOL.",
                  f("arial_bold", 44), NAVY)
    draw_centered(d, cx, 1010, "70 IBU  ·  16 FL OZ (473 mL)",
                  f("arial", 32), INK)

    # Producer
    draw_centered(d, cx, 1090, "Brewed and canned by",
                  f("baskerville", 22, index=2), INK_SOFT)
    draw_centered(d, cx, 1120, "Mountain Crest Brewing Co., Bend, Oregon",
                  f("baskerville", 28, index=2), INK)

    # Government warning — DELIBERATELY PARAPHRASED + TITLE CASE (this is the violation)
    draw_paragraph(d, 110, 1200, W - 220, WARNING_PARAPHRASED_TITLECASE,
                   f("arial", 18), INK, leading_extra=5)

    img.save(os.path.join(OUT, "03_fail_mountain_crest_ipa.png"), optimize=True)


# ============================================================
# LABEL 4 — UNREADABLE: Heritage Vineyards (rotated + glare + blur)
# ============================================================
def make_unreadable() -> None:
    # First, render a clean wine label at full quality
    LW, LH = 1000, 1400
    label = Image.new("RGB", (LW, LH), (245, 240, 228))
    d = ImageDraw.Draw(label)
    cx = LW // 2

    # Wine-label borders
    d.rectangle([50, 50, LW - 50, LH - 50], outline=BURGUNDY, width=3)
    d.line([(70, 80), (LW - 70, 80)], fill=BURGUNDY, width=1)
    d.line([(70, LH - 80), (LW - 70, LH - 80)], fill=BURGUNDY, width=1)

    # Crest (simple shield)
    sx, sy = cx, 200
    d.polygon([(sx - 50, sy - 60), (sx + 50, sy - 60),
               (sx + 50, sy + 30), (sx, sy + 80), (sx - 50, sy + 30)],
              outline=BURGUNDY, width=3)
    draw_centered(d, sx, sy - 20, "HV", f("times_bold", 36), BURGUNDY)

    # Estate name
    draw_centered(d, cx, 320, "HERITAGE", f("didot", 90, index=1), BURGUNDY)
    draw_centered(d, cx, 430, "VINEYARDS", f("didot", 70, index=1), BURGUNDY)

    # Decorative line
    d.line([(180, 540), (LW - 180, 540)], fill=BURGUNDY, width=1)

    # Vintage and varietal
    draw_centered(d, cx, 580, "2019", f("times", 80), INK)
    draw_centered(d, cx, 700, "Cabernet Sauvignon",
                  f("baskerville", 70, index=2), INK)

    # Appellation
    draw_centered(d, cx, 830, "NAPA  VALLEY",
                  f("copperplate", 36, index=2), BURGUNDY)

    # ABV and contents
    draw_centered(d, cx, 950, "14.2% Alc./Vol.",
                  f("times", 38), INK)
    draw_centered(d, cx, 1000, "750 mL",
                  f("times", 32), INK)

    # Bottler
    draw_centered(d, cx, 1080, "Produced and bottled by",
                  f("times_italic", 22), INK_SOFT)
    draw_centered(d, cx, 1115, "Heritage Vineyards · Oakville, CA",
                  f("times_italic", 26), INK)

    # Warning (verbatim — content is correct, the photo is the problem)
    draw_paragraph(d, 90, 1200, LW - 180, WARNING_VERBATIM,
                   f("arial_bold", 16), INK, leading_extra=4)

    # ---- Now degrade the image: rotate + glare + blur on one side ----
    # Pad to a larger canvas so rotation doesn't crop content
    canvas = Image.new("RGB", (1400, 1700), (32, 30, 28))
    canvas.paste(label, ((1400 - LW) // 2, (1700 - LH) // 2))

    # Rotate
    canvas = canvas.rotate(-7, resample=Image.BICUBIC, fillcolor=(32, 30, 28),
                           expand=False)

    # Add a directional glare (bright radial gradient on upper-left)
    glare = Image.new("L", canvas.size, 0)
    gd = ImageDraw.Draw(glare)
    for r, alpha in [(520, 220), (380, 240), (260, 250), (140, 255)]:
        gd.ellipse([350 - r, 200 - r, 350 + r, 200 + r], fill=alpha)
    glare = glare.filter(ImageFilter.GaussianBlur(120))
    glare_color = Image.new("RGB", canvas.size, (255, 248, 230))
    canvas = Image.composite(glare_color, canvas, glare)

    # Slight blur (out-of-focus camera)
    canvas = canvas.filter(ImageFilter.GaussianBlur(1.4))

    # Reduce contrast on glare side
    enhancer = ImageEnhance.Contrast(canvas)
    canvas = enhancer.enhance(0.92)

    # Add a subtle fingerprint smudge (low-opacity blur smear in lower-right)
    smudge = Image.new("L", canvas.size, 0)
    sd = ImageDraw.Draw(smudge)
    for i in range(6):
        sd.ellipse([1000 + i * 8, 1200 + i * 4,
                    1180 + i * 8, 1320 + i * 4], fill=60 - i * 8)
    smudge = smudge.filter(ImageFilter.GaussianBlur(40))
    smudge_color = Image.new("RGB", canvas.size, (180, 175, 165))
    canvas = Image.composite(smudge_color, canvas, smudge)

    canvas.save(os.path.join(OUT, "04_unreadable_heritage_vineyards.png"),
                optimize=True)


# ============================================================
# Master swatch — color palette + brand reference card
# ============================================================
def make_palette_card() -> None:
    W, H = 1200, 600
    img = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(img)

    d.text((60, 50), "Proofread — Brand Palette",
           font=f("arial_bold", 44), fill=NAVY)
    d.text((60, 110), "Visual identity for label verification UI and surfaces",
           font=f("arial", 22), fill=INK_SOFT)

    swatches = [
        ("Navy",           NAVY,        "#0E2A4A",  "Primary"),
        ("Amber",          AMBER,       "#AA7832",  "Accent / verification"),
        ("Cream",          CREAM,       "#F7F0DC",  "Background"),
        ("Pass green",     (63, 139, 90),  "#3F8B5A", "PASS state"),
        ("Warn yellow",    (230, 184, 71), "#E6B847", "WARN state"),
        ("Fail red",       (179, 58, 58),  "#B33A3A", "FAIL state"),
    ]
    swatch_w = (W - 120 - 5 * 24) // 6
    sy = 200
    for i, (name, rgb, hexv, role) in enumerate(swatches):
        sx = 60 + i * (swatch_w + 24)
        d.rectangle([sx, sy, sx + swatch_w, sy + 200], fill=rgb,
                    outline=INK, width=1)
        d.text((sx, sy + 220), name, font=f("arial_bold", 22), fill=INK)
        d.text((sx, sy + 252), hexv, font=f("arial", 18), fill=INK_SOFT)
        d.text((sx, sy + 278), role, font=f("arial", 16), fill=INK_SOFT)

    img.save(os.path.join(os.path.dirname(OUT), "brand", "palette.png"),
             optimize=True)


def main() -> None:
    print("Generating Proofread sample labels...")
    make_pass()
    print("  ✓ 01_pass_old_tom_distillery.png")
    make_warn()
    print("  ✓ 02_warn_stones_throw_gin.png")
    make_fail()
    print("  ✓ 03_fail_mountain_crest_ipa.png")
    make_unreadable()
    print("  ✓ 04_unreadable_heritage_vineyards.png")
    make_palette_card()
    print("  ✓ brand/palette.png")
    print(f"\nWritten to: {OUT}")


if __name__ == "__main__":
    main()
