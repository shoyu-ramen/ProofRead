"""Tests for the adversarial-input guards (SPEC §0.5).

The guards are deliberately conservative — false positives cost the user a
forced rescan, so the heuristics require strong evidence before firing.
These tests pin the behaviour at the calibration the verify path depends
on.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.services.adversarial import (
    AdversarialSignal,
    detect_foreign_language,
    detect_screen_capture,
    merge_signals,
)
from app.services.verify import VerifyInput, verify
from app.services.vision import MockVisionExtractor

CANONICAL = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


# ---------------------------------------------------------------------------
# detect_foreign_language
# ---------------------------------------------------------------------------


def test_english_label_is_not_flagged():
    """A normal English warning + producer line is fine."""
    assert (
        detect_foreign_language(
            CANONICAL,
            "Brewed and bottled by Anytown Brewing Co., Anytown, ST",
            "5.5% Alc./Vol. 12 FL OZ",
        )
        is None
    )


def test_short_text_is_never_flagged():
    """Avoid firing on a single-word brand like "Hofbräu"."""
    assert detect_foreign_language("Hofbräu") is None


def test_spanish_label_with_no_english_keywords_is_flagged():
    spanish = (
        "ADVERTENCIA: De acuerdo con el Cirujano General, las mujeres no deben "
        "beber bebidas alcohólicas durante el embarazo. El consumo de bebidas "
        "alcohólicas afecta la capacidad para conducir."
    )
    sig = detect_foreign_language(spanish)
    assert sig is not None
    assert sig.kind == "foreign_language"
    assert "English" in sig.detail
    assert sig.suggestion


def test_cyrillic_label_is_flagged_by_script():
    """Even if the text is short, a stretch of Cyrillic characters fires."""
    sig = detect_foreign_language(
        "ПРАВИТЕЛЬСТВЕННОЕ ПРЕДУПРЕЖДЕНИЕ: алкогольные напитки",
    )
    assert sig is not None
    assert "CYRILLIC" in sig.detail


def test_cjk_label_is_flagged_by_script():
    sig = detect_foreign_language(
        "政府警告 アルコール 啤酒 ワイン 蒸留酒 米国",
    )
    assert sig is not None
    assert "CJK" in sig.detail


def test_diacritics_alone_do_not_trigger():
    """A French-style label with English keywords should NOT be refused."""
    text = (
        "Élevé en fûts de chêne. Mis en bouteille à la propriété. "
        "Government Warning text follows. Alcohol 14% by Vol. 750 mL."
    )
    assert detect_foreign_language(text) is None


# ---------------------------------------------------------------------------
# detect_screen_capture
# ---------------------------------------------------------------------------


def _png_bytes(*, width: int, height: int, software: str | None = None) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_no_exif_with_phone_aspect_ratio_is_flagged():
    """16:9 portrait with no EXIF is suspicious — likely a screenshot."""
    sig = detect_screen_capture(_png_bytes(width=1080, height=1920))
    assert sig is not None
    assert sig.kind == "screenshot_suspected"


def test_unusual_aspect_ratio_with_no_exif_is_not_flagged():
    """A roughly square upload from a web form is not a phone screen."""
    sig = detect_screen_capture(_png_bytes(width=1200, height=1500))
    assert sig is None


def test_invalid_image_bytes_returns_none():
    """The guard must never crash the request on a corrupt upload."""
    assert detect_screen_capture(b"not-an-image") is None


# ---------------------------------------------------------------------------
# merge_signals
# ---------------------------------------------------------------------------


def test_merge_signals_combines_existing_notes():
    sig = AdversarialSignal(
        kind="foreign_language", detail="non-English text", suggestion="resubmit English"
    )
    merged = merge_signals((sig,), existing_notes="front: blurry")
    assert merged is not None
    assert "blurry" in merged
    assert "non-English text" in merged
    assert "resubmit English" in merged


def test_merge_signals_returns_none_when_nothing_to_merge():
    assert merge_signals((None, None), existing_notes=None) is None


# ---------------------------------------------------------------------------
# verify() integration: foreign-language guard short-circuits
# ---------------------------------------------------------------------------


def test_verify_short_circuits_on_foreign_language_label():
    """A label whose extracted text is overwhelmingly Spanish must come back
    `unreadable` with a clear refusal message — never a confusing edit-
    distance fail on the warning rule."""
    spanish_warning = (
        "ADVERTENCIA: De acuerdo con el Cirujano General, las mujeres "
        "embarazadas no deben beber bebidas alcohólicas durante el embarazo "
        "debido al riesgo de defectos de nacimiento. El consumo de bebidas "
        "alcohólicas afecta la capacidad para conducir."
    )
    extractor = MockVisionExtractor(
        {
            "brand_name": "TEQUILA EL PUEBLO",
            "class_type": "Tequila Reposado",
            "alcohol_content": "40% Alc/Vol",
            "net_contents": "750 mL",
            "name_address": "Embotellado por Destilería El Pueblo, Jalisco, México",
            "health_warning": spanish_warning,
        }
    )
    inp = VerifyInput(
        image_bytes=b"x",  # bypassed via skip_capture_quality
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=True,
        application={},
    )
    report = verify(inp, extractor=extractor, skip_capture_quality=True)
    assert report.overall == "unreadable"
    assert report.image_quality == "unreadable"
    assert report.image_quality_notes
    assert "english" in report.image_quality_notes.lower()


def test_verify_does_not_short_circuit_on_english_label():
    """An English label with normal content should not trip the foreign-
    language guard, even if some imported producer text contains diacritics."""
    extractor = MockVisionExtractor(
        {
            "brand_name": "Hofbräu Original",
            "class_type": "Münchner Helles",
            "alcohol_content": "5.1% Alc./Vol.",
            "net_contents": "12 FL OZ",
            "name_address": "Imported and bottled by Hofbräu USA, Bend, Oregon",
            "health_warning": CANONICAL,
        }
    )
    inp = VerifyInput(
        image_bytes=b"x",
        media_type="image/png",
        beverage_type="beer",
        container_size_ml=355,
        is_imported=True,
        application={},
    )
    report = verify(inp, extractor=extractor, skip_capture_quality=True)
    # Pass scenario; importantly NOT "unreadable" from a false-positive guard.
    assert report.overall != "unreadable"
