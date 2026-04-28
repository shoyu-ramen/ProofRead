"""Adversarial-input guards.

SPEC §0.5 calls out three adversarial input categories that the system
must handle honestly rather than guess at:

  * Counterfeit detection — explicitly OUT of scope. We never opine on
    authenticity. (No code needed beyond the model's prompt; documented
    here so reviewers can see we considered it.)
  * Foreign-language-only labels — refuse with a clear message rather
    than guessing at a Spanish label as if it were English.
  * Obscured / occluded labels (price stickers, security tags, photo of
    photo, screenshot) — surface the suspicion in image_quality_notes
    so the user can rescan.

The detection heuristics here are deliberately conservative. A false
"foreign language" call wastes the user's time; a false "ok English"
call lets the rule engine produce a guess. So both signals require
strong evidence before they fire.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Strong English-warning indicators. Any one of these in the joined text
# is enough to clear the foreign-language guard, because they show up in
# the canonical Government Warning sentence that *every* TTB-compliant
# label has to print. A label without any of them is suspicious.
_ENGLISH_WARNING_TOKENS = (
    "warning",
    "surgeon",
    "pregnancy",
    "birth",
    "defects",
    "machinery",
    "consumption",
    "operate",
)

# Generic English compliance vocabulary. Unit-of-measure tokens (mL, fl, oz,
# alc, vol) are intentionally NOT here — those appear on Spanish/French/
# German labels too and would create false negatives. We focus on tokens
# that are uniquely English in compliance context.
_ENGLISH_GENERIC_TOKENS = (
    "alcohol",
    "proof",
    "abv",
    "ounces",
    "brewed",
    "bottled",
    "produced",
    "distilled",
    "imported",
    "vintage",
    "appellation",
    "company",
    "inc",
    "corp",
)


_FOREIGN_SCRIPT_RANGES = (
    # Latin Extended-B and beyond is fine — Spanish/French/German use it.
    # The signals we care about are non-Latin scripts that imply non-English
    # primary text on the label.
    ("CYRILLIC", lambda c: "CYRILLIC" in unicodedata.name(c, "")),
    ("CJK", lambda c: any(
        kw in unicodedata.name(c, "")
        for kw in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")
    )),
    ("ARABIC", lambda c: "ARABIC" in unicodedata.name(c, "")),
    ("DEVANAGARI", lambda c: "DEVANAGARI" in unicodedata.name(c, "")),
    ("HEBREW", lambda c: "HEBREW" in unicodedata.name(c, "")),
    ("THAI", lambda c: "THAI" in unicodedata.name(c, "")),
    ("GREEK", lambda c: "GREEK" in unicodedata.name(c, "")),
)


@dataclass
class AdversarialSignal:
    """One detected guard outcome."""

    kind: str  # e.g. "foreign_language", "screenshot_suspected"
    detail: str
    suggestion: str


def detect_foreign_language(
    *texts: str | None,
    minimum_chars: int = 40,
) -> AdversarialSignal | None:
    """Flag a label whose dominant text isn't English.

    Two criteria; both must hold:

      1. The aggregated extracted text contains characters from a non-Latin
         script (Cyrillic, CJK, Arabic, Devanagari, Hebrew, Thai, Greek)
         OR has zero matches against the English keyword list.
      2. The text is long enough to make the call (avoids firing on a
         single-word brand name like "Hofbräu").

    Latin diacritics alone never trigger this — Spanish/French/German
    labels with English supplemental panels are out of scope for v1
    refusal.
    """
    joined = " ".join(t for t in texts if t).strip()

    # Non-Latin script is strong evidence even on short text — if we see
    # several CJK / Cyrillic / Arabic characters, the language guard fires
    # regardless of total length. The minimum_chars gate only protects the
    # "no English keywords" path from firing on a one-word brand.
    foreign_script = _detect_non_latin_script(joined)
    if foreign_script is not None:
        return AdversarialSignal(
            kind="foreign_language",
            detail=(
                f"Label appears to contain {foreign_script} text — outside the "
                "scope of v1 (English-only). The Government Warning rule "
                "requires a verbatim English statement; submit a label with "
                "English text or an English supplemental panel."
            ),
            suggestion=(
                "Submit a label with English text, or include an English "
                "supplemental panel for non-English imports."
            ),
        )

    if len(joined) < minimum_chars:
        return None

    lower = joined.lower()
    warning_hits = sum(
        1
        for tok in _ENGLISH_WARNING_TOKENS
        if re.search(rf"\b{re.escape(tok)}\b", lower)
    )
    generic_hits = sum(
        1
        for tok in _ENGLISH_GENERIC_TOKENS
        if re.search(rf"\b{re.escape(tok)}\b", lower)
    )

    # Strong English-warning evidence is sufficient.
    if warning_hits >= 1:
        return None
    # Many generic compliance tokens (≥3 distinct) are also sufficient —
    # accommodates labels with no warning visible but otherwise English.
    if generic_hits >= 3:
        return None

    return AdversarialSignal(
        kind="foreign_language",
        detail=(
            "Extracted text shows no English Government Warning vocabulary "
            "('warning', 'surgeon', 'pregnancy', 'machinery', 'consumption') "
            f"and only {generic_hits} generic English compliance keyword(s). "
            "Likely a foreign-language-only label."
        ),
        suggestion=(
            "Submit a label with English text, or include an English "
            "supplemental panel for non-English imports."
        ),
    )


def screenshot_signal_from_source(capture_source: str | None) -> AdversarialSignal | None:
    """Lift the sensor pre-check's `capture_source` classification into an
    `AdversarialSignal` for the verify path's `image_quality_notes`.

    Sensor pre-check (`sensor_check._classify_capture_source`) is the single
    source of truth for whether a frame looks like a phone screenshot — it
    already inspects EXIF Software, Make/Model, and the aspect ratio while
    decoding the image once for capture-quality analysis. Re-decoding here
    purely for an EXIF check would burn ~10–30ms per request for no signal
    the pre-check didn't already produce.

    The signal stays "soft" — a screenshot still goes through the rule
    engine; the user is told the upload looks like a screen capture so they
    can rescan if appropriate. Plenty of legitimate uploads (web uploads,
    certain Android cameras, browser-stripped EXIF) land here too.
    """
    if capture_source != "screenshot":
        return None
    return AdversarialSignal(
        kind="screenshot_suspected",
        detail=(
            "Image looks like a screenshot or screen render rather than a "
            "photo of the printed label (no camera EXIF metadata or a "
            "phone-screen aspect ratio). Compliance verification expects a "
            "photo of the printed label or original artwork."
        ),
        suggestion="Confirm the upload is a photo of the printed label.",
    )


def _detect_non_latin_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    for c in text:
        if not c.isalpha():
            continue
        for script_name, predicate in _FOREIGN_SCRIPT_RANGES:
            if predicate(c):
                counts[script_name] = counts.get(script_name, 0) + 1
                break
    if not counts:
        return None
    # Require at least 5 characters of the script to fire — otherwise a
    # single accented character could trip the heuristic.
    name, n = max(counts.items(), key=lambda kv: kv[1])
    return name if n >= 5 else None


def merge_signals(
    signals: Iterable[AdversarialSignal | None],
    existing_notes: str | None = None,
) -> str | None:
    """Combine adversarial signals + an existing image_quality_notes string."""
    parts: list[str] = []
    if existing_notes:
        parts.append(existing_notes)
    for s in signals:
        if s is None:
            continue
        parts.append(f"[{s.kind}] {s.detail} — {s.suggestion}")
    if not parts:
        return None
    return " | ".join(parts)
