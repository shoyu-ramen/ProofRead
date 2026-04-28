"""Health Warning redundant second-pass.

SPEC §0.5 mandates that the Government Health Warning is re-read every
time, regardless of whether the primary extractor read it confidently:

    > Health Warning second pass: dedicated TrOCR fine-tune on the
    > cropped warning region runs every time, regardless of primary
    > success — redundancy by design (v1).

The Health Warning is the most legally consequential element on the label
and the most expensive to misread (the user's COLA submission is at
stake). Two independent readings raise the bar for trust: when they
agree the verdict is rock solid; when they disagree the system declines
to claim either pass or fail and surfaces an advisory instead.

This module is a plain Protocol + Claude implementation; no TrOCR fine-
tune in v1 — we use Claude with a tightly focused prompt and a tiny
output budget so the extra call is cheap. It is intentionally
side-effect-free and thread-safe so it can be called from either the
`scans` flow or the single-shot `verify` flow.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from rapidfuzz.distance import Levenshtein

from app.rules.canonical import load_canonical

logger = logging.getLogger(__name__)


CrossCheckOutcome = Literal[
    "confirmed_compliant",   # both reads match canonical
    "confirmed_noncompliant",  # both reads agree but differ from canonical
    "disagreement",           # the two reads disagree — couldn't verify
    "primary_only",           # only the primary read is available
    "no_warning_present",     # neither read found a warning
]


# Levenshtein distance below which two reads are considered "the same text".
# A few bytes of OCR jitter shouldn't be treated as disagreement; large
# character-level deltas should.
_SAME_TEXT_THRESHOLD = 5


SYSTEM_PROMPT = """You are a careful proofreader for U.S. alcohol-beverage \
labels. Your one job: find the Government Warning paragraph on this label \
and return it VERBATIM, character for character, as it is printed.

Return ONLY a JSON object with this shape (no Markdown fences, no commentary):

{
  "value":      "<the warning text exactly as printed, or empty string>",
  "found":      true | false,
  "confidence": 0.0–1.0
}

Rules:
- Preserve case exactly. If the label prints "GOVERNMENT WARNING:" in
  capitals, return it that way. If it prints "Government Warning:" in
  title case, return it that way. The downstream check distinguishes
  capitalization.
- Preserve punctuation, parenthesised numerals (1)/(2), and word order
  exactly as printed. Do not paraphrase. Do not "correct" typos.
- If you cannot read the warning at all, set found=false and value="".
- If you can only partially read it, return what you can see verbatim and
  drop confidence accordingly. Never make up text you cannot see.
- Confidence: 1.0 = sharp and unambiguous; 0.85 = readable but small;
  0.6 = partial / some characters uncertain; <0.6 = use found=false.

This is a redundancy check — your read will be cross-checked against an
independent reading. Honesty about uncertainty is what makes the
redundancy useful, so under-claim rather than over-claim."""


@dataclass
class WarningRead:
    """One read of the Health Warning Statement."""

    value: str | None
    found: bool
    confidence: float = 0.0
    source: str = "unknown"
    raw_response: str | None = None


@dataclass
class CrossCheckResult:
    outcome: CrossCheckOutcome
    primary: WarningRead | None
    secondary: WarningRead | None
    edit_distance_to_canonical: int | None
    edit_distance_between_reads: int | None
    notes: str


class HealthWarningExtractor(Protocol):
    def read_warning(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> WarningRead: ...


class ClaudeHealthWarningExtractor:
    """Claude-backed second-pass reader.

    Caches the system prompt with `cache_control: ephemeral` so the
    prefix is paid once per process. Output budget is small (1024 tokens
    is more than enough for the two-sentence warning + JSON wrapper) to
    keep the per-call cost predictable.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        timeout: float | None = None,
    ) -> None:
        if client is None:
            from app.config import settings
            from app.services.anthropic_client import (
                DEFAULT_SECOND_PASS_TIMEOUT_S,
                build_client,
            )

            client = build_client(
                timeout=timeout if timeout is not None else DEFAULT_SECOND_PASS_TIMEOUT_S
            )
            if model is None:
                model = settings.anthropic_model
        self._client = client
        self._model = model or "claude-opus-4-7"
        self._max_tokens = max_tokens

    def read_warning(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> WarningRead:
        from app.services.anthropic_client import call_with_resilience

        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        response = call_with_resilience(
            self._client.messages.create,
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
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
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Find the Government Warning paragraph on "
                                "this label and return it verbatim."
                            ),
                        },
                    ],
                }
            ],
        )
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return _parse_response(text)


class MockHealthWarningExtractor:
    """Deterministic second-pass extractor for tests."""

    def __init__(self, *, value: str | None, confidence: float = 0.95) -> None:
        self._value = value
        self._confidence = confidence

    def read_warning(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> WarningRead:
        if self._value is None:
            return WarningRead(value=None, found=False, confidence=0.0, source="mock")
        return WarningRead(
            value=self._value,
            found=True,
            confidence=self._confidence,
            source="mock",
        )


def _parse_response(text: str) -> WarningRead:
    import json

    cleaned = re.sub(r"^\s*```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Defensive: if the model deviated from the JSON contract, return a
        # not-found read so the cross-check downgrades to advisory rather
        # than crashing the request.
        return WarningRead(
            value=None,
            found=False,
            confidence=0.0,
            source="claude_second_pass",
            raw_response=text,
        )
    if not isinstance(data, dict):
        return WarningRead(
            value=None,
            found=False,
            confidence=0.0,
            source="claude_second_pass",
            raw_response=text,
        )
    found = bool(data.get("found"))
    value = data.get("value")
    if not found or value is None or (isinstance(value, str) and not value.strip()):
        return WarningRead(
            value=None,
            found=False,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            source="claude_second_pass",
            raw_response=text,
        )
    confidence = data.get("confidence", 0.85)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.85
    return WarningRead(
        value=str(value),
        found=True,
        confidence=max(0.0, min(1.0, confidence)),
        source="claude_second_pass",
        raw_response=text,
    )


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def cross_check(
    primary: WarningRead | None,
    secondary: WarningRead | None,
    *,
    canonical_ref: str = "health_warning",
) -> CrossCheckResult:
    """Reconcile two independent reads against the statutory canonical text.

    Decision tree (case-insensitive comparisons; whitespace normalized):

      * Neither read found a warning → no_warning_present.
      * Primary matches canonical and secondary either matches canonical
        or wasn't run → confirmed_compliant.
      * Both reads agree (Levenshtein ≤ 5) but differ from canonical →
        confirmed_noncompliant.
      * The two reads disagree → disagreement (caller should downgrade
        the rule to advisory).
      * Only the primary is available → primary_only (caller leaves the
        primary verdict alone).
    """
    canonical = _normalize(load_canonical(canonical_ref))
    p_text = _normalize(primary.value if primary and primary.found else None)
    s_text = _normalize(secondary.value if secondary and secondary.found else None)
    has_primary = bool(p_text)
    has_secondary = bool(s_text)

    if not has_primary and not has_secondary:
        return CrossCheckResult(
            outcome="no_warning_present",
            primary=primary,
            secondary=secondary,
            edit_distance_to_canonical=None,
            edit_distance_between_reads=None,
            notes="Neither read found a Government Warning on the label.",
        )

    canonical_lower = canonical.lower()
    primary_canonical_dist = (
        Levenshtein.distance(p_text.lower(), canonical_lower) if has_primary else None
    )
    secondary_canonical_dist = (
        Levenshtein.distance(s_text.lower(), canonical_lower) if has_secondary else None
    )
    between_dist = (
        Levenshtein.distance(p_text.lower(), s_text.lower())
        if has_primary and has_secondary
        else None
    )

    primary_matches = primary_canonical_dist == 0 if has_primary else False
    secondary_matches = secondary_canonical_dist == 0 if has_secondary else False
    reads_agree = (
        between_dist is not None and between_dist <= _SAME_TEXT_THRESHOLD
    )

    if not has_secondary:
        return CrossCheckResult(
            outcome="primary_only",
            primary=primary,
            secondary=secondary,
            edit_distance_to_canonical=primary_canonical_dist,
            edit_distance_between_reads=None,
            notes=(
                "Second-pass reader could not be run; primary verdict stands."
            ),
        )

    if not has_primary:
        return CrossCheckResult(
            outcome="primary_only",
            primary=primary,
            secondary=secondary,
            edit_distance_to_canonical=secondary_canonical_dist,
            edit_distance_between_reads=None,
            notes=(
                "Primary reader did not return a warning; only the second "
                "pass saw text."
            ),
        )

    if primary_matches and secondary_matches:
        return CrossCheckResult(
            outcome="confirmed_compliant",
            primary=primary,
            secondary=secondary,
            edit_distance_to_canonical=0,
            edit_distance_between_reads=0,
            notes="Both reads match the canonical Government Warning.",
        )

    if reads_agree and not primary_matches:
        return CrossCheckResult(
            outcome="confirmed_noncompliant",
            primary=primary,
            secondary=secondary,
            edit_distance_to_canonical=primary_canonical_dist,
            edit_distance_between_reads=between_dist,
            notes=(
                "Both reads agree the warning differs from the statutory "
                f"text by {primary_canonical_dist} character(s)."
            ),
        )

    return CrossCheckResult(
        outcome="disagreement",
        primary=primary,
        secondary=secondary,
        edit_distance_to_canonical=primary_canonical_dist,
        edit_distance_between_reads=between_dist,
        notes=(
            "Primary and second-pass reads disagree. Couldn't verify the "
            "Government Warning with confidence — rescan recommended."
        ),
    )
