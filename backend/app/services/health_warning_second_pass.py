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
    "no_warning_present",     # neither read found a warning, frame is clean
    "unverifiable_obstructed",  # neither could read it, but obstruction signals
                                # (glare, blur, region_visible=true) say the
                                # warning is plausibly hidden — refuse to claim
                                # missing
]


# Levenshtein distance below which two reads are considered "the same text".
# A few bytes of OCR jitter shouldn't be treated as disagreement; large
# character-level deltas should.
_SAME_TEXT_THRESHOLD = 5

# A primary-only read this confident or below, combined with an obstruction
# signal, is treated as unverifiable rather than primary_only. The primary
# alone, partial and through glare, is not enough to FAIL on edit-distance —
# we cannot tell whether the missing characters were the model's or the
# label's. Tightened from a naive "any low confidence" rule: the cross-check
# only escalates when there's an INDEPENDENT signal (obstruction) that the
# label itself was hard to read.
_PRIMARY_LOW_CONFIDENCE_THRESHOLD = 0.7


SYSTEM_PROMPT = """You are a careful proofreader for U.S. alcohol-beverage \
labels. Your one job: find the Government Warning paragraph on this label \
and return it VERBATIM, character for character, as it is printed.

Return ONLY a JSON object with this shape (no Markdown fences, no commentary):

{
  "value":          "<the warning text exactly as printed, or empty string>",
  "found":          true | false,
  "confidence":     0.0–1.0,
  "region_visible": true | false
}

Rules:
- Preserve case exactly. If the label prints "GOVERNMENT WARNING:" in
  capitals, return it that way. If it prints "Government Warning:" in
  title case, return it that way. The downstream check distinguishes
  capitalization.
- Preserve punctuation, parenthesised numerals (1)/(2), and word order
  exactly as printed. Do not paraphrase. Do not "correct" typos.
- Recall is more important than completeness. Even ONE recognizable
  fragment is enough — the "GOVERNMENT WARNING:" trigger header alone,
  the phrase "Surgeon General", "(1) ... pregnancy ...", "(2) ...
  operate machinery ...", or any unambiguous chunk. Return whatever
  characters you can read verbatim and drop confidence to reflect the
  partial read. Do NOT refuse a partial transcription just because
  glare or smudging hides the rest — the cross-check will reconcile a
  partial read honestly.
- Never make up text you cannot see. Partial-but-honest beats full-but-
  guessed.
- `region_visible`: set true when you can see a block of fine-print text
  in a typical warning location (back, side, or lower edge of the
  label), EVEN IF glare, occlusion, smudging, or specular highlights
  make the characters themselves unreadable. This is the signal the
  downstream cross-check needs to distinguish "warning is missing from
  this label" from "warning is present but I cannot transcribe it
  through the glare". Set region_visible=false ONLY when you can see
  the surface clearly end-to-end and there is genuinely no fine-print
  warning paragraph anywhere on it.
- Decision matrix:
    Full clean read         → found=true,  region_visible=true,  confidence ≥ 0.85
    Partial read (fragment) → found=true,  region_visible=true,  confidence 0.4–0.8
    See block, cannot read  → found=false, region_visible=true,  confidence ≤ 0.3
    No warning anywhere     → found=false, region_visible=false, confidence ≥ 0.7

Confidence calibration: 1.0 = sharp and unambiguous; 0.85 = readable
but small; 0.6 = partial / some characters uncertain; ≤ 0.3 = saw the
region but the characters are not transcribable through the
obstruction.

This is a redundancy check — your read will be cross-checked against an
independent reading. Honesty about uncertainty is what makes the
redundancy useful, so under-claim rather than over-claim."""


@dataclass
class WarningRead:
    """One read of the Health Warning Statement.

    `region_visible` is the recall-preserving signal: True when the reader
    saw a block of fine-print text in a typical warning location even if
    the characters themselves were not transcribable through glare,
    occlusion, or smudging. Combined with `found=False`, this distinguishes
    "warning is genuinely missing from the label" from "warning is present
    but I cannot read it through the obstruction" — the cross-check uses
    that distinction to refuse a confident wrong-fail.
    """

    value: str | None
    found: bool
    confidence: float = 0.0
    source: str = "unknown"
    raw_response: str | None = None
    region_visible: bool = False


@dataclass
class ObstructionSignal:
    """Independent evidence that the warning region may be hidden.

    Sourced from the sensor pre-check (glare blob fraction over the label,
    label-region verdict, motion blur direction) and used by the cross-check
    to decide whether a "neither read found anything" conclusion is honest
    or merely a glare-cascade.

    The verify orchestrator constructs this from the per-panel
    `SurfaceCaptureQuality` metrics; tests can build it directly.
    """

    is_obstructed: bool
    reason: str = ""

    @classmethod
    def clear(cls) -> ObstructionSignal:
        return cls(is_obstructed=False, reason="")


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
                # Prefer the dedicated second-pass model — a small/fast
                # vision model is the right pick for one-paragraph re-reads
                # where the only job is to produce an independent
                # transcription. Fall back to the primary model only when
                # the dedicated setting is unset (older configs).
                model = (
                    getattr(settings, "anthropic_health_warning_model", None)
                    or settings.anthropic_model
                )
        self._client = client
        self._model = model or "claude-haiku-4-5-20251001"
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
    """Deterministic second-pass extractor for tests.

    `value=None` simulates a failed read. `region_visible` lets a test
    simulate the "I saw the warning region but couldn't transcribe it
    through glare" case — the recall-preserving signal the cross-check
    uses to refuse a confident "missing warning" verdict.
    """

    def __init__(
        self,
        *,
        value: str | None,
        confidence: float = 0.95,
        region_visible: bool | None = None,
    ) -> None:
        self._value = value
        self._confidence = confidence
        # If unset, default region_visible=True for found reads (we read
        # it, so it must be visible) and False for not-found reads (clean
        # frame with no warning anywhere). Tests can override either.
        if region_visible is None:
            region_visible = value is not None
        self._region_visible = region_visible

    def read_warning(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> WarningRead:
        if self._value is None:
            return WarningRead(
                value=None,
                found=False,
                confidence=self._confidence,
                source="mock",
                region_visible=self._region_visible,
            )
        return WarningRead(
            value=self._value,
            found=True,
            confidence=self._confidence,
            source="mock",
            region_visible=self._region_visible,
        )


def _parse_response(text: str) -> WarningRead:
    import json

    cleaned = re.sub(r"^\s*```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Haiku occasionally appends prose after the JSON object — our system
        # prompt forbids it, but the model honours the rule unevenly on
        # degraded images where it wants to qualify a low-confidence read.
        # Trying to outlaw the prose at prompt level would just push the
        # failure into a different shape, so we recover here instead: pull
        # the first balanced JSON object out of the text and parse THAT.
        # The previous behaviour — return found=False — silently discarded
        # otherwise-perfectly-valid 0.6+ reads, which is exactly the kind
        # of "couldn't read the label" failure the redundant second-pass
        # was designed to prevent.
        extracted = _extract_first_json_object(cleaned)
        if extracted is None:
            return WarningRead(
                value=None,
                found=False,
                confidence=0.0,
                source="claude_second_pass",
                raw_response=text,
            )
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError:
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
    region_visible = bool(data.get("region_visible", False))
    if not found or value is None or (isinstance(value, str) and not value.strip()):
        return WarningRead(
            value=None,
            found=False,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            source="claude_second_pass",
            raw_response=text,
            region_visible=region_visible,
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
        # The decision matrix says found=true implies region_visible=true.
        # If the model omitted region_visible while reporting a successful
        # read, infer it as True — recall is preserved either way.
        region_visible=region_visible or True,
    )


def _extract_first_json_object(text: str) -> str | None:
    """Return the substring of `text` covering the first balanced top-level
    JSON object, or None if no balanced object can be found.

    Walks the bytes once tracking brace depth, and is string-literal aware
    so braces inside quoted values don't confuse the depth counter. The
    Government Warning paragraph contains ":", "(1)", "(2)" but no real
    braces, so this is overkill on canonical content — but the same parser
    is used for paraphrased/garbled reads where the model could conceivably
    quote brace-bearing text, and the extra robustness costs almost
    nothing.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def cross_check(
    primary: WarningRead | None,
    secondary: WarningRead | None,
    *,
    canonical_ref: str = "health_warning",
    obstruction_signal: ObstructionSignal | None = None,
) -> CrossCheckResult:
    """Reconcile two independent reads against the statutory canonical text.

    Recall guarantee (SPEC §0.5 mandate): the cross-check refuses to
    return `no_warning_present` whenever there is ANY independent signal
    that the warning could be obscured rather than missing. The signals
    are:

      1. Either reader's `region_visible=True` (model saw a fine-print
         block in a typical warning location even if characters were
         not transcribable).
      2. `obstruction_signal.is_obstructed=True` (sensor pre-check found
         glare blobs, motion blur, or a degraded label region).

    Under either signal, what would have been `no_warning_present` is
    upgraded to `unverifiable_obstructed`, and what would have been
    `primary_only` with a low-confidence partial primary read is
    similarly upgraded — the upstream rule downgrades both to ADVISORY
    rather than letting the engine FAIL on "warning missing" or on a
    high edit-distance that's really a glare hole.

    Decision tree (case-insensitive comparisons; whitespace normalized):

      * Neither read found text:
          - region_visible OR obstruction_signal → unverifiable_obstructed
          - clean frame                          → no_warning_present
      * Both reads match canonical (or single read matches and other
        was unavailable) → confirmed_compliant.
      * Both reads agree (Levenshtein ≤ 5) but differ from canonical →
        confirmed_noncompliant.
      * The two reads disagree → disagreement (caller downgrades to
        advisory).
      * Only the primary is available:
          - primary confidence < 0.7 AND obstruction signal → unverifiable_obstructed
          - else → primary_only (caller leaves the primary verdict alone).
    """
    canonical = _normalize(load_canonical(canonical_ref))
    p_text = _normalize(primary.value if primary and primary.found else None)
    s_text = _normalize(secondary.value if secondary and secondary.found else None)
    has_primary = bool(p_text)
    has_secondary = bool(s_text)

    obstructed = bool(obstruction_signal and obstruction_signal.is_obstructed)
    region_visible_signal = bool(
        (primary and primary.region_visible)
        or (secondary and secondary.region_visible)
    )

    if not has_primary and not has_secondary:
        if region_visible_signal or obstructed:
            return CrossCheckResult(
                outcome="unverifiable_obstructed",
                primary=primary,
                secondary=secondary,
                edit_distance_to_canonical=None,
                edit_distance_between_reads=None,
                notes=_obstruction_note(
                    primary, secondary, obstruction_signal,
                    base="Neither reader could transcribe the warning, "
                    "but obstruction signals indicate it is plausibly hidden",
                ),
            )
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

    # Case-sensitive byte equality drives the `confirmed_compliant`
    # promotion. The engine's `health_warning.exact_text` rule FAILs a
    # title-case warning, so the cross-check cannot label the same input
    # `confirmed_compliant` — it's a contract drift that would let a
    # future caller use the cross-check verdict as the source of truth
    # and silently mis-pass a label the engine refuses. Case-folded
    # distance still drives `confirmed_noncompliant` and `disagreement`
    # (a few bytes of OCR jitter shouldn't be treated as case mismatch).
    primary_matches = (
        has_primary and p_text == canonical
    )
    secondary_matches = (
        has_secondary and s_text == canonical
    )
    reads_agree = (
        between_dist is not None and between_dist <= _SAME_TEXT_THRESHOLD
    )

    if not has_secondary:
        # Primary alone, low confidence, AND obstruction signal: refuse
        # to FAIL on edit-distance — we cannot tell whether the missing
        # characters were the model's or the label's. Without obstruction
        # signal, an under-confident primary is still the model's
        # judgment call and we honor it.
        primary_confidence = float(primary.confidence) if primary else 0.0
        if (
            obstructed
            and has_primary
            and primary_confidence < _PRIMARY_LOW_CONFIDENCE_THRESHOLD
        ):
            return CrossCheckResult(
                outcome="unverifiable_obstructed",
                primary=primary,
                secondary=secondary,
                edit_distance_to_canonical=primary_canonical_dist,
                edit_distance_between_reads=None,
                notes=_obstruction_note(
                    primary, secondary, obstruction_signal,
                    base=(
                        f"Primary returned a partial read at confidence "
                        f"{primary_confidence:.2f} and the second pass was "
                        "unavailable; obstruction signals indicate the "
                        "missing characters may be hidden rather than "
                        "absent"
                    ),
                ),
            )
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
        # Symmetrically: secondary alone with low confidence under an
        # obstruction signal escalates to unverifiable. The single
        # successful read is the second pass, so we read its confidence.
        secondary_confidence = float(secondary.confidence) if secondary else 0.0
        if (
            obstructed
            and has_secondary
            and secondary_confidence < _PRIMARY_LOW_CONFIDENCE_THRESHOLD
        ):
            return CrossCheckResult(
                outcome="unverifiable_obstructed",
                primary=primary,
                secondary=secondary,
                edit_distance_to_canonical=secondary_canonical_dist,
                edit_distance_between_reads=None,
                notes=_obstruction_note(
                    primary, secondary, obstruction_signal,
                    base=(
                        f"Only the second pass returned text, at confidence "
                        f"{secondary_confidence:.2f}; obstruction signals "
                        "indicate the warning is plausibly partially hidden"
                    ),
                ),
            )
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


def _obstruction_note(
    primary: WarningRead | None,
    secondary: WarningRead | None,
    obstruction_signal: ObstructionSignal | None,
    *,
    base: str,
) -> str:
    """Compose a human-readable explanation citing every obstruction
    signal the cross-check considered. Helps the agent UI tell the user
    *why* the verdict is advisory rather than the dread "couldn't verify"
    stub the original branch produced."""
    reasons: list[str] = []
    if obstruction_signal and obstruction_signal.is_obstructed:
        reasons.append(
            obstruction_signal.reason
            or "sensor pre-check observed obstruction over the label"
        )
    if primary and primary.region_visible and not primary.found:
        reasons.append("primary reader saw the warning region but could not read it")
    if secondary and secondary.region_visible and not secondary.found:
        reasons.append("second-pass reader saw the warning region but could not read it")
    if not reasons:
        return f"{base}. Reshoot recommended."
    return f"{base}: {'; '.join(reasons)}. Reshoot recommended."
