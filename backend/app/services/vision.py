"""Claude vision extractor — single VLM call to read TTB-relevant fields.

Replaces the OCR + regex-extractor combination for the /v1/verify path. The
system prompt defines the JSON output shape and is marked for prompt caching
since it's static across requests.

The extractor prefers `unreadable: true` over guessing on poor images — that
loops back through the rule engine's confidence-aware degradation so an
agent never sees a verdict that was guessed from a smudge.

SPEC §0.5 wiring:

  * The verify path runs `assess_capture_quality()` before this extractor
    is called. The resulting CaptureQualityReport is passed in via
    `capture_quality=` and inlined into the user prompt as an objective
    prior — same shape the scan-path extractor uses (label bbox, glare
    blobs, motion direction, sensor tier). The model decides per-region
    whether it can trust each part of the label.

  * The producer record (the user's COLA-style submission) is passed in
    via `producer_record=` and shown to the model as context, NOT as
    ground truth. The contract is unchanged: the extractor READS, the
    rule engine JUDGES. Knowing the claim only helps the model
    disambiguate ambiguous readings (e.g. choose between two volume
    statements on the same can).

  * The model returns its own image_quality verdict alongside per-field
    confidence/note/bbox; the verify orchestrator merges this with the
    sensor verdict pessimistically.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from app.config import settings
from app.rules.types import ExtractedField
from app.services.anthropic_client import (
    DEFAULT_VISION_TIMEOUT_S,
    ExtractorUnavailable,
    build_client,
    call_with_resilience,
)
from app.services.sensor_briefing import format_capture_quality

logger = logging.getLogger(__name__)

# Fields the extractor is asked to return. Mirrors the keys the rule engine
# (and producer_record cross-references) expect.
EXTRACTOR_FIELDS = (
    "brand_name",
    "class_type",
    "alcohol_content",
    "net_contents",
    "name_address",
    "country_of_origin",
    "health_warning",
)


# Anti-leakage note: the previous version of this prompt embedded specific
# fictional brand/class strings (e.g. "OLD TOM DISTILLERY",
# "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol. (90 Proof)") as
# per-field examples. Those strings are *exactly* what the four bundled
# sample labels print, so the model could "pass" the bundled samples by
# regurgitating its own prompt. Examples below are deliberately abstract —
# format rules and structural shapes only — so the model can only succeed
# by actually reading the label.
SYSTEM_PROMPT = """You are an OCR assistant for U.S. alcohol beverage labels. \
Read a label image and return a JSON object describing what is on the label.

Return ONLY a single JSON object — no Markdown fences (```), no commentary, \
no leading or trailing prose. The downstream parser is strict about that.

Return text VERBATIM as it appears on the label. Preserve the original case \
(uppercase, mixed-case, title-case), punctuation, and whitespace. The downstream \
compliance engine cares about exactness, so do not normalise, expand abbreviations, \
or "tidy up" the wording.

If a field is genuinely not present, or you cannot read it confidently, set \
`unreadable: true` and leave `value` as null. Prefer `unreadable: true` over \
guessing — the agent reviewing this label needs to trust your output.

CRITICAL: Read what is on THIS label. Field descriptions in this prompt are \
*format rules* only. Do NOT copy phrases from this prompt into your output; do \
NOT pattern-match on the user's claimed metadata if any is shown in the user \
message. The user message provides claim/sensor context as a *prior*, not \
ground truth.

Output shape (per field):

  value:        verbatim text from the label, or null if unreadable/absent
  confidence:   0.0–1.0
  unreadable:   include this key ONLY when the field is unreadable (set true).
                Omit it when value was successfully read; the downstream
                parser treats absent `unreadable` as false.
  note:         OMIT this key unless confidence < 0.85, in which case give a
                one-line reason (e.g. "partial occlusion lower-right").

Do NOT emit a `bbox` field — downstream consumers do not use it and emitting \
it costs latency. The schema permits absent bbox; do not fabricate one.

Top-level fields (always present):

  image_quality:           "good" | "degraded" | "unreadable" — your honest
                           assessment of whether the label is fully readable.
  image_quality_notes:     one or two sentences naming the specific reason
                           (e.g. "specular highlight covers right third of
                           label", "hand occludes lower text"). If the
                           sensor pre-check in the user message flagged
                           something, agree or disagree explicitly.
  beverage_type_observed:  "beer" | "wine" | "spirits" | "unknown" — what
                           the label itself indicates (e.g. "Brewing Co.",
                           "IBU", "Ale" → beer; "Vintage 2020", "AVA" →
                           wine; "Distilled", "Bourbon", "Gin" → spirits).
                           If the user's claimed beverage_type disagrees,
                           record the conflict in image_quality_notes.

Field definitions (FORMAT GUIDANCE — do not copy any wording from this list):

  brand_name        — the most prominent brand text on the front face,
                      transcribed verbatim. Usually large and stylised.

  class_type        — the TTB class/type designation (typically a noun
                      phrase like "<style> Lager" / "<varietal>" /
                      "<spirit subtype>"). Verbatim, original case.

  alcohol_content   — alcohol-content statement verbatim. Format usually
                      "<number>% ABV" or "<number>% Alc./Vol." sometimes
                      followed by "(<number> Proof)" in parentheses.

  net_contents      — net-contents statement verbatim, including units
                      (mL, L, FL OZ, fl oz). May include both metric and
                      U.S. customary in parentheses.

  name_address      — bottler / producer / brewer / importer statement
                      verbatim. Typically begins with a present-tense
                      verb phrase: "Bottled by", "Brewed by", "Distilled
                      by", "Imported by", "Produced by".

  country_of_origin — ONLY if the label EXPLICITLY declares a country of
                      origin (e.g. "Product of <country>", "Imported
                      from <country>"). For domestic U.S. labels with no
                      such statement, set unreadable: true.

  health_warning    — the full Government Warning paragraph verbatim,
                      INCLUDING the prefix exactly as it appears. If the
                      label uses "GOVERNMENT WARNING:" in capitals,
                      return it that way. If it uses a different case
                      (e.g. "Government Warning:"), return it that way.
                      The compliance check is character-exact.

Confidence scale:

  1.0  — text is large, sharp, and unambiguous
  0.85 — readable but smaller or partially stylised
  0.6  — partial read; some characters uncertain
  <0.6 — very uncertain; consider unreadable: true instead
"""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class _FieldOut(BaseModel):
    value: str | None = Field(
        None, description="Verbatim text as printed on the label, or null."
    )
    unreadable: bool = Field(
        False, description="True if the field cannot be read confidently."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int] | None = Field(
        None, description="[x, y, w, h] in image pixels, or null if unlocated."
    )
    note: str | None = Field(
        None, description="Required if confidence < 0.85."
    )


class VerifyLabelExtraction(BaseModel):
    """Structured output of one verify-path label inspection.

    Mirrors `extractors/claude_vision.LabelExtraction` but trimmed to the
    seven verify-path fields. Every field is always present; absent
    fields come back with value=null and unreadable=true.
    """

    image_quality: Literal["good", "degraded", "unreadable"] = "good"
    image_quality_notes: str = ""
    beverage_type_observed: Literal["beer", "wine", "spirits", "unknown"] = "unknown"
    brand_name: _FieldOut = Field(default_factory=_FieldOut)
    class_type: _FieldOut = Field(default_factory=_FieldOut)
    alcohol_content: _FieldOut = Field(default_factory=_FieldOut)
    net_contents: _FieldOut = Field(default_factory=_FieldOut)
    name_address: _FieldOut = Field(default_factory=_FieldOut)
    country_of_origin: _FieldOut = Field(default_factory=_FieldOut)
    health_warning: _FieldOut = Field(default_factory=_FieldOut)


# ---------------------------------------------------------------------------
# Extractor surface
# ---------------------------------------------------------------------------


@dataclass
class VisionExtraction:
    fields: dict[str, ExtractedField]
    unreadable: list[str]
    raw_response: str
    image_quality: str | None = None
    image_quality_notes: str | None = None
    beverage_type_observed: str | None = None


class VisionExtractor(Protocol):
    def extract(
        self,
        image_bytes: bytes,
        media_type: str = ...,
        *,
        capture_quality: Any | None = ...,
        producer_record: dict[str, Any] | None = ...,
        beverage_type: str | None = ...,
        container_size_ml: int | None = ...,
        is_imported: bool = ...,
    ) -> VisionExtraction: ...


def _format_producer_record(record: dict[str, Any] | None) -> str:
    """Render the producer claim as a context block.

    Shown to the model as a *prior*, not ground truth. The compliance
    engine still does the actual claim-vs-label cross-check.
    """
    if not record:
        return "No producer record was provided.\n"
    keys = (
        "brand_name",
        "class_type",
        "alcohol_content",
        "net_contents",
        "name_address",
        "country_of_origin",
    )
    lines = ["Producer record (claim from the submitter — context, not ground truth):"]
    seen_any = False
    for k in keys:
        v = record.get(k)
        if v in (None, ""):
            continue
        seen_any = True
        lines.append(f"  {k:18s} = {v!r}")
    if not seen_any:
        return "No producer record was provided.\n"
    return "\n".join(lines) + "\n"


def _format_claim_header(
    *,
    beverage_type: str | None,
    container_size_ml: int | None,
    is_imported: bool,
) -> str:
    parts: list[str] = []
    if beverage_type:
        parts.append(f"Beverage type (claimed): {beverage_type}")
    if container_size_ml:
        parts.append(f"Container size (claimed): {container_size_ml} mL")
    parts.append(f"Imported: {is_imported}")
    return "\n".join(parts) + "\n"


def _build_user_text(
    *,
    capture_quality: Any | None,
    producer_record: dict[str, Any] | None,
    beverage_type: str | None,
    container_size_ml: int | None,
    is_imported: bool,
) -> str:
    """Compose the text portion of the user message: claim header, producer
    record, sensor briefing, and the inspection instruction."""
    text_blocks: list[str] = [
        _format_claim_header(
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
        ),
        _format_producer_record(producer_record),
        format_capture_quality(capture_quality),
        (
            "Inspect this label and return the JSON object described above. "
            "Be honest about confidence — a low confidence is the correct "
            "answer when the image is degraded; the rule engine will downgrade "
            "those rules to advisory rather than guess pass/fail."
        ),
    ]
    return "\n".join(b for b in text_blocks if b)


class ClaudeVisionExtractor:
    """Production extractor backed by Claude Sonnet with vision input.

    Uses the centralised `build_client` factory so per-call timeout and
    retry budget come from one place; transient SDK errors are translated
    to `ExtractorUnavailable` so the pipeline can fall back to OCR.

    No extended thinking on this call path. The output schema (per-field
    value + confidence + unreadable flag) already disciplines the model
    into honest under-claiming on degraded labels; thinking buys little
    on a transcription task while costing a meaningful fraction of the
    end-to-end latency budget. The redundant Government-Warning second-
    pass — run concurrently in `verify()` — provides the fail-honestly
    redundancy SPEC §0.5 calls for instead.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        timeout: float = DEFAULT_VISION_TIMEOUT_S,
    ) -> None:
        self._client = client or build_client(timeout=timeout)
        self._model = model or settings.anthropic_model
        self._max_tokens = max_tokens

    def extract(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        *,
        capture_quality: Any | None = None,
        producer_record: dict[str, Any] | None = None,
        beverage_type: str | None = None,
        container_size_ml: int | None = None,
        is_imported: bool = False,
    ) -> VisionExtraction:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        user_text = _build_user_text(
            capture_quality=capture_quality,
            producer_record=producer_record,
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
        )
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            },
            {"type": "text", "text": user_text},
        ]

        # Anthropic's `messages.parse` (structured output) rejects this
        # ~38-property schema with "Schema is too complex." The system
        # prompt already constrains the JSON shape, so we use plain
        # `messages.create` and parse the response with the same
        # tolerant parser the Qwen fallback uses.
        # temperature=0 makes the transcription deterministic — same
        # bytes, same fields, same JSON. The cache layer relies on
        # determinism for hit-rate, and a temperature>0 sample on the
        # same image would bust caching upstream of us too.
        response = call_with_resilience(
            self._client.messages.create,
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return _parse_vision_response(text)


class MockVisionExtractor:
    """Deterministic extractor for tests and offline demos.

    Pass a fixture dict shaped like the JSON the real extractor returns:

        {"brand_name": {"value": "...", "confidence": 0.95}, ...}

    or shorthand:

        {"brand_name": "...", ...}

    Top-level keys `image_quality`, `image_quality_notes`, and
    `beverage_type_observed` are also honoured. The mock records the
    last context kwargs passed to `extract()` so tests can assert that
    the briefing wiring is intact.
    """

    def __init__(self, fixture: dict[str, Any]) -> None:
        self._fixture = fixture
        self.last_call: dict[str, Any] = {}

    def extract(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        *,
        capture_quality: Any | None = None,
        producer_record: dict[str, Any] | None = None,
        beverage_type: str | None = None,
        container_size_ml: int | None = None,
        is_imported: bool = False,
    ) -> VisionExtraction:
        self.last_call = {
            "media_type": media_type,
            "capture_quality": capture_quality,
            "producer_record": producer_record,
            "beverage_type": beverage_type,
            "container_size_ml": container_size_ml,
            "is_imported": is_imported,
        }
        normalised: dict[str, Any] = {}
        for name in EXTRACTOR_FIELDS:
            entry = self._fixture.get(name)
            if entry is None:
                continue
            if isinstance(entry, dict):
                normalised[name] = entry
            else:
                normalised[name] = {"value": str(entry), "unreadable": False, "confidence": 0.95}
        for top in ("image_quality", "image_quality_notes", "beverage_type_observed"):
            if top in self._fixture:
                normalised[top] = self._fixture[top]
        return _parse_vision_response(json.dumps(normalised))


def _from_pydantic(result: VerifyLabelExtraction) -> VisionExtraction:
    """Convert structured-output Pydantic result to the dataclass the verify
    orchestrator already consumes."""
    fields: dict[str, ExtractedField] = {}
    unreadable: list[str] = []
    for name in EXTRACTOR_FIELDS:
        fe: _FieldOut = getattr(result, name)
        if (
            fe.unreadable
            or fe.value is None
            or (isinstance(fe.value, str) and not fe.value.strip())
        ):
            unreadable.append(name)
            continue
        try:
            confidence = max(0.0, min(1.0, float(fe.confidence)))
        except (TypeError, ValueError):
            confidence = 0.85
        fields[name] = ExtractedField(
            value=fe.value,
            bbox=fe.bbox,
            confidence=confidence,
            source_image_id="front",
        )
    raw = result.model_dump_json()
    return VisionExtraction(
        fields=fields,
        unreadable=unreadable,
        raw_response=raw,
        image_quality=result.image_quality,
        image_quality_notes=result.image_quality_notes or None,
        beverage_type_observed=result.beverage_type_observed,
    )


def _parse_vision_response(text: str) -> VisionExtraction:
    """Parse a free-form JSON response (used by the Qwen fallback, which
    speaks an OpenAI-compatible API and can't bind to a Pydantic schema).

    Tolerates `image_quality` / `image_quality_notes` / `beverage_type_observed`
    at the top level — when missing, falls back to None so the verify
    orchestrator's existing inference rules apply unchanged.
    """
    cleaned = re.sub(r"^\s*```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Vision extractor returned non-JSON output: {text[:200]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Vision extractor returned a non-object JSON value: {type(data).__name__}"
        )

    fields: dict[str, ExtractedField] = {}
    unreadable: list[str] = []

    for name in EXTRACTOR_FIELDS:
        entry = data.get(name)
        if not isinstance(entry, dict):
            continue
        if entry.get("unreadable"):
            unreadable.append(name)
            continue
        value = entry.get("value")
        if value is None or (isinstance(value, str) and not value.strip()):
            unreadable.append(name)
            continue
        confidence = entry.get("confidence", 0.85)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.85
        bbox = entry.get("bbox")
        bbox_tuple: tuple[int, int, int, int] | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                bbox_tuple = (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                )
            except (TypeError, ValueError):
                bbox_tuple = None
        fields[name] = ExtractedField(
            value=str(value),
            bbox=bbox_tuple,
            confidence=max(0.0, min(1.0, confidence)),
            source_image_id="front",
        )

    image_quality_raw = data.get("image_quality")
    if (
        isinstance(image_quality_raw, str)
        and image_quality_raw in {"good", "degraded", "unreadable"}
    ):
        image_quality: str | None = image_quality_raw
    else:
        image_quality = None
    image_quality_notes = data.get("image_quality_notes")
    if not isinstance(image_quality_notes, str) or not image_quality_notes.strip():
        image_quality_notes = None
    beverage_type_observed = data.get("beverage_type_observed")
    if not isinstance(beverage_type_observed, str):
        beverage_type_observed = None

    return VisionExtraction(
        fields=fields,
        unreadable=unreadable,
        raw_response=text,
        image_quality=image_quality,
        image_quality_notes=image_quality_notes,
        beverage_type_observed=beverage_type_observed,
    )


def get_default_extractor() -> VisionExtractor:
    """Build the configured verify-path extractor.

    When `enable_qwen_fallback=True` AND a Qwen3-VL endpoint is configured,
    Claude is wrapped in a fallback chain — Claude first, Qwen second.
    Both `ExtractorUnavailable` from the Anthropic client and HTTP errors
    against the Qwen server are caught by the chain so a transient outage
    on either side keeps `/v1/verify` working as long as one extractor is
    reachable.
    """
    if settings.vision_extractor != "claude":
        raise RuntimeError(
            f"Unknown vision_extractor {settings.vision_extractor!r}; "
            "expected 'claude'."
        )
    if not settings.anthropic_api_key:
        raise ExtractorUnavailable(
            "ANTHROPIC_API_KEY is not set; required for vision_extractor=claude. "
            "Set the env var or override VISION_EXTRACTOR for tests."
        )

    primary = ClaudeVisionExtractor()
    if settings.enable_qwen_fallback and settings.qwen_vl_base_url:
        from app.services.qwen_vl import QwenVLExtractor
        from app.services.vision_chain import ChainedVerifyExtractor

        return ChainedVerifyExtractor([primary, QwenVLExtractor()])
    return primary
