"""Claude vision-based field extraction (the "SoTA Vision Agent").

This is the path that makes ProofRead robust under the extreme conditions
catalogued in SPEC §0.5 — direct sunlight, dim bars, wet bottles, foil,
shaky capture, holographic labels, and so on. Claude Opus 4.7's high-
resolution image support and adaptive thinking are well-matched to fine-
print elements like the Health Warning Statement; a substantial rotation
or glare situation triggers more reasoning before the model declares a
field unreadable.

Design principle (SPEC §0.5 "fail honestly"):

  * The model READS — it returns each TTB-mandated element with an
    explicit confidence and bbox. It does not decide pass/fail.
  * The rule engine JUDGES — it consumes the extraction context and
    applies the deterministic regulations in `app/rules/`.

Fields whose confidence is below the threshold are pushed into
`ctx.unreadable_fields`; the rule engine downgrades the matching rules
to ADVISORY rather than guessing. A wrong "pass" is the worst outcome
(SPEC §0.5); an honest "couldn't verify" is acceptable.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rules.types import ExtractedField, ExtractionContext
from app.services.sensor_briefing import format_capture_quality

logger = logging.getLogger(__name__)


# Cached system prompt — cache_control ephemeral on the messages call so a
# single process pays the prefix tokens once and reads them back across scans.
SYSTEM_PROMPT = """You are a TTB (Alcohol and Tobacco Tax and Trade Bureau) label
inspector. You inspect alcoholic-beverage labels and extract the elements that
TTB regulations care about. You do NOT decide whether the label complies — that
is the rule engine's job. Your job is to read what is on the label and report
each element with how confident you are.

For every label you receive:

  1. Assess image quality FIRST. If lighting (direct sun, dim bar, backlight,
     mixed colored light), glare, blur, motion, rotation, occlusion (price
     stickers, security tags), wet/condensed bottle, foil, embossing,
     holographic surface, or any other factor prevents a confident reading of
     any element, set `image_quality` to "degraded" or "unreadable" and explain
     in `image_quality_notes`.

  2. Extract each TTB-mandated element with:
       value:       verbatim text as it appears (preserve case + punctuation +
                    any spelling errors — do NOT normalize or auto-correct)
       confidence:  0.0–1.0
                      >= 0.85: "I am confident in this reading"
                      0.5–0.85: "probable but the rule engine should verify"
                      <  0.5: "unreadable — downgrade rule to advisory"
       bbox:        [x, y, w, h] in pixels of the image where you read it,
                    or null if the position cannot be localized
       surface:     which surface (e.g. "front", "back") you read it from
       note:        one-line reason for any confidence below 0.85

  3. The HEALTH WARNING STATEMENT must be returned VERBATIM. The rule engine
     compares it character-for-character to the canonical statutory text.
     Do NOT paraphrase. Do NOT fix the case. Do NOT correct typos. If you can
     only see part of the warning, return what you can read with the unread
     fraction reflected in confidence and explained in `note`.

  4. If an element is genuinely not present on the label, return value=null
     with confidence=0.0 and a note like "not present" or "occluded by glare".

  5. Bounding boxes are in image pixels of whichever supplied surface the
     value was read from. Origin (0,0) is the top-left.

Be HONEST. A wrong "I read it as X" is worse than "I couldn't read it." Low
confidence is the correct answer when the label is degraded. The rule engine
will downgrade unreadable rules to advisory; it will not guess pass/fail.

Adversarial guidance:
  - Counterfeit detection is OUT OF SCOPE; do not opine on authenticity.
  - For foreign-language-only labels with no English, set
    image_quality="unreadable" and note the language.
  - For embossed-only / foil / holographic labels where your reading is
    inherently uncertain, mark image_quality="degraded" and reflect that
    uncertainty in field-level confidence."""


class FieldExtraction(BaseModel):
    """A single extracted label element."""

    value: str | None = Field(
        None, description="Verbatim text as printed on the label, or null if not present."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int] | None = Field(
        None, description="[x, y, w, h] in image pixels, or null if unlocated."
    )
    surface: str | None = Field(
        None, description="Which surface (front/back/...) the value was read from."
    )
    note: str | None = Field(
        None, description="Required if confidence is below 0.85."
    )


class LabelExtraction(BaseModel):
    """Structured output of one label inspection.

    The model populates every field — those that are not present on the
    label come back with value=null, confidence=0.0, and a brief note.
    """

    beverage_type_observed: Literal["beer", "wine", "spirits", "unknown"]
    image_quality: Literal["good", "degraded", "unreadable"]
    image_quality_notes: str
    brand_name: FieldExtraction
    class_type: FieldExtraction
    alcohol_content: FieldExtraction
    net_contents: FieldExtraction
    name_address: FieldExtraction
    country_of_origin: FieldExtraction
    health_warning: FieldExtraction
    other_observations: str | None = Field(
        None,
        description=(
            "Free-form notes about non-extracted aspects of the label "
            "(foil-wrapped, embossed-only, holographic, etc.) that the rule "
            "engine cannot otherwise see."
        ),
    )


# Fields whose confidence is below this threshold are declared "unreadable" —
# the rule engine downgrades the matching rules to ADVISORY.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class ProducerRecord:
    """Producer-side metadata supplied by the user (or read from a COLA submission)."""

    brand: str | None = None
    class_type: str | None = None
    container_size_ml: int | None = None


class ClaudeVisionExtractor:
    """Claude Opus 4.7 vision extractor.

    The API key is consumed lazily — `__init__` does not require one if a
    pre-built `client` is passed (used in tests with a stubbed client).
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "claude-opus-4-7",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_tokens: int = 8192,
        timeout: float | None = None,
    ) -> None:
        if client is None:
            from app.services.anthropic_client import (
                DEFAULT_VISION_TIMEOUT_S,
                build_client,
            )

            client = build_client(
                timeout=timeout if timeout is not None else DEFAULT_VISION_TIMEOUT_S
            )
        self._client = client
        self._model = model
        self._threshold = confidence_threshold
        self._max_tokens = max_tokens

    def extract(
        self,
        *,
        beverage_type: str,
        container_size_ml: int,
        images: dict[str, bytes],
        producer_record: ProducerRecord | None = None,
        is_imported: bool = False,
        capture_quality: Any | None = None,
    ) -> ExtractionContext:
        if not images:
            raise ValueError("at least one image required")

        user_content: list[dict] = []
        for surface, data in images.items():
            user_content.append({"type": "text", "text": f"Surface: {surface}"})
            user_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _detect_media(data),
                        "data": base64.standard_b64encode(data).decode("ascii"),
                    },
                }
            )

        if producer_record:
            record_block = (
                f"Producer record (claimed by submitter):\n"
                f"  brand              = {producer_record.brand!r}\n"
                f"  class_type         = {producer_record.class_type!r}\n"
                f"  container_size_ml  = {producer_record.container_size_ml!r}\n"
            )
        else:
            record_block = "No producer record was provided.\n"

        capture_block = format_capture_quality(capture_quality)

        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Beverage type (claimed): {beverage_type}\n"
                    f"Container size (claimed): {container_size_ml} mL\n"
                    f"Imported: {is_imported}\n"
                    f"{record_block}"
                    f"{capture_block}\n"
                    "Inspect every supplied surface and return a structured "
                    "LabelExtraction. Be honest about confidence — a low "
                    "confidence is the correct answer when the image is "
                    "degraded; the rule engine will downgrade those rules to "
                    "advisory rather than guess pass/fail."
                ),
            }
        )

        from app.services.anthropic_client import call_with_resilience

        response = call_with_resilience(
            self._client.messages.parse,
            model=self._model,
            max_tokens=self._max_tokens,
            thinking={"type": "adaptive", "display": "summarized"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
            output_format=LabelExtraction,
        )
        result: LabelExtraction = response.parsed_output

        return _to_context(
            result,
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
            producer_record=producer_record,
            confidence_threshold=self._threshold,
        )


def _detect_media(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


_FIELD_NAMES = (
    "brand_name",
    "class_type",
    "alcohol_content",
    "net_contents",
    "name_address",
    "country_of_origin",
    "health_warning",
)


def _to_context(
    result: LabelExtraction,
    *,
    beverage_type: str,
    container_size_ml: int,
    is_imported: bool,
    producer_record: ProducerRecord | None,
    confidence_threshold: float,
) -> ExtractionContext:
    fields: dict[str, ExtractedField] = {}
    unreadable: list[str] = []

    for name in _FIELD_NAMES:
        fe: FieldExtraction = getattr(result, name)
        if fe is None:
            continue
        # Field genuinely absent — let presence checks see "not extracted" and
        # report FAIL with a real reason rather than ADVISORY.
        if fe.value is None and fe.confidence == 0.0:
            continue
        fields[name] = ExtractedField(
            value=fe.value,
            bbox=fe.bbox,
            confidence=fe.confidence,
            source_image_id=fe.surface,
        )
        # Below threshold: rule engine should downgrade this to advisory.
        if fe.confidence < confidence_threshold:
            unreadable.append(name)

    abv_pct = _abv_pct_from(fields.get("alcohol_content"))

    application: dict[str, Any] = {
        "model_provider": "claude_opus_4_7",
        "image_quality": result.image_quality,
        "image_quality_notes": result.image_quality_notes,
        "beverage_type_observed": result.beverage_type_observed,
        "model_observations": result.other_observations,
    }
    if producer_record is not None:
        application["producer_record"] = {
            "brand": producer_record.brand,
            "class_type": producer_record.class_type,
            "container_size_ml": producer_record.container_size_ml,
        }

    # Whole-image unreadable: every required field is unreliable. Push them
    # all into unreadable_fields so the engine produces ADVISORY across the
    # board rather than chasing phantom failures from a junk read.
    if result.image_quality == "unreadable":
        unreadable = list({*unreadable, *fields.keys()})

    return ExtractionContext(
        fields=fields,
        beverage_type=beverage_type,
        container_size_ml=container_size_ml,
        is_imported=is_imported,
        abv_pct=abv_pct,
        application=application,
        unreadable_fields=unreadable,
    )


_ABV_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _abv_pct_from(field: ExtractedField | None) -> float | None:
    if field is None or not field.value:
        return None
    m = _ABV_PCT_RE.search(field.value)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
