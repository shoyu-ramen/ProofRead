"""Claude vision extractor — single VLM call to read TTB-relevant fields.

Replaces the OCR + regex-extractor combination for the /v1/verify path. The
system prompt defines the JSON output shape and is marked for prompt caching
since it's static across requests.

The extractor prefers `unreadable: true` over guessing on poor images — that
loops back through the rule engine's confidence-aware degradation so an
agent never sees a verdict that was guessed from a smudge.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import Anthropic

from app.config import settings
from app.rules.types import ExtractedField
from app.services.anthropic_client import (
    DEFAULT_VISION_TIMEOUT_S,
    ExtractorUnavailable,
    build_client,
    call_with_resilience,
)

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


SYSTEM_PROMPT = """You are an OCR assistant for U.S. alcohol beverage labels. \
Read a label image and return a JSON object with the seven TTB-relevant fields \
listed below.

Return text VERBATIM as it appears on the label. Preserve the original case \
(uppercase, mixed-case, title-case), punctuation, and whitespace. The downstream \
compliance engine cares about exactness, so do not normalise, expand abbreviations, \
or "tidy up" the wording.

If a field is genuinely not present, or you cannot read it confidently, set \
`unreadable: true` and leave `value` as null. Prefer `unreadable: true` over \
guessing — the agent reviewing this label needs to trust your output.

Output ONLY a JSON object with this exact shape (no Markdown fences, no prose):

{
  "brand_name":        {"value": "<verbatim>", "unreadable": false, "confidence": 0.0-1.0},
  "class_type":        {"value": "<verbatim>", "unreadable": false, "confidence": 0.0-1.0},
  "alcohol_content":   {"value": "<verbatim>", "unreadable": false, "confidence": 0.0-1.0},
  "net_contents":      {"value": "<verbatim>", "unreadable": false, "confidence": 0.0-1.0},
  "name_address":      {"value": "<verbatim>", "unreadable": false, "confidence": 0.0-1.0},
  "country_of_origin": {"value": "<verbatim>", "unreadable": false, "confidence": 0.0-1.0},
  "health_warning":    {"value": "<verbatim>", "unreadable": false, "confidence": 0.0-1.0}
}

Field definitions:

- brand_name: The most prominent brand name on the front of the label \
(e.g. "OLD TOM DISTILLERY", "Stone's Throw", "Mountain Crest").

- class_type: The TTB class/type designation (e.g. "Kentucky Straight \
Bourbon Whiskey", "London Dry Gin", "India Pale Ale", "Cabernet Sauvignon").

- alcohol_content: The alcohol content statement verbatim, including the \
parenthetical proof statement when present (e.g. "45% Alc./Vol. (90 Proof)", \
"5.5% ABV", "14.2% Alc./Vol.").

- net_contents: The net contents statement verbatim (e.g. "750 mL", \
"12 FL OZ", "16 FL OZ (473 mL)").

- name_address: The bottler/producer/brewer/importer statement verbatim \
(e.g. "Bottled by Old Tom Distilling Co., Bardstown, Kentucky", "Brewed and \
canned by Mountain Crest Brewing Co., Bend, Oregon").

- country_of_origin: ONLY if the label explicitly declares a country of \
origin (e.g. "Product of Mexico", "Imported from Scotland"). For domestic \
U.S. labels with no such statement, set unreadable: true.

- health_warning: The full Government Warning paragraph verbatim, INCLUDING \
the prefix exactly as it appears on the label. If the label uses \
"GOVERNMENT WARNING:" in capitals, return it that way. If it uses any other \
case (e.g. "Government Warning:"), return it that way. Preserving the original \
case here is essential — the compliance check distinguishes capitalisation.

Confidence scale:
  1.0  — text is large, sharp, and unambiguous
  0.85 — readable but smaller or partially stylised
  0.6  — partial read; some characters uncertain
  <0.6 — very uncertain; consider unreadable: true instead
"""


@dataclass
class VisionExtraction:
    fields: dict[str, ExtractedField]
    unreadable: list[str]
    raw_response: str


class VisionExtractor(Protocol):
    def extract(self, image_bytes: bytes, media_type: str = ...) -> VisionExtraction: ...


class ClaudeVisionExtractor:
    """Production extractor backed by Claude Sonnet/Opus with vision input.

    Uses the centralised `build_client` factory so per-call timeout and
    retry budget come from one place; transient SDK errors are translated
    to `ExtractorUnavailable` so the pipeline can fall back to OCR.

    Adaptive thinking (SPEC §0.5 fail-honestly): the model decides per-label
    how much reasoning to spend before answering. A clean front-of-pack
    bourbon gets a fast answer; a glared/rotated bottle thinks longer
    before declaring a field unreadable. This is the same pattern the rich
    `extractors/claude_vision.py` extractor uses on the scan flow — wiring
    it into the single-shot verify path keeps the trust budget consistent.
    """

    def __init__(
        self,
        client: Anthropic | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        timeout: float = DEFAULT_VISION_TIMEOUT_S,
    ) -> None:
        self._client = client or build_client(timeout=timeout)
        self._model = model or settings.anthropic_model
        self._max_tokens = max_tokens

    def extract(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> VisionExtraction:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        response = call_with_resilience(
            self._client.messages.create,
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
                            "text": "Extract the TTB-relevant fields from this label.",
                        },
                    ],
                }
            ],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return _parse_vision_response(text)


class MockVisionExtractor:
    """Deterministic extractor for tests and offline demos.

    Pass a fixture dict shaped like the JSON the real extractor returns:

        {"brand_name": {"value": "...", "confidence": 0.95}, ...}

    or shorthand:

        {"brand_name": "...", ...}
    """

    def __init__(self, fixture: dict[str, Any]) -> None:
        self._fixture = fixture

    def extract(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> VisionExtraction:
        normalised: dict[str, Any] = {}
        for name in EXTRACTOR_FIELDS:
            entry = self._fixture.get(name)
            if entry is None:
                continue
            if isinstance(entry, dict):
                normalised[name] = entry
            else:
                normalised[name] = {"value": str(entry), "unreadable": False, "confidence": 0.95}
        return _parse_vision_response(json.dumps(normalised))


def _parse_vision_response(text: str) -> VisionExtraction:
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
            continue
        confidence = entry.get("confidence", 0.85)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.85
        fields[name] = ExtractedField(
            value=str(value),
            bbox=None,
            confidence=max(0.0, min(1.0, confidence)),
            source_image_id="front",
        )

    return VisionExtraction(fields=fields, unreadable=unreadable, raw_response=text)


def get_default_extractor() -> VisionExtractor:
    if settings.vision_extractor == "claude":
        if not settings.anthropic_api_key:
            raise ExtractorUnavailable(
                "ANTHROPIC_API_KEY is not set; required for vision_extractor=claude. "
                "Set the env var or override VISION_EXTRACTOR for tests."
            )
        return ClaudeVisionExtractor()
    raise RuntimeError(
        f"Unknown vision_extractor {settings.vision_extractor!r}; "
        "expected 'claude'."
    )
