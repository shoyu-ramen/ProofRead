"""Qwen3-VL fallback for the multi-image /v1/scans pipeline.

Mirrors `extractors/claude_vision.py` but talks to a local OpenAI-compatible
Qwen3-VL server. Used when the Claude vision call is unavailable; the chain
in `app.services.vision_chain.ChainedScanExtractor` tries Claude first and
this extractor second before the pipeline's existing OCR fallback kicks in.

Reads, never judges. Confidence below the threshold lands in
`ctx.unreadable_fields` so the rule engine downgrades to ADVISORY rather
than guessing pass/fail (SPEC §0.5 fail-honestly).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import settings
from app.rules.types import ExtractionContext
from app.services.anthropic_client import ExtractorUnavailable
from app.services.extractors.claude_vision import (
    SYSTEM_PROMPT,
    LabelExtraction,
    ProducerRecord,
    _default_confidence_threshold,
    _detect_media,
    _to_context,
)
from app.services.sensor_briefing import format_capture_quality

logger = logging.getLogger(__name__)

# Multi-surface payloads + larger model on a local box → wider budget. The
# scan flow is async-friendly (HTTP polling on the mobile side), so a
# longer tail is acceptable when the Claude path has already failed.
DEFAULT_QWEN_TIMEOUT_S = 60.0


# The base SYSTEM_PROMPT defines the field shape and confidence semantics;
# we tack a strict-JSON contract onto it because OpenAI-compatible servers
# don't honour Anthropic's `output_format=` structured-output binding.
SCAN_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

OUTPUT CONSTRAINT — return ONLY a single valid JSON object that exactly
conforms to this schema. Do not wrap it in markdown, do not add commentary:

{
  "beverage_type_observed": "beer" | "wine" | "spirits" | "unknown",
  "image_quality":          "good" | "degraded" | "unreadable",
  "image_quality_notes":    <string>,
  "brand_name":        <field>,
  "class_type":        <field>,
  "alcohol_content":   <field>,
  "net_contents":      <field>,
  "name_address":      <field>,
  "country_of_origin": <field>,
  "health_warning":    <field>,
  "other_observations": <string|null>
}

Each <field> object has the shape:
  {
    "value":      <string|null>,
    "confidence": <number between 0.0 and 1.0>,
    "bbox":       [x, y, w, h] | null,
    "surface":    <string|null>,
    "note":       <string|null>
  }
"""
)


class QwenVLExtractor:
    """Local Qwen3-VL extractor for the multi-image scan pipeline."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        confidence_threshold: float | None = None,
        max_tokens: int = 8192,
        timeout: float = DEFAULT_QWEN_TIMEOUT_S,
    ) -> None:
        self._base_url = (base_url or settings.qwen_vl_base_url or "").rstrip("/")
        self._model = model or settings.qwen_vl_model
        self._api_key = api_key or settings.qwen_vl_api_key
        self._threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else _default_confidence_threshold()
        )
        self._max_tokens = max_tokens
        self._timeout = timeout
        if not self._base_url:
            raise ExtractorUnavailable(
                "QWEN_VL_BASE_URL is not configured; cannot construct the "
                "Qwen3-VL scan-path fallback extractor."
            )

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

        user_content: list[dict[str, Any]] = []
        for surface, data in images.items():
            user_content.append({"type": "text", "text": f"Surface: {surface}"})
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{_detect_media(data)};base64,"
                            + base64.standard_b64encode(data).decode("ascii")
                        ),
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
                    "Inspect every supplied surface and return ONE JSON object "
                    "matching the schema above. Be honest about confidence; the "
                    "rule engine downgrades low-confidence fields to advisory."
                ),
            }
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SCAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"
        try:
            response = httpx.post(
                url, json=payload, headers=headers, timeout=self._timeout
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Qwen3-VL scan call failed: %s", exc)
            raise ExtractorUnavailable(
                f"Qwen3-VL fallback unavailable at {url}: {exc}"
            ) from exc

        data = response.json()
        text = _extract_message_text(data)

        try:
            obj = json.loads(_strip_fences(text))
        except json.JSONDecodeError as exc:
            raise ExtractorUnavailable(
                f"Qwen3-VL output was not valid JSON: {text[:200]!r}"
            ) from exc

        try:
            result = LabelExtraction.model_validate(obj)
        except ValidationError as exc:
            raise ExtractorUnavailable(
                f"Qwen3-VL output did not match LabelExtraction schema: {exc}"
            ) from exc

        ctx = _to_context(
            result,
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
            producer_record=producer_record,
            confidence_threshold=self._threshold,
        )
        ctx.application["model_provider"] = "qwen3_vl_local"
        return ctx


def _extract_message_text(data: Any) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExtractorUnavailable(
            f"Qwen3-VL returned an unexpected payload shape: {data!r}"
        ) from exc
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise ExtractorUnavailable(
            f"Qwen3-VL returned non-string content: {type(content).__name__}"
        )
    return content


def _strip_fences(text: str) -> str:
    """Defensive: a Qwen build that ignores the no-markdown instruction."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()
