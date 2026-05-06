"""Qwen3-VL local fallback for the single-shot /v1/verify path.

When the Anthropic Claude call is unavailable — flaky network, rate limit,
or upstream outage — the verify endpoint falls back to a locally hosted
Qwen3-VL server speaking the OpenAI-compatible chat-completions API
(vLLM, Ollama, LM Studio, llama.cpp `--api`). Same prompt, same field
shape, same fail-honestly contract; only the network hop changes.

Failure mode is `ExtractorUnavailable`, identical to the Anthropic client,
so a higher-level chain can treat both extractors uniformly.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from app.config import settings
from app.services.anthropic_client import ExtractorUnavailable
from app.services.vision import (
    SYSTEM_PROMPT,
    VisionExtraction,
    _build_user_text,
    _parse_vision_response,
)

logger = logging.getLogger(__name__)

# Local model + larger payload than Claude → give it slightly more room.
# The verify path's overall budget is ≤5 s for the agent UI, but the
# fallback only fires when Claude is already down, so we accept a longer
# tail rather than a hard timeout-cascade.
DEFAULT_QWEN_TIMEOUT_S = 30.0


# Qwen / OpenAI-compatible servers don't honour Anthropic's structured-output
# binding, so spell the JSON contract out in the prompt instead. The base
# SYSTEM_PROMPT names the per-field keys (value/confidence/unreadable/note)
# but smaller open-weight models — Nemotron-Nano-VL among them — read the
# schema loosely and return bare strings for each field. Spelling the
# object shape out with an example here keeps those models on the
# canonical schema; the parser also tolerates bare strings as a defence.
JSON_OUTPUT_REMINDER = """Return ONLY a JSON object — no Markdown fences, no prose. \
Use this exact shape, with each label field as a nested object (NOT a bare string):

{
  "image_quality": "good",
  "image_quality_notes": "...",
  "beverage_type_observed": "beer" | "wine" | "spirits" | "unknown",
  "brand_name":        {"value": "...", "confidence": 0.95},
  "class_type":        {"value": "...", "confidence": 0.92},
  "alcohol_content":   {"value": "...", "confidence": 0.96},
  "net_contents":      {"value": "...", "confidence": 0.95},
  "name_address":      {"value": "...", "confidence": 0.90},
  "country_of_origin": {"value": null,  "confidence": 0.0, "unreadable": true},
  "health_warning":    {"value": "...", "confidence": 0.94}
}"""


class QwenVLExtractor:
    """OpenAI-compatible client for a local Qwen3-VL server."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float | None = None,
    ) -> None:
        self._base_url = (base_url or settings.qwen_vl_base_url or "").rstrip("/")
        self._model = model or settings.qwen_vl_model
        self._api_key = api_key or settings.qwen_vl_api_key
        self._max_tokens = max_tokens
        # Default falls back to settings.qwen_vl_timeout_s so a slow local
        # Ollama (≥60 s on cold load) doesn't tank the request, while
        # hosted endpoints can keep the 30 s default.
        self._timeout = (
            timeout if timeout is not None else settings.qwen_vl_timeout_s
        )
        if not self._base_url:
            raise ExtractorUnavailable(
                "QWEN_VL_BASE_URL is not configured; cannot construct the "
                "Qwen3-VL fallback extractor. Point it at an OpenAI-compatible "
                "endpoint (e.g. http://localhost:8000/v1)."
            )

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
        # OpenAI-compatible endpoints don't bind structured output, so
        # remind the model what JSON shape we want at the very end of the
        # user message.
        user_text = f"{user_text}\n\n{JSON_OUTPUT_REMINDER}"
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
            # JSON mode is honoured by Ollama, vLLM, and most hosted
            # OpenAI-compatible providers (OpenRouter, DashScope). For a
            # quantised local Qwen3-VL it's the difference between a
            # malformed-JSON failure rate of ~10–20 % and 0 %, and it
            # also makes the model finish ~2× faster because it doesn't
            # spend tokens thinking about Markdown fences or commentary.
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}",
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
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
            logger.warning("Qwen3-VL verify call failed: %s", exc)
            raise ExtractorUnavailable(
                f"Qwen3-VL fallback unavailable at {url}: {exc}"
            ) from exc

        data = response.json()
        text = _extract_message_text(data)
        try:
            return _parse_vision_response(text)
        except ValueError as exc:
            raise ExtractorUnavailable(
                f"Qwen3-VL returned malformed JSON: {exc}"
            ) from exc


def _extract_message_text(data: Any) -> str:
    """Pull `choices[0].message.content` out of an OpenAI-style payload.

    Some servers return content as a list of `{type: text, text: ...}`
    parts (vLLM in multimodal mode); coalesce into a single string.
    """
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
