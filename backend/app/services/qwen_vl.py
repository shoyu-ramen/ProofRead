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
    VisionExtraction,
    _build_user_text,
    _parse_vision_response,
)

logger = logging.getLogger(__name__)

# Compact system prompt for the fallback path. The primary extractor's
# SYSTEM_PROMPT in vision.py is ~12k chars (~3k tokens) of Claude-grade
# guidance; local Ollama serves qwen3-vl with a 4096-token default
# context window, and (system + vision tokens + user text) overflowed
# it — Ollama silently truncates the prompt and the model returns empty
# content. This distilled prompt preserves the same JSON contract and
# the anti-fabrication rules in ~1/5 the tokens. Keep the field names
# and per-field object shape in lockstep with vision.SYSTEM_PROMPT /
# _parse_vision_response.
QWEN_SYSTEM_PROMPT = """You are an OCR assistant for U.S. alcohol beverage labels. \
Read the label image and return ONLY a single JSON object — no Markdown fences, no prose.

Rules:
- Transcribe text VERBATIM: preserve case, punctuation, spacing. Never normalise or tidy.
- If a field is absent or you cannot read it confidently: value null, unreadable true. \
Prefer unreadable over guessing.
- NEVER fabricate a company, city, state, or country that you cannot actually read — \
a partial verbatim fragment beats a plausible guess. Do not infer country of origin \
from brand, language, or style; only report an explicit statement \
("Product of ...", "Imported from ...", "Hecho en ...").
- alcohol_content and net_contents include their unit/marker text as part of the \
value ("5.5% ABV", "12 FL OZ"), never a bare number.
- health_warning is the GOVERNMENT WARNING statement, transcribed verbatim and complete.
- Do not emit bbox fields.

Per-field shape: {"value": <verbatim or null>, "confidence": 0.0-1.0, \
"unreadable": true only when unreadable}.

Top-level keys (always include): image_quality ("good" | "degraded" | "unreadable"), \
image_quality_notes (one sentence naming the limiting factor, or null), \
beverage_type_observed ("beer" | "wine" | "spirits" | "unknown"), and these label \
fields: brand_name, class_type, alcohol_content, net_contents, name_address, \
country_of_origin, health_warning; plus age_statement for spirits, and \
sulfite_declaration + organic_certification for wine."""

# Long-edge cap for images sent to the fallback. Label text is fully
# legible at this scale, and vision-token cost drops several-fold —
# the difference between fitting and overflowing a local Ollama's
# 4096-token default context. Anthropic's own guidance caps useful
# image input at ~1568px on the long edge, so the primary path loses
# nothing by comparison.
_MAX_IMAGE_LONG_EDGE = 1280


def _shrink_image(image_bytes: bytes, media_type: str) -> tuple[bytes, str]:
    """Downscale oversized captures before base64-encoding for the fallback.

    Returns (bytes, media_type) unchanged when the image is already
    within the cap or cannot be decoded (the endpoint will surface its
    own error for genuinely corrupt input).
    """
    try:
        import io

        from PIL import Image, ImageOps

        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= _MAX_IMAGE_LONG_EDGE:
            return image_bytes, media_type
        scale = _MAX_IMAGE_LONG_EDGE / long_edge
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        logger.info(
            "Qwen3-VL fallback: downscaled %dx%d image to %dx%d for context budget",
            w, h, *img.size,
        )
        return buf.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001 — best-effort; ship the original on any failure
        return image_bytes, media_type

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
        # Thinking-mode builds (Ollama qwen3-vl, OpenRouter reasoning
        # models) spend output budget on a `reasoning` channel before any
        # `content` arrives; 4096 was routinely exhausted mid-think on
        # full-label extractions, yielding empty content. 8192 leaves
        # room for both the think tail and the JSON payload.
        max_tokens: int = 8192,
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
        image_bytes, media_type = _shrink_image(image_bytes, media_type)
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
        # No `temperature` key: Ollama's /v1 endpoint (observed with
        # qwen3-vl, 2026-06) returns an HTTP 200 with empty content and
        # zeroed usage when ANY temperature value is present in the
        # payload. Determinism is not load-bearing here — recordings are
        # one-shot snapshots and replay-mode makes downstream runs
        # deterministic regardless.
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            # JSON mode is honoured by Ollama, vLLM, and most hosted
            # OpenAI-compatible providers (OpenRouter, DashScope). For a
            # quantised local Qwen3-VL it's the difference between a
            # malformed-JSON failure rate of ~10–20 % and 0 %, and it
            # also makes the model finish ~2× faster because it doesn't
            # spend tokens thinking about Markdown fences or commentary.
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": QWEN_SYSTEM_PROMPT},
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
        _assert_image_was_processed(data, url)
        text = _extract_message_text(data)
        try:
            return _parse_vision_response(text)
        except ValueError as exc:
            raise ExtractorUnavailable(
                f"Qwen3-VL returned malformed JSON: {exc}"
            ) from exc


# A text-only extraction prompt is ~350 tokens; any processed label image
# adds several hundred vision tokens on top. A reported prompt size below
# this floor means the server silently dropped the image — observed on a
# degraded local Ollama, which then answers the prompt BLIND, fabricating
# a plausible label (brand, address, even a Government Warning) at high
# confidence. A blind extraction reaching the rule engine is the worst
# possible fallback failure, so refuse it outright.
_MIN_PROMPT_TOKENS_WITH_IMAGE = 600


def _assert_image_was_processed(data: Any, url: str) -> None:
    """Reject responses whose usage proves the image never reached the model.

    Only fires when the server reports a positive prompt-token count
    below the with-image floor; servers that omit usage entirely are
    given the benefit of the doubt.
    """
    try:
        prompt_tokens = int(data["usage"]["prompt_tokens"])
    except (KeyError, TypeError, ValueError):
        return
    if 0 < prompt_tokens < _MIN_PROMPT_TOKENS_WITH_IMAGE:
        raise ExtractorUnavailable(
            f"Qwen3-VL at {url} reported prompt_tokens={prompt_tokens}, "
            "which is too small to include the label image — the server "
            "dropped the image and any answer would be fabricated blind. "
            "Refusing the extraction."
        )


def _extract_message_text(data: Any) -> str:
    """Pull `choices[0].message.content` out of an OpenAI-style payload.

    Some servers return content as a list of `{type: text, text: ...}`
    parts (vLLM in multimodal mode); coalesce into a single string.

    Thinking-mode models (Ollama qwen3-vl, DeepSeek-style endpoints)
    put their chain of thought in `message.reasoning` /
    `message.reasoning_content` and may run out of output budget before
    emitting any `content`. The think text usually *ends with* the JSON
    the model was about to commit, and `_parse_vision_response` already
    recovers a balanced JSON object from surrounding prose — so when
    content is empty, salvage the reasoning text rather than failing
    the whole extraction.
    """
    try:
        message = data["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExtractorUnavailable(
            f"Qwen3-VL returned an unexpected payload shape: {data!r}"
        ) from exc
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if content is not None and not isinstance(content, str):
        raise ExtractorUnavailable(
            f"Qwen3-VL returned non-string content: {type(content).__name__}"
        )
    if content and content.strip():
        return content
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        logger.info(
            "Qwen3-VL returned empty content; salvaging JSON from the "
            "reasoning channel (%d chars)", len(reasoning)
        )
        return reasoning
    return content or ""
