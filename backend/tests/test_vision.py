"""Tests for the single-shot Claude vision extractor used by /v1/verify.

The rich `extractors.claude_vision` extractor has its own coverage in
`test_claude_vision.py`; this file pins the simpler `services.vision`
extractor — specifically that we send adaptive thinking and a cached
system prompt on every call so the production verify path matches the
prototype's reasoning budget.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services.vision import (
    ClaudeVisionExtractor,
    _parse_vision_response,
)

_VALID_RESPONSE = json.dumps(
    {
        "brand_name": {"value": "ANYTOWN ALE", "unreadable": False, "confidence": 0.95},
        "class_type": {"value": "India Pale Ale", "unreadable": False, "confidence": 0.93},
        "alcohol_content": {"value": "5.5% ABV", "unreadable": False, "confidence": 0.97},
        "net_contents": {"value": "12 FL OZ", "unreadable": False, "confidence": 0.95},
        "name_address": {
            "value": "Brewed by Anytown Co.",
            "unreadable": False,
            "confidence": 0.9,
        },
        "country_of_origin": {"value": None, "unreadable": True, "confidence": 0.0},
        "health_warning": {
            "value": "GOVERNMENT WARNING: ...",
            "unreadable": False,
            "confidence": 0.94,
        },
    }
)


def _response_namespace(scripted: str) -> SimpleNamespace:
    """Pack a scripted JSON string into the same shape `messages.create`
    returns: a list of content blocks the extractor concatenates."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=scripted)],
    )


class _FakeMessages:
    def __init__(self, scripted: str) -> None:
        self._scripted = scripted
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _response_namespace(self._scripted)


class _FakeClient:
    def __init__(self, scripted: str) -> None:
        self.messages = _FakeMessages(scripted)


def _png_bytes() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63000100000005000100"
        "5e6c5dd80000000049454e44ae426082"
    )


def test_extract_sends_deterministic_call_with_cached_system_prompt():
    """Regression: /v1/verify must hit the model with a cached system prompt
    (so the static OCR instructions don't pay the prefix cost on every
    request) and a deterministic temperature (so identical bytes produce
    identical transcriptions, which the in-process verify cache and any
    upstream cache tier rely on).

    Adaptive thinking was intentionally removed: on a structured-JSON
    transcription task the schema already disciplines the model, and the
    end-to-end latency budget rewards a fast, deterministic single pass
    plus the redundant Government-Warning second-pass running concurrently
    in `verify()`.
    """
    fake = _FakeClient(_VALID_RESPONSE)
    extractor = ClaudeVisionExtractor(client=fake, model="claude-sonnet-4-6")

    result = extractor.extract(_png_bytes(), media_type="image/png")
    assert "brand_name" in result.fields

    [call] = fake.messages.calls
    assert "thinking" not in call, (
        "Adaptive thinking was dropped from the verify-path extractor; the "
        "presence of `thinking` here implies a regression that re-introduces "
        "the latency it was removed to save."
    )
    assert call["temperature"] == 0.0
    [system_block] = call["system"]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert call["model"] == "claude-sonnet-4-6"
    # Output budget: a typical seven-field response lands around 350
    # tokens, so 1024 is plenty of headroom while still capping tail
    # latency on labels where the model would otherwise generate filler
    # before stopping. The previous 4096 came from an older prompt that
    # asked for more verbose notes and is no longer needed.
    assert 700 <= call["max_tokens"] <= 2048


def test_extract_attaches_image_with_correct_media_type():
    fake = _FakeClient(_VALID_RESPONSE)
    extractor = ClaudeVisionExtractor(client=fake)
    extractor.extract(_png_bytes(), media_type="image/jpeg")

    [call] = fake.messages.calls
    [user_msg] = call["messages"]
    [image_block] = [b for b in user_msg["content"] if b["type"] == "image"]
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/jpeg"


def test_extract_concatenates_text_blocks_from_response():
    """`messages.create` returns a list of content blocks; thinking and tool
    blocks may be interleaved. The extractor must concatenate only the
    text blocks before parsing JSON — picking up a thinking block as
    body would tank parse rates."""

    class _FakeMessagesWithThinking:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", thinking="…"),
                    SimpleNamespace(type="text", text=_VALID_RESPONSE),
                ]
            )

    client = SimpleNamespace(messages=_FakeMessagesWithThinking())
    extractor = ClaudeVisionExtractor(client=client)
    result = extractor.extract(_png_bytes())
    assert result.fields["brand_name"].value == "ANYTOWN ALE"


def test_parse_vision_response_rejects_non_json():
    with pytest.raises(ValueError):
        _parse_vision_response("not json at all")


def test_parse_vision_response_recovers_json_with_trailing_prose():
    """Regression: Haiku occasionally appends qualifying commentary after
    the JSON object on degraded labels ("**Note:** the warning text was
    at an angle…"), even though the system prompt forbids it. The strict
    end-of-string fence regex used to bail on that case, raising
    ValueError → 500 from the verify endpoint, throwing away an
    otherwise-perfectly-valid extraction. Recover by walking the first
    balanced JSON object out of the response."""
    inner = json.dumps(
        {
            "brand_name": {"value": "GLARE-FOULED IPA", "confidence": 0.7},
            "class_type": {"value": None, "unreadable": True, "confidence": 0.0},
            "health_warning": {
                "value": (
                    "GOVERNMENT WARNING: (1) According to the Surgeon "
                    "General, women should not drink"
                ),
                "confidence": 0.65,
            },
            "image_quality": "degraded",
            "image_quality_notes": "specular highlight on right third",
            "beverage_type_observed": "beer",
        }
    )
    raw = (
        "```json\n"
        + inner
        + "\n```\n\n"
        + "**Note:** Image is at an angle and the warning text is partially "
        + "obscured by glare; transcribed what was legible."
    )
    result = _parse_vision_response(raw)
    assert "brand_name" in result.fields
    assert result.fields["brand_name"].value == "GLARE-FOULED IPA"
    assert "health_warning" in result.fields
    assert result.fields["health_warning"].value.startswith("GOVERNMENT WARNING")
    # The class_type field came back unreadable, so it's in the unreadable
    # list rather than fields.
    assert "class_type" in result.unreadable
