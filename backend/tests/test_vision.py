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

from app.services.vision import ClaudeVisionExtractor, _parse_vision_response

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


class _FakeMessages:
    def __init__(self, scripted: str) -> None:
        self._scripted = scripted
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._scripted)]
        )


class _FakeClient:
    def __init__(self, scripted: str) -> None:
        self.messages = _FakeMessages(scripted)


def _png_bytes() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63000100000005000100"
        "5e6c5dd80000000049454e44ae426082"
    )


def test_extract_sends_adaptive_thinking_and_cached_system_prompt():
    """Regression: /v1/verify must hit the model with adaptive thinking and
    cache the system prompt. Dropping either is the silent kind of regression
    that wouldn't show up in functional tests but would tank reasoning
    quality on degraded labels."""
    fake = _FakeClient(_VALID_RESPONSE)
    extractor = ClaudeVisionExtractor(client=fake, model="claude-opus-4-7")

    result = extractor.extract(_png_bytes(), media_type="image/png")
    assert "brand_name" in result.fields

    [call] = fake.messages.calls
    assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
    [system_block] = call["system"]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert call["model"] == "claude-opus-4-7"
    # max_tokens has to leave headroom for thinking on top of the JSON output.
    assert call["max_tokens"] >= 4096


def test_extract_attaches_image_with_correct_media_type():
    fake = _FakeClient(_VALID_RESPONSE)
    extractor = ClaudeVisionExtractor(client=fake)
    extractor.extract(_png_bytes(), media_type="image/jpeg")

    [call] = fake.messages.calls
    [user_msg] = call["messages"]
    [image_block] = [b for b in user_msg["content"] if b["type"] == "image"]
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/jpeg"


def test_extract_filters_thinking_blocks_from_response():
    """When adaptive thinking is on, the SDK returns content blocks of type
    'thinking' alongside the actual JSON 'text' block. The extractor must
    only concatenate 'text' blocks before parsing — feeding a thinking
    summary to the JSON parser would crash the request."""

    class _FakeMessagesWithThinking:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="thinking",
                        thinking="Considering the label looks fine; brand is large and clear.",
                    ),
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
