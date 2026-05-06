"""Tests for the Qwen3-VL local fallback path.

Covers both the verify-path extractor (`app.services.qwen_vl`) and the
scan-path extractor (`app.services.extractors.qwen_vl`), plus the chain
wrappers in `app.services.vision_chain`. Stubs `httpx.post` so no network
or running Qwen server is required.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.config import settings
from app.rules.types import ExtractedField
from app.services import qwen_vl as verify_qwen_mod
from app.services.anthropic_client import ExtractorUnavailable
from app.services.extractors import qwen_vl as scan_qwen_mod
from app.services.extractors.claude_vision import LabelExtraction, ProducerRecord
from app.services.vision import VisionExtraction
from app.services.vision_chain import ChainedScanExtractor, ChainedVerifyExtractor

CANONICAL_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


# ---------------------------------------------------------------------------
# httpx stubs
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "http://fake/v1/chat/completions"),
                response=httpx.Response(self.status_code),
            )


def _wrap_choices(content: str) -> dict[str, Any]:
    """Build the OpenAI chat-completions envelope around a content string."""
    return {"choices": [{"message": {"content": content}}]}


def _good_verify_payload() -> str:
    """JSON content the verify extractor expects to parse."""
    return json.dumps(
        {
            "brand_name": {
                "value": "Old Tom Distillery",
                "confidence": 0.95,
                "unreadable": False,
            },
            "class_type": {
                "value": "Kentucky Straight Bourbon Whiskey",
                "confidence": 0.92,
                "unreadable": False,
            },
            "alcohol_content": {
                "value": "45% Alc./Vol. (90 Proof)",
                "confidence": 0.96,
                "unreadable": False,
            },
            "net_contents": {
                "value": "750 mL",
                "confidence": 0.95,
                "unreadable": False,
            },
            "name_address": {
                "value": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
                "confidence": 0.9,
                "unreadable": False,
            },
            "country_of_origin": {"value": None, "unreadable": True, "confidence": 0.0},
            "health_warning": {
                "value": CANONICAL_WARNING,
                "confidence": 0.94,
                "unreadable": False,
            },
        }
    )


def _good_scan_payload() -> str:
    """LabelExtraction-shape JSON for the scan extractor."""
    return json.dumps(
        {
            "beverage_type_observed": "spirits",
            "image_quality": "good",
            "image_quality_notes": "Sharp, well-lit",
            "brand_name": {
                "value": "Old Tom Distillery",
                "confidence": 0.95,
                "bbox": [10, 20, 300, 80],
                "surface": "front",
                "note": None,
            },
            "class_type": {
                "value": "Kentucky Straight Bourbon Whiskey",
                "confidence": 0.92,
                "bbox": None,
                "surface": "front",
                "note": None,
            },
            "alcohol_content": {
                "value": "45% Alc./Vol.",
                "confidence": 0.96,
                "bbox": None,
                "surface": "front",
                "note": None,
            },
            "net_contents": {
                "value": "750 mL",
                "confidence": 0.95,
                "bbox": None,
                "surface": "front",
                "note": None,
            },
            "name_address": {
                "value": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
                "confidence": 0.9,
                "bbox": None,
                "surface": "back",
                "note": None,
            },
            "country_of_origin": {
                "value": None,
                "confidence": 0.0,
                "bbox": None,
                "surface": None,
                "note": "not present",
            },
            "health_warning": {
                "value": CANONICAL_WARNING,
                "confidence": 0.94,
                "bbox": None,
                "surface": "back",
                "note": None,
            },
            "other_observations": None,
        }
    )


def _png_bytes() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63000100000005000100"
        "5e6c5dd80000000049454e44ae426082"
    )


# ---------------------------------------------------------------------------
# Verify-path Qwen extractor
# ---------------------------------------------------------------------------


def test_verify_qwen_parses_chat_completion(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002 — httpx kw
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(payload=_wrap_choices(_good_verify_payload()))

    monkeypatch.setattr(verify_qwen_mod.httpx, "post", fake_post)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")
    monkeypatch.setattr(settings, "qwen_vl_model", "qwen3-vl")
    monkeypatch.setattr(settings, "qwen_vl_api_key", "sk-local")

    extractor = verify_qwen_mod.QwenVLExtractor()
    result = extractor.extract(_png_bytes(), media_type="image/png")

    assert isinstance(result, VisionExtraction)
    assert "brand_name" in result.fields
    assert result.fields["brand_name"].value == "Old Tom Distillery"
    # country_of_origin came back as unreadable — must be in the unreadable list
    assert "country_of_origin" in result.unreadable

    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-local"
    body = captured["json"]
    assert body["model"] == "qwen3-vl"
    image_blocks = [
        b for b in body["messages"][1]["content"] if b["type"] == "image_url"
    ]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_verify_qwen_raises_extractor_unavailable_on_http_error(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(verify_qwen_mod.httpx, "post", fake_post)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    extractor = verify_qwen_mod.QwenVLExtractor()
    with pytest.raises(ExtractorUnavailable):
        extractor.extract(_png_bytes())


def test_verify_qwen_raises_extractor_unavailable_on_garbage_json(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return _FakeResponse(payload=_wrap_choices("not json at all"))

    monkeypatch.setattr(verify_qwen_mod.httpx, "post", fake_post)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    extractor = verify_qwen_mod.QwenVLExtractor()
    with pytest.raises(ExtractorUnavailable):
        extractor.extract(_png_bytes())


def test_verify_qwen_constructor_requires_base_url(monkeypatch):
    monkeypatch.setattr(settings, "qwen_vl_base_url", None)
    with pytest.raises(ExtractorUnavailable):
        verify_qwen_mod.QwenVLExtractor()


def test_verify_qwen_tolerates_bare_string_field_values(monkeypatch):
    """Some open-weight VLMs (e.g. nvidia/nemotron-nano-12b-v2-vl on
    OpenRouter) emit each field as a bare string rather than the
    {value, confidence, unreadable} object the prompt asks for. The
    parser must wrap bare strings into the canonical shape so the
    fallback path stays usable on those models — without this, the
    response would be schema-correct JSON yet land with zero fields
    extracted."""
    bare_payload = json.dumps(
        {
            "brand_name": "OLD TOM DISTILLERY",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
            "name_address": "Bottled by Old Tom Distilling Co., Bardstown, KY",
            "country_of_origin": "",  # empty string → unreadable
            "health_warning": CANONICAL_WARNING,
            "image_quality": "good",
            "beverage_type_observed": "spirits",
        }
    )

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return _FakeResponse(payload=_wrap_choices(bare_payload))

    monkeypatch.setattr(verify_qwen_mod.httpx, "post", fake_post)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    extractor = verify_qwen_mod.QwenVLExtractor()
    result = extractor.extract(_png_bytes())
    assert result.fields["brand_name"].value == "OLD TOM DISTILLERY"
    assert result.fields["health_warning"].value == CANONICAL_WARNING
    # Empty bare string lands in unreadable, not in fields.
    assert "country_of_origin" in result.unreadable
    assert "country_of_origin" not in result.fields


def test_verify_qwen_handles_multimodal_content_list(monkeypatch):
    """Some servers return content as [{type, text}, ...] parts."""

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return _FakeResponse(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": _good_verify_payload()},
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(verify_qwen_mod.httpx, "post", fake_post)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    extractor = verify_qwen_mod.QwenVLExtractor()
    result = extractor.extract(_png_bytes())
    assert result.fields["brand_name"].value == "Old Tom Distillery"


# ---------------------------------------------------------------------------
# Scan-path Qwen extractor
# ---------------------------------------------------------------------------


def test_scan_qwen_parses_label_extraction(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(payload=_wrap_choices(_good_scan_payload()))

    monkeypatch.setattr(scan_qwen_mod.httpx, "post", fake_post)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")
    # Pin the model name explicitly — without this the test depends on
    # whatever the developer has in their local `.env`, which flakes when
    # someone repoints the fallback at e.g. `qwen3-vl:latest`.
    monkeypatch.setattr(settings, "qwen_vl_model", "qwen3-vl")

    extractor = scan_qwen_mod.QwenVLExtractor()
    ctx = extractor.extract(
        beverage_type="spirits",
        container_size_ml=750,
        images={"front": _png_bytes(), "back": _png_bytes()},
    )

    assert ctx.application["model_provider"] == "qwen3_vl_local"
    assert ctx.application["image_quality"] == "good"
    assert ctx.fields["brand_name"].value == "Old Tom Distillery"
    assert "country_of_origin" not in ctx.fields  # absent → skipped, not unreadable
    body = captured["json"]
    assert body["model"] == "qwen3-vl"
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.0
    image_blocks = [
        b for b in body["messages"][1]["content"] if b["type"] == "image_url"
    ]
    assert len(image_blocks) == 2


def test_scan_qwen_strips_markdown_fences(monkeypatch):
    """Robustness against a Qwen build that ignores the no-markdown rule."""
    fenced = "```json\n" + _good_scan_payload() + "\n```"

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return _FakeResponse(payload=_wrap_choices(fenced))

    monkeypatch.setattr(scan_qwen_mod.httpx, "post", fake_post)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    extractor = scan_qwen_mod.QwenVLExtractor()
    ctx = extractor.extract(
        beverage_type="spirits",
        container_size_ml=750,
        images={"front": _png_bytes()},
    )
    assert ctx.fields["brand_name"].value == "Old Tom Distillery"


def test_scan_qwen_attaches_producer_record(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return _FakeResponse(payload=_wrap_choices(_good_scan_payload()))

    monkeypatch.setattr(scan_qwen_mod.httpx, "post", fake_post)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    extractor = scan_qwen_mod.QwenVLExtractor()
    record = ProducerRecord(brand="Old Tom", class_type="Bourbon", container_size_ml=750)
    ctx = extractor.extract(
        beverage_type="spirits",
        container_size_ml=750,
        images={"front": _png_bytes()},
        producer_record=record,
    )
    assert ctx.application["producer_record"]["brand"] == "Old Tom"


def test_scan_qwen_raises_on_schema_mismatch(monkeypatch):
    bad = json.dumps({"oops": "wrong shape"})

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return _FakeResponse(payload=_wrap_choices(bad))

    monkeypatch.setattr(scan_qwen_mod.httpx, "post", fake_post)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    extractor = scan_qwen_mod.QwenVLExtractor()
    with pytest.raises(ExtractorUnavailable):
        extractor.extract(
            beverage_type="spirits",
            container_size_ml=750,
            images={"front": _png_bytes()},
        )


def test_scan_qwen_rejects_empty_images(monkeypatch):
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")
    extractor = scan_qwen_mod.QwenVLExtractor()
    with pytest.raises(ValueError):
        extractor.extract(
            beverage_type="spirits", container_size_ml=750, images={}
        )


# ---------------------------------------------------------------------------
# Chain wrappers
# ---------------------------------------------------------------------------


class _FakeVerifyOK:
    def extract(self, image_bytes, media_type="image/png"):
        return VisionExtraction(
            fields={
                "brand_name": ExtractedField(value="ALPHA", confidence=0.95)
            },
            unreadable=[],
            raw_response="{}",
        )


class _FakeVerifyDown:
    def extract(self, image_bytes, media_type="image/png"):
        raise ExtractorUnavailable("primary down")


class _FakeScanOK:
    def __init__(self, ctx):
        self._ctx = ctx

    def extract(self, **kw):
        return self._ctx


class _FakeScanDown:
    def extract(self, **kw):
        raise ExtractorUnavailable("primary down")


def test_chained_verify_falls_back_when_primary_unavailable():
    chain = ChainedVerifyExtractor([_FakeVerifyDown(), _FakeVerifyOK()])
    result = chain.extract(_png_bytes())
    assert result.fields["brand_name"].value == "ALPHA"


def test_chained_verify_raises_last_error_when_all_down():
    chain = ChainedVerifyExtractor([_FakeVerifyDown(), _FakeVerifyDown()])
    with pytest.raises(ExtractorUnavailable):
        chain.extract(_png_bytes())


def test_chained_verify_short_circuits_when_primary_succeeds():
    """Fallback must NOT be called when the primary extractor returns OK."""
    secondary_called = []

    class _FakeSecondary:
        def extract(self, image_bytes, media_type="image/png"):
            secondary_called.append(True)
            raise AssertionError("must not be called when primary succeeds")

    chain = ChainedVerifyExtractor([_FakeVerifyOK(), _FakeSecondary()])
    chain.extract(_png_bytes())
    assert secondary_called == []


def test_chained_verify_propagates_non_extractor_errors():
    """Bugs (anything other than ExtractorUnavailable) must NOT be swallowed."""

    class _FakeBuggy:
        def extract(self, image_bytes, media_type="image/png"):
            raise RuntimeError("logic bug")

    chain = ChainedVerifyExtractor([_FakeBuggy(), _FakeVerifyOK()])
    with pytest.raises(RuntimeError, match="logic bug"):
        chain.extract(_png_bytes())


def test_chained_scan_falls_back_when_primary_unavailable():
    label = LabelExtraction.model_validate(json.loads(_good_scan_payload()))
    from app.services.extractors.claude_vision import _to_context

    ctx = _to_context(
        label,
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        producer_record=None,
        confidence_threshold=0.6,
    )

    chain = ChainedScanExtractor([_FakeScanDown(), _FakeScanOK(ctx)])
    result = chain.extract(
        beverage_type="spirits", container_size_ml=750, images={"front": b""}
    )
    assert result is ctx


def test_chained_scan_raises_last_error_when_all_down():
    chain = ChainedScanExtractor([_FakeScanDown(), _FakeScanDown()])
    with pytest.raises(ExtractorUnavailable):
        chain.extract(
            beverage_type="spirits", container_size_ml=750, images={"front": b""}
        )


def test_chain_constructor_rejects_empty_list():
    with pytest.raises(ValueError):
        ChainedVerifyExtractor([])
    with pytest.raises(ValueError):
        ChainedScanExtractor([])


# ---------------------------------------------------------------------------
# Wiring: get_default_extractor() + scans.get_vision_extractor()
# ---------------------------------------------------------------------------


def test_get_default_extractor_returns_chain_when_qwen_enabled(monkeypatch):
    from app.services import vision

    monkeypatch.setattr(settings, "vision_extractor", "claude")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "enable_qwen_fallback", True)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    # Avoid actually constructing a real Anthropic client.
    monkeypatch.setattr(
        vision, "ClaudeVisionExtractor", lambda *a, **kw: _FakeVerifyOK()
    )

    result = vision.get_default_extractor()
    assert isinstance(result, ChainedVerifyExtractor)


def test_get_default_extractor_returns_claude_only_when_fallback_disabled(monkeypatch):
    from app.services import vision

    monkeypatch.setattr(settings, "vision_extractor", "claude")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "enable_qwen_fallback", False)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    sentinel = _FakeVerifyOK()
    monkeypatch.setattr(vision, "ClaudeVisionExtractor", lambda *a, **kw: sentinel)

    result = vision.get_default_extractor()
    assert result is sentinel


def test_scans_get_vision_extractor_returns_chain_when_qwen_enabled(monkeypatch):
    from app.api import scans

    monkeypatch.setattr(settings, "vision_extractor", "claude")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "enable_qwen_fallback", True)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    monkeypatch.setattr(
        scans, "ClaudeVisionExtractor", lambda *a, **kw: _FakeScanDown()
    )

    result = scans.get_vision_extractor()
    assert isinstance(result, ChainedScanExtractor)


def test_scans_get_vision_extractor_falls_back_to_primary_if_qwen_init_fails(
    monkeypatch,
):
    """If Qwen ctor blows up but Claude is fine, return Claude alone — the
    pipeline still has its OCR backstop, so a Qwen misconfiguration must
    not take the whole vision path down."""
    from app.api import scans

    monkeypatch.setattr(settings, "vision_extractor", "claude")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "enable_qwen_fallback", True)
    monkeypatch.setattr(settings, "qwen_vl_base_url", "http://localhost:8000/v1")

    primary = _FakeScanDown()
    monkeypatch.setattr(scans, "ClaudeVisionExtractor", lambda *a, **kw: primary)

    def boom():
        raise ExtractorUnavailable("missing dep")

    monkeypatch.setattr(
        "app.services.extractors.qwen_vl.QwenVLExtractor", lambda *a, **kw: boom()
    )

    result = scans.get_vision_extractor()
    assert result is primary
