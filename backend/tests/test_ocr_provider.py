"""Coverage for the OCR provider factory + MockOCRProvider file loading.

`app/services/ocr.py` sat at 54%: the `MockOCRProvider` JSON-file branch
and `get_default_provider`'s selection logic were never exercised (tests
inject a pre-built MockOCRProvider directly). The GoogleVision response-
parsing body still needs the `[google-vision]` extra and a live client,
so it stays out of scope here — we cover the factory dispatch by stubbing
the provider class.
"""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.services.ocr import MockOCRProvider, get_default_provider


def test_mock_provider_loads_fixture_from_json_file(tmp_path):
    fixture = {
        "full_text": "OLD TOM\n5.5% ABV",
        "blocks": [
            {"text": "OLD TOM", "bbox": [0, 0, 100, 20], "confidence": 0.9},
            {"text": "5.5% ABV", "bbox": [0, 30, 100, 20]},
        ],
    }
    path = tmp_path / "ocr.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    # str path (the `isinstance(fixture, (str, Path))` branch).
    provider = MockOCRProvider(str(path))
    result = provider.process(b"ignored")

    assert result.provider == "mock"
    assert result.full_text == "OLD TOM\n5.5% ABV"
    assert [b.text for b in result.blocks] == ["OLD TOM", "5.5% ABV"]
    # Second block omits confidence → defaults to 0.99.
    assert result.blocks[1].confidence == 0.99


def test_get_default_provider_rejects_mock_setting(monkeypatch):
    monkeypatch.setattr(settings, "ocr_provider", "mock")
    with pytest.raises(RuntimeError, match="MockOCRProvider"):
        get_default_provider()


def test_get_default_provider_selects_google_vision(monkeypatch):
    # Stub the provider class so we don't need the google-cloud-vision
    # extra or live ADC just to cover the dispatch branch.
    sentinel = object()
    monkeypatch.setattr(settings, "ocr_provider", "google_vision")
    monkeypatch.setattr(
        "app.services.ocr.GoogleVisionOCRProvider", lambda: sentinel
    )
    assert get_default_provider() is sentinel
