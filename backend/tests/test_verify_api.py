"""HTTP-level tests for /v1/verify error contracts.

Service-level coverage of `verify()` lives in `test_verify.py`. These tests
pin the *API* shape — specifically that an unconfigured / unreachable vision
extractor surfaces as a structured 503 the UI can render cleanly, not a 500
with a developer-facing env-var instruction (which is what the original
"fallback flow" did and what the screenshot shipped with this change captures).
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from app.api import verify as verify_api
from app.main import app
from app.services.anthropic_client import ExtractorUnavailable
from app.services.vision import MockVisionExtractor
from tests.conftest import _make_synthetic_png


@pytest.fixture(autouse=True)
def _clear_extractor_cache():
    """Reset the verify module's process-wide extractor cache between tests
    so one test's monkey-patched factory doesn't leak into the next."""
    verify_api._extractor_cache = None
    yield
    verify_api._extractor_cache = None
    app.dependency_overrides.clear()


def _form_payload() -> dict[str, object]:
    return {
        "beverage_type": "spirits",
        "container_size_ml": "750",
        "is_imported": "false",
        "application": json.dumps({"producer_record": {}}),
    }


def _png_file() -> tuple[str, io.BytesIO, str]:
    return ("label.png", io.BytesIO(_make_synthetic_png()), "image/png")


def test_verify_returns_503_when_extractor_unavailable(monkeypatch):
    """If `get_default_extractor` raises ExtractorUnavailable (e.g. no
    ANTHROPIC_API_KEY), the endpoint must return 503 with a structured
    payload. The UI keys off `detail.code` to render friendly copy and
    treats this as a transient/retryable condition."""

    def boom() -> object:
        raise ExtractorUnavailable(
            "ANTHROPIC_API_KEY is not set; required for vision_extractor=claude."
        )

    monkeypatch.setattr(verify_api, "get_default_extractor", boom)

    client = TestClient(app)
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files={"image": _png_file()},
    )

    assert res.status_code == 503, res.text
    body = res.json()
    assert body["detail"]["code"] == "vision_unavailable"
    assert "ANTHROPIC_API_KEY" in body["detail"]["message"]


def test_verify_returns_503_when_extractor_fails_mid_request(monkeypatch):
    """Construction succeeds but `extract()` raises ExtractorUnavailable —
    e.g. Anthropic rate-limit or socket error. Same 503 shape so the UI
    handles a single transient failure mode."""

    class _FlakyExtractor:
        def extract(self, image_bytes: bytes, media_type: str = "image/png"):
            raise ExtractorUnavailable("upstream rate limit (429)")

    monkeypatch.setattr(
        verify_api, "get_default_extractor", lambda: _FlakyExtractor()
    )

    client = TestClient(app)
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files={"image": _png_file()},
    )

    assert res.status_code == 503, res.text
    body = res.json()
    assert body["detail"]["code"] == "vision_unavailable"
    assert "rate limit" in body["detail"]["message"]


def test_verify_returns_500_for_config_runtime_error(monkeypatch):
    """A RuntimeError (config bug, e.g. unknown extractor name) is NOT a
    transient outage — keep 500 so misconfiguration is visible in alerts
    rather than masked as a flaky 503."""

    def boom() -> object:
        raise RuntimeError("Unknown vision_extractor 'made_up'")

    monkeypatch.setattr(verify_api, "get_default_extractor", boom)

    client = TestClient(app)
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files={"image": _png_file()},
    )

    assert res.status_code == 500, res.text


def test_verify_returns_200_when_extractor_succeeds(monkeypatch):
    """Sanity check: a healthy extractor produces a normal response shape so
    the new 503 branch isn't accidentally swallowing the success path."""
    fixture = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol.",
        "net_contents": "750 mL",
        "name_address": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
        "health_warning": (
            "GOVERNMENT WARNING: (1) According to the Surgeon General, women "
            "should not drink alcoholic beverages during pregnancy because of "
            "the risk of birth defects. (2) Consumption of alcoholic beverages "
            "impairs your ability to drive a car or operate machinery, and may "
            "cause health problems."
        ),
    }
    monkeypatch.setattr(
        verify_api, "get_default_extractor", lambda: MockVisionExtractor(fixture)
    )

    client = TestClient(app)
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files={"image": _png_file()},
    )

    assert res.status_code == 200, res.text
    body = res.json()
    assert "overall" in body
    assert "rule_results" in body
