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
    """Reset both module-level caches between tests.

    The extractor cache is reset because monkey-patched factories must
    not leak. The verify cache is reset because tests rely on cold-path
    behavior (e.g. assertions on which extractor was called) which a
    leftover entry would short-circuit.
    """
    verify_api._extractor_cache = None
    verify_api._reset_verify_cache()
    yield
    verify_api._extractor_cache = None
    verify_api._reset_verify_cache()
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
        def extract(self, image_bytes: bytes, media_type: str = "image/png", **_: object):
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


def test_verify_rejects_wine_with_422(monkeypatch):
    """Wine isn't supported in v1; the rule engine has no wine.yaml. Gate
    here with a structured 422 (`code: beverage_type_unsupported`) so the
    UI can render a friendly "wine arrives in v2" affordance instead of
    treating it as a malformed request or letting it 500 inside verify().
    """
    client = TestClient(app)
    payload = _form_payload()
    payload["beverage_type"] = "wine"
    res = client.post(
        "/v1/verify",
        data=payload,
        files={"image": _png_file()},
    )
    assert res.status_code == 422, res.text
    body = res.json()
    assert body["detail"]["code"] == "beverage_type_unsupported"
    assert "v2" in body["detail"]["message"].lower()


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
    # Cold path: cache_hit must be False. Asserts the field is wired into
    # the response model and not silently dropped by Pydantic.
    assert body.get("cache_hit") is False


def test_verify_second_identical_request_hits_cache(monkeypatch):
    """End-to-end: two POSTs with identical body must produce the same
    verdict; the second comes back with `cache_hit=True` and elapses well
    under the 50 ms iterative-design budget. This is the user-facing
    contract — a brewer re-submitting the same Illustrator export gets
    an instant verdict."""

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
    # Same image bytes for both requests so they share a cache key.
    image_bytes = _make_synthetic_png()

    payload = _form_payload()
    cold = client.post(
        "/v1/verify",
        data=payload,
        files={"image": ("label.png", io.BytesIO(image_bytes), "image/png")},
    )
    warm = client.post(
        "/v1/verify",
        data=payload,
        files={"image": ("label.png", io.BytesIO(image_bytes), "image/png")},
    )

    assert cold.status_code == 200
    assert warm.status_code == 200
    cold_body = cold.json()
    warm_body = warm.json()

    assert cold_body["cache_hit"] is False
    assert warm_body["cache_hit"] is True
    assert cold_body["overall"] == warm_body["overall"]
    # The reported elapsed_ms must reflect the warm path, not the cold one.
    # Sensor pre-check is ~60 ms on the cold path; the warm path skips it.
    assert warm_body["elapsed_ms"] < 50, (
        f"warm path elapsed_ms={warm_body['elapsed_ms']} exceeds 50 ms budget"
    )


# ---------------------------------------------------------------------------
# Phase-1 safety net: image-size cap, request timeout, observability,
# rule_result.surface field, model-independence config.
# ---------------------------------------------------------------------------


def test_verify_rejects_oversized_image(monkeypatch):
    """A POST with a body larger than `settings.max_image_bytes` must be
    rejected at the API layer with 413 + a structured payload — never
    forwarded to the (memory-hungry) sensor pre-check."""
    from app.config import settings

    monkeypatch.setattr(settings, "max_image_bytes", 1024)

    client = TestClient(app)
    big = b"\x00" * 4096  # 4 KiB > 1 KiB cap
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files={"image": ("big.png", io.BytesIO(big), "image/png")},
    )

    assert res.status_code == 413, res.text
    body = res.json()
    assert body["detail"]["code"] == "image_too_large"
    assert "1024" in body["detail"]["message"]


def test_verify_returns_504_on_timeout(monkeypatch):
    """Hard wall-clock cap: when the orchestrator outlives
    `verify_request_timeout_s`, the endpoint converts the asyncio
    TimeoutError to a 504 with a structured payload — never lets the
    request hang the worker indefinitely."""
    from app.config import settings

    monkeypatch.setattr(settings, "verify_request_timeout_s", 0.05)

    class _SlowExtractor:
        def extract(self, image_bytes, media_type="image/png", **_):
            import time
            time.sleep(0.5)  # well over the 50 ms cap
            return None  # never reached

    monkeypatch.setattr(
        verify_api, "get_default_extractor", lambda: _SlowExtractor()
    )

    client = TestClient(app)
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files={"image": _png_file()},
    )
    assert res.status_code == 504, res.text
    body = res.json()
    assert body["detail"]["code"] == "verify_timeout"


def test_verify_response_carries_surface_on_rule_results(monkeypatch):
    """Every rule_result with a referenced field must carry the `surface`
    field. The v1 mobile happy path uploads one unrolled-label panorama,
    so every field-tied rule_result reports surface='panorama'."""
    fixture = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol.",
        "net_contents": "750 mL",
        "name_address": (
            "Distilled and bottled by Old Tom Distilling Company, "
            "Bardstown, Kentucky"
        ),
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
    rule_results = body["rule_results"]
    # At least one rule_result must surface the source panel — we just
    # established the contract, so any field-tied rule should have it.
    surfaced = [r for r in rule_results if r.get("surface") is not None]
    assert surfaced, (
        f"No rule_results carry surface; payload: {rule_results}"
    )
    assert all(r["surface"] == "panorama" for r in surfaced), (
        f"Single-image path should report surface='panorama'; got "
        f"{[r['surface'] for r in surfaced]}"
    )


def test_verify_stats_endpoint_returns_counters(monkeypatch):
    """The `/v1/verify/_stats` endpoint exposes the in-process counter
    snapshot. After one cold + one warm verify, both counts > 0 and the
    overall verdict map is populated."""
    from app.services import verify_stats

    verify_stats.reset()
    fixture = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol.",
        "net_contents": "750 mL",
        "name_address": (
            "Distilled and bottled by Old Tom Distilling Company, "
            "Bardstown, Kentucky"
        ),
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
    image_bytes = _make_synthetic_png()
    payload = _form_payload()
    # cold
    client.post(
        "/v1/verify",
        data=payload,
        files={"image": ("label.png", io.BytesIO(image_bytes), "image/png")},
    )
    # warm
    client.post(
        "/v1/verify",
        data=payload,
        files={"image": ("label.png", io.BytesIO(image_bytes), "image/png")},
    )

    res = client.get("/v1/verify/_stats")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["cold_count"] == 1
    assert body["warm_count"] == 1
    assert body["overall_verdicts"]
    # The cache section is non-None when verify_cache_max_entries > 0.
    assert body["cache"] is not None
    assert body["cache"]["hits"] >= 1


def test_second_pass_model_default_differs_from_primary(monkeypatch):
    """Item #3: SPEC §0.5 calls for two *independent* reads of the
    Government Warning. The default config must route the second pass to
    a different model family than the primary so the two reads are not
    correlated by sharing an OCR backend.

    Test pins the *defaults* — operators can still tune them via env
    overrides — so a future code change that accidentally collapses
    both defaults to the same model fails this test loudly. We use
    `model_fields` rather than instantiating `Settings()` to avoid
    picking up the developer's local `.env` overrides."""
    from app.config import Settings

    primary_default = Settings.model_fields["anthropic_model"].default
    second_default = Settings.model_fields["anthropic_health_warning_model"].default
    assert primary_default != second_default, (
        f"Primary and second-pass models must default to different "
        f"families (SPEC §0.5 redundancy); both default to {primary_default!r}"
    )
