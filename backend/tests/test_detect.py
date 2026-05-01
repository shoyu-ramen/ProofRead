"""Tests for the /v1/detect-container pre-capture gate.

Two layers:

  * Service-level unit tests stub the Anthropic client (matches the
    `_FakeMessages` / `_FakeClient` pattern from
    `tests/test_health_warning_second_pass.py` and `test_claude_vision.py`).
  * Route-level tests use FastAPI's TestClient + multipart upload to pin
    the HTTP contract (200/422/503) the way `tests/test_verify_api.py`
    does for /v1/verify.

The point of the gate is to refuse to run the 10.8 s cylindrical-scan
chain on a frame that doesn't even contain a container, so the tests
focus on the two outcomes the UI keys off (`detected=true` with a
bbox, or `detected=false` with a reason) and the failure modes that
the UI must render cleanly (timeout → 503).
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.anthropic_client import ExtractorUnavailable
from app.services.container_check import (
    ContainerCheckService,
    ContainerDetection,
    _override_service,
    _reset_cache,
    detect_container,
)
from tests.conftest import _make_synthetic_png

# ---------------------------------------------------------------------------
# Fakes — match the pattern from test_claude_vision.py / test_health_warning_*.
# ---------------------------------------------------------------------------


class _FakeMessages:
    """Records every messages.parse() call and returns a scripted result."""

    def __init__(self, scripted: ContainerDetection | Exception) -> None:
        self._scripted = scripted
        self.calls: list[dict] = []

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if isinstance(self._scripted, Exception):
            raise self._scripted
        return SimpleNamespace(parsed_output=self._scripted, usage=None)


class _FakeClient:
    def __init__(self, scripted: ContainerDetection | Exception) -> None:
        self.messages = _FakeMessages(scripted)


class _FakeRequest:
    """Stub the anthropic exception constructors expect."""

    def __init__(self) -> None:
        self.method = "POST"
        self.url = "https://api.anthropic.com/v1/messages"


# ---------------------------------------------------------------------------
# Autouse: reset cache + service singleton between tests so a leaked
# stub doesn't pollute the next test, and a cached entry from one test
# doesn't short-circuit another's stub call.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_state():
    _reset_cache()
    _override_service(None)
    yield
    _reset_cache()
    _override_service(None)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# ContainerDetection model_validator
# ---------------------------------------------------------------------------


def test_detected_true_requires_container_type_and_bbox():
    with pytest.raises(ValueError, match="container_type"):
        ContainerDetection(detected=True, bbox=(0.1, 0.1, 0.9, 0.9), confidence=0.9)
    with pytest.raises(ValueError, match="bbox"):
        ContainerDetection(detected=True, container_type="bottle", confidence=0.9)


def test_detected_false_requires_reason():
    with pytest.raises(ValueError, match="reason"):
        ContainerDetection(detected=False, confidence=0.1)
    with pytest.raises(ValueError, match="reason"):
        # Empty string is not a meaningful reason.
        ContainerDetection(detected=False, reason="   ", confidence=0.1)


def test_detected_true_rejects_inverted_bbox():
    with pytest.raises(ValueError, match="bbox"):
        ContainerDetection(
            detected=True,
            container_type="bottle",
            bbox=(0.9, 0.1, 0.1, 0.9),  # x0 > x1
            confidence=0.9,
        )


def test_detected_true_rejects_zero_area_bbox():
    """A degenerate bbox (zero width or zero height) carries no spatial
    information — the validator must reject so the mobile overlay never
    receives a stroke that collapses to a line."""
    with pytest.raises(ValueError, match="bbox"):
        ContainerDetection(
            detected=True,
            container_type="bottle",
            bbox=(0.5, 0.1, 0.5, 0.9),  # x0 == x1
            confidence=0.9,
        )
    with pytest.raises(ValueError, match="bbox"):
        ContainerDetection(
            detected=True,
            container_type="bottle",
            bbox=(0.1, 0.5, 0.9, 0.5),  # y0 == y1
            confidence=0.9,
        )


def test_detected_true_accepts_full_frame_bbox():
    """A container that fills the frame is still a valid bbox at the
    boundaries (inclusive 0.0 / 1.0)."""
    det = ContainerDetection(
        detected=True,
        container_type="can",
        bbox=(0.0, 0.0, 1.0, 1.0),
        confidence=0.95,
    )
    assert det.bbox == (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Service-level: stubbed Anthropic client
# ---------------------------------------------------------------------------


def _png_bytes() -> bytes:
    """Synthetic PNG of a sensible size for the test (real bytes hash uniquely)."""
    return _make_synthetic_png()


def test_service_returns_detected_true_with_bbox():
    scripted = ContainerDetection(
        detected=True,
        container_type="bottle",
        bbox=(0.2, 0.1, 0.7, 0.95),
        confidence=0.92,
    )
    client = _FakeClient(scripted)
    service = ContainerCheckService(client=client, model="claude-haiku-test")
    _override_service(service)

    result = detect_container(_png_bytes(), media_type="image/png")
    assert result.detected is True
    assert result.container_type == "bottle"
    assert result.bbox == (0.2, 0.1, 0.7, 0.95)
    assert result.confidence == pytest.approx(0.92)
    # Wiring: the call MUST send the structured-output schema, the cached
    # system prompt, and one image block.
    [call] = client.messages.calls
    assert call["model"] == "claude-haiku-test"
    assert call["temperature"] == 0.0
    assert call["output_format"] is ContainerDetection
    [system_block] = call["system"]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    [user_msg] = call["messages"]
    image_block = next(b for b in user_msg["content"] if b["type"] == "image")
    assert image_block["source"]["media_type"] == "image/png"


def test_service_returns_detected_false_with_reason():
    scripted = ContainerDetection(
        detected=False,
        reason="appears to be a selfie",
        confidence=0.05,
    )
    client = _FakeClient(scripted)
    _override_service(ContainerCheckService(client=client))

    result = detect_container(_png_bytes(), media_type="image/png")
    assert result.detected is False
    assert result.reason == "appears to be a selfie"
    assert result.container_type is None
    assert result.bbox is None


def test_cache_hits_on_repeat_call_for_same_bytes():
    scripted = ContainerDetection(
        detected=True,
        container_type="can",
        bbox=(0.1, 0.05, 0.9, 0.95),
        confidence=0.88,
    )
    client = _FakeClient(scripted)
    _override_service(ContainerCheckService(client=client))

    image = _png_bytes()
    first = detect_container(image, media_type="image/png")
    second = detect_container(image, media_type="image/png")

    # Same bytes, same result — but the SDK was called only once.
    assert len(client.messages.calls) == 1, (
        f"Expected exactly one upstream call on cache hit; got "
        f"{len(client.messages.calls)}."
    )
    # Caller-visible result is equivalent (not the same object — we hand
    # out a fresh copy on hit).
    assert first.detected == second.detected
    assert first.bbox == second.bbox
    assert first.container_type == second.container_type
    assert first is not second


def test_cache_miss_for_different_bytes():
    scripted = ContainerDetection(
        detected=True,
        container_type="bottle",
        bbox=(0.1, 0.05, 0.9, 0.95),
        confidence=0.9,
    )
    client = _FakeClient(scripted)
    _override_service(ContainerCheckService(client=client))

    detect_container(_make_synthetic_png(text="A"), media_type="image/png")
    detect_container(_make_synthetic_png(text="B"), media_type="image/png")

    # Different bytes, two separate upstream calls.
    assert len(client.messages.calls) == 2


def test_timeout_maps_to_extractor_unavailable():
    """The pre-capture timeout (no retries) must surface as
    ExtractorUnavailable so the route can render a 503 envelope. The
    SDK's `APITimeoutError` is the thing `call_with_resilience`
    translates."""
    request = _FakeRequest()
    timeout_exc = anthropic.APITimeoutError(request=request)
    client = _FakeClient(timeout_exc)
    _override_service(ContainerCheckService(client=client))

    with pytest.raises(ExtractorUnavailable):
        detect_container(_png_bytes(), media_type="image/png")


def test_connection_error_maps_to_extractor_unavailable():
    """Same translation for connection errors — single failure surface."""
    request = _FakeRequest()
    conn_exc = anthropic.APIConnectionError(request=request)
    client = _FakeClient(conn_exc)
    _override_service(ContainerCheckService(client=client))

    with pytest.raises(ExtractorUnavailable):
        detect_container(_png_bytes(), media_type="image/png")


def test_empty_image_bytes_rejected_at_service_layer():
    """Defensive: the service should not even attempt an upstream call
    on empty bytes (the route layer already 400s, but the service
    contract guards it too)."""
    _override_service(ContainerCheckService(client=_FakeClient(
        ContainerDetection(detected=False, reason="x", confidence=0.0)
    )))
    with pytest.raises(ValueError):
        detect_container(b"", media_type="image/png")


# ---------------------------------------------------------------------------
# Route-level: HTTP contract via multipart upload
# ---------------------------------------------------------------------------


def _png_file() -> tuple[str, io.BytesIO, str]:
    return ("frame.png", io.BytesIO(_png_bytes()), "image/png")


def test_route_returns_200_and_detection_payload(monkeypatch):
    scripted = ContainerDetection(
        detected=True,
        container_type="bottle",
        bbox=(0.15, 0.05, 0.85, 0.95),
        confidence=0.91,
    )
    _override_service(ContainerCheckService(client=_FakeClient(scripted)))

    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["detected"] is True
    assert body["container_type"] == "bottle"
    assert body["bbox"] == [0.15, 0.05, 0.85, 0.95]
    assert body["confidence"] == pytest.approx(0.91)
    assert body["reason"] is None


def test_route_returns_200_for_negative_detection(monkeypatch):
    scripted = ContainerDetection(
        detected=False,
        reason="wall and floor only, no container in frame",
        confidence=0.02,
    )
    _override_service(ContainerCheckService(client=_FakeClient(scripted)))

    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["detected"] is False
    assert body["container_type"] is None
    assert body["bbox"] is None
    assert "no container" in body["reason"]


def test_route_returns_422_on_missing_file():
    """FastAPI raises 422 when the required `image` field is missing.
    The UI keys off the status code to render a "no file uploaded"
    diagnostic separately from the 503 transient-failure path."""
    client = TestClient(app)
    res = client.post("/v1/detect-container")
    assert res.status_code == 422, res.text


def test_route_returns_503_when_service_raises_extractor_unavailable():
    """An upstream timeout / connection failure must surface as 503 with
    the structured envelope. The mobile UI maps this to "tap to retry"
    rather than dumping a stack trace."""

    class _Boom:
        def detect(self, image_bytes, media_type="image/png"):
            raise ExtractorUnavailable("upstream rate limit (429)")

    _override_service(_Boom())

    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": _png_file()},
    )
    assert res.status_code == 503, res.text
    body = res.json()
    assert body["detail"]["code"] == "vision_unavailable"
    assert "rate limit" in body["detail"]["message"]


def test_route_returns_400_on_empty_upload(monkeypatch):
    """Defensive: a zero-byte multipart part must 400 rather than crash
    the service layer."""
    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": ("frame.png", io.BytesIO(b""), "image/png")},
    )
    assert res.status_code == 400, res.text


def test_route_returns_400_on_non_image_content_type():
    """The route validates `image/*` so a misrouted upload (text, PDF)
    fails fast rather than spending an Anthropic call on garbage bytes."""
    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": ("frame.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert res.status_code == 400, res.text
    assert "image/" in res.json()["detail"]


def test_route_returns_413_on_oversize_upload(monkeypatch):
    """The route caps uploads at `settings.max_image_bytes` so a giant
    POST cannot blow worker RSS — same envelope as /v1/verify."""
    from app.config import settings

    monkeypatch.setattr(settings, "max_image_bytes", 1024)

    big = b"x" * 4096
    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": ("big.png", io.BytesIO(big), "image/png")},
    )
    assert res.status_code == 413, res.text
    assert res.json()["detail"]["code"] == "image_too_large"


# ---------------------------------------------------------------------------
# Known-label recognition response shape
# ---------------------------------------------------------------------------


def test_route_returns_image_dhash_on_detected_true(monkeypatch):
    """Every detect-container response should carry the upload's dhash
    as a 16-char lowercase hex string. The mobile UI forwards this to
    /v1/scans/{id}/finalize so enrich_verdict can stamp it onto the L3
    row."""
    scripted = ContainerDetection(
        detected=True,
        container_type="bottle",
        bbox=(0.1, 0.1, 0.9, 0.9),
        confidence=0.9,
    )
    _override_service(ContainerCheckService(client=_FakeClient(scripted)))

    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["image_dhash"] is not None
    assert isinstance(body["image_dhash"], str)
    # 16-char lowercase hex format from signature_to_hex.
    assert len(body["image_dhash"]) <= 16
    assert body["image_dhash"] == body["image_dhash"].lower()


def test_route_returns_known_label_null_on_miss(monkeypatch):
    """No L3 cache configured → known_label is null. Detect-container
    must always return the field; null on a miss, populated on a hit."""
    scripted = ContainerDetection(
        detected=True,
        container_type="bottle",
        bbox=(0.1, 0.1, 0.9, 0.9),
        confidence=0.9,
        brand_name="Unknown Brand",
        net_contents="12 FL OZ",
    )
    _override_service(ContainerCheckService(client=_FakeClient(scripted)))

    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["known_label"] is None
    assert body["brand_name"] == "Unknown Brand"
    assert body["net_contents"] == "12 FL OZ"


def test_route_returns_known_label_null_on_undetected(monkeypatch):
    """detected=False → known_label is null even if the rest of the
    response is well-formed (no bbox, no brand_name). The recognition
    block short-circuits before any L3 work."""
    scripted = ContainerDetection(
        detected=False, reason="appears to be a selfie"
    )
    _override_service(ContainerCheckService(client=_FakeClient(scripted)))

    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["known_label"] is None
    assert body["brand_name"] is None
    assert body["net_contents"] is None


def test_route_returns_known_label_on_brand_hit(
    monkeypatch, db_setup, tmp_path
):
    """Seed the L3 cache with a brand match, then assert the
    detect-container response carries the assembled known_label payload
    with `source: "brand"` and a freshly-evaluated verdict_summary."""
    from app.api import verify as verify_api
    from app.config import settings
    from app.rules.types import ExtractedField
    from app.services.persisted_cache import PersistedLabelCache
    from app.services.vision import VisionExtraction

    # Enable the L3 cache and reset its singleton so the test's
    # SQLite-backed instance is the one detect-container queries.
    monkeypatch.setattr(settings, "persisted_label_cache_enabled", True)
    verify_api._reset_persisted_label_cache()

    async def _seed():
        cache = PersistedLabelCache(hamming_threshold=6)
        extraction = VisionExtraction(
            fields={
                "brand_name": ExtractedField(
                    value="Sierra Nevada",
                    bbox=None,
                    confidence=0.95,
                ),
                "net_contents": ExtractedField(
                    value="12 FL OZ",
                    bbox=None,
                    confidence=0.95,
                ),
                "country_of_origin": ExtractedField(
                    value="USA", bbox=None, confidence=0.9
                ),
            },
            unreadable=[],
            raw_response="{}",
            image_quality="good",
            beverage_type_observed="beer",
        )
        return await cache.upsert(
            signature=(0xCAFE,),
            beverage_type="beer",
            extraction=extraction,
        )

    import asyncio as _asyncio

    _asyncio.run(_seed())

    scripted = ContainerDetection(
        detected=True,
        container_type="bottle",
        bbox=(0.1, 0.1, 0.9, 0.9),
        confidence=0.9,
        brand_name="Sierra Nevada",
        net_contents="12 FL OZ",
    )
    _override_service(ContainerCheckService(client=_FakeClient(scripted)))

    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["detected"] is True
    assert body["known_label"] is not None
    payload = body["known_label"]
    assert payload["source"] == "brand"
    assert payload["beverage_type"] == "beer"
    assert payload["container_size_ml"] == 355  # parsed from "12 FL OZ"
    assert payload["brand_name"] == "Sierra Nevada"
    assert payload["is_imported"] is False  # USA in country_of_origin
    summary = payload["verdict_summary"]
    assert summary["overall"] in {"pass", "warn", "fail", "advisory", "unreadable"}
    assert isinstance(summary["rule_results"], list)
    # The rule engine ran fresh — every result has a status, not a
    # frozen "verified" tag.
    for rr in summary["rule_results"]:
        assert "status" in rr
        assert "rule_id" in rr


# ---------------------------------------------------------------------------
# Shadow-mode inference seam (MODEL_INTEGRATION_PLAN §3.2)
# ---------------------------------------------------------------------------


def test_route_fires_shadow_predict_on_detected_response(monkeypatch):
    """The detect-container route MUST invoke `_spawn_shadow_predict`
    AFTER the response is assembled. The shadow prediction's output is
    log-only — never reflected in the body — so this test asserts the
    spawn happens (with the right inputs) and that the response shape
    is unchanged."""
    import app.api.detect as detect_module

    captured: dict = {}

    def _fake_spawn(image_bytes: bytes, vlm_brand: str | None) -> None:
        captured["image_bytes_len"] = len(image_bytes)
        captured["vlm_brand"] = vlm_brand

    monkeypatch.setattr(detect_module, "_spawn_shadow_predict", _fake_spawn)

    scripted = ContainerDetection(
        detected=True,
        container_type="bottle",
        bbox=(0.1, 0.1, 0.9, 0.9),
        confidence=0.9,
        brand_name="Anytown Ale",
    )
    _override_service(ContainerCheckService(client=_FakeClient(scripted)))

    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text
    # Spawn was called with the upload bytes + the VLM-extracted brand.
    assert captured["image_bytes_len"] > 0
    assert captured["vlm_brand"] == "Anytown Ale"
    # Response shape unchanged — no `shadow_*` field leaked into the
    # body, regardless of spawn outcome.
    body = res.json()
    assert "shadow_prediction" not in body
    assert "shadow_predicted_brand" not in body


def test_route_fires_shadow_predict_on_negative_response(monkeypatch):
    """Even on `detected=False`, the shadow seam still fires (so the
    agreement-rate metric has a denominator that reflects all traffic).
    `vlm_brand` is None when the VLM rejected the frame."""
    import app.api.detect as detect_module

    captured: dict = {}

    def _fake_spawn(image_bytes: bytes, vlm_brand: str | None) -> None:
        captured["called"] = True
        captured["vlm_brand"] = vlm_brand

    monkeypatch.setattr(detect_module, "_spawn_shadow_predict", _fake_spawn)

    scripted = ContainerDetection(
        detected=False,
        reason="no container in frame",
        confidence=0.05,
    )
    _override_service(ContainerCheckService(client=_FakeClient(scripted)))

    client = TestClient(app)
    res = client.post(
        "/v1/detect-container",
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text
    assert captured.get("called") is True
    assert captured.get("vlm_brand") is None


def test_spawn_shadow_predict_runs_without_blocking(monkeypatch):
    """Direct test: `_spawn_shadow_predict` must return immediately and
    the daemon thread must run the prediction + telemetry. The caller
    can't race-detect the thread, so we use a Lock the predict path
    releases to confirm it ran."""
    import threading as _threading

    import app.api.detect as detect_module
    import app.services.shadow_model as sm

    ran = _threading.Event()

    def _fake_predict(_b: bytes, _bev: str | None):
        ran.set()
        return sm.ShadowPrediction()

    monkeypatch.setattr(sm, "shadow_predict", _fake_predict)
    detect_module._spawn_shadow_predict(b"x" * 64, "ipa-co")
    # The daemon thread should run shortly after spawn. A 2-second
    # ceiling keeps a regression in the threading wire-up from
    # silently passing.
    assert ran.wait(timeout=2.0), (
        "shadow_predict did not run inside the daemon thread"
    )


def test_spawn_shadow_predict_swallows_exception(monkeypatch):
    """A future model that raises in the daemon thread must NOT
    propagate. `_timed_shadow_predict` catches; the spawn returns
    nothing; the route response is unchanged."""
    import threading as _threading

    import app.api.detect as detect_module
    import app.services.shadow_model as sm

    ran = _threading.Event()

    def _fake_predict(_b: bytes, _bev: str | None):
        ran.set()
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(sm, "shadow_predict", _fake_predict)
    # Should not raise, even though `shadow_predict` itself raises.
    detect_module._spawn_shadow_predict(b"x" * 64, "ipa-co")
    assert ran.wait(timeout=2.0)
