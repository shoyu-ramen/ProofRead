"""Tests for ``GET /v1/admin/cache-health``.

Pins the auth contract (503 when ADMIN_API_TOKEN is unset, 401 for a
wrong token, 200 for a correct token), the response schema, and the
counter increments after a real verify call. The verify call is run
through the same ``MockVisionExtractor``-based fixture pattern as
``test_verify_api.py`` so we don't need a live Anthropic key.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from app.api import verify as verify_api
from app.config import settings
from app.main import app
from app.services.external.adapter import _clear_registry_for_tests
from app.services.external.ttb_cola import TTBColaAdapter
from app.services.vision import MockVisionExtractor
from tests.conftest import _make_synthetic_png


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_caches_and_extractor():
    """Reset every process-wide cache + extractor between tests.

    Same shape as the autouse fixture in ``test_verify_api.py``: a
    leftover entry would short-circuit the cold-path assertions and
    silently break the counter-increment test.
    """
    verify_api._extractor_cache = None
    verify_api._reset_verify_cache()
    verify_api._reset_reverse_lookup_cache()
    verify_api._reset_persisted_label_cache()
    _clear_registry_for_tests()
    yield
    verify_api._extractor_cache = None
    verify_api._reset_verify_cache()
    verify_api._reset_reverse_lookup_cache()
    verify_api._reset_persisted_label_cache()
    _clear_registry_for_tests()
    app.dependency_overrides.clear()


_PASS_FIXTURE = {
    "brand_name": "Old Tom Distillery",
    "class_type": "Kentucky Straight Bourbon Whiskey",
    "alcohol_content": "45% Alc./Vol.",
    "net_contents": "750 mL",
    "name_address": (
        "Bottled by Old Tom Distilling Co., Bardstown, Kentucky"
    ),
    "health_warning": (
        "GOVERNMENT WARNING: (1) According to the Surgeon General, women "
        "should not drink alcoholic beverages during pregnancy because of "
        "the risk of birth defects. (2) Consumption of alcoholic beverages "
        "impairs your ability to drive a car or operate machinery, and may "
        "cause health problems."
    ),
}


def _form_payload() -> dict[str, object]:
    return {
        "beverage_type": "spirits",
        "container_size_ml": "750",
        "is_imported": "false",
        "application": json.dumps(
            {
                "producer_record": {
                    "brand_name": "Old Tom Distillery",
                    "class_type": "Kentucky Straight Bourbon Whiskey",
                    "alcohol_content": "45",
                    "net_contents": "750 mL",
                    "name_address": (
                        "Old Tom Distilling Co., Bardstown, Kentucky"
                    ),
                    "country_of_origin": "USA",
                }
            }
        ),
    }


def _png_file() -> tuple[str, io.BytesIO, str]:
    return ("label.png", io.BytesIO(_make_synthetic_png()), "image/png")


# ---------------------------------------------------------------------------
# Auth contract
# ---------------------------------------------------------------------------


def test_admin_cache_health_503_when_token_unset(monkeypatch):
    """Unset ``ADMIN_API_TOKEN`` → 503 ``admin_disabled``.

    Distinct from 401: an unset token is a configuration state, not
    an authentication failure. The response code lets a deploy
    pipeline distinguish "I haven't enabled admin yet" from "I'm
    sending the wrong header".
    """
    monkeypatch.setattr(settings, "admin_api_token", None)
    client = TestClient(app)
    res = client.get("/v1/admin/cache-health")
    assert res.status_code == 503, res.text
    body = res.json()
    assert body["detail"]["code"] == "admin_disabled"


def test_admin_cache_health_401_when_token_wrong(monkeypatch):
    """Token configured + wrong header → 401."""
    monkeypatch.setattr(settings, "admin_api_token", "expected-token")
    client = TestClient(app)
    res = client.get(
        "/v1/admin/cache-health",
        headers={"X-Admin-Token": "wrong"},
    )
    assert res.status_code == 401, res.text
    body = res.json()
    assert body["detail"]["code"] == "admin_unauthorized"


def test_admin_cache_health_401_when_token_missing(monkeypatch):
    """Token configured but header absent → 401."""
    monkeypatch.setattr(settings, "admin_api_token", "expected-token")
    client = TestClient(app)
    res = client.get("/v1/admin/cache-health")
    assert res.status_code == 401, res.text


def test_admin_cache_health_200_with_correct_token(monkeypatch):
    """Correct token → 200 with the documented schema."""
    monkeypatch.setattr(settings, "admin_api_token", "expected-token")
    client = TestClient(app)
    res = client.get(
        "/v1/admin/cache-health",
        headers={"X-Admin-Token": "expected-token"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert set(body.keys()) == {"l1", "l2", "l3", "ttb_cola"}
    # ttb_cola is always present (zero-state when the adapter isn't
    # registered) so the dashboard never has to handle a missing key.
    ttb = body["ttb_cola"]
    assert {
        "enabled",
        "last_request_at",
        "request_count",
        "error_count",
        "circuit_open",
    }.issubset(ttb.keys())


# ---------------------------------------------------------------------------
# Schema details on the cache tiers
# ---------------------------------------------------------------------------


def test_l1_l2_payloads_present_after_verify(monkeypatch):
    """A real verify call constructs the lazy L1/L2 caches.

    Before any verify request, both caches are ``None`` (lazy init).
    After one cold call they exist, and the admin payload reports
    ``size=1`` for L1 and a hit/miss count consistent with the
    flow.
    """
    monkeypatch.setattr(settings, "admin_api_token", "expected-token")
    monkeypatch.setattr(
        verify_api,
        "get_default_extractor",
        lambda: MockVisionExtractor(_PASS_FIXTURE),
    )
    # Disable persisted cache + explanations + TTB so the cold path
    # doesn't reach for the DB / Anthropic / network and stays
    # hermetic in CI.
    monkeypatch.setattr(settings, "persisted_label_cache_enabled", False)
    monkeypatch.setattr(settings, "explanation_enabled", False)
    monkeypatch.setattr(settings, "ttb_cola_lookup_enabled", False)

    client = TestClient(app)
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text

    health = client.get(
        "/v1/admin/cache-health",
        headers={"X-Admin-Token": "expected-token"},
    )
    assert health.status_code == 200, health.text
    body = health.json()
    # L1 was constructed and now holds the cold-path entry.
    assert body["l1"] is not None
    assert body["l1"]["size"] >= 1
    assert body["l1"]["misses"] >= 1
    # L2 also constructed; one miss recorded by the orchestrator.
    assert body["l2"] is not None
    assert body["l2"]["size"] >= 0


def test_counters_increment_on_repeat_verify(monkeypatch):
    """Two identical verify calls → L1 hits counter advances.

    Pins the integration: the admin endpoint reflects the in-process
    counters that the verify path bumps. A regression that swapped
    the L1 cache for a non-counting impl would surface here.
    """
    monkeypatch.setattr(settings, "admin_api_token", "expected-token")
    monkeypatch.setattr(
        verify_api,
        "get_default_extractor",
        lambda: MockVisionExtractor(_PASS_FIXTURE),
    )
    monkeypatch.setattr(settings, "persisted_label_cache_enabled", False)
    monkeypatch.setattr(settings, "explanation_enabled", False)
    monkeypatch.setattr(settings, "ttb_cola_lookup_enabled", False)

    client = TestClient(app)

    # Cold call.
    image_bytes = _make_synthetic_png()

    def _png() -> tuple[str, io.BytesIO, str]:
        return ("label.png", io.BytesIO(image_bytes), "image/png")

    r1 = client.post(
        "/v1/verify", data=_form_payload(), files={"image": _png()}
    )
    assert r1.status_code == 200, r1.text

    # Warm call (identical bytes + form) — should hit L1.
    r2 = client.post(
        "/v1/verify", data=_form_payload(), files={"image": _png()}
    )
    assert r2.status_code == 200, r2.text

    health = client.get(
        "/v1/admin/cache-health",
        headers={"X-Admin-Token": "expected-token"},
    )
    body = health.json()
    assert body["l1"]["hits"] >= 1, body["l1"]


def test_l3_payload_disabled_when_flag_off(monkeypatch):
    """Persisted cache flag off → ``{"enabled": false}`` on L3."""
    monkeypatch.setattr(settings, "admin_api_token", "expected-token")
    monkeypatch.setattr(settings, "persisted_label_cache_enabled", False)
    client = TestClient(app)
    res = client.get(
        "/v1/admin/cache-health",
        headers={"X-Admin-Token": "expected-token"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["l3"] == {"enabled": False}


def test_ttb_cola_stats_when_adapter_registered(monkeypatch):
    """When the TTB adapter is in the registry, its stats() drives
    the ``ttb_cola`` payload — including the lifetime request and
    error counters.
    """
    monkeypatch.setattr(settings, "admin_api_token", "expected-token")
    monkeypatch.setattr(settings, "ttb_cola_lookup_enabled", True)
    # Register an adapter into the (cleared-by-fixture) registry so
    # the admin endpoint can find it.
    from app.services.external import register_adapter

    register_adapter(TTBColaAdapter())

    client = TestClient(app)
    res = client.get(
        "/v1/admin/cache-health",
        headers={"X-Admin-Token": "expected-token"},
    )
    body = res.json()
    ttb = body["ttb_cola"]
    assert ttb["enabled"] is True
    # Zero-state before any lookup happens.
    assert ttb["request_count"] == 0
    assert ttb["error_count"] == 0
    assert ttb["circuit_open"] is False
    assert ttb["last_request_at"] is None
