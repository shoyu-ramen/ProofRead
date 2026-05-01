"""Tests for `scripts/verify_latency.py --probe-cache-write`.

The probe is the smoke check we run after a deploy. It should:

  1. Wait for /healthz before issuing the verify calls.
  2. Issue a cold + warm verify with the same image bytes.
  3. Pass when the warm call reports `cache_hit=true` AND the
     cold→warm savings exceed the configured threshold.
  4. Fail (exit 6) on any of: transport error, non-200, savings below
     threshold, or warm without cache_hit.

These tests use httpx's `MockTransport` so we never make a real HTTP
call — a probe failure here is a regression in the script's logic, not
a flaky network.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_latency.py"


@pytest.fixture(scope="module")
def latency_module():
    """Load `verify_latency.py` as a module under a stable name so tests
    can address its functions even though the file lives outside the
    `app/` package tree."""
    spec = importlib.util.spec_from_file_location("verify_latency", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_latency"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_transport(*, cold_ms: int, warm_ms: int, warm_cache_hit: bool = True):
    """Synthesize an httpx MockTransport that simulates a healthy server.

    Sequence is positional: first /healthz returns 200 immediately; the
    next two POSTs return cold + warm payloads.
    """
    state = {"calls": 0, "healthz": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/healthz"):
            state["healthz"] = True
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path.endswith("/v1/verify"):
            state["calls"] += 1
            if state["calls"] == 1:
                # Cold response — simulate a real server elapsed time.
                return httpx.Response(
                    200,
                    json={
                        "overall": "pass",
                        "rule_results": [],
                        "extracted": {},
                        "unreadable_fields": [],
                        "elapsed_ms": cold_ms,
                        "cache_hit": False,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "overall": "pass",
                    "rule_results": [],
                    "extracted": {},
                    "unreadable_fields": [],
                    "elapsed_ms": warm_ms,
                    "cache_hit": warm_cache_hit,
                },
            )
        return httpx.Response(404)

    return handler, state


def test_probe_passes_when_savings_exceed_threshold(monkeypatch, latency_module):
    """Realistic deploy scenario: cold ~3.5s, warm ~25ms (cache hit).
    Savings = 3475ms, well above the 150ms default threshold."""
    handler, state = _build_transport(cold_ms=3500, warm_ms=25)
    transport = httpx.MockTransport(handler)

    # Patch httpx.AsyncClient to use our mock transport in the probe runner.
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs.pop("limits", None)
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(latency_module.httpx, "AsyncClient", fake_client)

    cold, warm, hz = asyncio.run(
        latency_module._run_cache_write_probe(
            base="https://example.test",
            image_bytes=b"\x89PNG\r\n\x1a\nfake",
            image_name="probe.png",
            media_type="image/png",
            beverage_type="spirits",
            container_size_ml=750,
            is_imported=False,
            timeout=5.0,
            healthz_timeout_s=2.0,
        )
    )

    assert state["healthz"] is True
    assert cold is not None and cold.status == 200
    assert warm is not None and warm.status == 200
    assert warm.cache_hit is True
    passed = latency_module._print_probe(cold, warm, hz, min_savings_ms=150)
    assert passed is True


def test_probe_fails_when_savings_below_threshold(monkeypatch, latency_module):
    """Cold and warm both ~50ms — savings only 0-ish, below the
    150ms threshold. The whole point of the probe is to catch the
    case where the cache silently failed to write."""
    handler, _ = _build_transport(cold_ms=50, warm_ms=45)
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs.pop("limits", None)
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(latency_module.httpx, "AsyncClient", fake_client)

    cold, warm, hz = asyncio.run(
        latency_module._run_cache_write_probe(
            base="https://example.test",
            image_bytes=b"\x89PNG\r\n\x1a\nfake",
            image_name="probe.png",
            media_type="image/png",
            beverage_type="spirits",
            container_size_ml=750,
            is_imported=False,
            timeout=5.0,
            healthz_timeout_s=2.0,
        )
    )

    passed = latency_module._print_probe(cold, warm, hz, min_savings_ms=150)
    assert passed is False


def test_probe_fails_when_warm_misses_cache(monkeypatch, latency_module):
    """Even with massive savings (cold→warm), if the warm call doesn't
    report `cache_hit=true`, the probe fails: that's a determinism bug
    or a cache-key drift, not "the cache worked"."""
    handler, _ = _build_transport(cold_ms=3500, warm_ms=25, warm_cache_hit=False)
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs.pop("limits", None)
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(latency_module.httpx, "AsyncClient", fake_client)

    cold, warm, hz = asyncio.run(
        latency_module._run_cache_write_probe(
            base="https://example.test",
            image_bytes=b"\x89PNG\r\n\x1a\nfake",
            image_name="probe.png",
            media_type="image/png",
            beverage_type="spirits",
            container_size_ml=750,
            is_imported=False,
            timeout=5.0,
            healthz_timeout_s=2.0,
        )
    )

    passed = latency_module._print_probe(cold, warm, hz, min_savings_ms=150)
    assert passed is False


def test_probe_fails_when_healthz_never_ready(monkeypatch, latency_module):
    """If /healthz never returns 200 before the timeout, the probe
    must report failure rather than charge ahead with calls that would
    obviously fail anyway."""

    def handler(request: httpx.Request) -> httpx.Response:
        # /healthz always 503 — instance never ready
        return httpx.Response(503)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs.pop("limits", None)
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(latency_module.httpx, "AsyncClient", fake_client)

    cold, warm, hz = asyncio.run(
        latency_module._run_cache_write_probe(
            base="https://example.test",
            image_bytes=b"\x89PNG\r\n\x1a\nfake",
            image_name="probe.png",
            media_type="image/png",
            beverage_type="spirits",
            container_size_ml=750,
            is_imported=False,
            timeout=5.0,
            # Tight healthz timeout so the test runs fast.
            healthz_timeout_s=0.5,
        )
    )

    assert cold is None
    assert warm is None
    assert hz >= 0.5
    passed = latency_module._print_probe(cold, warm, hz, min_savings_ms=150)
    assert passed is False
