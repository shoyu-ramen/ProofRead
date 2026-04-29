"""Tests for the Sentry + OpenTelemetry init wiring.

Both init helpers must:
- No-op silently when env vars are missing (local dev / CI default).
- No-op silently when packages are missing (telemetry extra not installed).
- Init exactly once per process when env vars + packages are present.

The tests use `monkeypatch` to flip the settings on/off and `unittest.mock`
to intercept the SDK init calls — no real Sentry / Honeycomb traffic.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import telemetry
from app.config import settings


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    """Each test starts with a fresh telemetry module state."""
    telemetry._reset_for_tests()
    yield
    telemetry._reset_for_tests()


def test_init_sentry_noop_when_dsn_missing(monkeypatch):
    monkeypatch.setattr(settings, "sentry_dsn", None)
    assert telemetry.init_sentry() is False


def test_init_sentry_noop_when_package_missing(monkeypatch):
    """A deploy that sets SENTRY_DSN but never installed the optional
    `telemetry` extra must not crash startup. Init silently returns
    False so the rest of the lifespan continues."""
    monkeypatch.setattr(settings, "sentry_dsn", "https://fake@sentry.io/123")
    # Force ImportError by stashing a sentinel that fails to import.
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    assert telemetry.init_sentry() is False


def test_init_sentry_calls_sdk_when_dsn_set(monkeypatch):
    """When DSN is set AND the package imports cleanly, the SDK's
    init() runs with our config (env, release, conservative sample
    rate, send_default_pii=False)."""
    monkeypatch.setattr(settings, "sentry_dsn", "https://fake@sentry.io/123")
    monkeypatch.setattr(settings, "deploy_environment", "test-env")
    monkeypatch.setattr(settings, "deploy_release", "abc123")

    fake_sdk = MagicMock()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake_sdk)

    assert telemetry.init_sentry() is True
    fake_sdk.init.assert_called_once()
    kwargs = fake_sdk.init.call_args.kwargs
    assert kwargs["dsn"] == "https://fake@sentry.io/123"
    assert kwargs["environment"] == "test-env"
    assert kwargs["release"] == "abc123"
    assert kwargs["send_default_pii"] is False
    # Conservative sample rate — keep us inside the free tier.
    assert 0.0 < kwargs["traces_sample_rate"] <= 0.1


def test_init_sentry_idempotent(monkeypatch):
    """Lifespan can be re-entered (graceful restart, test fixture
    teardown / setup) — init must not double-fire."""
    monkeypatch.setattr(settings, "sentry_dsn", "https://fake@sentry.io/123")
    fake_sdk = MagicMock()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake_sdk)

    assert telemetry.init_sentry() is True
    assert telemetry.init_sentry() is True
    fake_sdk.init.assert_called_once()


def test_capture_exception_noop_when_sentry_off(monkeypatch):
    """When Sentry isn't initialized, `capture_exception` must not
    attempt to import or call the SDK — keeps the verify orchestrator's
    swallow path identical when telemetry is off."""
    monkeypatch.setattr(settings, "sentry_dsn", None)
    # Should not raise, should not import sentry_sdk.
    telemetry.capture_exception(RuntimeError("test"))


def test_capture_exception_routes_to_sentry_when_initialized(monkeypatch):
    monkeypatch.setattr(settings, "sentry_dsn", "https://fake@sentry.io/123")
    fake_sdk = MagicMock()
    fake_scope = MagicMock()
    fake_sdk.push_scope.return_value.__enter__ = lambda *_: fake_scope
    fake_sdk.push_scope.return_value.__exit__ = lambda *_: None
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake_sdk)

    telemetry.init_sentry()
    exc = RuntimeError("boom")
    telemetry.capture_exception(exc, outcome="rate_limit", component="x")

    fake_sdk.capture_exception.assert_called_once_with(exc)
    # Tags should be set on the scope so dashboards can filter.
    set_tag_calls = [c.args for c in fake_scope.set_tag.call_args_list]
    assert ("outcome", "rate_limit") in set_tag_calls
    assert ("component", "x") in set_tag_calls


def test_init_otel_noop_when_keys_missing(monkeypatch):
    monkeypatch.setattr(settings, "honeycomb_api_key", None)
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", None)
    assert telemetry.init_otel(app=None) is False


def test_init_otel_noop_when_only_one_env_var(monkeypatch):
    """Both keys are required — set one but not the other and we still
    no-op so a partially-configured deploy doesn't ship traces nowhere."""
    monkeypatch.setattr(settings, "honeycomb_api_key", "fake-key")
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", None)
    assert telemetry.init_otel(app=None) is False

    monkeypatch.setattr(settings, "honeycomb_api_key", None)
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", "https://api.honeycomb.io")
    assert telemetry.init_otel(app=None) is False


def test_init_otel_noop_when_packages_missing(monkeypatch):
    monkeypatch.setattr(settings, "honeycomb_api_key", "fake-key")
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", "https://api.honeycomb.io")
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    assert telemetry.init_otel(app=None) is False


def test_traced_span_no_op_when_otel_off():
    """`traced_span` must work as a context manager when OTel isn't
    initialized — callers don't gate on a `if otel_enabled:` check."""
    with telemetry.traced_span("test.span", attr1="value", attr2=42) as span:
        assert span is None  # no real span when OTel off


def test_traced_span_drops_bytes_attrs(monkeypatch):
    """Image bytes must never end up as span attributes (PII /
    proprietary artwork). Helper drops them at the boundary."""
    fake_tracer = MagicMock()
    fake_span_cm = MagicMock()
    fake_span_cm.__enter__ = MagicMock(return_value=MagicMock())
    fake_span_cm.__exit__ = MagicMock(return_value=False)
    fake_tracer.start_as_current_span.return_value = fake_span_cm
    monkeypatch.setattr(telemetry, "_otel_tracer", fake_tracer)

    big_bytes = b"\x89PNG..." * 1000
    with telemetry.traced_span(
        "test.span", safe_attr="ok", image_bytes=big_bytes, also_bytes=bytearray(b"x")
    ):
        pass

    # The tracer should have been called WITHOUT the bytes attributes.
    call = fake_tracer.start_as_current_span.call_args
    attrs = call.kwargs.get("attributes", {})
    assert "safe_attr" in attrs
    assert "image_bytes" not in attrs
    assert "also_bytes" not in attrs


def test_current_trace_id_returns_none_when_off():
    assert telemetry.current_trace_id() is None


def test_current_trace_id_formats_hex_when_active(monkeypatch):
    """When OTel is on AND there's an active span, return the trace id
    as a 32-char zero-padded hex string (the standard wire format)."""
    fake_ctx = SimpleNamespace(is_valid=True, trace_id=0xDEADBEEFCAFEBABE)
    fake_span = MagicMock()
    fake_span.get_span_context.return_value = fake_ctx
    fake_tracer = MagicMock()
    monkeypatch.setattr(telemetry, "_otel_tracer", fake_tracer)

    fake_trace_module = MagicMock()
    fake_trace_module.get_current_span.return_value = fake_span
    monkeypatch.setitem(sys.modules, "opentelemetry", MagicMock(trace=fake_trace_module))
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", fake_trace_module)

    with patch.dict(sys.modules, {"opentelemetry": SimpleNamespace(trace=fake_trace_module)}):
        # Re-import path: our function does `from opentelemetry import trace`
        # at call time. The MagicMock-as-module above intercepts that.
        tid = telemetry.current_trace_id()

    assert tid == f"{0xDEADBEEFCAFEBABE:032x}"
    assert len(tid) == 32
