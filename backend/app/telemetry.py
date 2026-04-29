"""Sentry + OpenTelemetry wiring.

Three contracts:

  1. **Local dev / CI: zero-config no-op.** Both `init_sentry` and
     `init_otel` are safe to call when the corresponding env vars are
     unset; they return early and the rest of the app behaves identically.
     The optional dependency packages (sentry-sdk, opentelemetry-*) are
     imported lazily so a missing install doesn't crash startup either.

  2. **Trace-id correlation with Phase-1 logs.** `current_trace_id()`
     returns the active OTel trace id formatted as the standard 32-char
     hex string, or `None` when tracing is off. The Phase-1 verify
     structured-log line will pull this and append it so log lines and
     spans correlate one-to-one.

  3. **Span helpers stay minimal.** `traced_span(name, **attrs)` is a
     context-manager that creates a span when OTel is configured and is
     a no-op `nullcontext` otherwise. Callers don't need to gate on
     `otel_enabled` themselves — the helper handles both states with the
     same call site.

NO image bytes in span attributes. The verify pipeline carries
`image_bytes` through nearly every layer; an accidental attribute set
would push PII / proprietary artwork to Honeycomb. Helpers below only
accept primitive attribute values; a `bytes` attribute is dropped at
the helper boundary, never the SDK boundary.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Any

logger = logging.getLogger(__name__)


# Module-level state — set by the init functions, read by the helpers.
_sentry_initialized = False
_otel_tracer: Any | None = None


def init_sentry() -> bool:
    """Initialize the Sentry SDK if `SENTRY_DSN` is set.

    Returns True when init actually fired; False when it was a no-op
    (missing DSN or missing dependency).
    """
    global _sentry_initialized
    if _sentry_initialized:
        return True

    from app.config import settings

    if not settings.sentry_dsn:
        return False

    try:
        import sentry_sdk
    except ImportError:
        logger.warning(
            "SENTRY_DSN is set but sentry-sdk is not installed; "
            "skipping init. Add the `telemetry` extra to enable."
        )
        return False

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.deploy_environment,
        release=settings.deploy_release,
        # Default samples — Sentry's free tier is small; stay conservative
        # until traffic justifies tuning. Errors always sampled at 1.0.
        traces_sample_rate=0.05,
        # Don't ship request bodies (image multipart is too large + has
        # PII/proprietary artwork). Default off; the FastAPI integration
        # captures URL + status code which is enough for triage.
        send_default_pii=False,
    )
    _sentry_initialized = True
    logger.info(
        "Sentry initialized (env=%s release=%s)",
        settings.deploy_environment,
        settings.deploy_release,
    )
    return True


def capture_exception(exc: BaseException, **tags: str) -> None:
    """Best-effort capture to Sentry.

    Used by the verify orchestrator's swallowed-exception paths
    (`_secondary_for_panel`) where we want a counter bump locally AND a
    Sentry breadcrumb so a spike is visible in the dashboard. No-op when
    Sentry is not initialized.
    """
    if not _sentry_initialized:
        return
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for k, v in tags.items():
                scope.set_tag(k, v)
            sentry_sdk.capture_exception(exc)
    except Exception as scope_exc:  # pragma: no cover — defensive
        logger.debug("Sentry capture failed: %s", scope_exc)


def init_otel(app: Any | None = None) -> bool:
    """Initialize OTel tracing → Honeycomb if both env vars are set.

    `app` is the FastAPI instance; when supplied AND OTel init succeeds,
    the FastAPI auto-instrumentor is attached so every HTTP route is
    automatically a span.

    Returns True when init actually fired; False when it was a no-op.
    """
    global _otel_tracer
    if _otel_tracer is not None:
        return True

    from app.config import settings

    if not settings.honeycomb_api_key or not settings.otel_exporter_otlp_endpoint:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "Honeycomb env vars set but opentelemetry packages not "
            "installed; skipping init. Add the `telemetry` extra to enable."
        )
        return False

    resource = Resource.create(
        {
            "service.name": "proofread-backend",
            "deployment.environment": settings.deploy_environment,
            **(
                {"service.version": settings.deploy_release}
                if settings.deploy_release
                else {}
            ),
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint,
        headers={"x-honeycomb-team": settings.honeycomb_api_key},
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _otel_tracer = trace.get_tracer("proofread-backend")

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import (
                FastAPIInstrumentor,
            )

            FastAPIInstrumentor.instrument_app(app)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("FastAPI auto-instrumentation skipped: %s", exc)

    logger.info(
        "OpenTelemetry initialized → %s (env=%s)",
        settings.otel_exporter_otlp_endpoint,
        settings.deploy_environment,
    )
    return True


@contextmanager
def traced_span(name: str, **attrs: Any):
    """Context-manager that opens an OTel span when tracing is on, no-op
    when off. Drops any `bytes` attribute values — they're not safe to
    ship to Honeycomb (PII / proprietary artwork)."""
    if _otel_tracer is None:
        with nullcontext():
            yield None
        return
    safe = {
        k: v
        for k, v in attrs.items()
        if not isinstance(v, (bytes, bytearray, memoryview))
    }
    with _otel_tracer.start_as_current_span(name, attributes=safe) as span:
        yield span


def current_trace_id() -> str | None:
    """Return the current OTel trace id as a 32-char hex string, or None
    when tracing is off / there is no active span."""
    if _otel_tracer is None:
        return None
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    span = trace.get_current_span()
    ctx = span.get_span_context() if span is not None else None
    if ctx is None or not ctx.is_valid:
        return None
    # Trace id is a 128-bit int; OTel formats as 32-char zero-padded hex.
    return f"{ctx.trace_id:032x}"


def _reset_for_tests() -> None:
    """Test hook: drop module state so init helpers can be re-run with
    different env settings. Production never calls this."""
    global _sentry_initialized, _otel_tracer
    _sentry_initialized = False
    _otel_tracer = None
