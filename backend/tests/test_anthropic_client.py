"""Tests for the centralised Anthropic client factory + resilience wrapper.

Bounded latency, predictable retries, single domain exception are the
properties we want — exercised here with stubs so no API key is required.
"""

from __future__ import annotations

import pytest

import anthropic

from app.services import anthropic_client
from app.services.anthropic_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_VISION_TIMEOUT_S,
    ExtractorUnavailable,
    build_client,
    call_with_resilience,
)


# ---------------------------------------------------------------------------
# build_client
# ---------------------------------------------------------------------------


def test_build_client_passes_through_timeout_and_retries(monkeypatch):
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(anthropic_client, "anthropic", anthropic, raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

    build_client(api_key="sk-test", timeout=5.0, max_retries=3)
    assert captured["api_key"] == "sk-test"
    assert captured["timeout"] == 5.0
    assert captured["max_retries"] == 3


def test_build_client_uses_settings_when_no_api_key_given(monkeypatch):
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-from-settings")
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

    build_client(timeout=DEFAULT_VISION_TIMEOUT_S)
    assert captured["api_key"] == "sk-from-settings"
    assert captured["max_retries"] == DEFAULT_MAX_RETRIES


def test_build_client_raises_extractor_unavailable_without_api_key(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", None)
    with pytest.raises(ExtractorUnavailable):
        build_client()


# ---------------------------------------------------------------------------
# call_with_resilience
# ---------------------------------------------------------------------------


def _stub_response():
    """Sentinel returned by the SDK on success — opaque object is fine."""
    return object()


def test_call_with_resilience_passes_through_on_success():
    sentinel = _stub_response()
    result = call_with_resilience(lambda **kw: sentinel, foo=1)
    assert result is sentinel


class _FakeRequest:
    """Minimum stub the anthropic exceptions accept as their `request` kw."""

    def __init__(self) -> None:
        self.method = "POST"
        self.url = "https://api.anthropic.com/v1/messages"


def _raise(exc_cls):
    request = _FakeRequest()
    if exc_cls is anthropic.APIConnectionError:
        return exc_cls(request=request)
    if exc_cls is anthropic.APITimeoutError:
        return exc_cls(request=request)
    body = {"error": {"message": "boom"}}
    if exc_cls is anthropic.RateLimitError:
        return anthropic.RateLimitError(
            message="rate limited", response=_FakeResponse(429), body=body
        )
    if exc_cls is anthropic.InternalServerError:
        return anthropic.InternalServerError(
            message="upstream blew up", response=_FakeResponse(500), body=body
        )
    if exc_cls is anthropic.BadRequestError:
        return anthropic.BadRequestError(
            message="bad request", response=_FakeResponse(400), body=body
        )
    raise AssertionError(f"unhandled exc_cls {exc_cls}")


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers = {}
        self.request = _FakeRequest()


@pytest.mark.parametrize(
    "exc_cls",
    [
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    ],
)
def test_call_with_resilience_translates_transient_errors(exc_cls):
    """Each transient SDK error type must surface as ExtractorUnavailable so
    pipeline callers can fall back without binding to the SDK's hierarchy."""

    def _boom(**kwargs):
        raise _raise(exc_cls)

    with pytest.raises(ExtractorUnavailable):
        call_with_resilience(_boom)


def test_call_with_resilience_does_not_swallow_4xx_request_errors():
    """A 400 (or other non-transient client error) is a request-shape bug we
    want to fail loud on, not silently fall back."""

    def _boom(**kwargs):
        raise _raise(anthropic.BadRequestError)

    with pytest.raises(anthropic.APIStatusError):
        call_with_resilience(_boom)
