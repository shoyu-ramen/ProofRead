"""Tests for the centralised Anthropic client factory + resilience wrapper.

Bounded latency, predictable retries, single domain exception are the
properties we want — exercised here with stubs so no API key is required.
"""

from __future__ import annotations

import anthropic
import pytest

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


def _raise(exc_cls, *, message: str = "boom", body: dict | None = None):
    request = _FakeRequest()
    if exc_cls is anthropic.APIConnectionError:
        return exc_cls(request=request)
    if exc_cls is anthropic.APITimeoutError:
        return exc_cls(request=request)
    body = body if body is not None else {"error": {"message": message}}
    status = {
        anthropic.RateLimitError: 429,
        anthropic.InternalServerError: 500,
        anthropic.BadRequestError: 400,
        anthropic.AuthenticationError: 401,
        anthropic.PermissionDeniedError: 403,
    }.get(exc_cls)
    if status is None:
        raise AssertionError(f"unhandled exc_cls {exc_cls}")
    return exc_cls(message=message, response=_FakeResponse(status), body=body)


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
    """A 400 caused by a malformed request (non-billing) is a code bug we
    want to fail loud on, not silently fall back to a backup that would
    hit the same problem."""

    def _boom(**kwargs):
        raise _raise(anthropic.BadRequestError, message="messages.0: invalid")

    with pytest.raises(anthropic.APIStatusError):
        call_with_resilience(_boom)


@pytest.mark.parametrize(
    "billing_message",
    [
        "Your credit balance is too low to access the Anthropic API.",
        "credit balance is too low",
        "insufficient credit on this account",
        "Billing error: please update your payment method",
    ],
)
def test_call_with_resilience_translates_billing_errors_to_extractor_unavailable(
    billing_message,
):
    """Anthropic returns out-of-funds as a 400 BadRequestError. Without
    translation, the chain wouldn't fall through to the local-model leg,
    so a deploy with an out-of-funds key would 500 instead of degrading
    gracefully."""

    def _boom(**kwargs):
        raise _raise(anthropic.BadRequestError, message=billing_message)

    with pytest.raises(ExtractorUnavailable):
        call_with_resilience(_boom)


def test_call_with_resilience_inspects_body_for_billing_keyword():
    """Some SDK versions surface only the body's error.message rather than
    embedding it in the exception's stringification — the helper has to
    look at both."""

    def _boom(**kwargs):
        raise _raise(
            anthropic.BadRequestError,
            message="bad request",
            body={
                "error": {
                    "type": "invalid_request_error",
                    "message": "Your credit balance is too low.",
                }
            },
        )

    with pytest.raises(ExtractorUnavailable):
        call_with_resilience(_boom)


@pytest.mark.parametrize(
    "exc_cls",
    [anthropic.AuthenticationError, anthropic.PermissionDeniedError],
)
def test_call_with_resilience_translates_auth_errors_to_extractor_unavailable(
    exc_cls,
):
    """An invalid/expired/revoked key is environmental, not a code bug —
    the fallback should fire so the rest of the deploy keeps serving."""

    def _boom(**kwargs):
        raise _raise(exc_cls, message="invalid x-api-key")

    with pytest.raises(ExtractorUnavailable):
        call_with_resilience(_boom)
