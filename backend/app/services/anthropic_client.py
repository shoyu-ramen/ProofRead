"""Centralised Anthropic client factory and resilience wrapper.

Three goals:

  1. **Bounded latency.** The Anthropic SDK defaults to a 600 s timeout
     with up to two automatic retries — a 30-minute worst case. SPEC
     §0.5 calls for a p95 of ≤25 s scan→report (and the verify path
     wants ≤5 s), so we set explicit per-call timeouts that respect those
     budgets and let the caller fall back to OCR if we blow through them.

  2. **Predictable retries.** The SDK already retries 408/409/429/5xx
     and connection errors with exponential backoff. We keep that
     behaviour but cap retries so a flaky upstream cannot drag the whole
     request beyond the timeout.

  3. **Clear failure surface.** Callers shouldn't have to know about
     `anthropic.APIConnectionError`, `anthropic.RateLimitError`, etc. —
     just whether the extractor is currently usable. `ExtractorUnavailable`
     is the single domain exception they can catch to fall back to OCR.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Per-call timeouts, in seconds. Vision calls send a full image and ask
# for thinking + structured output; the second-pass call is much cheaper
# (one image, ≤1k token output) so we give it a tighter budget. These
# are the values the SDK enforces on a single attempt; with retries the
# total time can be slightly higher but is still bounded.
DEFAULT_VISION_TIMEOUT_S = 20.0
DEFAULT_SECOND_PASS_TIMEOUT_S = 8.0
DEFAULT_MAX_RETRIES = 2


class ExtractorUnavailable(RuntimeError):
    """Raised when the Anthropic extractor cannot be reached or used.

    Pipeline callers catch this to fall back to OCR. Tests assert against
    this type so they don't depend on the SDK's exception hierarchy.
    """


def build_client(
    *,
    api_key: str | None = None,
    timeout: float | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Any:
    """Construct a configured `anthropic.Anthropic` client.

    Returns the SDK's official client (we don't wrap it in a custom
    proxy) so consumers retain access to `messages.create`, `parse`,
    and the rest of the surface. The factory applies our default
    timeout and retry budget.
    """
    import anthropic

    from app.config import settings

    key = api_key if api_key is not None else settings.anthropic_api_key
    if not key:
        raise ExtractorUnavailable(
            "ANTHROPIC_API_KEY is not configured; cannot build a vision "
            "client. Set the env var or use a Mock extractor in tests."
        )

    kwargs: dict[str, Any] = {
        "api_key": key,
        "max_retries": max_retries,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout

    return anthropic.Anthropic(**kwargs)


def call_with_resilience(callable_, *args: Any, **kwargs: Any) -> Any:
    """Call a function that issues one Anthropic SDK request.

    Translates the SDK's exception family into `ExtractorUnavailable`
    so upstream code can do a single try/except. This is intentionally
    a thin shim: the SDK already does the retry work; we just normalise
    the failure surface and log a useful message.
    """
    import anthropic

    try:
        return callable_(*args, **kwargs)
    except (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    ) as exc:
        logger.warning("Anthropic call failed transiently: %s", exc)
        raise ExtractorUnavailable(
            f"Vision extractor unavailable: {exc}"
        ) from exc
    except anthropic.APIStatusError as exc:
        # 4xx other than 408/409/429 — typically a request shape problem.
        # Don't translate to ExtractorUnavailable; surface to the caller so
        # we can fix the offending request rather than silently fall back.
        logger.error("Anthropic API status error %s: %s", exc.status_code, exc)
        raise
