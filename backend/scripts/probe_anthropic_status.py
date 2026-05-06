"""Probe ANTHROPIC_API_KEY against the live API to confirm fallback trigger.

Issues a minimal `messages.create` call with the same key the deployed
service uses (read from the env). Reports whether the call succeeds or
which exception surfaces — useful for confirming an "out of funds" /
billing failure is actually translating to ExtractorUnavailable.

Usage:
    railway run -- python backend/scripts/probe_anthropic_status.py
"""

from __future__ import annotations

import sys

import anthropic

from app.config import settings
from app.services.anthropic_client import (
    ExtractorUnavailable,
    build_client,
    call_with_resilience,
)


def main() -> int:
    print("=== Anthropic key probe ===")
    print(f"  model = {settings.anthropic_model}")
    print(f"  key   = {(settings.anthropic_api_key or '<unset>')[:10]}…")

    try:
        client = build_client(timeout=10.0, max_retries=0)
    except ExtractorUnavailable as exc:
        print(f"build_client raised ExtractorUnavailable: {exc}")
        return 1

    def _call(**_kw):
        return client.messages.create(
            model=settings.anthropic_model,
            max_tokens=8,
            messages=[{"role": "user", "content": "ping"}],
        )

    try:
        response = call_with_resilience(_call)
    except ExtractorUnavailable as exc:
        print(f"PRIMARY DOWN — call_with_resilience raised ExtractorUnavailable")
        print(f"  underlying cause: {exc.__cause__ and type(exc.__cause__).__name__}")
        print(f"  message: {exc}")
        print()
        print("=> Fallback chain WILL trigger on /v1/verify in production.")
        return 0
    except anthropic.APIStatusError as exc:
        print(f"PRIMARY DOWN — APIStatusError {exc.status_code} surfaced (NOT translated)")
        print(f"  body: {getattr(exc, 'body', None)}")
        print(f"  message: {exc}")
        print()
        print("=> Fallback chain will NOT trigger; this error needs translation.")
        return 1
    print("PRIMARY OK — Anthropic responded.")
    print(f"  response.stop_reason = {response.stop_reason}")
    print()
    print("=> Fallback chain will only trigger if Anthropic later fails.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
