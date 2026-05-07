"""Pytest configuration for the validation harness.

Registers the `real_ocr` mark and makes tests under it skipped by
default. To run real-OCR / vision-extractor tests, opt in with:

    pytest validation/ -m real_ocr

This is the standard pattern for tests that require external services
(here: Google Vision credentials or an Anthropic API key).
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_ocr: hits a real OCR / vision provider (skipped by default; opt in with -m real_ocr)",
    )
    # Replay-mode harness against the hand-labeled corpus
    # (validation/real_labels/). Zero recurring cost — runs against
    # committed `recorded_extraction.json` payloads, no live API calls.
    # Always-run: gated only by corpus presence.
    config.addinivalue_line(
        "markers",
        "real_corpus: replay-mode harness against the real-labels corpus (zero-cost; CI-runnable)",
    )
    # Smoke test for `record_extraction.py` against a real local Qwen3-VL
    # endpoint. Skipped by default — opted into via the QWEN_VL_BASE_URL
    # env var. The mark exists so the recorder can be exercised without
    # forcing every CI run to spin up a model server.
    config.addinivalue_line(
        "markers",
        "live_recorder: hits a local Qwen3-VL server (skipped without QWEN_VL_BASE_URL)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip `real_ocr`-marked tests unless the user explicitly opts in.

    Opt-in detection: pytest reports the active mark expression via
    `config.getoption("markexpr")`. We skip when the user did not name
    `real_ocr` in `-m`.

    `live_recorder` tests follow a different rule — they're auto-skipped
    only when `QWEN_VL_BASE_URL` is unset, so a contributor with a local
    server doesn't have to remember to opt in.
    """
    import os

    markexpr = config.getoption("markexpr") or ""
    if "real_ocr" not in markexpr:
        skip_real_ocr = pytest.mark.skip(
            reason=(
                "real_ocr tests skipped by default — opt in with "
                "`pytest validation/ -m real_ocr`"
            )
        )
        for item in items:
            if "real_ocr" in item.keywords:
                item.add_marker(skip_real_ocr)

    if not os.environ.get("QWEN_VL_BASE_URL"):
        skip_live_recorder = pytest.mark.skip(
            reason="QWEN_VL_BASE_URL not set; live recorder smoke skipped"
        )
        for item in items:
            if "live_recorder" in item.keywords:
                item.add_marker(skip_live_recorder)
