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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip `real_ocr`-marked tests unless the user explicitly opts in.

    Opt-in detection: pytest reports the active mark expression via
    `config.getoption("markexpr")`. We skip when the user did not name
    `real_ocr` in `-m`.
    """
    markexpr = config.getoption("markexpr") or ""
    if "real_ocr" in markexpr:
        return  # user explicitly asked for these
    skip_marker = pytest.mark.skip(
        reason="real_ocr tests skipped by default — opt in with `pytest validation/ -m real_ocr`"
    )
    for item in items:
        if "real_ocr" in item.keywords:
            item.add_marker(skip_marker)
