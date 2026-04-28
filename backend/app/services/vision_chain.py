"""Ordered fallback chain for vision extractors.

Two parallel chains because the verify path and the scan pipeline use
slightly different `extract()` signatures (single image bytes vs.
keyword-only multi-surface), and we don't want to leak that detail into
either consumer:

  * `ChainedVerifyExtractor` — VisionExtractor protocol from `vision.py`
  * `ChainedScanExtractor`   — VisionExtractor protocol from `pipeline.py`

Both raise the last `ExtractorUnavailable` if every link is down, so
callers can either surface the error (verify path) or fall through to
OCR (scan path's existing behaviour). Non-`ExtractorUnavailable`
exceptions propagate unchanged — those are bugs we want to see, not
candidates for silent fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.anthropic_client import ExtractorUnavailable

logger = logging.getLogger(__name__)


class ChainedVerifyExtractor:
    """Try each verify-path VisionExtractor in order."""

    def __init__(self, extractors: list[Any]) -> None:
        if not extractors:
            raise ValueError("ChainedVerifyExtractor requires at least one extractor")
        self._extractors = extractors

    def extract(self, image_bytes: bytes, media_type: str = "image/png"):
        last: ExtractorUnavailable | None = None
        for ex in self._extractors:
            try:
                return ex.extract(image_bytes, media_type=media_type)
            except ExtractorUnavailable as exc:
                last = exc
                logger.warning(
                    "Verify extractor %s unavailable; falling back: %s",
                    type(ex).__name__,
                    exc,
                )
        assert last is not None  # loop guarantees we set this on every iteration
        raise last


class ChainedScanExtractor:
    """Try each scan-path VisionExtractor in order."""

    def __init__(self, extractors: list[Any]) -> None:
        if not extractors:
            raise ValueError("ChainedScanExtractor requires at least one extractor")
        self._extractors = extractors

    def extract(self, **kwargs: Any):
        last: ExtractorUnavailable | None = None
        for ex in self._extractors:
            try:
                return ex.extract(**kwargs)
            except ExtractorUnavailable as exc:
                last = exc
                logger.warning(
                    "Scan extractor %s unavailable; falling back: %s",
                    type(ex).__name__,
                    exc,
                )
        assert last is not None
        raise last
