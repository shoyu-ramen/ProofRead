"""Pre-capture beverage-container detector.

Single-image gate that fires BEFORE the cylindrical-scan capture begins,
so the user can never run the 10.8 s verify chain on a selfie. The model
inspects one frame and answers two questions:

  1. Is the primary subject a beverage container (bottle, can, or box)?
  2. If yes, where is it (a tight normalized bounding box)?

The contract is intentionally minimal — no field extraction, no
compliance reasoning, no second-pass. The downstream cylindrical-scan
flow is what reads labels; this module's only job is to refuse to start
that flow when the camera is pointed at a face, a wall, or a hand.

Latency budget (pre-capture, on the user's clock):
  * 8 s timeout, 0 retries — the user feels every retry as dead time
    while the scan animation hasn't started.
  * In-process LRU keyed on `sha256(image_bytes)` so the rapid
    "tap-tap-tap" retry pattern hits the cache instead of round-tripping.

Conservatism: the prompt biases the model toward `detected=False` on any
ambiguity. A false negative makes the user re-aim the camera; a false
positive lets a 10.8 s scan run on a non-container, which is the worse
failure mode.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.config import settings
from app.services.anthropic_client import (
    build_client,
    call_with_resilience,
)

logger = logging.getLogger(__name__)


# Pre-capture latency budget: retries hurt the user more than they help.
# A flaky upstream is better surfaced as a one-shot 503 the UI can
# render as "tap to retry" than as an 8 s × 3 cascade the user stares at.
#
# Timeout sized for cold-start tolerance: warm calls return in 2-4 s but
# the first request after a deploy can hit a cold worker and a cold
# Anthropic connection together, pushing past 8 s. The default
# second-pass budget (8 s) was tripping false 503s in prod and — paired
# with the UI's older fail-open posture — let selfies leak through to
# /v1/verify. 15 s gives cold paths headroom while still capping the
# "Checking…" hint at something a user will actually wait through.
_PRE_CAPTURE_TIMEOUT_S = 15.0
_PRE_CAPTURE_MAX_RETRIES = 0

# In-process LRU sized for the rapid-retry pattern: a user tapping the
# capture button repeatedly on the same camera frame should hit the
# cache, but two consecutive captures milliseconds apart will never
# share bytes (the camera frame moves). 64 entries covers the long
# tail without bloating RSS — each value is a small Pydantic snapshot.
_CACHE_MAX_ENTRIES = 64


SYSTEM_PROMPT = """You are a pre-capture gate for a label-compliance scanner. \
You receive a single image. Your job: decide whether a beverage container \
(bottle, can, or box) is the primary subject of the image.

If yes:
  * Return a tight bounding box in normalized 0.0-1.0 coordinates measured \
from the top-left corner: x0 (left), y0 (top), x1 (right), y1 (bottom). \
0.0 is the top/left edge of the frame; 1.0 is the bottom/right edge. \
The box should hug the container, not the whole frame.
  * Classify the container as exactly one of: "bottle", "can", or "box".
  * Provide a confidence between 0.0 and 1.0.

If no - the subject is a person, a hand, a wall, ambiguous, multiple \
containers, or anything that is not a single beverage container - set \
detected=false and explain in `reason` (one short sentence: "appears to \
be a selfie", "wall and floor only, no container in frame", "two bottles \
in frame, cannot pick a primary subject").

Be conservative. If you would describe this as "might be a bottle but \
unclear", "container partially in frame but cropped", or "could be a \
container or could be a vase", set detected=false. The downstream scan is \
expensive (10+ seconds); a false negative just makes the user re-aim, but \
a false positive runs the whole pipeline on garbage.

Do not hallucinate coordinates. If you cannot see the container clearly \
enough to draw a tight box around it, say detected=false rather than \
guessing the box.

Output format: a single JSON object matching the provided schema. No \
Markdown fences, no commentary, no prose."""


class ContainerDetection(BaseModel):
    """Result of one pre-capture container-detection pass.

    `detected=True` requires both `container_type` and `bbox` to be
    non-null. `detected=False` requires `reason` to be non-null. The
    validator below enforces this invariant — callers and the API
    layer can rely on the shape rather than re-checking it.

    `bbox` is `(x0, y0, x1, y1)` in normalized image coordinates with
    the top-left corner at `(0.0, 0.0)` and the bottom-right at
    `(1.0, 1.0)`. Values are clamped to `[0.0, 1.0]` by the model
    schema constraints.
    """

    detected: bool
    container_type: Literal["bottle", "can", "box"] | None = None
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="[x0, y0, x1, y1] normalized 0.0-1.0 from top-left corner.",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str | None = Field(
        default=None,
        description="Required when detected=False; explains the rejection.",
    )

    @model_validator(mode="after")
    def _enforce_shape(self) -> ContainerDetection:
        if self.detected:
            if self.container_type is None:
                raise ValueError(
                    "detected=True requires container_type to be one of "
                    "'bottle' | 'can' | 'box'."
                )
            if self.bbox is None:
                raise ValueError(
                    "detected=True requires bbox to be a 4-tuple of normalized "
                    "coordinates."
                )
            # Validate bbox shape: 0.0 <= x0 < x1 <= 1.0, same for y.
            x0, y0, x1, y1 = self.bbox
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"bbox must satisfy 0<=x0<x1<=1 and 0<=y0<y1<=1; got {self.bbox!r}."
                )
        else:
            if self.reason is None or not self.reason.strip():
                raise ValueError(
                    "detected=False requires `reason` to be a non-empty string."
                )
        return self


# ---------------------------------------------------------------------------
# In-process LRU cache (L1 only — no L2/L3 here)
# ---------------------------------------------------------------------------


class _DetectCache:
    """Thread-safe LRU keyed on `sha256(image_bytes)`.

    Stored under a `threading.Lock` for the same reason `VerifyCache`
    is — FastAPI's threadpool can run two `detect_container` calls
    concurrently, and the GIL alone does not protect a plain `dict`
    against a concurrent `move_to_end` + `popitem` race.

    Values are frozen Pydantic snapshots; on hit we materialize a
    fresh copy via `model_copy()` so a caller mutating the result
    (the API layer doesn't, but defensively) can't corrupt the
    cache slot.
    """

    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._cache: OrderedDict[str, ContainerDetection] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def get(self, key: str) -> ContainerDetection | None:
        with self._lock:
            snap = self._cache.get(key)
            if snap is None:
                return None
            # LRU bookkeeping.
            self._cache.move_to_end(key)
            # Materialize a fresh copy so the caller cannot mutate cache state.
            return snap.model_copy(deep=True)

    def put(self, key: str, value: ContainerDetection) -> None:
        # Snapshot a deep copy so subsequent caller mutations cannot
        # contaminate the cached entry. Pydantic models are mostly
        # immutable in practice but the bbox tuple makes this cheap.
        snap = value.model_copy(deep=True)
        with self._lock:
            self._cache[key] = snap
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_cache = _DetectCache()


def _reset_cache() -> None:
    """Test hook: drop every entry without rebuilding the cache instance."""
    _cache.clear()


def _cache_key(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Service surface
# ---------------------------------------------------------------------------


class ContainerCheckService:
    """Stateless wrapper around the Anthropic call.

    The class exists primarily so tests can inject a stubbed client; the
    module-level `detect_container` helper constructs a process-wide
    singleton on first use (and is what the route handler calls).
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        timeout: float = _PRE_CAPTURE_TIMEOUT_S,
        max_retries: int = _PRE_CAPTURE_MAX_RETRIES,
        max_tokens: int = 256,
    ) -> None:
        if client is None:
            client = build_client(timeout=timeout, max_retries=max_retries)
        self._client = client
        # Pre-capture gate has a sub-2s latency budget, so route to the
        # primary model setting (Haiku 4.5 by default) rather than the
        # second-pass setting (Sonnet 4.6) — the structured-output schema
        # is small enough that Haiku handles it reliably.
        self._model = model or settings.anthropic_model
        self._max_tokens = max_tokens

    def detect(
        self, image_bytes: bytes, media_type: str = "image/jpeg"
    ) -> ContainerDetection:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        # `messages.parse` (structured output) bound to the small
        # ContainerDetection schema. The schema is small enough that the
        # SDK accepts it (the bigger label schema in vision.py:544 was
        # rejected for being "too complex").
        response = call_with_resilience(
            self._client.messages.parse,
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Decide whether the primary subject of this "
                                "image is a beverage container."
                            ),
                        },
                    ],
                }
            ],
            output_format=ContainerDetection,
        )
        result: ContainerDetection = response.parsed_output
        return result


# Module-level singleton — the route handler calls `detect_container()` and
# this instance is built lazily on first use. Tests can install a stub via
# `_override_service()` or by clearing `_service` directly.
_service: ContainerCheckService | None = None
_service_lock = threading.Lock()


def _get_service() -> ContainerCheckService:
    """Lazily construct the process-wide ContainerCheckService.

    `ExtractorUnavailable` is intentionally NOT cached — that lets a
    missing env var be fixed and recovered without restarting the
    process, matching the verify path's pattern.
    """
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            _service = ContainerCheckService()
    return _service


def _override_service(service: ContainerCheckService | None) -> None:
    """Test hook: install (or clear with None) a custom service."""
    global _service
    with _service_lock:
        _service = service


def detect_container(
    image_bytes: bytes, media_type: str
) -> ContainerDetection:
    """Run the pre-capture container-detection gate on a single image.

    Hits the in-process LRU first; on miss, calls the Anthropic-backed
    service. Raises `ExtractorUnavailable` if the upstream is
    unreachable or times out — the route handler maps that to a clean
    503 envelope so the UI can render "tap to retry" rather than dump
    a stack trace.
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty")
    key = _cache_key(image_bytes)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    service = _get_service()
    result = service.detect(image_bytes, media_type=media_type)
    _cache.put(key, result)
    return result
