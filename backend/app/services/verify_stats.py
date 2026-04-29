"""In-process counters for the verify pipeline's observability story.

The audit's HIGH item was "we can't tell if our cache hit rate is 5 % or
95 %, whether second-pass is failing on 1 % or 30 % of requests, what the
cold-vs-warm latency split looks like." This module is the small, fast,
in-process answer: a thread-safe counter store that the verify
orchestrator bumps on every interesting event, exposed via the
`/v1/verify/_stats` admin endpoint and folded into the per-request
structured log.

Not a replacement for OpenTelemetry / Honeycomb (those land later per
SPEC §0). This is the "we need the number right now to debug a
production blip" tier.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass
from typing import Literal


# Outcome classes for the second-pass swallow path. Tracked separately so
# operators can distinguish a systemic Anthropic outage (`extractor_unavailable`
# spikes) from a one-off bad response (`malformed_json`) from a user-facing
# upload that simply lacks a warning paragraph (`no_warning_found` is on the
# read side, not here).
SecondPassOutcome = Literal[
    "success",                # second-pass returned a parsed read
    "extractor_unavailable",  # ExtractorUnavailable (rate-limit, network, auth)
    "rate_limit",             # anthropic.RateLimitError specifically
    "connection_error",       # anthropic.APIConnectionError specifically
    "malformed_json",         # JSON couldn't parse / wrong shape
    "other_error",            # everything else (logged with type)
]


# Cap the recent-outcomes window; long enough to spot a ~5-minute trend on a
# busy box, short enough that a memory blip doesn't matter.
_RECENT_WINDOW = 256


@dataclass
class VerifyStats:
    """Snapshot returned by `get_stats()`. Plain dataclass for trivial JSON
    serialization in the `_stats` endpoint; not the live counter."""

    cold_count: int
    warm_count: int
    cold_elapsed_ms_recent: list[int]
    warm_elapsed_ms_recent: list[int]
    second_pass_outcomes: dict[str, int]
    overall_verdicts: dict[str, int]
    # Perceptual reverse-lookup outcomes, separate from the byte-exact
    # warm/cold counters because they answer different questions: a
    # SHA-256 cache hit means "I already verified these exact bytes",
    # a reverse-lookup hit means "I already verified a perceptually-
    # equivalent image". Tracking both lets dashboards split the
    # latency win between exact-resubmit and similar-resubmit flows.
    reverse_lookup_hits: int
    reverse_lookup_misses: int
    reverse_lookup_elapsed_ms_recent: list[int]


class _Counters:
    """Mutable counter set guarded by a single lock.

    Read paths return immutable snapshots so callers don't see mid-write
    state. Writes are O(1) amortized so the lock is uncontended in
    practice; even at 100 RPS the bump cost is negligible against the VLM
    call's wall-clock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cold = 0
        self._warm = 0
        self._cold_elapsed: deque[int] = deque(maxlen=_RECENT_WINDOW)
        self._warm_elapsed: deque[int] = deque(maxlen=_RECENT_WINDOW)
        self._second_pass: Counter[str] = Counter()
        self._verdicts: Counter[str] = Counter()
        self._reverse_hits = 0
        self._reverse_misses = 0
        self._reverse_elapsed: deque[int] = deque(maxlen=_RECENT_WINDOW)

    def record_cold(self, *, elapsed_ms: int, overall: str) -> None:
        with self._lock:
            self._cold += 1
            self._cold_elapsed.append(elapsed_ms)
            self._verdicts[overall] += 1

    def record_warm(self, *, elapsed_ms: int, overall: str) -> None:
        with self._lock:
            self._warm += 1
            self._warm_elapsed.append(elapsed_ms)
            self._verdicts[overall] += 1

    def record_second_pass(self, outcome: SecondPassOutcome) -> None:
        with self._lock:
            self._second_pass[outcome] += 1

    def record_reverse_lookup_hit(self, *, elapsed_ms: int) -> None:
        """Bump the reverse-lookup hit counter and record latency.

        `elapsed_ms` is the *whole verify call* on a reverse-hit
        path — i.e. sensor pre-check + dhash + lookup + rule engine,
        but no VLM call. Tracking it separately from the warm-path
        timing window keeps the cold/warm split honest (a reverse-
        hit isn't a cache "warm" in the SHA-256 sense; it skipped
        the VLM but ran the rule engine fresh)."""
        with self._lock:
            self._reverse_hits += 1
            self._reverse_elapsed.append(elapsed_ms)

    def record_reverse_lookup_miss(self) -> None:
        with self._lock:
            self._reverse_misses += 1

    def snapshot(self) -> VerifyStats:
        with self._lock:
            return VerifyStats(
                cold_count=self._cold,
                warm_count=self._warm,
                cold_elapsed_ms_recent=list(self._cold_elapsed),
                warm_elapsed_ms_recent=list(self._warm_elapsed),
                second_pass_outcomes=dict(self._second_pass),
                overall_verdicts=dict(self._verdicts),
                reverse_lookup_hits=self._reverse_hits,
                reverse_lookup_misses=self._reverse_misses,
                reverse_lookup_elapsed_ms_recent=list(self._reverse_elapsed),
            )

    def reset(self) -> None:
        """Test hook: drop all counters. Production never calls this."""
        with self._lock:
            self._cold = 0
            self._warm = 0
            self._cold_elapsed.clear()
            self._warm_elapsed.clear()
            self._second_pass.clear()
            self._verdicts.clear()
            self._reverse_hits = 0
            self._reverse_misses = 0
            self._reverse_elapsed.clear()


_singleton = _Counters()


def record_cold(*, elapsed_ms: int, overall: str) -> None:
    _singleton.record_cold(elapsed_ms=elapsed_ms, overall=overall)


def record_warm(*, elapsed_ms: int, overall: str) -> None:
    _singleton.record_warm(elapsed_ms=elapsed_ms, overall=overall)


def record_second_pass(outcome: SecondPassOutcome) -> None:
    _singleton.record_second_pass(outcome)


def record_reverse_lookup_hit(*, elapsed_ms: int) -> None:
    _singleton.record_reverse_lookup_hit(elapsed_ms=elapsed_ms)


def record_reverse_lookup_miss() -> None:
    _singleton.record_reverse_lookup_miss()


def snapshot() -> VerifyStats:
    return _singleton.snapshot()


def reset() -> None:
    """Test hook only."""
    _singleton.reset()


def classify_second_pass_exception(exc: BaseException) -> SecondPassOutcome:
    """Map a swallowed exception to an outcome class.

    Imports are lazy because `anthropic` is an optional runtime dep — the
    test suite runs without it installed.
    """
    # JSON decode failures bubble through `_parse_response` as a returned
    # `WarningRead(found=False)`, not as exceptions, so the caller doesn't
    # currently route those through here. Kept in the enum so the stats
    # surface is forward-compatible if that contract changes.
    try:
        import anthropic
    except ImportError:
        anthropic = None  # type: ignore[assignment]

    if anthropic is not None:
        if isinstance(exc, anthropic.RateLimitError):
            return "rate_limit"
        if isinstance(exc, anthropic.APIConnectionError):
            return "connection_error"

    # `ExtractorUnavailable` is the orchestrator's own surface for any
    # transient SDK failure — keep distinct so the dashboard can graph
    # "we couldn't reach Anthropic at all" separately from concrete causes.
    from app.services.anthropic_client import ExtractorUnavailable

    if isinstance(exc, ExtractorUnavailable):
        return "extractor_unavailable"
    return "other_error"
