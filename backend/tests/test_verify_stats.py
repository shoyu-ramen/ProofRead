"""Tests for the in-process verify-pipeline counter store."""

from __future__ import annotations

import json

from app.services import verify_stats
from app.services.anthropic_client import ExtractorUnavailable


def setup_function() -> None:
    """Each test starts with a fresh counter set."""
    verify_stats.reset()


def test_record_cold_and_warm_split():
    """Cold and warm counts are tracked separately so dashboards can
    compute hit rate without subtracting overlapping totals."""
    verify_stats.record_cold(elapsed_ms=2_500, overall="pass")
    verify_stats.record_warm(elapsed_ms=12, overall="pass")
    verify_stats.record_warm(elapsed_ms=8, overall="pass")

    snap = verify_stats.snapshot()
    assert snap.cold_count == 1
    assert snap.warm_count == 2
    assert snap.cold_elapsed_ms_recent == [2_500]
    assert snap.warm_elapsed_ms_recent == [12, 8]
    assert snap.overall_verdicts == {"pass": 3}


def test_recent_window_is_bounded():
    """The recent-elapsed deque must not grow without bound — a long-
    running instance shouldn't accumulate megabytes of latency samples."""
    for i in range(500):
        verify_stats.record_cold(elapsed_ms=i, overall="pass")

    snap = verify_stats.snapshot()
    assert snap.cold_count == 500
    # _RECENT_WINDOW = 256; only the most recent 256 should remain.
    assert len(snap.cold_elapsed_ms_recent) == 256
    # The newest sample should be 499 (latest write).
    assert snap.cold_elapsed_ms_recent[-1] == 499
    # The oldest retained sample should be 500 - 256 = 244.
    assert snap.cold_elapsed_ms_recent[0] == 244


def test_record_second_pass_outcomes_are_counted():
    verify_stats.record_second_pass("success")
    verify_stats.record_second_pass("success")
    verify_stats.record_second_pass("rate_limit")
    verify_stats.record_second_pass("connection_error")

    snap = verify_stats.snapshot()
    assert snap.second_pass_outcomes == {
        "success": 2,
        "rate_limit": 1,
        "connection_error": 1,
    }


def test_classify_extractor_unavailable():
    """The orchestrator's `ExtractorUnavailable` is a domain wrapper —
    callers above the SDK never see anthropic.* directly. The classifier
    must map it to its own bucket so dashboards can graph 'we couldn't
    reach Anthropic at all' separately from concrete causes."""
    exc = ExtractorUnavailable("upstream rate limit")
    assert verify_stats.classify_second_pass_exception(exc) == "extractor_unavailable"


def test_classify_other_exception():
    """Anything not in the known set falls into `other_error` so the bucket
    monotonically captures unknown failure modes — a spike in
    `other_error` is the signal to add a new classifier branch."""

    class _Strange(RuntimeError):
        pass

    assert verify_stats.classify_second_pass_exception(_Strange("?")) == "other_error"


def test_snapshot_serializes_to_json():
    """The snapshot is dict-shaped for direct JSON return from the
    `_stats` endpoint. A regression that puts non-serializable values
    (e.g. raw Counter type bleeding through) would break the route."""
    verify_stats.record_cold(elapsed_ms=100, overall="advisory")
    verify_stats.record_second_pass("success")

    snap = verify_stats.snapshot()
    payload = {
        "cold_count": snap.cold_count,
        "warm_count": snap.warm_count,
        "cold_elapsed_ms_recent": snap.cold_elapsed_ms_recent,
        "warm_elapsed_ms_recent": snap.warm_elapsed_ms_recent,
        "second_pass_outcomes": snap.second_pass_outcomes,
        "overall_verdicts": snap.overall_verdicts,
    }
    # Round-trip through JSON to lock the shape.
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["cold_count"] == 1
    assert decoded["second_pass_outcomes"] == {"success": 1}
    assert decoded["overall_verdicts"] == {"advisory": 1}
