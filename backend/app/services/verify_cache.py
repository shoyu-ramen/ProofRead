"""In-process LRU cache for `/v1/verify` results.

The cold path through `verify()` is dominated by the vision-model call
(hundreds of milliseconds at minimum) and the sensor pre-check (~60 ms
on a typical 2 MP frame). For iterative workflows — a brewer tweaking
artwork in Illustrator, re-exporting, re-submitting — the *same image
bytes* land at the API repeatedly within minutes. Re-running the VLM
on identical bytes is wasteful both on latency and spend.

Cache contract:

  * Key: SHA-256 over the image bytes plus the small set of inputs that
    can legitimately change the verdict for those bytes (beverage type,
    container size, imported flag, claim payload, the rule-set version
    fingerprint). Anything that changes the verdict invalidates the
    entry; anything that doesn't, hits.
  * Value: the full `VerifyReport` produced by the cold path, minus
    `elapsed_ms` (which we restamp on every read). The cached report
    already contains the rule_results, the cross-check verdict, the
    image-quality summary, and the extracted-fields summary — exactly
    what the API surfaces to the UI.
  * Eviction: LRU at a bounded entry count. The default keeps a few
    thousand recent labels resident — small enough to never threaten
    process memory on a 256 MB Fly machine, large enough to span a
    full design iteration.

Mutation isolation (freeze pattern, replaces the prior `copy.deepcopy`):
  * `put` stores the entry as a frozen snapshot — `rule_results` is
    captured as a tuple, the mutable mappings (`extracted`,
    `unreadable_fields`, `health_warning_cross_check`) are captured
    via the snapshot helpers below.
  * `get` materializes a fresh `VerifyReport` from the snapshot, so
    callers receive normal mutable Python collections they can sort,
    append to, etc. Their mutations target the freshly-materialized
    collections, not the cache's internal storage — so the cache stays
    quarantined without paying the deepcopy cost on every hit.

SPEC §0.5 fail-honestly is preserved: a cache hit returns the same
verdict the cold path produced, including any "advisory" /
"unreadable" downgrades. We never *upgrade* a previously honest
verdict by skipping the pre-check.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from app.rules.types import Rule, RuleResult

if TYPE_CHECKING:
    # `VerifyReport` lives in `app.services.verify`, which itself imports this
    # module — a runtime import would be circular. Annotation-only is fine
    # because every call site that touches the cache already lives inside the
    # verify service, where the dataclass is concretely available.
    from app.services.verify import VerifyReport


@dataclass
class CacheStats:
    hits: int
    misses: int
    size: int
    max_entries: int


@dataclass(frozen=True)
class _Snapshot:
    """Immutable form of `VerifyReport` held in the cache.

    Frozen at the dataclass level so a stray attribute reassignment
    (e.g. `snap.overall = "fail"`) raises rather than silently
    contaminating future hits. The mutable-shaped fields (lists, dicts)
    are captured as tuples / new dicts on `put` and re-built into
    fresh mutables on `get`, so the cache surface to callers is
    indistinguishable from the prior `copy.deepcopy` contract.
    """

    overall: str
    rule_results: tuple[RuleResult, ...]
    extracted: dict[str, dict[str, Any]]
    unreadable_fields: tuple[str, ...]
    image_quality: str
    image_quality_notes: str | None
    health_warning_cross_check: dict[str, Any] | None
    elapsed_ms: int
    cache_hit: bool
    reverse_lookup_hit: bool


def _snapshot(report: VerifyReport) -> _Snapshot:
    """Capture the cold-path report as an immutable snapshot.

    `rule_results` becomes a tuple — `RuleResult` is a regular
    dataclass with only scalar / tuple fields, so a tuple of references
    is sufficient to lock the read shape against caller mutation. The
    nested dicts are shallow-copied (single-level) since their values
    are also scalar / tuple. Anything deeper (e.g. a list-of-objects
    inside one of the entries) would need a recursive copy.
    """
    return _Snapshot(
        overall=report.overall,
        rule_results=tuple(report.rule_results),
        extracted={
            name: dict(info) if isinstance(info, dict) else info
            for name, info in report.extracted.items()
        },
        unreadable_fields=tuple(report.unreadable_fields),
        image_quality=report.image_quality,
        image_quality_notes=report.image_quality_notes,
        health_warning_cross_check=(
            dict(report.health_warning_cross_check)
            if isinstance(report.health_warning_cross_check, dict)
            else report.health_warning_cross_check
        ),
        elapsed_ms=report.elapsed_ms,
        cache_hit=report.cache_hit,
        reverse_lookup_hit=report.reverse_lookup_hit,
    )


def _materialize(snap: _Snapshot) -> VerifyReport:
    """Build a fresh `VerifyReport` from a frozen snapshot.

    Returns mutable collections (list, dict) so the caller can
    sort / append / clear without surprises. Mutations land on the
    freshly-built containers, not the cache's frozen storage — that's
    the whole point of the freeze pattern: isolation by construction
    rather than by recursive copy.
    """
    from app.services.verify import VerifyReport

    return VerifyReport(
        overall=snap.overall,
        rule_results=list(snap.rule_results),
        extracted={
            name: dict(info) if isinstance(info, dict) else info
            for name, info in snap.extracted.items()
        },
        unreadable_fields=list(snap.unreadable_fields),
        image_quality=snap.image_quality,
        image_quality_notes=snap.image_quality_notes,
        health_warning_cross_check=(
            dict(snap.health_warning_cross_check)
            if isinstance(snap.health_warning_cross_check, dict)
            else snap.health_warning_cross_check
        ),
        elapsed_ms=snap.elapsed_ms,
        cache_hit=snap.cache_hit,
        reverse_lookup_hit=snap.reverse_lookup_hit,
    )


class VerifyCache:
    """Thread-safe LRU of `VerifyReport`s keyed on image+context fingerprints.

    Stored under a `threading.Lock` rather than `asyncio.Lock` because
    FastAPI's threadpool can run two `verify_label` handlers
    concurrently for sync helpers; the GIL alone does not protect a
    plain `dict` against a concurrent `move_to_end` + `popitem` race.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._cache: OrderedDict[str, _Snapshot] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> VerifyReport | None:
        """Return a fresh `VerifyReport` materialized from the cached
        frozen snapshot, or None on miss.

        The snapshot's collections are immutable (tuple of
        rule_results, frozen dict views), so re-materializing into
        normal mutable lists/dicts here is enough to isolate the caller
        from the cache: any mutation lands on the freshly-built
        collection, not the cached storage. Cheaper than `copy.deepcopy`
        because we only re-build the top-level container shape, not
        every leaf value.
        """
        with self._lock:
            snap = self._cache.get(key)
            if snap is None:
                self._misses += 1
                return None
            # LRU bookkeeping.
            self._cache.move_to_end(key)
            self._hits += 1
            return _materialize(snap)

    def put(self, key: str, value: VerifyReport) -> None:
        """Store a frozen snapshot of `value` so the cache holds an
        island of state nobody else can reach into.

        The snapshot replaces the prior `copy.deepcopy` because the
        cached report's mutable surface is shallow (top-level lists +
        the cross-check dict + the extracted-fields dict-of-dicts). A
        tuple-of-rule_results plus copied dicts gives the same isolation
        guarantee at a fraction of the deepcopy cost, and on
        materialization we hand the caller fresh mutables instead of
        forcing them through a frozen-collection API at every site.
        """
        snap = _snapshot(value)
        with self._lock:
            self._cache[key] = snap
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                size=len(self._cache),
                max_entries=self._max,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


def make_cache_key(
    *,
    panels: list[tuple[bytes, str]],
    beverage_type: str,
    container_size_ml: int,
    is_imported: bool,
    application: dict[str, Any] | None,
    rules: list[Rule],
) -> str:
    """Stable hash over every input that can change the verdict.

    `panels` is a list of `(image_bytes, media_type)` for each label panel
    that contributed to the verdict, in submission order. Order matters —
    a request that sends `[front, back]` is a different verdict input than
    `[back, front]` because the merge picks "panel_0" / "panel_1" surface
    IDs that propagate into the response. `media_type` is hashed alongside
    each panel for the same reason it was on the single-image path: the
    model receives it via the multimodal payload and could conceivably
    treat the same bytes differently if the declared MIME changes.
    `rules` is fingerprinted by `(id, version)` so a rule edit in
    `definitions/*.yaml` invalidates every cached verdict for that
    beverage type without us having to bump a manual version constant.
    """
    if not panels:
        raise ValueError("make_cache_key requires at least one panel")
    h = hashlib.sha256()
    # Schema version bumped from v1 → v2 on the multi-panel cutover so the
    # in-process cache discards any single-image entries hashed under the
    # old layout (different field ordering would otherwise produce
    # spurious "hits" with the wrong verdict structure).
    h.update(b"v2\x00")
    h.update(str(len(panels)).encode("ascii"))
    h.update(b"\x00")
    for image_bytes, media_type in panels:
        h.update(image_bytes)
        h.update(b"\x00")
        h.update(media_type.encode("utf-8"))
        h.update(b"\x00")
    h.update(beverage_type.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(container_size_ml).encode("ascii"))
    h.update(b"\x00")
    h.update(b"1" if is_imported else b"0")
    h.update(b"\x00")
    h.update(_canonical_application(application).encode("utf-8"))
    h.update(b"\x00")
    h.update(_rules_fingerprint(rules).encode("ascii"))
    return h.hexdigest()


def restamp_report(cached: VerifyReport, elapsed_ms: int) -> VerifyReport:
    """Return a copy of `cached` with `elapsed_ms` set to the current run.

    `elapsed_ms` is the only field we deliberately don't cache — every
    request gets its own honest measurement so dashboards and the UI
    show the actual wall-clock cost of *this* request, not the cold
    one we cached.
    """
    return replace(cached, elapsed_ms=elapsed_ms)


def _canonical_application(application: dict[str, Any] | None) -> str:
    """Stable JSON serialization for cache keying.

    `sort_keys=True` makes `{"a":1,"b":2}` and `{"b":2,"a":1}` produce
    the same key; `default=str` covers any caller that smuggled in a
    UUID or datetime.
    """
    if not application:
        return "{}"
    return json.dumps(application, sort_keys=True, default=str, separators=(",", ":"))


def _rules_fingerprint(rules: list[Rule]) -> str:
    """`(id, version)` pairs sorted by id — cheap, stable, change-sensitive."""
    pairs = sorted((r.id, r.version) for r in rules)
    return ";".join(f"{i}@{v}" for i, v in pairs)
