"""Perceptual-hash reverse-image lookup for /v1/verify.

The byte-exact `VerifyCache` (see `verify_cache.py`) keys on SHA-256 of
the upload bytes. That nails the iterative-design "re-submitted the
same Illustrator export" case but misses every visually-equivalent
input that doesn't byte-match:

  * The same artwork re-exported at a different JPEG quality.
  * PNG-of-the-master vs JPEG-of-the-master from the same source PSD.
  * The same physical bottle photographed from a slightly different
    distance, angle, or under different lighting.
  * Identical pixels with one bit of EXIF-stripping or color-profile
    rewrite from an intermediate tool.

For all of those, the *label content* is the same — so the VLM call
that reads the label fields is the same, and we should not pay for it.
This module owns that second-tier lookup.

Hash choice: dhash (difference hash, 8×9 grayscale → 64 boolean
left-vs-right comparisons → 64-bit signature). Robust to JPEG
re-encode, gentle scaling, white-balance shifts, and small crops; not
robust to large rotations or full re-cropping. The Hamming-distance
threshold below is tuned for "very likely the same source image" —
six bit-flips out of sixty-four is the standard cutoff in the
imagehash literature for the false-positive rate to drop below ~1 %.

Cache contract:

  * Key: per-panel signature (one dhash per submitted panel) plus
    `beverage_type` and panel count. Beverage type is a scoping guard:
    the rule engine evaluates a different rule set for beer vs.
    spirits, and an extraction we read on a beer label is not a
    legitimate spirits extraction. Panel count keeps the
    `panel_N` / `panorama` source-image-id contract intact across
    promote/replay.

  * Value: the *raw merged* `VisionExtraction` produced by the cold
    path before the rule engine ran. We keep the extraction rather
    than the full report so the same image at a different
    `container_size_ml` / `is_imported` / claim still hits — only the
    rule engine's evaluation depends on those.

  * Eviction: linear-scan LRU bounded at `max_entries`. Linear scan
    is O(N×panels×64-bit-popcount) per lookup; on a 4096-entry cache
    with up to 4 panels per query that lands well under 1 ms even on
    the slow path, so a separate VP-tree / BK-tree index is not worth
    the build/maintenance complexity at the cache sizes we run.

SPEC §0.5 fail-honestly contract:

  * We DO NOT promote unreadable runs (no extraction value to reuse).

  * We DO NOT promote runs whose cross-check landed on a
    `disagreement` or `unverifiable_obstructed` outcome — those
    verdicts already downgraded the warning rule to ADVISORY on the
    cold path because the two reads disagreed, and silently reusing
    the extraction without re-running the cross-check would mean a
    *next* request gets a cleaner-looking verdict than the cold path
    actually produced. Better to make the next request pay for its
    own cold path and arrive at the same honest verdict.

  * On a reverse-lookup hit we re-run the rule engine fresh with the
    current request's container_size_ml / is_imported / claim / rule
    fingerprint, so a verdict-affecting change to any of those still
    produces the right answer. We deliberately skip the second-pass
    Government-Warning reader on a hit to recover the latency win;
    the obstruction-backstop downgrade still runs on the new request's
    sensor verdict so a re-photograph that introduced new glare over
    the warning still gets the ADVISORY safety net.
"""

from __future__ import annotations

import io
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

    from app.services.vision import VisionExtraction


logger = logging.getLogger(__name__)


# 64-bit dhash — bumping this to 256-bit (16×16 grid) buys a touch of
# extra precision but doubles the per-entry signature storage and the
# Hamming-popcount work without measurably moving the false-positive
# rate at our threshold. 64-bit is the standard.
_HASH_GRID = 8


# Default Hamming-distance threshold. ≤6 bit-flips out of 64 → ~91 %
# bit similarity → in practice >99 % chance the two images share a
# source. Surfaced as a config knob so an operator can tighten this if
# false-positive verdict-reuse ever shows up in the dashboard, or
# loosen it for "obviously similar but I want it to hit anyway"
# diagnostic flows.
DEFAULT_HAMMING_THRESHOLD = 6


@dataclass
class LookupStats:
    """Snapshot of cache behaviour for the /v1/verify/_stats endpoint.

    `hits` and `misses` are tracked alongside the byte-exact cache's
    counters because they answer different questions: a SHA-256 hit
    means "exact same upload"; a perceptual hit means "perceptually
    same upload". The split lets dashboards see how much of the
    iterative-design flow vs. the photographer-resubmits-similar flow
    is actually saving cold paths.
    """

    hits: int
    misses: int
    size: int
    max_entries: int
    hamming_threshold: int


@dataclass(frozen=True)
class _Snapshot:
    """Frozen, immutable extraction held in the cache.

    Same shape as `verify_cache._Snapshot` — frozen so a stray caller
    mutation can't contaminate future hits. We hold the raw bytes of
    the second-pass-skipped extraction; on hit, the caller materialises
    a fresh `VisionExtraction` mutable so caller mutations land on the
    fresh container.

    `signature` is the per-panel dhash tuple captured at promotion
    time; `beverage_type` and `panel_count` scope the entry. Only
    entries whose scope matches the query are eligible for Hamming
    comparison.
    """

    signature: tuple[int, ...]
    beverage_type: str
    panel_count: int
    fields: dict[str, Any]
    unreadable: tuple[str, ...]
    raw_response: str
    image_quality: str | None
    image_quality_notes: str | None
    beverage_type_observed: str | None


def _snapshot(
    *,
    signature: tuple[int, ...],
    beverage_type: str,
    extraction: VisionExtraction,
) -> _Snapshot:
    """Capture the cold-path extraction as an immutable snapshot.

    `ExtractedField` is a *mutable* dataclass — the verify orchestrator
    rewrites `confidence` after the merge step (capping it at the
    surface verdict). To prevent that rewrite from contaminating the
    cached entry on a subsequent reverse-hit, we deep-copy each field
    via `dataclasses.replace` at promotion time. The fresh copies
    insulate the cache from any post-promotion mutation the caller
    may have done to the extraction it handed us.
    """
    from dataclasses import replace as dc_replace

    return _Snapshot(
        signature=signature,
        beverage_type=beverage_type,
        panel_count=len(signature),
        fields={name: dc_replace(f) for name, f in extraction.fields.items()},
        unreadable=tuple(extraction.unreadable),
        raw_response=extraction.raw_response,
        image_quality=extraction.image_quality,
        image_quality_notes=extraction.image_quality_notes,
        beverage_type_observed=extraction.beverage_type_observed,
    )


def _materialize(snap: _Snapshot) -> VisionExtraction:
    """Build a fresh `VisionExtraction` mutable from a frozen snapshot.

    Each `ExtractedField` is re-copied via `dataclasses.replace` so
    the fresh extraction's mutations (confidence cap, bbox shift) land
    on its own field objects, not the cached snapshot's. Same
    isolation pattern `verify_cache._materialize` uses for
    `RuleResult` — frozen snapshot in storage, fresh mutables on
    every read.
    """
    from dataclasses import replace as dc_replace

    from app.services.vision import VisionExtraction

    return VisionExtraction(
        fields={name: dc_replace(f) for name, f in snap.fields.items()},
        unreadable=list(snap.unreadable),
        raw_response=snap.raw_response,
        image_quality=snap.image_quality,
        image_quality_notes=snap.image_quality_notes,
        beverage_type_observed=snap.beverage_type_observed,
    )


def compute_dhash(img: Image | None) -> int | None:
    """Return a 64-bit dhash of `img` (or None on failure).

    The standard 8×9 difference-hash recipe:

      1. Convert to grayscale (luminance).
      2. Resize to (HASH_GRID + 1) × HASH_GRID = 9×8 with a fast
         resampling filter — the algorithm only cares about coarse
         intensity gradients so LANCZOS is wasted cycles; BILINEAR is
         the right choice on the speed/quality curve.
      3. For each row, compare pixel[i] < pixel[i+1] across the 9
         columns. That gives 8 booleans per row × 8 rows = 64 bits.
      4. Pack into a 64-bit int (MSB = top-left, LSB = bottom-right).

    The hash is invariant to overall brightness shifts and
    monotonic-gamma changes (because we compare neighbour pixels, not
    absolute values), and stable under JPEG re-encode at any quality
    above ~50 (verified empirically; the 9×8 grid quantises away
    block-DCT artifacts well below that).

    None on failure — the orchestrator treats that as "skip the
    reverse-lookup for this panel" and falls through to the regular
    cold path. Hashing must never break the request.
    """
    if img is None:
        return None
    try:
        gray = img.convert("L")
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("dhash failed at convert: %s", exc)
        return None
    try:
        from PIL import Image as PILImage

        # +1 column so we get HASH_GRID horizontal comparisons per row.
        small = gray.resize(
            (_HASH_GRID + 1, _HASH_GRID), PILImage.Resampling.BILINEAR
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("dhash failed at resize: %s", exc)
        return None

    try:
        import numpy as np

        arr = np.asarray(small, dtype=np.int16)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("dhash failed at array: %s", exc)
        return None

    # Compare each pixel with its right neighbour. `arr[:, :-1] <
    # arr[:, 1:]` gives an 8×8 boolean grid that we flatten row-major
    # and pack MSB-first. Pure-numpy is fast (≈100 µs per panel on a
    # modern x86) and the only allocation is the small int16 array.
    diffs = arr[:, :-1] < arr[:, 1:]
    bits = diffs.flatten()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def compute_dhash_bytes(image_bytes: bytes) -> int | None:
    """Convenience: decode `image_bytes` once and dhash the result.

    Used by callers that only have raw bytes on hand (e.g. the panel
    upload path before normalisation has run). The verify orchestrator
    already decodes inside `_normalize_for_vision` and lifts the dhash
    from there; this is the fall-back for paths that don't.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as exc:
        logger.debug("dhash decode failed: %s", exc)
        return None
    return compute_dhash(img)


def hamming(a: int, b: int) -> int:
    """64-bit Hamming distance between two dhash signatures.

    `bin(a ^ b).count("1")` is the textbook approach and clocks in
    around 100 ns on CPython 3.12 — fast enough that an 8-panel × 4096-
    entry linear scan completes inside a millisecond. Python 3.10+
    also exposes `int.bit_count()` which is roughly 3× faster on
    micro-benchmarks; using it where available keeps hot-loop perf
    predictable as cache fill grows.
    """
    return (a ^ b).bit_count()


@dataclass
class LookupHit:
    """Result of a successful reverse-lookup query.

    `extraction` is a freshly-materialised `VisionExtraction` (fresh
    mutable collections, frozen-snapshot-isolation pattern shared with
    `VerifyCache`). `min_distance` is the worst-case per-panel Hamming
    distance — the orchestrator can log it for tuning.
    """

    extraction: VisionExtraction
    min_distance: int


class ReverseLookupCache:
    """Thread-safe LRU keyed on per-panel perceptual signatures.

    The store is an `OrderedDict` of `_Snapshot`s, with insertion
    order tracking access order (move_to_end on hit). Lookup is a
    linear scan filtered by `beverage_type` and panel count, with the
    minimum-Hamming match returned when within threshold.

    Locking strategy mirrors `VerifyCache`: a single `threading.Lock`
    guards the scan + LRU bookkeeping. The verify orchestrator calls
    into the cache once per request (one `get`, optionally one `put`)
    so contention is bounded by request concurrency, not by
    per-comparison work.
    """

    def __init__(
        self,
        *,
        max_entries: int = 4096,
        hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if hamming_threshold < 0 or hamming_threshold > 64:
            raise ValueError("hamming_threshold must be in [0, 64]")
        self._cache: OrderedDict[int, _Snapshot] = OrderedDict()
        self._max = max_entries
        self._threshold = hamming_threshold
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        # Monotonic counter for entry keys — we don't use the
        # signature itself as the dict key because near-collisions
        # (e.g. a sequence of similar labels) would clobber each other
        # in a hash-keyed dict, defeating the LRU. A monotonic int
        # gives us an LRU ordering without dropping entries to
        # collision.
        self._next_id = 0

    def get(
        self,
        *,
        signature: tuple[int, ...],
        beverage_type: str,
    ) -> LookupHit | None:
        """Return the closest-matching extraction within Hamming threshold.

        "Closest" is measured as the maximum per-panel Hamming
        distance (a hit must satisfy every panel independently;
        the worst-of-N panels is the binding constraint). Returns
        None when no eligible entry is in scope or every eligible
        entry exceeds the threshold on at least one panel.

        On hit: bumps the entry to the LRU head and increments the
        hit counter; the returned `extraction` is a fresh mutable
        the caller can mutate freely.
        """
        if not signature or any(s is None for s in signature):
            with self._lock:
                self._misses += 1
            return None

        panel_count = len(signature)
        best: tuple[int, _Snapshot] | None = None
        best_distance = self._threshold + 1

        with self._lock:
            for entry_id, snap in self._cache.items():
                if snap.beverage_type != beverage_type:
                    continue
                if snap.panel_count != panel_count:
                    continue
                # Per-panel Hamming distance, taking the worst (max)
                # across panels as the binding constraint. An early-
                # exit lets us skip the rest as soon as one panel
                # exceeds threshold.
                worst = 0
                ok = True
                for query_h, entry_h in zip(signature, snap.signature):
                    d = hamming(query_h, entry_h)
                    if d > self._threshold:
                        ok = False
                        break
                    if d > worst:
                        worst = d
                if not ok:
                    continue
                if worst < best_distance:
                    best_distance = worst
                    best = (entry_id, snap)
                    if worst == 0:
                        # Exact-match short-circuit: no eligible
                        # entry can do better than 0 across all
                        # panels, so stop scanning.
                        break

            if best is None:
                self._misses += 1
                return None

            entry_id, snap = best
            self._cache.move_to_end(entry_id)
            self._hits += 1
            extraction = _materialize(snap)
        return LookupHit(extraction=extraction, min_distance=best_distance)

    def put(
        self,
        *,
        signature: tuple[int, ...],
        beverage_type: str,
        extraction: VisionExtraction,
    ) -> None:
        """Promote a cold-path extraction into the cache.

        Callers are expected to gate this themselves — the cache
        accepts whatever is handed to it. Specifically:

          * Don't promote unreadable extractions (no fields).
          * Don't promote a verdict whose cross-check disagreed.
          * Don't promote when `signature` has any None entries
            (decode failed on at least one panel; signature is
            unreliable).

        These are invariants the verify orchestrator enforces; the
        cache's job is to be a thread-safe LRU, not to second-guess
        the orchestrator's promotion policy.
        """
        if not signature or any(s is None for s in signature):
            return
        snap = _snapshot(
            signature=signature,
            beverage_type=beverage_type,
            extraction=extraction,
        )
        with self._lock:
            entry_id = self._next_id
            self._next_id += 1
            self._cache[entry_id] = snap
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._next_id = 0

    def stats(self) -> LookupStats:
        with self._lock:
            return LookupStats(
                hits=self._hits,
                misses=self._misses,
                size=len(self._cache),
                max_entries=self._max,
                hamming_threshold=self._threshold,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
