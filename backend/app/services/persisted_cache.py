"""L3 Postgres-backed perceptual cache for /v1/verify.

Three-tier verify cache architecture:

  L1  byte-exact LRU (verify_cache.py)            — SHA-256 keyed
  L2  perceptual LRU (reverse_lookup.py)          — dhash + Hamming
  L3  persisted perceptual store (this module)    — dhash + Hamming + DB

L1 and L2 are both in-process and die with the worker. L3 survives
process restart, scales beyond a single deploy's RAM, and carries
sibling-workstream enrichment data (TTB COLA matches, AI explanations)
that the verify path stitches in after the extraction is computed.

Why a separate persisted layer instead of just bigger L2:

  * Cold-start hit-rate. After a Fly redeploy or Railway rolling
    restart, the in-process caches are empty. L3 lets the new worker
    start serving cached verdicts immediately for any label the deploy
    has ever extracted.

  * Multi-worker coherence. A 256 MB Fly machine runs one Python
    process; a Railway deploy may run several. Each in-process cache
    is local to a worker, so a worker that processed a label can't
    share its hit with its peers without a shared store.

  * Enrichment pivot. The COLA reverse-search and the rule-explanation
    generator both run downstream of the extraction and write back to
    the same row. Holding that row in Postgres keeps the cache, the
    enrichment, and the audit trail aligned by ``entry_id``.

The cache contract mirrors L2's reverse-lookup layer: per-panel signature
plus ``beverage_type`` plus panel count must all match, with worst-case
per-panel Hamming distance ≤ ``hamming_threshold``. Linear scan of
candidate rows in Python; on the deploy's expected corpus (≤100K labels)
the popcount work lands well inside the latency budget. SQLite locally
(test setup) and Postgres in production both handle the candidate
filter via the composite ``(beverage_type, panel_count)`` index.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import get_session_factory
from app.models import LabelCacheEntry
from app.rules.types import ExtractedField
from app.services.vision import VisionExtraction

# Default Hamming-distance threshold mirrors L2's tuning (see
# reverse_lookup.DEFAULT_HAMMING_THRESHOLD): 6 bit-flips out of 64 keeps
# the false-positive rate under ~1 % across the imagehash literature's
# benchmarks. Surfaced as a constructor knob so an operator can tighten
# it if the dashboard ever shows reused-verdict drift.
DEFAULT_HAMMING_THRESHOLD = 6


@dataclass(frozen=True)
class PersistedCacheStats:
    """Snapshot of the persisted cache for the verify-path stats endpoint.

    ``total_hits`` aggregates the per-row ``hit_count`` column rather
    than tracking an in-process counter — the persisted store outlives
    any single worker, so the durable row counter is the only source
    that survives restart.
    """

    total_entries: int
    total_hits: int


@dataclass(frozen=True)
class PersistedHit:
    """Result of a successful L3 lookup.

    ``extraction`` is rebuilt from ``extraction_json`` with fresh mutable
    collections — the verify orchestrator caps confidence on the
    returned extraction post-merge, and we don't want that mutation to
    contaminate the JSON column on the next round-trip. ``min_distance``
    is the worst-case per-panel Hamming distance, useful for tuning.
    """

    entry_id: uuid.UUID
    extraction: VisionExtraction
    external_match: dict | None
    explanations: dict[str, str] | None
    min_distance: int
    signature: tuple[int, ...]


# ---------------------------------------------------------------------------
# Signature / Hamming helpers
# ---------------------------------------------------------------------------


def signature_to_hex(signature: tuple[int, ...]) -> str:
    """Encode a tuple of 64-bit dhash ints as comma-separated lower-case hex.

    Lower-case is enforced so a round-trip through the DB column never
    flips capitalisation between writes — the column is a plain string,
    so a stable canonical encoding lets us do dhash-exact matching by
    string equality on ``upsert``.
    """
    return ",".join(f"{h:x}" for h in signature)


def signature_from_hex(hex_str: str) -> tuple[int, ...]:
    """Decode the comma-separated hex format produced by ``signature_to_hex``."""
    if not hex_str:
        return ()
    return tuple(int(part, 16) for part in hex_str.split(","))


def hamming(a: int, b: int) -> int:
    """64-bit Hamming distance between two dhash signatures.

    ``int.bit_count`` (Python 3.10+) is roughly 3× faster than
    ``bin(...).count('1')`` on micro-benchmarks; preferred here because
    the lookup loop runs popcount per candidate × per panel.
    """
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# VisionExtraction <-> dict round-trip
# ---------------------------------------------------------------------------


def extraction_to_dict(extraction: VisionExtraction) -> dict[str, Any]:
    """Serialize a ``VisionExtraction`` to a JSON-compatible dict.

    Lives here rather than in ``vision.py`` because the cache layer is
    the only consumer of the round-trip — the verify path itself never
    serializes/deserializes its in-flight extraction. Keeping this
    helper module-local also keeps ``vision.py`` free of the JSON
    column's encoding concerns.
    """
    return {
        "fields": {
            name: {
                "value": fe.value,
                "bbox": list(fe.bbox) if fe.bbox is not None else None,
                "confidence": fe.confidence,
                "source_image_id": fe.source_image_id,
            }
            for name, fe in extraction.fields.items()
        },
        "unreadable": list(extraction.unreadable),
        "raw_response": extraction.raw_response,
        "image_quality": extraction.image_quality,
        "image_quality_notes": extraction.image_quality_notes,
        "beverage_type_observed": extraction.beverage_type_observed,
    }


def extraction_from_dict(data: dict[str, Any]) -> VisionExtraction:
    """Rebuild a ``VisionExtraction`` from the ``extraction_to_dict`` form.

    Builds fresh mutable collections so the caller can mutate without
    touching the original dict (the verify orchestrator caps confidence
    on the returned extraction post-merge).
    """
    fields: dict[str, ExtractedField] = {}
    for name, raw in data.get("fields", {}).items():
        bbox_raw = raw.get("bbox")
        bbox: tuple[int, int, int, int] | None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox = (
                int(bbox_raw[0]),
                int(bbox_raw[1]),
                int(bbox_raw[2]),
                int(bbox_raw[3]),
            )
        else:
            bbox = None
        fields[name] = ExtractedField(
            value=raw.get("value"),
            bbox=bbox,
            confidence=float(raw.get("confidence", 1.0)),
            source_image_id=raw.get("source_image_id"),
        )
    return VisionExtraction(
        fields=fields,
        unreadable=list(data.get("unreadable", [])),
        raw_response=str(data.get("raw_response", "")),
        image_quality=data.get("image_quality"),
        image_quality_notes=data.get("image_quality_notes"),
        beverage_type_observed=data.get("beverage_type_observed"),
    )


# ---------------------------------------------------------------------------
# PersistedLabelCache
# ---------------------------------------------------------------------------


class PersistedLabelCache:
    """Postgres-backed L3 perceptual cache.

    Async throughout: every public method opens its own session via the
    shared session factory so callers don't have to pass a transaction
    in. Each operation is a single short transaction — read-modify-write
    on hit (bump counter, refresh ``last_seen_at``) and a simple
    insert/update on ``upsert``. The candidate-set scan happens in
    Python after a SQL pre-filter on ``(beverage_type, panel_count)``,
    indexed at table-create time.
    """

    def __init__(self, *, hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD) -> None:
        if hamming_threshold < 0 or hamming_threshold > 64:
            raise ValueError("hamming_threshold must be in [0, 64]")
        self._threshold = hamming_threshold

    @property
    def hamming_threshold(self) -> int:
        return self._threshold

    def _factory(self) -> async_sessionmaker[AsyncSession]:
        # Resolved per call so tests that reconfigure the engine mid-test
        # (the SQLite-override pattern) pick up the new session factory
        # without having to reach into instance state.
        return get_session_factory()

    async def lookup(
        self,
        *,
        signature: tuple[int, ...],
        beverage_type: str,
    ) -> PersistedHit | None:
        """Return the best-matching entry within ``hamming_threshold``.

        The match contract mirrors L2: every panel must lie within the
        threshold; the binding constraint is the worst-of-N panel
        distance. Closest match (lowest worst-panel distance) wins when
        multiple eligible candidates exist. On a hit, the row's
        ``hit_count`` is bumped and ``last_seen_at`` refreshed so an
        operator can see which entries are durably valuable.
        """
        if not signature or any(s is None for s in signature):
            return None

        panel_count = len(signature)
        factory = self._factory()
        async with factory() as session:
            stmt = select(LabelCacheEntry).where(
                LabelCacheEntry.beverage_type == beverage_type,
                LabelCacheEntry.panel_count == panel_count,
            )
            rows = (await session.scalars(stmt)).all()

            best_row: LabelCacheEntry | None = None
            best_distance = self._threshold + 1
            best_signature: tuple[int, ...] = ()

            for row in rows:
                row_sig = signature_from_hex(row.signature_hex)
                if len(row_sig) != panel_count:
                    # Defensive: a row with a corrupted signature_hex
                    # should not be considered a candidate.
                    continue
                worst = 0
                ok = True
                for query_h, entry_h in zip(signature, row_sig, strict=True):
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
                    best_row = row
                    best_signature = row_sig
                    if worst == 0:
                        break

            if best_row is None:
                return None

            now = datetime.now(UTC)
            await session.execute(
                update(LabelCacheEntry)
                .where(LabelCacheEntry.id == best_row.id)
                .values(
                    hit_count=LabelCacheEntry.hit_count + 1,
                    last_seen_at=now,
                )
            )
            await session.commit()

            extraction = extraction_from_dict(best_row.extraction_json)
            return PersistedHit(
                entry_id=best_row.id,
                extraction=extraction,
                external_match=(
                    dict(best_row.external_match_json)
                    if best_row.external_match_json is not None
                    else None
                ),
                explanations=(
                    dict(best_row.explanations_json)
                    if best_row.explanations_json is not None
                    else None
                ),
                min_distance=best_distance,
                signature=best_signature,
            )

    async def upsert(
        self,
        *,
        signature: tuple[int, ...],
        beverage_type: str,
        extraction: VisionExtraction,
    ) -> uuid.UUID:
        """Insert a new entry, or update extraction on dhash-exact match.

        "dhash-exact" means the comma-separated hex of the signature is
        identical to a row already in the table — same labels in the
        same panel order. On exact-match we refresh the extraction
        (the cold path may have produced a higher-confidence read
        than the cached one), update ``updated_at`` / ``last_seen_at``,
        and re-derive ``brand_name_normalized`` from the freshly stored
        extraction so the index never goes stale relative to the JSON.
        Other enrichment columns (TTB COLA, AI explanations,
        ``first_frame_signature_hex``) are left alone — they belong to
        other workstreams and are written via the dedicated update
        methods / ``enrich_verdict``'s post-upsert stamp.
        """
        sig_hex = signature_to_hex(signature)
        extraction_json = extraction_to_dict(_freeze_extraction(extraction))
        brand_norm = _brand_normalized_from_extraction(extraction_json)
        factory = self._factory()
        async with factory() as session:
            stmt = select(LabelCacheEntry).where(
                LabelCacheEntry.signature_hex == sig_hex,
                LabelCacheEntry.beverage_type == beverage_type,
                LabelCacheEntry.panel_count == len(signature),
            )
            existing = (await session.scalars(stmt)).first()
            now = datetime.now(UTC)
            if existing is not None:
                await session.execute(
                    update(LabelCacheEntry)
                    .where(LabelCacheEntry.id == existing.id)
                    .values(
                        extraction_json=extraction_json,
                        brand_name_normalized=brand_norm,
                        last_seen_at=now,
                    )
                )
                await session.commit()
                return existing.id

            entry = LabelCacheEntry(
                beverage_type=beverage_type,
                panel_count=len(signature),
                signature_hex=sig_hex,
                brand_name_normalized=brand_norm,
                first_frame_signature_hex=None,
                extraction_json=extraction_json,
                external_match_json=None,
                explanations_json=None,
                hit_count=0,
            )
            session.add(entry)
            await session.commit()
            return entry.id

    async def lookup_by_brand(
        self,
        beverage_type: str | None,
        brand_name: str,
    ) -> PersistedHit | None:
        """Return the best-matching row whose ``brand_name_normalized``
        equals the supplied brand_name (case-insensitive, stripped).

        ``beverage_type=None`` queries across every type — the
        detect-container gate has the user's declared brand but doesn't
        always know the beverage type at probe time.

        Fail-honestly (SPEC §0.5): a candidate row is ineligible if its
        cached extraction was a low-confidence read. Proxy:
        ``extraction_json["fields"]["brand_name"]["confidence"] < 0.5``.
        Returning a recognition payload built off a downgraded extraction
        would let a stale or disputed read drive the user past the scan
        flow with a confident-looking "view results" affordance.
        """
        if not brand_name or not brand_name.strip():
            return None
        normalized = brand_name.strip().lower()
        factory = self._factory()
        async with factory() as session:
            stmt = (
                select(LabelCacheEntry)
                .where(LabelCacheEntry.brand_name_normalized == normalized)
                .order_by(
                    LabelCacheEntry.hit_count.desc(),
                    LabelCacheEntry.last_seen_at.desc(),
                )
                .limit(5)
            )
            if beverage_type is not None:
                stmt = stmt.where(LabelCacheEntry.beverage_type == beverage_type)
            rows = (await session.scalars(stmt)).all()
            if not rows:
                return None

            chosen: LabelCacheEntry | None = None
            for row in rows:
                if _row_extraction_is_eligible(row.extraction_json):
                    chosen = row
                    break
            if chosen is None:
                return None

            now = datetime.now(UTC)
            await session.execute(
                update(LabelCacheEntry)
                .where(LabelCacheEntry.id == chosen.id)
                .values(
                    hit_count=LabelCacheEntry.hit_count + 1,
                    last_seen_at=now,
                )
            )
            await session.commit()

            extraction = extraction_from_dict(chosen.extraction_json)
            row_signature = signature_from_hex(chosen.signature_hex)
            return PersistedHit(
                entry_id=chosen.id,
                extraction=extraction,
                external_match=(
                    dict(chosen.external_match_json)
                    if chosen.external_match_json is not None
                    else None
                ),
                explanations=(
                    dict(chosen.explanations_json)
                    if chosen.explanations_json is not None
                    else None
                ),
                # Brand match is not a Hamming probe — distance is N/A,
                # surface a sentinel zero so downstream tuning logic
                # doesn't misread the field.
                min_distance=0,
                signature=row_signature,
            )

    async def lookup_by_first_frame(
        self,
        signature_hex: str,
        beverage_type: str | None = None,
    ) -> PersistedHit | None:
        """Return any row whose ``first_frame_signature_hex`` lies within
        ``self._threshold`` of ``signature_hex``.

        Decodes both hashes to 64-bit ints and uses the same Hamming
        helper as the panel-signature lookup so the tuning pivot
        (``self._threshold``) stays consistent.

        Fail-honestly: same eligibility filter as ``lookup_by_brand``
        — skip rows whose cached brand_name extraction was a
        low-confidence read.
        """
        if not signature_hex or not signature_hex.strip():
            return None
        try:
            query_int = int(signature_hex.strip(), 16)
        except ValueError:
            return None

        factory = self._factory()
        async with factory() as session:
            stmt = select(LabelCacheEntry).where(
                LabelCacheEntry.first_frame_signature_hex.is_not(None)
            )
            if beverage_type is not None:
                stmt = stmt.where(LabelCacheEntry.beverage_type == beverage_type)
            rows = (await session.scalars(stmt)).all()

            best_row: LabelCacheEntry | None = None
            best_distance = self._threshold + 1

            for row in rows:
                row_hex = (row.first_frame_signature_hex or "").strip()
                if not row_hex:
                    continue
                try:
                    row_int = int(row_hex, 16)
                except ValueError:
                    continue
                d = hamming(query_int, row_int)
                if d > self._threshold:
                    continue
                if not _row_extraction_is_eligible(row.extraction_json):
                    continue
                if d < best_distance:
                    best_distance = d
                    best_row = row
                    if d == 0:
                        break

            if best_row is None:
                return None

            now = datetime.now(UTC)
            await session.execute(
                update(LabelCacheEntry)
                .where(LabelCacheEntry.id == best_row.id)
                .values(
                    hit_count=LabelCacheEntry.hit_count + 1,
                    last_seen_at=now,
                )
            )
            await session.commit()

            extraction = extraction_from_dict(best_row.extraction_json)
            row_signature = signature_from_hex(best_row.signature_hex)
            return PersistedHit(
                entry_id=best_row.id,
                extraction=extraction,
                external_match=(
                    dict(best_row.external_match_json)
                    if best_row.external_match_json is not None
                    else None
                ),
                explanations=(
                    dict(best_row.explanations_json)
                    if best_row.explanations_json is not None
                    else None
                ),
                min_distance=best_distance,
                signature=row_signature,
            )

    async def stamp_first_frame_signature(
        self, entry_id: uuid.UUID, signature_hex: str
    ) -> None:
        """Idempotently stamp ``first_frame_signature_hex`` on a row.

        ``IS NULL`` guard so a label scanned multiple times only ever
        records the first observed frame hash — subsequent finalizes
        no-op. Centralizing the write here keeps ``finalize_scan`` from
        touching the L3 row directly (only ``enrich_verdict`` does).
        """
        if not signature_hex or not signature_hex.strip():
            return
        normalized = signature_hex.strip().lower()
        factory = self._factory()
        async with factory() as session:
            await session.execute(
                update(LabelCacheEntry)
                .where(
                    LabelCacheEntry.id == entry_id,
                    LabelCacheEntry.first_frame_signature_hex.is_(None),
                )
                .values(first_frame_signature_hex=normalized)
            )
            await session.commit()

    async def stamp_brand_name_normalized(
        self, entry_id: uuid.UUID, brand_name: str | None
    ) -> None:
        """Idempotently stamp ``brand_name_normalized`` on a row.

        ``IS NULL`` guard so the column never silently rewrites itself
        on subsequent scans. The ``upsert`` path also writes this
        column from the same extraction; this helper exists so a
        downstream caller (``enrich_verdict``) can stamp the field
        without re-running ``upsert`` when the row already existed.
        """
        if brand_name is None:
            return
        normalized = brand_name.strip().lower()
        if not normalized:
            return
        factory = self._factory()
        async with factory() as session:
            await session.execute(
                update(LabelCacheEntry)
                .where(
                    LabelCacheEntry.id == entry_id,
                    LabelCacheEntry.brand_name_normalized.is_(None),
                )
                .values(brand_name_normalized=normalized)
            )
            await session.commit()

    async def update_external_match(
        self, entry_id: uuid.UUID, match: dict | None
    ) -> None:
        """Stitch a TTB COLA match payload onto the cached entry."""
        factory = self._factory()
        async with factory() as session:
            await session.execute(
                update(LabelCacheEntry)
                .where(LabelCacheEntry.id == entry_id)
                .values(external_match_json=match)
            )
            await session.commit()

    async def update_explanations(
        self, entry_id: uuid.UUID, explanations: dict[str, str]
    ) -> None:
        """Stitch the rule-id → explanation map onto the cached entry."""
        factory = self._factory()
        async with factory() as session:
            await session.execute(
                update(LabelCacheEntry)
                .where(LabelCacheEntry.id == entry_id)
                .values(explanations_json=explanations)
            )
            await session.commit()

    async def stats(self) -> PersistedCacheStats:
        """Aggregate counters across the persisted store.

        ``total_hits`` sums the per-row ``hit_count`` column instead of
        tracking an in-process counter — the row counter survives
        restart and is the only one that reflects lifetime cache value.
        """
        from sqlalchemy import func as sa_func

        factory = self._factory()
        async with factory() as session:
            count_stmt = select(sa_func.count(LabelCacheEntry.id))
            sum_stmt = select(sa_func.coalesce(sa_func.sum(LabelCacheEntry.hit_count), 0))
            total_entries = int((await session.execute(count_stmt)).scalar() or 0)
            total_hits = int((await session.execute(sum_stmt)).scalar() or 0)
            return PersistedCacheStats(
                total_entries=total_entries,
                total_hits=total_hits,
            )

    async def top_hit_entries(
        self, *, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return the top-`limit` most-hit entries for admin inspection.

        Each entry is shaped ``{"signature_prefix", "hit_count",
        "last_seen_at"}`` — the prefix-only signature lets an operator
        eyeball "this is the same label" without leaking the full
        perceptual fingerprint of every cached row to whoever has the
        admin token. Sorted by ``hit_count`` descending; ties broken
        by ``last_seen_at`` (most recent first) so a hot recent entry
        ranks above a cold historical one with the same hit count.
        """
        factory = self._factory()
        async with factory() as session:
            stmt = (
                select(LabelCacheEntry)
                .order_by(
                    LabelCacheEntry.hit_count.desc(),
                    LabelCacheEntry.last_seen_at.desc(),
                )
                .limit(limit)
            )
            rows = (await session.scalars(stmt)).all()
            results: list[dict[str, Any]] = []
            for row in rows:
                # First 16 hex chars = first 64 bits = first panel's
                # full dhash. That's enough for an operator to
                # disambiguate two near-collision entries by eye while
                # not leaking the entire perceptual fingerprint.
                sig_prefix = (row.signature_hex or "")[:16]
                last_seen = row.last_seen_at
                results.append(
                    {
                        "signature_prefix": sig_prefix,
                        "hit_count": int(row.hit_count or 0),
                        "last_seen_at": (
                            last_seen.isoformat()
                            if last_seen is not None
                            else None
                        ),
                    }
                )
            return results


def _brand_normalized_from_extraction(
    extraction_json: dict[str, Any],
) -> str | None:
    """Lift ``fields.brand_name.value`` out of the JSON form, lower + strip.

    Returns ``None`` when the brand isn't present, isn't a string, or
    normalizes to empty. Matched at upsert time (when we already have
    the dict form) and re-derived in ``stamp_brand_name_normalized`` /
    enrichment to keep the index aligned with the row's content.
    """
    fields = extraction_json.get("fields") if isinstance(extraction_json, dict) else None
    if not isinstance(fields, dict):
        return None
    info = fields.get("brand_name")
    if not isinstance(info, dict):
        return None
    value = info.get("value")
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _row_extraction_is_eligible(extraction_json: Any) -> bool:
    """Fail-honestly gate for the known-label lookups (SPEC §0.5).

    A cached row is ineligible when its ``brand_name`` confidence
    indicates a low-quality or downgraded read — proxy for the
    cross-check disagreement / health-warning advisory cases. Returning
    a recognition payload built off such a row would let the user
    skip the panorama capture on a row whose verdict we already
    refused to stand behind.
    """
    if not isinstance(extraction_json, dict):
        return False
    fields = extraction_json.get("fields")
    if not isinstance(fields, dict):
        return False
    info = fields.get("brand_name")
    if not isinstance(info, dict):
        return False
    confidence = info.get("confidence")
    try:
        return float(confidence) >= 0.5
    except (TypeError, ValueError):
        return False


def _freeze_extraction(extraction: VisionExtraction) -> VisionExtraction:
    """Deep-copy each ExtractedField so a caller mutation post-upsert
    can't reach into the JSON column we're about to serialize.

    ``ExtractedField`` is a mutable dataclass; the verify orchestrator
    rewrites ``confidence`` after the merge step. We materialise fresh
    field copies before serialisation so the JSON snapshot reflects the
    extraction at upsert time, not whatever the caller mutates after.
    """
    return VisionExtraction(
        fields={name: dc_replace(f) for name, f in extraction.fields.items()},
        unreadable=list(extraction.unreadable),
        raw_response=extraction.raw_response,
        image_quality=extraction.image_quality,
        image_quality_notes=extraction.image_quality_notes,
        beverage_type_observed=extraction.beverage_type_observed,
    )
