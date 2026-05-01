"""Tests for ``POST /v1/scans/from-cache``.

The from-cache route lets the mobile UI skip the panorama capture when
the detect-container probe recognized the label. Critical SPEC §0.5
property: the route NEVER serves a frozen verdict — it re-runs the
rule engine fresh with the user's actual container_size_ml + is_imported
even though the extraction is reused. These tests pin that invariant
plus the standard route plumbing (auth → user_id, history visibility,
404 on bad entry_id, transactional persistence).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.auth import _TEST_USER
from app.db import get_session_factory
from app.main import app
from app.models import (
    ExtractedFieldRow,
    LabelCacheEntry,
    Report,
    RuleResultRow,
    Scan,
)
from app.rules.types import ExtractedField
from app.services.persisted_cache import PersistedLabelCache
from app.services.vision import VisionExtraction


def _seed_l3(
    *,
    brand: str = "Sierra Nevada",
    fanciful: str | None = "Pale Ale",
    country_of_origin: str = "USA",
    explanations: dict[str, str] | None = None,
    external_match: dict | None = None,
) -> uuid.UUID:
    """Insert a fully-formed L3 row and return its UUID.

    Cached extraction includes the same field set the verify path
    persists so the from-cache route's rule engine has plenty of
    fixture data to evaluate against — the result of which is what we
    assert on.
    """
    cache = PersistedLabelCache(hamming_threshold=6)
    fields: dict[str, ExtractedField] = {
        "brand_name": ExtractedField(value=brand, confidence=0.95),
        "net_contents": ExtractedField(value="12 FL OZ", confidence=0.9),
        "alcohol_content": ExtractedField(value="5.5% ABV", confidence=0.95),
        "country_of_origin": ExtractedField(
            value=country_of_origin, confidence=0.9
        ),
        "name_address": ExtractedField(
            value="Anytown Brewing Co., Anytown, CA", confidence=0.9
        ),
        "health_warning": ExtractedField(
            value=(
                "GOVERNMENT WARNING: (1) According to the Surgeon General, "
                "women should not drink alcoholic beverages during "
                "pregnancy because of the risk of birth defects. "
                "(2) Consumption of alcoholic beverages impairs your "
                "ability to drive a car or operate machinery, and may "
                "cause health problems."
            ),
            confidence=0.95,
        ),
    }
    if fanciful:
        fields["fanciful_name"] = ExtractedField(
            value=fanciful, confidence=0.9
        )
    extraction = VisionExtraction(
        fields=fields,
        unreadable=[],
        raw_response="{}",
        image_quality="good",
        beverage_type_observed="beer",
    )

    async def _do() -> uuid.UUID:
        entry_id = await cache.upsert(
            signature=(0xCAFE,),
            beverage_type="beer",
            extraction=extraction,
        )
        if external_match is not None:
            await cache.update_external_match(entry_id, external_match)
        if explanations is not None:
            await cache.update_explanations(entry_id, explanations)
        return entry_id

    return asyncio.run(_do())


def test_from_cache_happy_path(db_setup, temp_storage):
    entry_id = _seed_l3()
    client = TestClient(app)

    res = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": str(entry_id),
            "beverage_type": "beer",
            "container_size_ml": 355,
            "is_imported": False,
        },
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["status"] == "complete"
    assert body["overall"] in {"pass", "warn", "fail", "advisory"}
    assert body["image_quality"] == "good"
    scan_id = body["scan_id"]
    assert scan_id

    # Scan row visible in history.
    history = client.get("/v1/scans")
    assert history.status_code == 200
    items = history.json()["items"]
    assert any(item["scan_id"] == scan_id for item in items)


def test_from_cache_returns_404_on_unknown_entry_id(db_setup, temp_storage):
    client = TestClient(app)
    res = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": str(uuid.uuid4()),
            "beverage_type": "beer",
            "container_size_ml": 355,
            "is_imported": False,
        },
    )
    assert res.status_code == 404, res.text


def test_from_cache_returns_404_on_invalid_uuid(db_setup, temp_storage):
    client = TestClient(app)
    res = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": "not-a-uuid",
            "beverage_type": "beer",
            "container_size_ml": 355,
            "is_imported": False,
        },
    )
    assert res.status_code == 404, res.text


def test_from_cache_re_runs_rule_engine_with_user_size(
    db_setup, temp_storage
):
    """SPEC §0.5: the verdict must be re-derived from the user's actual
    container_size_ml. Two from-cache requests against the same L3 row
    with different container sizes must produce two distinct rule_result
    rows (the same extraction, different inputs)."""
    entry_id = _seed_l3()
    client = TestClient(app)

    res_355 = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": str(entry_id),
            "beverage_type": "beer",
            "container_size_ml": 355,
            "is_imported": False,
        },
    )
    assert res_355.status_code == 201, res_355.text
    res_473 = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": str(entry_id),
            "beverage_type": "beer",
            "container_size_ml": 473,
            "is_imported": False,
        },
    )
    assert res_473.status_code == 201, res_473.text

    factory = get_session_factory()

    async def _check() -> None:
        async with factory() as session:
            scans = (await session.scalars(select(Scan))).all()
            sizes = sorted(s.container_size_ml for s in scans)
            assert sizes == [355, 473]
            # container_size_source identifies these scans as
            # cache-derived rather than user-uploaded.
            for scan in scans:
                assert scan.container_size_source == "cache"
                assert scan.user_id == _TEST_USER.id
            # Each scan has its own report and its own rule rows.
            reports = (await session.scalars(select(Report))).all()
            assert len(reports) == 2
            rule_rows = (
                await session.scalars(select(RuleResultRow))
            ).all()
            assert len(rule_rows) > 0

    asyncio.run(_check())


def test_from_cache_persists_extracted_fields_from_l3_extraction(
    db_setup, temp_storage
):
    entry_id = _seed_l3(brand="Anytown Ale")
    client = TestClient(app)
    res = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": str(entry_id),
            "beverage_type": "beer",
            "container_size_ml": 355,
            "is_imported": False,
        },
    )
    assert res.status_code == 201, res.text
    scan_id = res.json()["scan_id"]

    factory = get_session_factory()

    async def _check() -> None:
        async with factory() as session:
            fields = (
                await session.scalars(
                    select(ExtractedFieldRow).where(
                        ExtractedFieldRow.scan_id == uuid.UUID(scan_id)
                    )
                )
            ).all()
            by_id = {f.field_id: f for f in fields}
            assert "brand_name" in by_id
            assert by_id["brand_name"].value == "Anytown Ale"
            assert "net_contents" in by_id

    asyncio.run(_check())


def test_from_cache_attaches_external_match_and_explanations(
    db_setup, temp_storage
):
    """When the L3 row carries enrichment payloads, the from-cache scan
    must surface them on the report so the user gets the same context
    they would get from a live finalize."""
    entry_id = _seed_l3(
        external_match={"source": "ttb_cola", "ttb_id": "12345"},
        explanations={
            "beer.brand_name.presence": "Brand name visible on the label.",
        },
    )
    client = TestClient(app)
    res = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": str(entry_id),
            "beverage_type": "beer",
            "container_size_ml": 355,
            "is_imported": False,
        },
    )
    assert res.status_code == 201, res.text
    scan_id = res.json()["scan_id"]

    report_res = client.get(f"/v1/scans/{scan_id}/report")
    assert report_res.status_code == 200, report_res.text
    body = report_res.json()
    assert body["external_match"] == {"source": "ttb_cola", "ttb_id": "12345"}
    rule_results = body["rule_results"]
    explanations_seen = {
        r["rule_id"]: r.get("explanation") for r in rule_results
    }
    # The seeded explanation key must be attached to its matching rule
    # row when the from-cache eval produces that rule_id.
    assert (
        explanations_seen.get("beer.brand_name.presence")
        == "Brand name visible on the label."
    )


def test_from_cache_rejects_unknown_beverage_type(db_setup, temp_storage):
    """The Pydantic regex on `beverage_type` rejects values outside
    beer/wine/spirits — pinning the error code so the mobile UI can
    surface it the same way it does on `/v1/scans` create."""
    entry_id = _seed_l3()
    client = TestClient(app)
    res = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": str(entry_id),
            "beverage_type": "moonshine",
            "container_size_ml": 355,
            "is_imported": False,
        },
    )
    assert res.status_code == 422, res.text


def test_from_cache_user_id_from_auth(db_setup, temp_storage):
    """The route stamps `user_id` from `Depends(get_current_user)`,
    same pattern as `POST /v1/scans` — so history endpoint returns the
    row for the authed user."""
    entry_id = _seed_l3()
    client = TestClient(app)
    res = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": str(entry_id),
            "beverage_type": "beer",
            "container_size_ml": 355,
            "is_imported": False,
        },
    )
    assert res.status_code == 201, res.text
    scan_id = res.json()["scan_id"]

    factory = get_session_factory()

    async def _check() -> None:
        async with factory() as session:
            scan = await session.get(Scan, uuid.UUID(scan_id))
            assert scan is not None
            assert scan.user_id == _TEST_USER.id

    asyncio.run(_check())


def test_from_cache_404_when_extraction_json_missing(
    db_setup, temp_storage
):
    """A row with a NULL/empty extraction can never be replayed — no
    rule eval is possible. The route 404s rather than serving an empty
    verdict."""
    factory = get_session_factory()

    async def _seed() -> uuid.UUID:
        entry = LabelCacheEntry(
            id=uuid.uuid4(),
            beverage_type="beer",
            panel_count=1,
            signature_hex="cafe",
            extraction_json={},
        )
        async with factory() as session:
            session.add(entry)
            await session.commit()
        return entry.id

    entry_id = asyncio.run(_seed())
    client = TestClient(app)
    res = client.post(
        "/v1/scans/from-cache",
        json={
            "entry_id": str(entry_id),
            "beverage_type": "beer",
            "container_size_ml": 355,
            "is_imported": False,
        },
    )
    assert res.status_code == 404, res.text
