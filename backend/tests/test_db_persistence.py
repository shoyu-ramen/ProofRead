"""Tests that the persistence layer actually writes rows.

Goes through the public API and then asserts on the underlying tables
to confirm that scans, scan_images, ocr_results, extracted_fields,
reports, and rule_results all get the expected inserts at the right
lifecycle steps.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.scans import get_ocr_provider, get_vision_extractor
from app.db import get_session_factory
from app.main import app
from app.models import (
    ExtractedFieldRow,
    OCRResultRow,
    Report,
    RuleResultRow,
    Scan,
    ScanImage,
)
from app.services.ocr import MockOCRProvider
from tests.conftest import _make_synthetic_png

CANONICAL_HW = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


def _ocr_fixture(text: str) -> dict:
    return {
        "full_text": text,
        "blocks": [
            {"text": line, "bbox": [0, i * 30, 400, 25], "confidence": 0.99}
            for i, line in enumerate(text.split("\n"))
        ],
    }


def _build_provider(front: str, back: str):
    fixtures = {"front": _ocr_fixture(front), "back": _ocr_fixture(back)}

    class _Multi:
        def process(self, image_bytes, hint=None):
            return MockOCRProvider(fixtures[hint]).process(image_bytes, hint)

    return _Multi()


@pytest.fixture(autouse=True)
def _wipe_overrides():
    # Pin vision to None so the OCR fallback drives persistence: the API
    # endpoint resolves a vision extractor from settings, which would otherwise
    # call the real ClaudeVisionExtractor when ANTHROPIC_API_KEY is set in .env
    # and skip the OCR row inserts these tests are asserting on.
    app.dependency_overrides[get_vision_extractor] = lambda: None
    yield
    app.dependency_overrides.clear()


def test_create_scan_persists_row(db_setup, temp_storage):
    client = TestClient(app)
    r = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = r.json()["scan_id"]

    import asyncio

    async def _check():
        factory = get_session_factory()
        async with factory() as s:
            scans = (await s.scalars(select(Scan))).all()
            assert len(scans) == 1
            assert str(scans[0].id) == scan_id
            assert scans[0].status == "uploading"

    asyncio.run(_check())


def test_upload_persists_scan_image_and_storage_blob(db_setup, temp_storage):
    client = TestClient(app)
    r = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )

    upload_url = r.json()["upload_urls"][0]
    path = upload_url["signed_url"].replace(str(client.base_url), "")
    put = client.put(path, content=b"front-bytes")
    assert put.status_code == 204

    import asyncio

    async def _check():
        factory = get_session_factory()
        async with factory() as s:
            images = (await s.scalars(select(ScanImage))).all()
            assert len(images) == 1
            assert images[0].surface == "front"
            blob = await temp_storage.get(images[0].s3_key)
            assert blob == b"front-bytes"

    asyncio.run(_check())


def test_finalize_persists_full_report_chain(db_setup, temp_storage):
    front = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    back = "Brewed and bottled by Anytown Brewing Co., Anytown, ST\n" + CANONICAL_HW
    app.dependency_overrides[get_ocr_provider] = lambda: _build_provider(front, back)

    client = TestClient(app)
    r = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = r.json()["scan_id"]
    png = _make_synthetic_png()
    for url in r.json()["upload_urls"]:
        path = url["signed_url"].replace(str(client.base_url), "")
        client.put(path, content=png)

    fin = client.post(f"/v1/scans/{scan_id}/finalize")
    assert fin.status_code == 200

    import asyncio

    async def _check():
        factory = get_session_factory()
        async with factory() as s:
            assert (await s.scalars(select(OCRResultRow))).all(), "ocr_results empty"
            assert (await s.scalars(select(ExtractedFieldRow))).all(), "fields empty"
            reports = (await s.scalars(select(Report))).all()
            assert len(reports) == 1
            rules = (await s.scalars(select(RuleResultRow))).all()
            assert len(rules) >= 8, "expected all v1 beer rules persisted"

    asyncio.run(_check())


def test_get_scan_404_on_unknown_id(db_setup, temp_storage):
    client = TestClient(app)
    r = client.get("/v1/scans/00000000-0000-0000-0000-000000000999")
    assert r.status_code == 404


def test_get_scan_404_on_garbled_id(db_setup, temp_storage):
    client = TestClient(app)
    r = client.get("/v1/scans/not-a-uuid")
    assert r.status_code == 404


def test_report_409_when_not_finalized(db_setup, temp_storage):
    client = TestClient(app)
    r = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = r.json()["scan_id"]
    rep = client.get(f"/v1/scans/{scan_id}/report")
    assert rep.status_code == 409
