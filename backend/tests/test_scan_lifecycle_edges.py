"""Edge-case + error-contract coverage for the /v1/scans lifecycle.

`test_api.py` covers the happy path (create → upload → finalize → report)
and a couple of rejections. These tests fill the remaining branches the
mobile client actually depends on but that no test exercised:

* `GET /v1/scans/{id}` status polling — the processing screen polls this,
  and the `status == "complete"` branch (which loads the report to attach
  `overall` + `image_quality`) was never hit.
* `POST .../finalize` re-entry guard (409) once a scan is already complete.
* `PUT .../upload/{surface}` rejections: unknown surface (400) and empty
  body (400).
* `POST .../rule-results/{rule_id}/flag` not-found branches (404 for a
  missing report and for a missing rule id).

All async route handlers run inside Starlette's TestClient worker thread;
see `pyproject.toml`'s `[tool.coverage.run] concurrency` for why coverage
must be told to trace it.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.scans import get_ocr_provider, get_vision_extractor
from app.main import app
from tests.test_api import CANONICAL_HW, _build_provider, _png_bytes


@pytest.fixture(autouse=True)
def _pin_vision_none():
    # Mirror test_api: pin the vision extractor to None so the OCR
    # fallback drives the verdict regardless of local .env keys.
    app.dependency_overrides[get_vision_extractor] = lambda: None
    yield
    app.dependency_overrides.clear()


def _compliant_provider():
    front = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    back = "Brewed and bottled by Anytown Brewing Co., Anytown, ST\n" + CANONICAL_HW
    return _build_provider(front, back)


def _create_and_finalize(client: TestClient) -> str:
    """Run a scan all the way to `complete` and return its id."""
    app.dependency_overrides[get_ocr_provider] = _compliant_provider
    create = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    assert create.status_code == 201, create.text
    scan_id = create.json()["scan_id"]
    for url in create.json()["upload_urls"]:
        path = url["signed_url"].replace(str(client.base_url), "")
        assert client.put(path, content=_png_bytes()).status_code == 204
    assert client.post(f"/v1/scans/{scan_id}/finalize").status_code == 200
    return scan_id


# --- GET /v1/scans/{id} (status polling) -----------------------------------


def test_get_scan_status_pending_before_finalize(db_setup, temp_storage):
    client = TestClient(app)
    create = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = create.json()["scan_id"]

    res = client.get(f"/v1/scans/{scan_id}")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["scan_id"] == scan_id
    # Freshly created, no report yet → uploading, no verdict attached.
    assert body["status"] == "uploading"
    assert body["overall"] is None
    assert body["image_quality"] is None


def test_get_scan_status_complete_attaches_verdict(db_setup, temp_storage):
    client = TestClient(app)
    scan_id = _create_and_finalize(client)

    res = client.get(f"/v1/scans/{scan_id}")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "complete"
    # The complete branch reads the report and surfaces the verdict so the
    # processing screen can route straight to the result.
    assert body["overall"] in {"pass", "advisory", "warn", "fail"}
    assert body["image_quality"] in {"good", "degraded"}


def test_get_scan_unknown_id_returns_404(db_setup, temp_storage):
    client = TestClient(app)
    assert client.get(f"/v1/scans/{uuid.uuid4()}").status_code == 404
    # A non-UUID path segment resolves through the same guard.
    assert client.get("/v1/scans/not-a-uuid").status_code == 404


# --- POST /v1/scans/{id}/finalize (re-entry guard) -------------------------


def test_finalize_twice_conflicts(db_setup, temp_storage):
    client = TestClient(app)
    scan_id = _create_and_finalize(client)

    # Second finalize: scan is already `complete`, not in {uploading, failed}.
    again = client.post(f"/v1/scans/{scan_id}/finalize")
    assert again.status_code == 409, again.text
    assert "complete" in again.json()["detail"]


# --- PUT /v1/scans/{id}/upload/{surface} (rejections) ----------------------


def test_upload_unknown_surface_rejected(db_setup, temp_storage):
    client = TestClient(app)
    create = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = create.json()["scan_id"]

    res = client.put(f"/v1/scans/{scan_id}/upload/sideways", content=_png_bytes())
    assert res.status_code == 400, res.text
    assert "sideways" in res.json()["detail"]


def test_upload_empty_body_rejected(db_setup, temp_storage):
    client = TestClient(app)
    create = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = create.json()["scan_id"]

    res = client.put(f"/v1/scans/{scan_id}/upload/panorama", content=b"")
    assert res.status_code == 400, res.text
    assert "empty body" in res.json()["detail"]


def test_upload_to_unknown_scan_returns_404(db_setup, temp_storage):
    client = TestClient(app)
    res = client.put(f"/v1/scans/{uuid.uuid4()}/upload/panorama", content=_png_bytes())
    assert res.status_code == 404


# --- POST /v1/scans/{id}/rule-results/{rule_id}/flag (not-found) -----------


def test_flag_returns_404_when_report_missing(db_setup, temp_storage):
    client = TestClient(app)
    # Created but never finalized → no Report row yet.
    create = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = create.json()["scan_id"]

    res = client.post(
        f"/v1/scans/{scan_id}/rule-results/beer.health_warning.exact_text/flag",
        json={"comment": "nope"},
    )
    assert res.status_code == 404, res.text
    assert "Report not found" in res.json()["detail"]


def test_flag_returns_404_for_unknown_rule_id(db_setup, temp_storage):
    client = TestClient(app)
    scan_id = _create_and_finalize(client)

    res = client.post(
        f"/v1/scans/{scan_id}/rule-results/beer.does.not.exist/flag",
        json={"comment": "nope"},
    )
    assert res.status_code == 404, res.text
    assert "Rule result not found" in res.json()["detail"]
