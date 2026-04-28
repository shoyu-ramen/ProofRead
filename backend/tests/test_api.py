"""FastAPI smoke test: full request lifecycle through the app."""

import pytest
from fastapi.testclient import TestClient

from app.api.scans import get_ocr_provider, get_vision_extractor
from app.main import app
from app.services.ocr import MockOCRProvider
from tests.conftest import _make_synthetic_png


def _png_bytes() -> bytes:
    """Real PNG bytes that pass the sensor capture-quality pre-check.

    The pipeline now decodes every upload through `sensor_check` before OCR;
    a flat single-color frame would be flagged unreadable (low contrast, low
    resolution). `_make_synthetic_png` produces a sharp, contrasty 2 MP+
    frame that passes the pre-check so these end-to-end tests exercise the
    rule-engine path, not the capture-rejection path.
    """
    return _make_synthetic_png()


def _ocr_fixture(text: str) -> dict:
    return {
        "full_text": text,
        "blocks": [
            {"text": line, "bbox": [0, i * 30, 400, 25], "confidence": 0.99}
            for i, line in enumerate(text.split("\n"))
        ],
    }


def _build_provider(front_text: str, back_text: str) -> object:
    fixtures = {
        "front": _ocr_fixture(front_text),
        "back": _ocr_fixture(back_text),
    }

    class _Multi:
        def process(self, image_bytes: bytes, hint: str | None = None):
            return MockOCRProvider(fixtures[hint]).process(image_bytes, hint)

    return _Multi()


CANONICAL_HW = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


@pytest.fixture(autouse=True)
def _wipe_overrides():
    # The API integration tests exercise the OCR + rule-engine path. The scan
    # endpoint resolves a vision extractor from settings, so when an
    # ANTHROPIC_API_KEY is present in the local .env the production
    # ClaudeVisionExtractor would actually be called against the synthetic test
    # PNG and (correctly) report it as unreadable. Pin vision to None for every
    # test so the pipeline's OCR fallback drives the verdict.
    app.dependency_overrides[get_vision_extractor] = lambda: None
    yield
    app.dependency_overrides.clear()


def test_health_endpoint(db_setup, temp_storage):
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_scan_lifecycle_compliant_label_passes(db_setup, temp_storage):
    front = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    back = "Brewed and bottled by Anytown Brewing Co., Anytown, ST\n" + CANONICAL_HW

    app.dependency_overrides[get_ocr_provider] = lambda: _build_provider(front, back)
    client = TestClient(app)

    create = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355, "is_imported": False},
    )
    assert create.status_code == 201, create.text
    scan_id = create.json()["scan_id"]

    for url in create.json()["upload_urls"]:
        path = url["signed_url"].replace(str(client.base_url), "")
        r = client.put(path, content=_png_bytes())
        assert r.status_code == 204, r.text

    finalize = client.post(f"/v1/scans/{scan_id}/finalize")
    assert finalize.status_code == 200, finalize.text
    body = finalize.json()
    assert body["status"] == "complete"
    assert body["overall"] in {"pass", "advisory"}
    # SPEC §0.5: image quality must be visible in the API surface so the
    # mobile client can show "couldn't verify — rescan" rather than guessing.
    assert body["image_quality"] in {"good", "degraded"}

    report = client.get(f"/v1/scans/{scan_id}/report")
    assert report.status_code == 200
    body = report.json()
    failures = [r for r in body["rule_results"] if r["status"] == "fail"]
    assert not failures, f"Unexpected failures: {failures}"
    assert body["image_quality"] in {"good", "degraded"}
    assert body["extractor"]  # populated


def test_scan_lifecycle_typo_in_warning_fails(db_setup, temp_storage):
    front = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    bad_back = (
        "Brewed and bottled by Anytown Brewing Co., Anytown, ST\n"
        + CANONICAL_HW.replace("Surgeon", "Sergent")
    )

    app.dependency_overrides[get_ocr_provider] = lambda: _build_provider(front, bad_back)
    client = TestClient(app)

    create = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = create.json()["scan_id"]
    for url in create.json()["upload_urls"]:
        path = url["signed_url"].replace(str(client.base_url), "")
        client.put(path, content=_png_bytes())

    finalize = client.post(f"/v1/scans/{scan_id}/finalize")
    assert finalize.status_code == 200
    assert finalize.json()["overall"] == "fail"

    report = client.get(f"/v1/scans/{scan_id}/report")
    hw = next(
        r for r in report.json()["rule_results"]
        if r["rule_id"] == "beer.health_warning.exact_text"
    )
    assert hw["status"] == "fail"
    assert hw["citation"] == "27 CFR 16.21"
    assert hw["expected"]
    assert hw["fix_suggestion"]


def test_create_scan_rejects_wine(db_setup, temp_storage):
    client = TestClient(app)
    r = client.post(
        "/v1/scans",
        json={"beverage_type": "wine", "container_size_ml": 750},
    )
    assert r.status_code == 400
    assert "v1 supports beer only" in r.json()["detail"]


def test_unreadable_capture_surfaces_in_api_response(db_setup, temp_storage):
    """A flat-grey upload (the canonical 'fail honestly' case) must produce an
    'unreadable' overall verdict that the API exposes alongside per-rule
    advisories — never a wrong-pass / wrong-fail."""
    flat = _make_synthetic_png(flat=True)

    front = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    back = "Brewed and bottled by Anytown Brewing Co., Anytown, ST\n" + CANONICAL_HW
    app.dependency_overrides[get_ocr_provider] = lambda: _build_provider(front, back)

    client = TestClient(app)
    create = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = create.json()["scan_id"]
    for url in create.json()["upload_urls"]:
        path = url["signed_url"].replace(str(client.base_url), "")
        client.put(path, content=flat)

    finalize = client.post(f"/v1/scans/{scan_id}/finalize")
    assert finalize.status_code == 200
    body = finalize.json()
    assert body["overall"] == "unreadable"
    assert body["image_quality"] == "unreadable"

    report = client.get(f"/v1/scans/{scan_id}/report").json()
    assert all(r["status"] != "fail" for r in report["rule_results"])
    assert report["image_quality_notes"]


def test_finalize_rejects_when_uploads_missing(db_setup, temp_storage):
    front = "ANYTOWN ALE"
    back = "anything"
    app.dependency_overrides[get_ocr_provider] = lambda: _build_provider(front, back)
    client = TestClient(app)

    create = client.post(
        "/v1/scans",
        json={"beverage_type": "beer", "container_size_ml": 355},
    )
    scan_id = create.json()["scan_id"]

    finalize = client.post(f"/v1/scans/{scan_id}/finalize")
    assert finalize.status_code == 400
    assert "missing surfaces" in finalize.json()["detail"]
