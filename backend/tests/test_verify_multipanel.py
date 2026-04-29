"""Multi-panel verify path: per-panel extraction, merge, and HTTP contract.

The single-shot `verify()` path is exercised in `test_verify.py`. These
tests pin the multi-panel behavior:

  * Service-level merge: per-field source_image_id, highest-confidence
    wins on overlap, panel-level unreadable bookkeeping.
  * Endpoint shape: `images=` list takes precedence, legacy `image=`
    still works, panel_count is reported back, source_image_id round-
    trips through the API, validation rejects empty / over-long lists.

Why a panel-aware mock instead of `MockVisionExtractor`: the existing
mock returns the same fixture every call, so it can't distinguish
"front sees brand" from "back sees warning". This file's `_PanelMock`
indexes its fixtures by call order so the merge logic actually has
disagreement to resolve.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import verify as verify_api
from app.main import app
from app.rules.types import ExtractedField
from app.services.verify import Panel, VerifyInput, verify
from app.services.vision import VisionExtraction
from tests.conftest import _make_synthetic_png

CANONICAL_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


# A real PNG that passes the sensor pre-check. Reused so every test
# starts from a frame the sensor module won't short-circuit on.
_GOOD_PNG = _make_synthetic_png()
# Second panel needs different bytes so panel_0 and panel_1 don't
# collide on the cache key (and so the sensor pre-check sees two
# distinct surfaces). The text difference here is purely a marker —
# the mock extractor doesn't read it.
_GOOD_PNG_BACK = _make_synthetic_png(text="BACK PANEL")


def _bourbon_application(**overrides: Any) -> dict[str, Any]:
    record = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45",
        "net_contents": "750 mL",
        "name_address": "Old Tom Distilling Co., Bardstown, Kentucky",
        "country_of_origin": "USA",
    }
    record.update(overrides)
    return {"producer_record": record}


class _PanelMock:
    """Vision extractor that returns a different fixture per call.

    `fixtures` is a list of `{field_name: {value, confidence}}` dicts in
    panel order. Call N returns fixtures[N]. Anything beyond the list
    returns an empty extraction so a too-many-call test surfaces as
    "no fields read" rather than mysteriously matching the last fixture.
    The mock tracks each call's image_bytes so tests can assert which
    panel the extractor was actually given.
    """

    def __init__(self, fixtures: list[dict[str, dict[str, Any]]]):
        self._fixtures = fixtures
        self.calls: list[dict[str, Any]] = []

    def extract(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        **kwargs: Any,
    ) -> VisionExtraction:
        idx = len(self.calls)
        self.calls.append(
            {"image_bytes_len": len(image_bytes), "media_type": media_type, **kwargs}
        )
        spec = self._fixtures[idx] if idx < len(self._fixtures) else {}

        fields: dict[str, ExtractedField] = {}
        unreadable: list[str] = []
        for name, info in spec.items():
            value = info.get("value")
            if value is None:
                unreadable.append(name)
                continue
            fields[name] = ExtractedField(
                value=str(value),
                bbox=info.get("bbox"),
                confidence=float(info.get("confidence", 0.9)),
                # Per-panel mocks tag with "front"/"back" historically;
                # the merge step rewrites these to "panel_N" so the
                # pre-merge tag is irrelevant to the test contract.
                source_image_id=info.get("source_image_id", "front"),
            )
        return VisionExtraction(
            fields=fields,
            unreadable=unreadable,
            raw_response="",
            image_quality=spec.get("__image_quality"),
            image_quality_notes=spec.get("__image_quality_notes"),
            beverage_type_observed=spec.get("__beverage_type_observed", "unknown"),
        )


# ---------------------------------------------------------------------------
# Service-level merge
# ---------------------------------------------------------------------------


def test_two_panels_merge_disjoint_fields():
    """Front carries brand + ABV; back carries the warning + name/address.
    The merged extraction has every field, each tagged with its source panel."""
    extractor = _PanelMock(
        [
            {  # panel_0 (front)
                "brand_name": {"value": "Old Tom Distillery", "confidence": 0.97},
                "class_type": {
                    "value": "Kentucky Straight Bourbon Whiskey",
                    "confidence": 0.96,
                },
                "alcohol_content": {
                    "value": "45% Alc./Vol. (90 Proof)",
                    "confidence": 0.95,
                },
                "net_contents": {"value": "750 mL", "confidence": 0.94},
            },
            {  # panel_1 (back)
                "name_address": {
                    "value": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
                    "confidence": 0.92,
                },
                "health_warning": {"value": CANONICAL_WARNING, "confidence": 0.96},
            },
        ]
    )

    inp = VerifyInput(
        image_bytes=_GOOD_PNG,
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=_bourbon_application(),
        extra_panels=[Panel(image_bytes=_GOOD_PNG_BACK, media_type="image/png")],
    )
    report = verify(inp, extractor=extractor)

    # Both panels were extracted — multi-panel parallelism kicked in.
    assert len(extractor.calls) == 2

    sources = {
        name: info.get("source_image_id") for name, info in report.extracted.items()
    }
    assert sources["brand_name"] == "panel_0"
    assert sources["alcohol_content"] == "panel_0"
    assert sources["health_warning"] == "panel_1"
    assert sources["name_address"] == "panel_1"
    # Verdict should be `pass` — every required field came from some panel.
    assert report.overall == "pass", [
        (r.rule_id, r.status.value, r.finding) for r in report.rule_results
    ]


def test_two_panels_merge_overlap_picks_higher_confidence():
    """Both panels return brand_name; the higher-confidence read wins and
    its source_image_id is the winning panel."""
    extractor = _PanelMock(
        [
            {  # panel_0: lower-confidence brand
                "brand_name": {"value": "Old Tom Distillery", "confidence": 0.70},
                "class_type": {
                    "value": "Kentucky Straight Bourbon Whiskey",
                    "confidence": 0.95,
                },
                "alcohol_content": {
                    "value": "45% Alc./Vol. (90 Proof)",
                    "confidence": 0.95,
                },
                "net_contents": {"value": "750 mL", "confidence": 0.95},
                "name_address": {
                    "value": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
                    "confidence": 0.92,
                },
                "health_warning": {"value": CANONICAL_WARNING, "confidence": 0.95},
            },
            {  # panel_1: higher-confidence brand → wins on merge
                "brand_name": {"value": "Old Tom Distillery", "confidence": 0.99},
            },
        ]
    )

    inp = VerifyInput(
        image_bytes=_GOOD_PNG,
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=_bourbon_application(),
        extra_panels=[Panel(image_bytes=_GOOD_PNG_BACK, media_type="image/png")],
    )
    report = verify(inp, extractor=extractor)

    brand_info = report.extracted["brand_name"]
    assert brand_info["source_image_id"] == "panel_1"
    # Confidence is capped at the surface confidence; we only assert the
    # *direction* — the merged confidence reflects panel_1's read, which
    # has at least the cap applied to 0.99.
    assert brand_info["confidence"] >= 0.9


def test_single_panel_via_panels_property_tags_panel_zero():
    """The single-panel path still produces source_image_id='panel_0' for
    every field — multi-panel response shape is the new uniform contract."""
    extractor = _PanelMock(
        [
            {
                "brand_name": {"value": "Old Tom Distillery", "confidence": 0.95},
                "class_type": {
                    "value": "Kentucky Straight Bourbon Whiskey",
                    "confidence": 0.95,
                },
                "alcohol_content": {
                    "value": "45% Alc./Vol. (90 Proof)",
                    "confidence": 0.95,
                },
                "net_contents": {"value": "750 mL", "confidence": 0.95},
                "name_address": {
                    "value": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
                    "confidence": 0.92,
                },
                "health_warning": {"value": CANONICAL_WARNING, "confidence": 0.95},
            }
        ]
    )

    inp = VerifyInput(
        image_bytes=_GOOD_PNG,
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=_bourbon_application(),
    )
    report = verify(inp, extractor=extractor)

    assert len(extractor.calls) == 1
    for name, info in report.extracted.items():
        if info.get("value") is not None:
            assert info["source_image_id"] == "panel_0", (name, info)


def test_field_unreadable_only_when_no_panel_finds_it():
    """If panel_0 says brand is unreadable but panel_1 reads it, the
    merged extraction must NOT list brand as unreadable — we have a value."""
    extractor = _PanelMock(
        [
            {
                # panel_0: brand_name unreadable
                "brand_name": {"value": None},
                "class_type": {
                    "value": "Kentucky Straight Bourbon Whiskey",
                    "confidence": 0.95,
                },
                "alcohol_content": {
                    "value": "45% Alc./Vol. (90 Proof)",
                    "confidence": 0.95,
                },
                "net_contents": {"value": "750 mL", "confidence": 0.95},
                "name_address": {
                    "value": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
                    "confidence": 0.92,
                },
                "health_warning": {"value": CANONICAL_WARNING, "confidence": 0.95},
            },
            {
                # panel_1: brand_name readable; nothing else relevant
                "brand_name": {"value": "Old Tom Distillery", "confidence": 0.95},
            },
        ]
    )

    inp = VerifyInput(
        image_bytes=_GOOD_PNG,
        media_type="image/png",
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
        application=_bourbon_application(),
        extra_panels=[Panel(image_bytes=_GOOD_PNG_BACK, media_type="image/png")],
    )
    report = verify(inp, extractor=extractor)

    assert "brand_name" not in report.unreadable_fields
    assert report.extracted["brand_name"]["value"] == "Old Tom Distillery"


# ---------------------------------------------------------------------------
# HTTP contract
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_extractor_cache():
    verify_api._extractor_cache = None
    verify_api._reset_verify_cache()
    yield
    verify_api._extractor_cache = None
    verify_api._reset_verify_cache()
    app.dependency_overrides.clear()


def _form_payload() -> dict[str, object]:
    return {
        "beverage_type": "spirits",
        "container_size_ml": "750",
        "is_imported": "false",
        "application": json.dumps({"producer_record": {}}),
    }


def _png_file(name: str = "label.png", body: bytes | None = None) -> tuple[str, io.BytesIO, str]:
    return (name, io.BytesIO(body or _GOOD_PNG), "image/png")


def test_endpoint_legacy_single_image_still_works(monkeypatch):
    """Backwards compat: a single `image=` upload reports panel_count=1
    and tags every extracted field as panel_0."""
    extractor = _PanelMock(
        [
            {
                "brand_name": {"value": "Old Tom Distillery", "confidence": 0.95},
                "class_type": {
                    "value": "Kentucky Straight Bourbon Whiskey",
                    "confidence": 0.95,
                },
                "alcohol_content": {
                    "value": "45% Alc./Vol. (90 Proof)",
                    "confidence": 0.95,
                },
                "net_contents": {"value": "750 mL", "confidence": 0.95},
                "name_address": {
                    "value": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
                    "confidence": 0.92,
                },
                "health_warning": {"value": CANONICAL_WARNING, "confidence": 0.95},
            }
        ]
    )
    monkeypatch.setattr(verify_api, "get_default_extractor", lambda: extractor)

    client = TestClient(app)
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files={"image": _png_file()},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["panel_count"] == 1
    for name, info in body["extracted"].items():
        if info.get("value") is not None:
            assert info["source_image_id"] == "panel_0"


def test_endpoint_multipanel_via_images_list(monkeypatch):
    """Two `images=` uploads → panel_count=2; brand from front, warning
    from back; source_image_ids round-trip in the response."""
    extractor = _PanelMock(
        [
            {
                "brand_name": {"value": "Old Tom Distillery", "confidence": 0.95},
                "class_type": {
                    "value": "Kentucky Straight Bourbon Whiskey",
                    "confidence": 0.95,
                },
                "alcohol_content": {
                    "value": "45% Alc./Vol. (90 Proof)",
                    "confidence": 0.95,
                },
                "net_contents": {"value": "750 mL", "confidence": 0.95},
            },
            {
                "name_address": {
                    "value": "Bottled by Old Tom Distilling Co., Bardstown, Kentucky",
                    "confidence": 0.92,
                },
                "health_warning": {"value": CANONICAL_WARNING, "confidence": 0.95},
            },
        ]
    )
    monkeypatch.setattr(verify_api, "get_default_extractor", lambda: extractor)

    client = TestClient(app)
    # `httpx`'s multipart shape: list of (field_name, value) tuples lets
    # us send two files under the same `images` field.
    files = [
        ("images", _png_file("front.png", _GOOD_PNG)),
        ("images", _png_file("back.png", _GOOD_PNG_BACK)),
    ]
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files=files,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["panel_count"] == 2
    assert body["extracted"]["brand_name"]["source_image_id"] == "panel_0"
    assert body["extracted"]["health_warning"]["source_image_id"] == "panel_1"
    # The mock should have been called once per panel.
    assert len(extractor.calls) == 2


def test_endpoint_rejects_no_image(monkeypatch):
    """Neither `image` nor `images` provided → 400 with a useful message."""
    monkeypatch.setattr(
        verify_api, "get_default_extractor", lambda: _PanelMock([{}])
    )
    client = TestClient(app)
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
    )
    assert res.status_code == 400, res.text
    # FastAPI wraps the HTTPException detail under "detail".
    assert "image" in res.json().get("detail", "").lower()


def test_endpoint_rejects_too_many_panels(monkeypatch):
    """5 panels → 400. Cap is 4 (front + back + neck + base on the worst-
    shaped bottle); beyond that the UX stops being a quick scan."""
    monkeypatch.setattr(
        verify_api, "get_default_extractor", lambda: _PanelMock([{}] * 5)
    )
    client = TestClient(app)
    files = [("images", _png_file(f"p{i}.png", _GOOD_PNG)) for i in range(5)]
    res = client.post(
        "/v1/verify",
        data=_form_payload(),
        files=files,
    )
    assert res.status_code == 400, res.text
    assert "panels" in res.json()["detail"].lower()
