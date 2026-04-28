"""Tests for the Claude vision extractor.

Stubs the anthropic SDK so we exercise the prompt-construction, payload
shape, and result-mapping logic without making a real API call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.rules.types import ExtractionContext
from app.services.extractors.claude_vision import (
    ClaudeVisionExtractor,
    FieldExtraction,
    LabelExtraction,
    ProducerRecord,
    _to_context,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeMessages:
    """Records every messages.parse() call and returns a scripted result."""

    def __init__(self, scripted: LabelExtraction) -> None:
        self._scripted = scripted
        self.calls: list[dict] = []

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self._scripted, usage=None)


class FakeAnthropic:
    def __init__(self, scripted: LabelExtraction) -> None:
        self.messages = FakeMessages(scripted)


def _good_label() -> LabelExtraction:
    """A well-read, fully-populated label."""
    return LabelExtraction(
        beverage_type_observed="beer",
        image_quality="good",
        image_quality_notes="Sharp, well-lit, all elements visible.",
        brand_name=FieldExtraction(
            value="ANYTOWN ALE", confidence=0.96, surface="front", bbox=(10, 20, 300, 80)
        ),
        class_type=FieldExtraction(
            value="India Pale Ale", confidence=0.93, surface="front"
        ),
        alcohol_content=FieldExtraction(
            value="5.5% ABV", confidence=0.97, surface="front"
        ),
        net_contents=FieldExtraction(
            value="12 FL OZ", confidence=0.95, surface="front"
        ),
        name_address=FieldExtraction(
            value="Brewed and bottled by Anytown Brewing Co., Anytown, ST",
            confidence=0.92,
            surface="back",
        ),
        country_of_origin=FieldExtraction(
            value=None, confidence=0.0, note="not present"
        ),
        health_warning=FieldExtraction(
            value=(
                "GOVERNMENT WARNING: (1) According to the Surgeon General, "
                "women should not drink alcoholic beverages during pregnancy "
                "because of the risk of birth defects. (2) Consumption of "
                "alcoholic beverages impairs your ability to drive a car or "
                "operate machinery, and may cause health problems."
            ),
            confidence=0.94,
            surface="back",
        ),
        other_observations=None,
    )


def _png_bytes() -> bytes:
    """Smallest valid PNG bytes the API will accept; content irrelevant here."""
    # 1x1 transparent PNG.
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63000100000005000100"
        "5e6c5dd80000000049454e44ae426082"
    )


# ---------------------------------------------------------------------------
# Extract → ExtractionContext mapping
# ---------------------------------------------------------------------------


def test_good_label_maps_to_clean_extraction_context():
    extractor = ClaudeVisionExtractor(client=FakeAnthropic(_good_label()))
    ctx = extractor.extract(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": _png_bytes(), "back": _png_bytes()},
    )

    assert isinstance(ctx, ExtractionContext)
    assert ctx.application["model_provider"] == "claude_opus_4_7"
    assert ctx.application["image_quality"] == "good"
    assert ctx.unreadable_fields == []
    assert "brand_name" in ctx.fields
    assert ctx.fields["brand_name"].value == "ANYTOWN ALE"
    assert ctx.fields["brand_name"].source_image_id == "front"
    assert ctx.fields["health_warning"].value.startswith("GOVERNMENT WARNING:")
    # country_of_origin reported absent — must NOT appear in ctx.fields so the
    # presence check sees "missing" not "low confidence".
    assert "country_of_origin" not in ctx.fields
    assert ctx.abv_pct == pytest.approx(5.5)


def test_low_confidence_field_lands_in_unreadable_fields():
    label = _good_label()
    label.health_warning = FieldExtraction(
        value="GOVERNMENT WARNING: ... (mostly washed out)",
        confidence=0.4,
        surface="back",
        note="Glare across lower half of label",
    )
    extractor = ClaudeVisionExtractor(client=FakeAnthropic(label))
    ctx = extractor.extract(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": _png_bytes(), "back": _png_bytes()},
    )
    assert "health_warning" in ctx.unreadable_fields
    assert ctx.fields["health_warning"].confidence == 0.4


def test_unreadable_image_pushes_every_field_to_unreadable():
    label = _good_label()
    label.image_quality = "unreadable"
    label.image_quality_notes = "Bottle photo is rotated 30°, severe glare on top half."
    label.health_warning.confidence = 0.7  # individually OK, but image_quality wins
    extractor = ClaudeVisionExtractor(client=FakeAnthropic(label))
    ctx = extractor.extract(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": _png_bytes(), "back": _png_bytes()},
    )
    # Every populated field should be unreadable when image_quality is unreadable.
    assert set(ctx.unreadable_fields) >= set(ctx.fields.keys())


def test_producer_record_is_attached_to_application_context():
    label = _good_label()
    extractor = ClaudeVisionExtractor(client=FakeAnthropic(label))
    record = ProducerRecord(
        brand="Anytown Ale", class_type="IPA", container_size_ml=355
    )
    ctx = extractor.extract(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": _png_bytes()},
        producer_record=record,
    )
    assert ctx.application["producer_record"]["brand"] == "Anytown Ale"
    assert ctx.application["producer_record"]["class_type"] == "IPA"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_extract_call_uses_claude_opus_4_7_with_caching_and_thinking():
    fake = FakeAnthropic(_good_label())
    extractor = ClaudeVisionExtractor(client=fake, model="claude-opus-4-7")
    extractor.extract(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": _png_bytes()},
    )

    [call] = fake.messages.calls
    assert call["model"] == "claude-opus-4-7"
    # Adaptive thinking — let the model decide how much reasoning each label
    # needs (a clean front gets a fast answer, a degraded one thinks more).
    assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
    # System prompt is sent under cache_control so the prefix is paid once.
    [system_block] = call["system"]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert "TTB" in system_block["text"]
    # Structured output binding.
    assert call["output_format"] is LabelExtraction


def test_extract_attaches_every_supplied_surface_image_to_user_content():
    fake = FakeAnthropic(_good_label())
    extractor = ClaudeVisionExtractor(client=fake)
    extractor.extract(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": _png_bytes(), "back": _png_bytes()},
    )
    [call] = fake.messages.calls
    user_msg = call["messages"][0]
    image_blocks = [b for b in user_msg["content"] if b["type"] == "image"]
    assert len(image_blocks) == 2
    for block in image_blocks:
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"


def test_extract_briefs_model_with_capture_quality_prior():
    fake = FakeAnthropic(_good_label())
    extractor = ClaudeVisionExtractor(client=fake)
    fake_capture = SimpleNamespace(
        overall_verdict="degraded",
        overall_confidence=0.55,
        surfaces=[
            SimpleNamespace(
                surface="front",
                verdict="good",
                confidence=0.92,
                issues=[],
                sensor=SimpleNamespace(describe=lambda: "Apple iPhone 15 · 12 MP"),
            ),
            SimpleNamespace(
                surface="back",
                verdict="degraded",
                confidence=0.4,
                issues=["Noticeable glare on label (28% clipped pixels)"],
                sensor=SimpleNamespace(describe=lambda: "Apple iPhone 15 · 12 MP"),
            ),
        ],
    )
    extractor.extract(
        beverage_type="beer",
        container_size_ml=355,
        images={"front": _png_bytes(), "back": _png_bytes()},
        capture_quality=fake_capture,
    )
    [call] = fake.messages.calls
    text_blocks = [
        b["text"] for b in call["messages"][0]["content"] if b["type"] == "text"
    ]
    body = "\n".join(text_blocks)
    assert "Sensor pre-check" in body
    assert "verdict=degraded" in body
    assert "glare" in body.lower()


def test_extract_rejects_empty_images():
    fake = FakeAnthropic(_good_label())
    extractor = ClaudeVisionExtractor(client=fake)
    with pytest.raises(ValueError):
        extractor.extract(
            beverage_type="beer",
            container_size_ml=355,
            images={},
        )


# ---------------------------------------------------------------------------
# _to_context unit-tests (mapping logic without the SDK at all)
# ---------------------------------------------------------------------------


def test_to_context_skips_genuinely_absent_fields():
    label = _good_label()
    label.country_of_origin = FieldExtraction(value=None, confidence=0.0)
    ctx = _to_context(
        label,
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        producer_record=None,
        confidence_threshold=0.6,
    )
    assert "country_of_origin" not in ctx.fields
    assert "country_of_origin" not in ctx.unreadable_fields


def test_to_context_threshold_governs_unreadable_classification():
    label = _good_label()
    label.brand_name = FieldExtraction(value="ANYTOWN", confidence=0.55)
    ctx_strict = _to_context(
        label,
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        producer_record=None,
        confidence_threshold=0.6,
    )
    ctx_lax = _to_context(
        label,
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        producer_record=None,
        confidence_threshold=0.5,
    )
    assert "brand_name" in ctx_strict.unreadable_fields
    assert "brand_name" not in ctx_lax.unreadable_fields
