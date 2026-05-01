"""Tests for the single-shot Claude vision extractor used by /v1/verify.

The rich `extractors.claude_vision` extractor has its own coverage in
`test_claude_vision.py`; this file pins the simpler `services.vision`
extractor — specifically that we send adaptive thinking and a cached
system prompt on every call so the production verify path matches the
prototype's reasoning budget.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services.vision import (
    EXTRACTOR_BASE_FIELDS,
    ClaudeVisionExtractor,
    MockVisionExtractor,
    _build_user_text,
    _parse_vision_response,
    fields_for_beverage,
)

_VALID_RESPONSE = json.dumps(
    {
        "brand_name": {"value": "ANYTOWN ALE", "unreadable": False, "confidence": 0.95},
        "class_type": {"value": "India Pale Ale", "unreadable": False, "confidence": 0.93},
        "alcohol_content": {"value": "5.5% ABV", "unreadable": False, "confidence": 0.97},
        "net_contents": {"value": "12 FL OZ", "unreadable": False, "confidence": 0.95},
        "name_address": {
            "value": "Brewed by Anytown Co.",
            "unreadable": False,
            "confidence": 0.9,
        },
        "country_of_origin": {"value": None, "unreadable": True, "confidence": 0.0},
        "health_warning": {
            "value": "GOVERNMENT WARNING: ...",
            "unreadable": False,
            "confidence": 0.94,
        },
    }
)


def _response_namespace(scripted: str) -> SimpleNamespace:
    """Pack a scripted JSON string into the same shape `messages.create`
    returns: a list of content blocks the extractor concatenates."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=scripted)],
    )


class _FakeMessages:
    def __init__(self, scripted: str) -> None:
        self._scripted = scripted
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _response_namespace(self._scripted)


class _FakeClient:
    def __init__(self, scripted: str) -> None:
        self.messages = _FakeMessages(scripted)


def _png_bytes() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63000100000005000100"
        "5e6c5dd80000000049454e44ae426082"
    )


def test_extract_sends_deterministic_call_with_cached_system_prompt():
    """Regression: /v1/verify must hit the model with a cached system prompt
    (so the static OCR instructions don't pay the prefix cost on every
    request) and a deterministic temperature (so identical bytes produce
    identical transcriptions, which the in-process verify cache and any
    upstream cache tier rely on).

    Adaptive thinking was intentionally removed: on a structured-JSON
    transcription task the schema already disciplines the model, and the
    end-to-end latency budget rewards a fast, deterministic single pass
    plus the redundant Government-Warning second-pass running concurrently
    in `verify()`.
    """
    fake = _FakeClient(_VALID_RESPONSE)
    extractor = ClaudeVisionExtractor(client=fake, model="claude-sonnet-4-6")

    result = extractor.extract(_png_bytes(), media_type="image/png")
    assert "brand_name" in result.fields

    [call] = fake.messages.calls
    assert "thinking" not in call, (
        "Adaptive thinking was dropped from the verify-path extractor; the "
        "presence of `thinking` here implies a regression that re-introduces "
        "the latency it was removed to save."
    )
    assert call["temperature"] == 0.0
    [system_block] = call["system"]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert call["model"] == "claude-sonnet-4-6"
    # Output budget: a typical seven-field response lands around 350
    # tokens, so 1024 is plenty of headroom while still capping tail
    # latency on labels where the model would otherwise generate filler
    # before stopping. The previous 4096 came from an older prompt that
    # asked for more verbose notes and is no longer needed.
    assert 700 <= call["max_tokens"] <= 2048


def test_extract_attaches_image_with_correct_media_type():
    fake = _FakeClient(_VALID_RESPONSE)
    extractor = ClaudeVisionExtractor(client=fake)
    extractor.extract(_png_bytes(), media_type="image/jpeg")

    [call] = fake.messages.calls
    [user_msg] = call["messages"]
    [image_block] = [b for b in user_msg["content"] if b["type"] == "image"]
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/jpeg"


def test_extract_concatenates_text_blocks_from_response():
    """`messages.create` returns a list of content blocks; thinking and tool
    blocks may be interleaved. The extractor must concatenate only the
    text blocks before parsing JSON — picking up a thinking block as
    body would tank parse rates."""

    class _FakeMessagesWithThinking:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", thinking="…"),
                    SimpleNamespace(type="text", text=_VALID_RESPONSE),
                ]
            )

    client = SimpleNamespace(messages=_FakeMessagesWithThinking())
    extractor = ClaudeVisionExtractor(client=client)
    result = extractor.extract(_png_bytes())
    assert result.fields["brand_name"].value == "ANYTOWN ALE"


def test_parse_vision_response_rejects_non_json():
    with pytest.raises(ValueError):
        _parse_vision_response("not json at all")


def test_parse_vision_response_preserves_partial_alcohol_content_value():
    """Regression: when the vision model returns a partial value like "4.8"
    (digits without the "%") the verify path must still surface the
    extracted value, not strip it. The format rule will FAIL on that input
    with an actionable "Express ABV as ... '5.5% ABV'" message — that is
    informative; replacing the value with "(unreadable)" hides what the
    model actually read and forces the user to guess what was identified."""
    raw = json.dumps(
        {
            "alcohol_content": {"value": "4.8", "confidence": 0.65},
            "net_contents": {"value": "12", "confidence": 0.65},
            "brand_name": {"value": "NOTCH'D", "confidence": 0.76},
        }
    )
    result = _parse_vision_response(raw)
    assert result.fields["alcohol_content"].value == "4.8"
    assert result.fields["net_contents"].value == "12"
    assert result.fields["brand_name"].value == "NOTCH'D"
    assert "alcohol_content" not in result.unreadable
    assert "net_contents" not in result.unreadable


def test_parse_vision_response_recovers_json_with_trailing_prose():
    """Regression: Haiku occasionally appends qualifying commentary after
    the JSON object on degraded labels ("**Note:** the warning text was
    at an angle…"), even though the system prompt forbids it. The strict
    end-of-string fence regex used to bail on that case, raising
    ValueError → 500 from the verify endpoint, throwing away an
    otherwise-perfectly-valid extraction. Recover by walking the first
    balanced JSON object out of the response."""
    inner = json.dumps(
        {
            "brand_name": {"value": "GLARE-FOULED IPA", "confidence": 0.7},
            "class_type": {"value": None, "unreadable": True, "confidence": 0.0},
            "health_warning": {
                "value": (
                    "GOVERNMENT WARNING: (1) According to the Surgeon "
                    "General, women should not drink"
                ),
                "confidence": 0.65,
            },
            "image_quality": "degraded",
            "image_quality_notes": "specular highlight on right third",
            "beverage_type_observed": "beer",
        }
    )
    raw = (
        "```json\n"
        + inner
        + "\n```\n\n"
        + "**Note:** Image is at an angle and the warning text is partially "
        + "obscured by glare; transcribed what was legible."
    )
    result = _parse_vision_response(raw)
    assert "brand_name" in result.fields
    assert result.fields["brand_name"].value == "GLARE-FOULED IPA"
    assert "health_warning" in result.fields
    assert result.fields["health_warning"].value.startswith("GOVERNMENT WARNING")
    # The class_type field came back unreadable, so it's in the unreadable
    # list rather than fields.
    assert "class_type" in result.unreadable


# ---------------------------------------------------------------------------
# Beverage-type-conditional field routing
# ---------------------------------------------------------------------------


def test_fields_for_beverage_routes_per_type():
    """Each beverage_type pulls in its own conditional fields without
    leaking into the others. Beer gets the seven base fields only;
    wine adds sulfite/organic; spirits adds age_statement.
    """
    base = set(EXTRACTOR_BASE_FIELDS)
    assert set(fields_for_beverage("beer")) == base
    assert set(fields_for_beverage("wine")) == base | {
        "sulfite_declaration",
        "organic_certification",
    }
    assert set(fields_for_beverage("spirits")) == base | {"age_statement"}
    # Unknown beverage type / None falls back to base.
    assert set(fields_for_beverage("absinthe")) == base
    assert set(fields_for_beverage(None)) == base


def test_user_text_lists_beverage_specific_fields_for_wine():
    """The per-request user message must enumerate the wine-only
    conditional fields so the model knows to look for them.
    """
    text = _build_user_text(
        capture_quality=None,
        producer_record=None,
        beverage_type="wine",
        container_size_ml=750,
        is_imported=False,
    )
    assert "sulfite_declaration" in text
    assert "organic_certification" in text
    # No spirits-only field on a wine prompt.
    assert "age_statement" not in text


def test_user_text_lists_age_statement_for_spirits():
    text = _build_user_text(
        capture_quality=None,
        producer_record=None,
        beverage_type="spirits",
        container_size_ml=750,
        is_imported=False,
    )
    assert "age_statement" in text
    # No wine fields on a spirits prompt.
    assert "sulfite_declaration" not in text
    assert "organic_certification" not in text


def test_user_text_omits_conditional_fields_for_beer():
    """Beer panels must not be asked for sulfites, organic, or age — the
    prompt explicitly tells the model NOT to emit those keys."""
    text = _build_user_text(
        capture_quality=None,
        producer_record=None,
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
    )
    # Beer text should reference the don't-emit instruction by name.
    assert "do NOT" in text
    assert "sulfite_declaration" in text
    # The beverage-type-specific extraction header (used for wine/spirits)
    # must NOT fire for beer — there are no beverage-type-specific fields
    # on a beer panel to extract.
    assert "Beverage-type-specific fields (wine)" not in text
    assert "Beverage-type-specific fields (spirits)" not in text


def test_parse_vision_response_extracts_wine_conditional_fields():
    """A wine-shaped response with sulfite + organic must surface those
    fields in the parsed extraction.
    """
    raw = json.dumps(
        {
            "brand_name": {"value": "MOCKINGBIRD VINEYARDS", "confidence": 0.95},
            "class_type": {"value": "Cabernet Sauvignon", "confidence": 0.93},
            "alcohol_content": {"value": "13.8% Alc./Vol.", "confidence": 0.96},
            "net_contents": {"value": "750 mL", "confidence": 0.97},
            "name_address": {
                "value": "Bottled by Mockingbird Vineyards, Napa, California",
                "confidence": 0.91,
            },
            "country_of_origin": {"value": None, "unreadable": True},
            "health_warning": {"value": "GOVERNMENT WARNING: ...", "confidence": 0.92},
            "sulfite_declaration": {"value": "CONTAINS SULFITES", "confidence": 0.94},
            "organic_certification": {
                "value": "Made with Organic Grapes",
                "confidence": 0.88,
            },
        }
    )
    result = _parse_vision_response(raw)
    assert result.fields["sulfite_declaration"].value == "CONTAINS SULFITES"
    assert (
        result.fields["organic_certification"].value == "Made with Organic Grapes"
    )


def test_parse_vision_response_marks_unreadable_organic_when_absent():
    """When a wine label has no organic claim, the model returns
    unreadable: true and the parser routes it to `unreadable`.
    """
    raw = json.dumps(
        {
            "brand_name": {"value": "MOCKINGBIRD", "confidence": 0.95},
            "sulfite_declaration": {
                "value": "Contains Sulfites",
                "confidence": 0.93,
            },
            "organic_certification": {
                "value": None,
                "unreadable": True,
                "confidence": 0.0,
            },
        }
    )
    result = _parse_vision_response(raw)
    assert "organic_certification" in result.unreadable
    assert "sulfite_declaration" not in result.unreadable


def test_parse_vision_response_extracts_spirits_age_statement():
    raw = json.dumps(
        {
            "brand_name": {"value": "Old Tom Distillery", "confidence": 0.97},
            "class_type": {
                "value": "Kentucky Straight Bourbon Whiskey",
                "confidence": 0.97,
            },
            "age_statement": {"value": "Aged 4 Years", "confidence": 0.95},
        }
    )
    result = _parse_vision_response(raw)
    assert result.fields["age_statement"].value == "Aged 4 Years"


def test_mock_extractor_propagates_new_fields():
    """`MockVisionExtractor` must round-trip the three new fields so
    end-to-end tests can drive the rule engine with wine + spirits
    fixtures.
    """
    fixture = {
        "brand_name": "Mockingbird Vineyards",
        "sulfite_declaration": "Contains Sulfites",
        "organic_certification": "USDA Organic",
        "age_statement": "Aged 4 Years",
    }
    extractor = MockVisionExtractor(fixture)
    extraction = extractor.extract(b"\x00", media_type="image/png", beverage_type="wine")
    # Beverage-type-conditional fields the test fixture provides survive
    # the parse round-trip.
    assert extraction.fields["sulfite_declaration"].value == "Contains Sulfites"
    assert extraction.fields["organic_certification"].value == "USDA Organic"
    assert extraction.fields["age_statement"].value == "Aged 4 Years"
    assert extraction.fields["brand_name"].value == "Mockingbird Vineyards"
