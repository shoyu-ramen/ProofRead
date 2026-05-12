"""Replay-mode VisionExtractor — reads a recorded extraction from disk.

The recorded JSON shape mirrors what
`app.services.extractors.claude_vision._to_context(...)` produces — it's the
JSON-encodable serialisation of an `ExtractionContext`. Once an item has been
recorded by the live recorder (`validation/scripts/record_extraction.py`,
which uses the local Qwen3-VL fallback by default), every subsequent harness
run can score the rule engine against the same fixed extraction without
ever calling a vision model.

Why replay-mode is the harness default:

  1. Cost: zero recurring API spend during dev or CI.
  2. Determinism: the rule engine and verify orchestrator are scored
     against a frozen extraction, so a measured precision/recall change
     is unambiguously a rule-engine regression, not extractor drift.
  3. Speed: a CI run is bounded by Pillow + the rule engine — sub-second
     per item — instead of multi-second VLM RTTs.

Extractor drift is surveilled separately by re-recording weekly through
the local Qwen3-VL fallback (`record_extraction.py`); diffing the new
recording against the committed one surfaces drift without paying for
API calls.

JSON shape (one file per item, written next to truth.json):

    {
      "schema_version": 1,
      "model_provider": "qwen3_vl_local | claude_opus_4_7 | synth_from_truth",
      "fields": {
        "brand_name": {
          "value": "Caymus" | null,
          "bbox": [x, y, w, h] | null,
          "confidence": 0.95,
          "source_image_id": "front" | null
        },
        ...
      },
      "unreadable_fields": ["country_of_origin", ...],
      "application": {
        "image_quality": "good | degraded | unreadable",
        "image_quality_notes": "...",
        "model_provider": "...",
        "beverage_type_observed": "beer | wine | spirits | unknown",
        "model_observations": "..." | null
      },
      "abv_pct": 14.6 | null,
      "raw_ocr_texts": {}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rules.types import ExtractedField, ExtractionContext

RECORDING_SCHEMA_VERSION = 1


class RecordingSchemaError(ValueError):
    """Raised when a recorded_extraction.json fails schema validation.

    Hard-fails at the loader, never at the rule engine — a malformed
    recording silently producing an empty extraction would skew
    precision/recall numbers in a way that's painful to debug later.
    """


def _validate_payload(payload: dict[str, Any], source: str) -> None:
    schema = payload.get("schema_version")
    if schema != RECORDING_SCHEMA_VERSION:
        raise RecordingSchemaError(
            f"{source}: recording schema_version is {schema!r}, expected "
            f"{RECORDING_SCHEMA_VERSION}"
        )
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise RecordingSchemaError(f"{source}: `fields` must be an object")
    application = payload.get("application")
    if not isinstance(application, dict):
        raise RecordingSchemaError(f"{source}: `application` must be an object")


def _build_field(name: str, raw: dict[str, Any], source: str) -> ExtractedField:
    if not isinstance(raw, dict):
        raise RecordingSchemaError(
            f"{source}: field {name!r} must be an object, got {type(raw).__name__}"
        )
    bbox_raw = raw.get("bbox")
    bbox: tuple[int, int, int, int] | None
    if bbox_raw is None:
        bbox = None
    else:
        if not (isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4):
            raise RecordingSchemaError(
                f"{source}: field {name!r} bbox must be [x, y, w, h] or null"
            )
        bbox = tuple(int(v) for v in bbox_raw)  # type: ignore[assignment]
    return ExtractedField(
        value=raw.get("value"),
        bbox=bbox,
        confidence=float(raw.get("confidence", 0.0)),
        source_image_id=raw.get("source_image_id"),
    )


@dataclass
class ReplayVisionExtractor:
    """Drop-in `VisionExtractor` for the `process_scan` pipeline.

    Bound to a single recorded payload at construction. Use the class
    method `from_path` to load from disk, or `from_payload` to bind a
    parsed dict.

    The extractor is *single-image* shaped — it returns the same
    extraction regardless of which images dict is passed in. That's
    intentional: a recording corresponds to a specific item, and the
    item carries its own image bytes already; replay just hands back
    what was recorded for that item. Callers that need per-item
    recordings should construct one extractor per item (cheap — it's
    just a dict reference).
    """

    payload: dict[str, Any]
    source: str = "<inline>"

    @classmethod
    def from_path(cls, path: Path) -> ReplayVisionExtractor:
        if not path.exists():
            raise FileNotFoundError(
                f"recording not found at {path}; run record_extraction.py "
                "or synth_from_truth.py first"
            )
        payload = json.loads(path.read_text())
        return cls.from_payload(payload, source=str(path))

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, source: str = "<inline>"
    ) -> ReplayVisionExtractor:
        _validate_payload(payload, source)
        return cls(payload=payload, source=source)

    def extract(
        self,
        *,
        beverage_type: str,
        container_size_ml: int,
        images: dict[str, bytes],
        producer_record: Any | None = None,
        is_imported: bool = False,
        capture_quality: Any | None = None,
    ) -> ExtractionContext:
        # Build fields from the recording. `_to_context()` in the live
        # extractor drops fields with value=null AND confidence=0 so the
        # rule engine treats them as genuinely absent (FAIL on presence
        # checks) rather than unreadable (ADVISORY). Mirror that here so
        # the replay path produces identical rule outcomes to a live
        # extraction with the same payload.
        fields: dict[str, ExtractedField] = {}
        for name, raw in self.payload.get("fields", {}).items():
            field = _build_field(name, raw, self.source)
            if field.value is None and field.confidence == 0.0:
                continue
            fields[name] = field

        unreadable = list(self.payload.get("unreadable_fields", []) or [])

        # Compose the rule-engine `application` payload. Start from the
        # recorded application (model verdict on image_quality, observed
        # beverage type, etc.), then echo in `producer_record` from the
        # call args — same shape `_to_context()` produces.
        application: dict[str, Any] = dict(self.payload.get("application") or {})
        application.setdefault("model_provider", "replay")
        if producer_record is not None:
            application["producer_record"] = {
                "brand": getattr(producer_record, "brand", None),
                "class_type": getattr(producer_record, "class_type", None),
                "container_size_ml": getattr(
                    producer_record, "container_size_ml", None
                ),
            }

        abv_pct_raw = self.payload.get("abv_pct")
        abv_pct = float(abv_pct_raw) if abv_pct_raw is not None else None

        return ExtractionContext(
            fields=fields,
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
            abv_pct=abv_pct,
            raw_ocr_texts=dict(self.payload.get("raw_ocr_texts") or {}),
            application=application,
            unreadable_fields=unreadable,
        )
