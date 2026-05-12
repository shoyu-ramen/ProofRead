"""Live recorder — replaces `synth_from_truth.py` once an item has photos.

Produces a `recorded_extraction.json` for an item by calling a real
extractor on its `front.jpg` + `back.jpg` and serialising the merged
result into the recording schema. The recording is then committed and
re-read by `ReplayVisionExtractor` on every measurement run.

Modes (selected with `--mode`):

  * `qwen` (default) — calls a local OpenAI-compatible Qwen3-VL endpoint
    via `QWEN_VL_BASE_URL`. Free; runs on whatever box hosts the model
    (Ollama / vLLM / LM Studio / llama.cpp). Falls back to `synth` if
    the endpoint isn't configured or the call fails — the operator is
    told explicitly so it's not silent.

  * `synth`  — delegates to `synth_from_truth` for a perfect-recall stub.
    Useful when an annotator hasn't finished label_spec yet but wants the
    plumbing test to pass; never the source of truth for measurement.

  * `anthropic` — calls the production Claude vision extractor. Costs
    money. Gated behind `--i-know-this-costs-money` so accidental use
    can't drain the budget. NOT the recommended path.

Multi-panel handling: front/back are read separately, then merged
field-by-field on highest confidence (mirrors `_merge_panel_extractions`
in `app/services/verify.py`). The merged record carries `source_image_id`
per field so a downstream consumer can tell which surface a value came
from.

Run from `backend/`:

    # One item, default mode (qwen if configured, else synth)
    python -m validation.scripts.record_extraction validation/real_labels/lbl-0007

    # Whole corpus, force qwen
    python -m validation.scripts.record_extraction --all --mode qwen

    # Re-record everything from scratch (overwrites existing recordings)
    python -m validation.scripts.record_extraction --all --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Cost-gating sentinel: the only way to invoke the Anthropic recorder
# is with this flag. Keeps an absent-minded `--mode anthropic` from
# silently spending. The string is intentionally awkward to type.
_COST_GATE_FLAG = "--i-know-this-costs-money"

_VALIDATION_ROOT = Path(__file__).resolve().parents[1]
_REAL_LABELS = _VALIDATION_ROOT / "real_labels"

_ABV_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# Quality ordering for pessimistic merging — "worse wins" so a degraded
# panel doesn't silently get overruled by a "good" verdict on a different
# panel. Mirrors `_worse_quality` in `app/services/pipeline.py`.
_QUALITY_RANK = {"good": 0, "degraded": 1, "unreadable": 2}


# ---------------------------------------------------------------------------
# Recording assembly
# ---------------------------------------------------------------------------


def _detect_media_type(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/jpeg"


def _abv_pct_from_field(value: str | None) -> float | None:
    if not value:
        return None
    m = _ABV_PCT_RE.search(value)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _merge_panel_extractions(
    panels: dict[str, Any],
    *,
    model_provider: str,
) -> dict[str, Any]:
    """Combine per-panel `VisionExtraction`s into one recording payload.

    Per-field rule: highest-confidence read wins; ties keep the earlier
    panel ("front" first per the dict iteration order in CPython 3.7+).
    Panel-level rule: pessimistic image_quality (worst across panels).
    Unreadable union: a field is unreadable iff every panel that
    extracted it called it unreadable.
    """
    merged_fields: dict[str, dict[str, Any]] = {}
    unreadable_per_field: dict[str, list[bool]] = {}
    notes: list[str] = []
    worst_quality = "good"
    beverage_observed: str | None = None

    for surface, ext in panels.items():
        # Track worst image quality across panels
        q = (ext.image_quality or "good").lower()
        if _QUALITY_RANK.get(q, 0) > _QUALITY_RANK.get(worst_quality, 0):
            worst_quality = q
        if ext.image_quality_notes:
            notes.append(f"[{surface}] {ext.image_quality_notes}")
        if ext.beverage_type_observed and beverage_observed is None:
            beverage_observed = ext.beverage_type_observed

        for fname, field in ext.fields.items():
            entry = {
                "value": field.value,
                "bbox": list(field.bbox) if field.bbox else None,
                "confidence": float(field.confidence),
                "source_image_id": surface,
            }
            existing = merged_fields.get(fname)
            if existing is None or entry["confidence"] > existing["confidence"]:
                merged_fields[fname] = entry
            unreadable_per_field.setdefault(fname, []).append(
                fname in ext.unreadable
            )

    # A field is unreadable in the merged record only if every panel that
    # tried to read it called it unreadable — partial reads on some panels
    # still count as evidence.
    unreadable_merged: list[str] = sorted(
        f for f, flags in unreadable_per_field.items() if all(flags)
    )

    abv_pct = _abv_pct_from_field(
        merged_fields.get("alcohol_content", {}).get("value")
    )

    return {
        "schema_version": 1,
        "model_provider": model_provider,
        "fields": merged_fields,
        "unreadable_fields": unreadable_merged,
        "application": {
            "image_quality": worst_quality,
            "image_quality_notes": " | ".join(notes) or None,
            "model_provider": model_provider,
            "beverage_type_observed": beverage_observed,
            "model_observations": None,
        },
        "abv_pct": abv_pct,
        "raw_ocr_texts": {},
    }


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


def _qwen_record(
    item_dir: Path, truth: dict[str, Any]
) -> dict[str, Any]:
    """Call Qwen3-VL on each panel and merge the results."""
    from app.services.anthropic_client import ExtractorUnavailable
    from app.services.qwen_vl import QwenVLExtractor

    extractor = QwenVLExtractor()  # raises ExtractorUnavailable if no base_url
    panels: dict[str, Any] = {}
    for surface in ("front", "back"):
        path = item_dir / f"{surface}.jpg"
        if not path.exists():
            continue
        data = path.read_bytes()
        try:
            ext = extractor.extract(
                data,
                media_type=_detect_media_type(data),
                beverage_type=truth.get("beverage_type"),
                container_size_ml=truth.get("container_size_ml"),
                is_imported=truth.get("is_imported", False),
                producer_record=truth.get("application") or None,
            )
        except ExtractorUnavailable:
            raise
        except Exception as exc:
            raise ExtractorUnavailable(
                f"Qwen3-VL extraction failed on {surface}.jpg: {exc}"
            ) from exc
        panels[surface] = ext
    if not panels:
        raise FileNotFoundError(
            f"{item_dir}: no front.jpg/back.jpg to record from"
        )
    return _merge_panel_extractions(panels, model_provider="qwen3_vl_local")


def _anthropic_record(
    item_dir: Path, truth: dict[str, Any]
) -> dict[str, Any]:
    """Call the structured Claude extractor (ExtractionContext-shaped output).

    Gated behind the cost-gate flag in `main()`. Uses the pipeline-shaped
    extractor in `app/services/extractors/claude_vision.py` because it
    natively handles multi-image input and returns an
    `ExtractionContext` directly — no merge needed.
    """
    from app.config import settings
    from app.services.extractors.claude_vision import (
        ClaudeVisionExtractor,
        ProducerRecord,
    )

    images: dict[str, bytes] = {}
    for surface in ("front", "back"):
        path = item_dir / f"{surface}.jpg"
        if path.exists():
            images[surface] = path.read_bytes()
    if not images:
        raise FileNotFoundError(
            f"{item_dir}: no front.jpg/back.jpg to record from"
        )

    application = truth.get("application") or {}
    producer = ProducerRecord(
        brand=application.get("brand_name"),
        class_type=application.get("class_type"),
        container_size_ml=truth.get("container_size_ml"),
    )
    extractor = ClaudeVisionExtractor()
    ctx = extractor.extract(
        beverage_type=truth.get("beverage_type"),
        container_size_ml=truth.get("container_size_ml"),
        images=images,
        producer_record=producer,
        is_imported=truth.get("is_imported", False),
        capture_quality=None,
    )
    fields_serialised = {
        name: {
            "value": f.value,
            "bbox": list(f.bbox) if f.bbox else None,
            "confidence": float(f.confidence),
            "source_image_id": f.source_image_id,
        }
        for name, f in ctx.fields.items()
    }
    return {
        "schema_version": 1,
        "model_provider": settings.anthropic_model,
        "fields": fields_serialised,
        "unreadable_fields": list(ctx.unreadable_fields),
        "application": {
            "image_quality": ctx.application.get("image_quality", "good"),
            "image_quality_notes": ctx.application.get("image_quality_notes"),
            "model_provider": settings.anthropic_model,
            "beverage_type_observed": ctx.application.get(
                "beverage_type_observed"
            ),
            "model_observations": ctx.application.get("model_observations"),
        },
        "abv_pct": ctx.abv_pct,
        "raw_ocr_texts": {},
    }


def _synth_record(item_dir: Path, truth: dict[str, Any]) -> dict[str, Any]:
    """Reuse synth_from_truth's payload builder for the stub path."""
    from validation.scripts.synth_from_truth import synthesise_recording

    return synthesise_recording(truth)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def record_one(
    item_dir: Path,
    *,
    mode: str,
    force: bool,
) -> Path:
    truth_path = item_dir / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"missing truth.json in {item_dir}")
    truth = json.loads(truth_path.read_text())

    out = item_dir / "recorded_extraction.json"
    if out.exists() and not force:
        raise FileExistsError(
            f"{out} already exists; pass --force to overwrite"
        )

    if mode == "qwen":
        try:
            payload = _qwen_record(item_dir, truth)
        except Exception as exc:
            print(
                f"qwen mode failed on {item_dir.name}: {exc} — falling back to synth",
                file=sys.stderr,
            )
            payload = _synth_record(item_dir, truth)
    elif mode == "anthropic":
        payload = _anthropic_record(item_dir, truth)
    elif mode == "synth":
        payload = _synth_record(item_dir, truth)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a real extraction for one or more items."
    )
    parser.add_argument(
        "items",
        nargs="*",
        type=Path,
        help="One or more item directories. Ignored when --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Walk validation/real_labels/ and record every item.",
    )
    parser.add_argument(
        "--mode",
        choices=["qwen", "synth", "anthropic"],
        default="qwen",
        help=(
            "Extractor to use. `qwen` = local OpenAI-compatible Qwen3-VL "
            "(zero $); `synth` = perfect-recall stub from truth.json; "
            "`anthropic` = Claude (costs money; requires "
            f"{_COST_GATE_FLAG})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing recorded_extraction.json files.",
    )
    parser.add_argument(
        _COST_GATE_FLAG,
        action="store_true",
        dest="cost_gate_acknowledged",
        help="Acknowledge that --mode anthropic costs money before running.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "anthropic" and not args.cost_gate_acknowledged:
        print(
            "refusing to run --mode anthropic without "
            f"{_COST_GATE_FLAG} (the project has a $0 budget; "
            "the qwen path is the supported recorder)",
            file=sys.stderr,
        )
        return 2

    if args.all:
        if not _REAL_LABELS.exists():
            print(f"no real_labels directory at {_REAL_LABELS}", file=sys.stderr)
            return 1
        targets = [
            d for d in sorted(_REAL_LABELS.iterdir())
            if d.is_dir() and d.name.startswith("lbl-")
        ]
    else:
        targets = list(args.items)
    if not targets:
        print("no items to record (pass paths or --all)", file=sys.stderr)
        return 1

    recorded = 0
    for item_dir in targets:
        try:
            out = record_one(item_dir, mode=args.mode, force=args.force)
        except FileExistsError as exc:
            print(f"skip: {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"error on {item_dir}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {out}")
        recorded += 1
    print(f"{recorded} recordings written (mode={args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
