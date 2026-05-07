"""Day-1 zero-cost stub recorder.

Generates a perfect-recall `recorded_extraction.json` from a `truth.json`
by reading `gold_extracted_fields`. The result lets the harness measure
the rule engine end-to-end without ever calling a vision model.

Day 3+ replaces these synthesised recordings with real Qwen3-VL recordings
via `record_extraction.py` (which is the live local-CPU recorder). The
synthesised recordings are retained only to validate plumbing — they
score 100% by construction (the recording carries the same values the
truth file declares as gold), so they prove the loader/replay/harness
loop is wired correctly. They do NOT exercise the rule engine the way
a real recording would, since real recordings carry confidence < 1,
unreadable fields, and degraded-quality verdicts.

Usage:

    # One item
    python -m validation.scripts.synth_from_truth path/to/lbl-0001/

    # Whole corpus
    python -m validation.scripts.synth_from_truth --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Mirror the perfect-mock confidence baseline so a synthesised recording
# behaves like the synthetic-corpus harness's mock OCR provider does:
# every field that has a value reports near-1 confidence, every absent
# field reports 0.0. Stays below 1.0 so confidence-aware downstream
# logic (cap-at-surface-confidence, etc.) doesn't get triggered.
_PRESENT_CONFIDENCE = 0.99

_VALIDATION_ROOT = Path(__file__).resolve().parents[1]
_REAL_LABELS = _VALIDATION_ROOT / "real_labels"

_ABV_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _abv_pct_from(label_spec: dict[str, Any]) -> float | None:
    abv = label_spec.get("abv")
    if not abv:
        return None
    m = _ABV_PCT_RE.search(abv)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _build_fields(
    gold: dict[str, Any], label_spec: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Translate `gold_extracted_fields` into the recording's `fields` map.

    Falls back to `label_spec` for any field key not present in `gold`,
    so partially-annotated items still produce a usable recording.
    """
    fields: dict[str, dict[str, Any]] = {}
    unreadable: list[str] = []

    # Map from extractor field name -> label_spec key, for the fall-through.
    spec_aliases = {
        "brand_name": "brand",
        "class_type": "class_type",
        "alcohol_content": "abv",
        "net_contents": "net_contents",
        "name_address": "name_address",
        "country_of_origin": "country",
        "health_warning": "health_warning_text",
        "sulfite_declaration": "sulfite_declaration",
        "organic_certification": "organic_certification",
        "age_statement": "age_statement",
    }

    for fname, spec_key in spec_aliases.items():
        gold_entry = gold.get(fname)
        if gold_entry is None:
            spec_value = label_spec.get(spec_key)
            if spec_value is None:
                continue
            value = spec_value
            bbox = None
            is_unreadable = False
        else:
            value = gold_entry.get("value")
            bbox = gold_entry.get("bbox")
            is_unreadable = bool(gold_entry.get("unreadable", False))

        if value is None and not is_unreadable:
            # Field genuinely absent on the label — present as
            # value=null, confidence=0 so the rule engine treats it
            # as absent (FAIL on presence checks) rather than unreadable
            # (ADVISORY). Mirrors `_to_context()` in claude_vision.py.
            fields[fname] = {
                "value": None,
                "bbox": None,
                "confidence": 0.0,
                "source_image_id": None,
            }
            continue

        if is_unreadable:
            fields[fname] = {
                "value": value,
                "bbox": bbox,
                # Below the rule engine's low-confidence threshold so it
                # downgrades the matching rule to ADVISORY.
                "confidence": 0.3,
                "source_image_id": _surface_for(fname),
            }
            unreadable.append(fname)
        else:
            fields[fname] = {
                "value": value,
                "bbox": bbox,
                "confidence": _PRESENT_CONFIDENCE,
                "source_image_id": _surface_for(fname),
            }
    return fields, unreadable


def _surface_for(field_name: str) -> str:
    # Conventional split: anything to do with the warning lives on the
    # back panel; everything else on the front. Real recordings will
    # come from the live recorder which inspects the actual frames, but
    # for the synth path this convention matches how the synthesiser
    # in `validation/synthesize.py` lays out front vs back.
    return "back" if "warning" in field_name else "front"


def synthesise_recording(truth: dict[str, Any]) -> dict[str, Any]:
    """Build a recorded_extraction payload from a truth.json dict."""
    label_spec = truth.get("label_spec", {}) or {}
    gold = truth.get("gold_extracted_fields", {}) or {}
    fields, unreadable = _build_fields(gold, label_spec)
    return {
        "schema_version": 1,
        "model_provider": "synth_from_truth",
        "fields": fields,
        "unreadable_fields": unreadable,
        "application": {
            "image_quality": "good",
            "image_quality_notes": (
                "synth_from_truth stub: capture quality assumed good for "
                "plumbing validation only. Real recordings carry the "
                "live model's verdict."
            ),
            "model_provider": "synth_from_truth",
            "beverage_type_observed": truth.get("beverage_type"),
            "model_observations": None,
        },
        "abv_pct": _abv_pct_from(label_spec),
        "raw_ocr_texts": {},
    }


def write_recording(item_dir: Path, *, force: bool = False) -> Path:
    truth_path = item_dir / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"missing truth.json in {item_dir}")
    truth = json.loads(truth_path.read_text())
    payload = synthesise_recording(truth)
    out = item_dir / "recorded_extraction.json"
    if out.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite {out}; pass --force to replace"
        )
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesise recorded_extraction.json from truth.json"
    )
    parser.add_argument(
        "items",
        nargs="*",
        type=Path,
        help="One or more item directories (lbl-XXXX/). Ignored when --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Walk validation/real_labels/ and write a recording for every item.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing recorded_extraction.json files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
        print("no items to synthesise (pass paths or --all)", file=sys.stderr)
        return 1
    written = 0
    for item_dir in targets:
        try:
            out = write_recording(item_dir, force=args.force)
        except FileExistsError as exc:
            print(f"skip: {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"error on {item_dir}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {out}")
        written += 1
    print(f"{written} recordings written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
