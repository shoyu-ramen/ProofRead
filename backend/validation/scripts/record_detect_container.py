"""Live recorder for the /v1/detect-container pre-capture gate.

Runs `app.services.container_check.detect_container` against an item's
`front.jpg` and writes the result to `recorded_detect_container.json`
beside the existing `recorded_extraction.json`. Downstream measurement
replays from this committed payload at $0 cost; this script is the
only place that pays for a live call.

Mirrors the CLI shape and cost-gate of `record_extraction.py` so the
operator's muscle memory carries over.

Run from `backend/`:

    # One item
    python -m validation.scripts.record_detect_container \\
        validation/real_labels/lbl-0001 --i-know-this-costs-money

    # Whole corpus (real_labels + tests_fixtures), force overwrite
    python -m validation.scripts.record_detect_container --all --force \\
        --i-know-this-costs-money
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.container_check import detect_container

# Cost-gating sentinel — same string as record_extraction.py so anyone
# who's already typed it once doesn't have to learn a new flag.
_COST_GATE_FLAG = "--i-know-this-costs-money"

_VALIDATION_ROOT = Path(__file__).resolve().parents[1]
_REAL_LABELS = _VALIDATION_ROOT / "real_labels"
_FIXTURES = _VALIDATION_ROOT / "tests_fixtures"


def _detect_media_type(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/jpeg"


def record_one(item_dir: Path, *, force: bool) -> Path:
    """Record one item's detect-container response.

    Reads `front.jpg` (the pre-capture gate is single-image, so back.jpg
    is intentionally ignored). Writes the merged JSON to
    `recorded_detect_container.json`. Raises `FileNotFoundError` when
    the directory lacks `front.jpg`, and `FileExistsError` when the
    output already exists and `force` is false — same ergonomics as
    `record_extraction.py`.
    """
    front = item_dir / "front.jpg"
    if not front.exists():
        raise FileNotFoundError(f"{item_dir}: no front.jpg to record from")

    out = item_dir / "recorded_detect_container.json"
    if out.exists() and not force:
        raise FileExistsError(
            f"{out} already exists; pass --force to overwrite"
        )

    image_bytes = front.read_bytes()
    media_type = _detect_media_type(image_bytes)
    detection = detect_container(image_bytes, media_type)

    payload: dict[str, Any] = {
        "schema_version": 1,
        # The pre-capture gate routes to `settings.anthropic_model`
        # (Haiku 4.5 by default). Stamp the live model so the recording
        # carries provenance — if a future tightening of the
        # `anthropic_model` setting changes which model writes the
        # recording, the diff makes that visible.
        "model_provider": settings.anthropic_model,
        "image": "front.jpg",
        "detection": {
            "detected": detection.detected,
            "container_type": detection.container_type,
            "bbox": list(detection.bbox) if detection.bbox else None,
            "confidence": float(detection.confidence),
            "reason": detection.reason,
            "brand_name": detection.brand_name,
            "net_contents": detection.net_contents,
        },
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record /v1/detect-container responses for corpus items. "
            "Costs money (each call routes to Anthropic)."
        )
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
        help=(
            "Walk validation/real_labels/ and validation/tests_fixtures/ "
            "and record every item."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing recorded_detect_container.json files.",
    )
    parser.add_argument(
        _COST_GATE_FLAG,
        action="store_true",
        dest="cost_gate_acknowledged",
        help=(
            "Acknowledge that this script costs money before running. "
            "Required — no $0 fallback for detect-container today."
        ),
    )
    return parser.parse_args(argv)


def _iter_corpus_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in (_REAL_LABELS, _FIXTURES):
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "truth.json").exists():
                dirs.append(child)
    return dirs


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.cost_gate_acknowledged:
        print(
            f"refusing to record without {_COST_GATE_FLAG} (every call "
            "routes to Anthropic and costs money)",
            file=sys.stderr,
        )
        return 2

    if args.all:
        targets = _iter_corpus_dirs()
        if not targets:
            print("no corpus items found to record", file=sys.stderr)
            return 1
    else:
        if not args.items:
            print("pass --all or one or more item dirs", file=sys.stderr)
            return 2
        targets = [p.resolve() for p in args.items]

    written = 0
    for target in targets:
        try:
            out = record_one(target, force=args.force)
        except FileExistsError as exc:
            print(f"skip: {exc}", file=sys.stderr)
            continue
        except FileNotFoundError as exc:
            print(f"skip: {exc}", file=sys.stderr)
            continue
        print(f"wrote {out}")
        written += 1
    print(f"{written} detect-container recordings written")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
