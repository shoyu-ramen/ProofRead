"""One-shot migration: v1 truth.json → schema v2.

Idempotent. Reads each existing `truth.json` under
`validation/real_labels/`, adds:

  * `schema_version: 2`
  * `source_kind: "cola_artwork"` for the seed corpus
  * `split: "test"` (the COLA artwork serves as a regression baseline,
    not a dev-iteration target)
  * `gold_extracted_fields` derived from `label_spec`
  * `application: {}` placeholder (beer base rules don't use it)

Run from `backend/`:

    python -m validation.scripts.migrate_truth_v2

The script is designed to run once on the seed corpus; new items
should already be authored in v2 directly. Re-running on already-v2
files is a no-op (the script reports "already v2; skipping").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_VALIDATION_ROOT = Path(__file__).resolve().parents[1]
_REAL_LABELS = _VALIDATION_ROOT / "real_labels"


# Map from extractor field name -> label_spec key.
_FIELD_MAP = {
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


def _gold_from_label_spec(label_spec: dict) -> dict:
    gold: dict = {}
    for field_name, spec_key in _FIELD_MAP.items():
        if spec_key not in label_spec:
            continue
        value = label_spec.get(spec_key)
        # Carry every present spec field through, including null
        # (which means "genuinely absent on the label" per the schema
        # — the synth recorder treats null + unreadable=False as
        # "FAIL on presence checks", same as the live extractor).
        gold[field_name] = {"value": value}
    return gold


def migrate_one(truth_path: Path) -> bool:
    """Returns True iff the file was migrated, False if already v2."""
    truth = json.loads(truth_path.read_text())
    if truth.get("schema_version") == 2:
        return False
    if "schema_version" in truth and truth["schema_version"] != 2:
        raise ValueError(
            f"{truth_path}: unexpected schema_version {truth['schema_version']!r}"
        )
    truth["schema_version"] = 2
    truth.setdefault("source_kind", "cola_artwork")
    truth.setdefault("split", "test")
    truth.setdefault("application", {})
    if "gold_extracted_fields" not in truth:
        truth["gold_extracted_fields"] = _gold_from_label_spec(
            truth.get("label_spec", {}) or {}
        )

    # Reorder for readability: top-level identity, then source/split,
    # then capture, then truth + spec + application + gold, then
    # provenance. Stable order so diffs stay clean across migrations.
    ordered_keys = [
        "schema_version",
        "id",
        "beverage_type",
        "container_size_ml",
        "is_imported",
        "source_kind",
        "split",
        "capture_conditions",
        "ground_truth",
        "label_spec",
        "application",
        "gold_extracted_fields",
        "annotator",
        "annotated_at",
        "two_annotator_check",
    ]
    ordered = {k: truth[k] for k in ordered_keys if k in truth}
    # Preserve any unknown keys at the end — never drop content silently.
    for k, v in truth.items():
        if k not in ordered:
            ordered[k] = v

    truth_path.write_text(json.dumps(ordered, indent=2) + "\n")
    return True


def main() -> int:
    if not _REAL_LABELS.exists():
        print(f"no real_labels directory at {_REAL_LABELS}", file=sys.stderr)
        return 1
    migrated = 0
    skipped = 0
    for item_dir in sorted(_REAL_LABELS.iterdir()):
        if not item_dir.is_dir() or not item_dir.name.startswith("lbl-"):
            continue
        truth_path = item_dir / "truth.json"
        if not truth_path.exists():
            continue
        if migrate_one(truth_path):
            print(f"migrated {truth_path}")
            migrated += 1
        else:
            print(f"already v2; skipping {truth_path}")
            skipped += 1
    print(f"migrated={migrated} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
