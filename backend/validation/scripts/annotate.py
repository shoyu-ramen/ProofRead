"""Interactive annotator for `validation/real_labels/<id>/truth.json`.

Walks the operator through every field of the v2 schema with sensible
defaults so a typical "happy-path" item takes ~2 minutes to annotate.
After writing, runs `validate_truth.py` against the new file so a
schema slip surfaces immediately rather than on the next CI run.

Run from `backend/`:

    # Annotate a brand-new item (creates the directory)
    python -m validation.scripts.annotate validation/real_labels/lbl-0042

    # Edit an existing item (reads and offers existing values as defaults)
    python -m validation.scripts.annotate validation/real_labels/lbl-0001 --edit

The annotator does NOT touch the photos themselves; the operator drops
`front.jpg` and `back.jpg` into the directory before or after running
this. The linter checks both exist.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from validation.real_corpus import RULE_IDS_BY_BEVERAGE
from validation.scripts.validate_truth import lint_item

_VALIDATION_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _prompt(label: str, default: Any | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    raw = input(f"{label}{suffix}: ").strip()
    if raw == "" and default is not None:
        return str(default)
    return raw


def _prompt_choice(
    label: str, choices: Iterable[str], default: str | None = None
) -> str:
    options = list(choices)
    while True:
        text = "/".join(
            f"[{c}]" if c == default else c for c in options
        )
        raw = input(f"{label} ({text}): ").strip().lower()
        if raw == "" and default is not None:
            return default
        if raw in options:
            return raw
        print(f"  pick one of {options}")


def _prompt_bool(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} ({suffix}): ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def _prompt_int(label: str, default: int | None = None) -> int:
    while True:
        raw = _prompt(label, default)
        try:
            return int(raw)
        except ValueError:
            print("  expected an integer")


def _prompt_optional(label: str, default: str | None = None) -> str | None:
    """Empty string returns None; literal `null` also returns None."""
    raw = _prompt(label, default if default is not None else "")
    if raw in ("", "null", "None"):
        return None
    return raw


# ---------------------------------------------------------------------------
# Field collectors
# ---------------------------------------------------------------------------


_BEVERAGES = ["beer", "wine", "spirits"]
_SOURCE_KINDS = ["wikimedia_commons", "wikimedia_synth", "cola_artwork"]
_SPLITS = ["train", "dev", "test"]


def _collect_capture_conditions(default: dict[str, Any]) -> dict[str, Any]:
    print("\n--- capture conditions ---")
    return {
        "lighting": _prompt(
            "lighting (e.g. studio, indoor_office, bar_dim, outdoor_bright, mixed, backlit)",
            default.get("lighting", "studio"),
        ),
        "surface": _prompt(
            "surface (dry, wet_condensation, foggy, dirty, torn, foil, embossed)",
            default.get("surface", "dry"),
        ),
        "container": _prompt(
            "container (12oz_can, 12oz_bottle, 750ml_wine, 750ml_spirits, 1750ml_handle, growler)",
            default.get("container", "12oz_bottle"),
        ),
        "device": _prompt("device (free-form)", default.get("device", "Wikimedia upload")),
        "notes": _prompt("notes (free-form, no PII)", default.get("notes", "")),
    }


def _collect_label_spec(beverage: str, default: dict[str, Any]) -> dict[str, Any]:
    print("\n--- label_spec (verbatim transcription) ---")
    spec: dict[str, Any] = {
        "brand": _prompt_optional("brand", default.get("brand")),
        "class_type": _prompt_optional("class_type", default.get("class_type")),
        "abv": _prompt_optional("abv (with unit, e.g. '5.5% ABV')", default.get("abv")),
        "net_contents": _prompt_optional(
            "net_contents (with unit, e.g. '12 FL OZ')", default.get("net_contents")
        ),
        "name_address": _prompt_optional("name_address", default.get("name_address")),
        "health_warning_text": _prompt_optional(
            "health_warning_text (verbatim, including any typo; null if absent)",
            default.get("health_warning_text"),
        ),
        "country": _prompt_optional(
            "country (only if imported; otherwise blank)", default.get("country")
        ),
    }
    if beverage == "wine":
        spec["sulfite_declaration"] = _prompt_optional(
            "sulfite_declaration (e.g. 'Contains Sulfites'; null if absent)",
            default.get("sulfite_declaration"),
        )
        spec["organic_certification"] = _prompt_optional(
            "organic_certification (e.g. 'USDA Organic'; null if absent)",
            default.get("organic_certification"),
        )
    if beverage == "spirits":
        spec["age_statement"] = _prompt_optional(
            "age_statement (e.g. 'Aged 4 Years'; null if absent)",
            default.get("age_statement"),
        )
    return spec


def _collect_application(beverage: str, default: dict[str, Any]) -> dict[str, Any]:
    if beverage != "spirits":
        return dict(default)  # beer/wine: optional, default empty
    print("\n--- application (claimed by submitter — drives spirits cross-checks) ---")
    print("Use the same value as the label by default; flip a field to seed a")
    print("deliberate-mismatch fail-case test.")
    return {
        "brand_name": _prompt_optional("application.brand_name", default.get("brand_name")),
        "class_type": _prompt_optional("application.class_type", default.get("class_type")),
        "alcohol_content": _prompt_optional(
            "application.alcohol_content (e.g. '45%')", default.get("alcohol_content")
        ),
        "net_contents": _prompt_optional(
            "application.net_contents (e.g. '750 mL')", default.get("net_contents")
        ),
    }


def _default_verdict(
    rule_id: str, *, is_imported: bool, country: str | None, hw_text: str | None
) -> str:
    """Best-guess verdict to pre-fill before the operator confirms."""
    if rule_id.endswith("country_of_origin.presence_if_imported"):
        if not is_imported:
            return "na"
        return "pass" if country else "fail"
    if rule_id.endswith("health_warning.size"):
        return "advisory"
    if rule_id.endswith("health_warning.exact_text") or rule_id.endswith(
        "health_warning.compliance"
    ):
        return "pass" if hw_text else "fail"
    if rule_id.endswith("organic.format"):
        return "advisory"
    return "pass"


def _collect_ground_truth(
    beverage: str,
    *,
    is_imported: bool,
    country: str | None,
    hw_text: str | None,
    default: dict[str, str],
) -> dict[str, str]:
    print("\n--- ground_truth (verdict per rule_id) ---")
    print("Defaults are best-guesses. Press Enter to accept, or type")
    print("pass/fail/advisory/na to override.")
    rule_ids = sorted(RULE_IDS_BY_BEVERAGE[beverage])
    out: dict[str, str] = {}
    for rid in rule_ids:
        d = default.get(rid) or _default_verdict(
            rid, is_imported=is_imported, country=country, hw_text=hw_text
        )
        v = _prompt_choice(
            rid, ["pass", "fail", "advisory", "na"], default=d
        )
        out[rid] = v
    return out


def _collect_gold_fields(
    beverage: str, label_spec: dict[str, Any], default: dict[str, Any]
) -> dict[str, Any]:
    """gold_extracted_fields default is derived from label_spec.

    Operator can edit on a per-field basis; this is the field-level
    shadow eval input. For day-1-style smoke runs the defaults are
    perfect, so we don't force the operator to type each one.
    """
    print("\n--- gold_extracted_fields (defaults derived from label_spec) ---")
    aliases = {
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
    out: dict[str, Any] = {}
    for field_name, spec_key in aliases.items():
        if spec_key not in label_spec:
            continue  # field doesn't apply to this beverage
        spec_value = label_spec.get(spec_key)
        existing = (default.get(field_name) or {}).get("value")
        # Default to label_spec value; allow override.
        proposed = existing if existing is not None else spec_value
        ans = _prompt_optional(
            f"gold.{field_name}.value", proposed
        )
        out[field_name] = {"value": ans}
    return out


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_email() -> str | None:
    try:
        out = subprocess.run(
            ["git", "config", "--get", "user.email"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def annotate(item_dir: Path, *, edit_existing: bool) -> Path:
    item_dir.mkdir(parents=True, exist_ok=True)
    truth_path = item_dir / "truth.json"

    existing: dict[str, Any] = {}
    if truth_path.exists():
        if not edit_existing:
            raise FileExistsError(
                f"{truth_path} already exists; pass --edit to load and update it"
            )
        existing = json.loads(truth_path.read_text())
        print(f"loaded existing truth.json from {truth_path}\n")

    print("=== ProofRead truth.json annotator (schema v2) ===\n")

    item_id = _prompt("id", existing.get("id") or item_dir.name)
    beverage = _prompt_choice(
        "beverage_type", _BEVERAGES, default=existing.get("beverage_type") or "beer"
    )
    container_size_ml = _prompt_int(
        "container_size_ml", existing.get("container_size_ml") or 355
    )
    is_imported = _prompt_bool(
        "is_imported", default=bool(existing.get("is_imported", False))
    )
    source_kind = _prompt_choice(
        "source_kind",
        _SOURCE_KINDS,
        default=existing.get("source_kind") or "wikimedia_commons",
    )
    split = _prompt_choice(
        "split", _SPLITS, default=existing.get("split") or "train"
    )

    capture = _collect_capture_conditions(existing.get("capture_conditions") or {})
    label_spec = _collect_label_spec(beverage, existing.get("label_spec") or {})
    application = _collect_application(beverage, existing.get("application") or {})
    ground_truth = _collect_ground_truth(
        beverage,
        is_imported=is_imported,
        country=label_spec.get("country"),
        hw_text=label_spec.get("health_warning_text"),
        default=existing.get("ground_truth") or {},
    )
    gold = _collect_gold_fields(
        beverage, label_spec, existing.get("gold_extracted_fields") or {}
    )

    annotator = _prompt(
        "annotator (email)", existing.get("annotator") or _git_email() or ""
    )
    annotated_at = _prompt(
        "annotated_at (YYYY-MM-DD)",
        existing.get("annotated_at") or dt.date.today().isoformat(),
    )

    payload: dict[str, Any] = {
        "schema_version": 2,
        "id": item_id,
        "beverage_type": beverage,
        "container_size_ml": container_size_ml,
        "is_imported": is_imported,
        "source_kind": source_kind,
        "split": split,
        "capture_conditions": capture,
        "ground_truth": ground_truth,
        "label_spec": label_spec,
        "application": application,
        "gold_extracted_fields": gold,
        "annotator": annotator,
        "annotated_at": annotated_at,
    }
    if "two_annotator_check" in existing:
        payload["two_annotator_check"] = existing["two_annotator_check"]

    truth_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {truth_path}")
    return truth_path


def _run_linter(item_dir: Path) -> bool:
    print("\n--- running validate_truth ---")
    findings = lint_item(item_dir)
    findings.emit()
    if findings.has_errors:
        print(f"linter rejected {item_dir.name}: {len(findings.errors)} errors")
        return False
    print(
        f"linter clean ({len(findings.warnings)} warnings)"
        if findings.warnings
        else "linter clean"
    )
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive truth.json annotator")
    parser.add_argument(
        "item_dir", type=Path, help="Item directory (e.g. validation/real_labels/lbl-0042)"
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="If truth.json already exists, load it and use existing values as defaults.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        truth_path = annotate(args.item_dir, edit_existing=args.edit)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\naborted (no file written)", file=sys.stderr)
        return 130
    ok = _run_linter(truth_path.parent)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
