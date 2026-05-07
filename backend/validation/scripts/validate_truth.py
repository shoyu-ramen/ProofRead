"""Comprehensive linter for `truth.json` + sibling files.

The loader (`validation.real_corpus.load_item`) already validates the
truth.json schema enough to keep the harness from blowing up. This
linter does the rest — the things that matter for *corpus quality*
rather than runtime safety:

  * `schema_version == 2` and required keys present per beverage type.
  * `front.jpg` and `back.jpg` exist and are ≥ 1024 px on the long edge.
  * `application` is non-empty for spirits items (cross-reference rules
    score zero signal otherwise).
  * `health_warning_text`, when present, either matches the canonical
    statutory text (zero edits) or the diff is reflected in the
    `beer.health_warning.exact_text` ground-truth verdict
    (`pass` ↔ canonical, `fail` ↔ deviation).
  * `country_of_origin.presence_if_imported` verdict is consistent with
    `is_imported` and the presence of `country` in `label_spec`.
  * `capture_conditions.notes` does not embed obvious PII (emails,
    phone numbers, US SSNs).
  * `gold_extracted_fields` keys are all extractor-recognised names.

Run from `backend/`:

    # Lint the whole real-labels corpus (+ optional fixtures dir).
    python -m validation.scripts.validate_truth

    # Lint a single item directory or a custom root.
    python -m validation.scripts.validate_truth --root path/to/dir
    python -m validation.scripts.validate_truth path/to/lbl-XXXX

Exit code is 0 on clean, 1 on lint failures, 2 on internal error. Designed
for pre-commit and CI consumption — output is one line per finding,
prefixed with the item id.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from validation.real_corpus import (
    RULE_IDS_BY_BEVERAGE,
    SCHEMA_VERSION,
    TruthSchemaError,
    load_item,
)

_VALIDATION_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _VALIDATION_ROOT / "real_labels"
_CANONICAL_HW_PATH = _VALIDATION_ROOT.parent / "app" / "canonical" / "health_warning.txt"

# Photo resolution floor — long edge ≥ 1024 px. Below this the OCR /
# vision extractor reads carry too much noise to score canonical-text
# health-warning compliance reliably.
MIN_LONG_EDGE_PX = 1024

# Extractor-recognised field names. Items that introduce other keys in
# `gold_extracted_fields` will silently never be scored, so reject the
# extra at lint time.
_EXTRACTOR_FIELDS = frozenset(
    {
        "brand_name",
        "class_type",
        "alcohol_content",
        "net_contents",
        "name_address",
        "country_of_origin",
        "health_warning",
        "sulfite_declaration",
        "organic_certification",
        "age_statement",
    }
)

# Cheap PII heuristics. Not exhaustive — annotators get a clear signal
# to scrub the note rather than a clever regex hand-tuned to the corpus.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)),
    ("phone", re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]


def _load_canonical_health_warning() -> str | None:
    if not _CANONICAL_HW_PATH.exists():
        return None
    return _CANONICAL_HW_PATH.read_text().strip()


CANONICAL_HW = _load_canonical_health_warning()


# ---------------------------------------------------------------------------
# Per-item findings
# ---------------------------------------------------------------------------


class _Findings:
    """Accumulator for one item's lint findings.

    Severities collapse to two categories:
      * `error`   — fails the item, fails the whole run.
      * `warning` — surfaced to the operator, does not fail the run.

    Warnings exist for things like "PINT not in the rule regex" that are
    legitimate annotator decisions rather than bugs in the corpus.
    """

    def __init__(self, item_id: str) -> None:
        self.item_id = item_id
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def emit(self) -> None:
        for msg in self.errors:
            print(f"ERROR  [{self.item_id}] {msg}")
        for msg in self.warnings:
            print(f"WARN   [{self.item_id}] {msg}")

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_image_resolution(item_dir: Path, findings: _Findings) -> None:
    for name in ("front.jpg", "back.jpg"):
        path = item_dir / name
        if not path.exists():
            findings.err(f"missing {name}")
            continue
        try:
            with Image.open(path) as img:
                long_edge = max(img.width, img.height)
        except Exception as exc:
            findings.err(f"{name}: cannot read image ({exc})")
            continue
        if long_edge < MIN_LONG_EDGE_PX:
            findings.err(
                f"{name}: long edge {long_edge}px < {MIN_LONG_EDGE_PX}px floor"
            )


def _check_application_present_for_spirits(
    truth: dict[str, Any], findings: _Findings
) -> None:
    if truth.get("beverage_type") != "spirits":
        return
    application = truth.get("application") or {}
    if not application:
        findings.err(
            "spirits items must include a non-empty `application` payload "
            "(cross-reference rules score zero signal without it)"
        )


def _check_country_of_origin_consistency(
    truth: dict[str, Any], findings: _Findings
) -> None:
    """`country_of_origin.presence_if_imported` verdict must match state.

    is_imported=False  → verdict must be `na`.
    is_imported=True + country present → verdict should be `pass`.
    is_imported=True + country absent  → verdict should be `fail`.
    """
    bev = truth.get("beverage_type")
    rule_id = f"{bev}.country_of_origin.presence_if_imported"
    gt = truth.get("ground_truth", {}) or {}
    verdict = gt.get(rule_id)
    if verdict is None:
        return  # rule_id check covered elsewhere
    label_spec = truth.get("label_spec") or {}
    is_imported = bool(truth.get("is_imported", False))
    country = label_spec.get("country")
    if not is_imported:
        if verdict != "na":
            findings.err(
                f"{rule_id}: is_imported=False so verdict must be `na`, "
                f"got {verdict!r}"
            )
        return
    if country:
        if verdict not in {"pass", "advisory"}:
            findings.err(
                f"{rule_id}: is_imported=True with country={country!r} but "
                f"verdict is {verdict!r}; expected `pass` (or `advisory` "
                f"for degraded captures)"
            )
    else:
        if verdict not in {"fail", "advisory"}:
            findings.err(
                f"{rule_id}: is_imported=True with no country in label_spec, "
                f"verdict is {verdict!r}; expected `fail` (or `advisory` for "
                f"degraded captures)"
            )


def _check_health_warning_consistency(
    truth: dict[str, Any], findings: _Findings
) -> None:
    """Canonical-text rule verdict must agree with the transcribed text.

    Verdict `pass` ↔ transcribed text equals the canonical statutory text.
    Verdict `fail` ↔ transcribed text differs (typo, missing, wrong case
    on prefix, etc.). `advisory` is reserved for degraded captures and is
    accepted without further check.
    """
    if CANONICAL_HW is None:
        return  # canonical file missing — out of scope for the linter
    bev = truth.get("beverage_type")
    if bev == "spirits":
        rule_id = "spirits.health_warning.compliance"
    else:
        rule_id = f"{bev}.health_warning.exact_text"
    gt = truth.get("ground_truth", {}) or {}
    verdict = gt.get(rule_id)
    if verdict is None or verdict == "advisory":
        return
    label_spec = truth.get("label_spec") or {}
    transcribed = label_spec.get("health_warning_text")
    if transcribed is None:
        if verdict != "fail":
            findings.err(
                f"{rule_id}: label_spec.health_warning_text is null but "
                f"verdict is {verdict!r}; expected `fail` (no warning present)"
            )
        return
    matches_canonical = transcribed.strip() == CANONICAL_HW
    if verdict == "pass" and not matches_canonical:
        findings.err(
            f"{rule_id}: verdict is `pass` but transcribed text does not "
            f"match canonical statutory text — pick `fail` or fix the "
            f"transcription"
        )
    if verdict == "fail" and matches_canonical:
        findings.err(
            f"{rule_id}: verdict is `fail` but transcribed text matches "
            f"canonical statutory text — pick `pass` or note the deviation"
        )


def _check_pii(truth: dict[str, Any], findings: _Findings) -> None:
    notes = (truth.get("capture_conditions") or {}).get("notes") or ""
    annotator = truth.get("annotator") or ""
    # Annotator email is a contractually-allowed field, so skip it. Only
    # the free-form notes block is scanned.
    if not notes:
        return
    for kind, pattern in _PII_PATTERNS:
        m = pattern.search(notes)
        if m and m.group(0) != annotator:
            findings.err(
                f"capture_conditions.notes contains {kind} pattern "
                f"({m.group(0)!r}) — scrub before commit"
            )


def _check_gold_field_keys(truth: dict[str, Any], findings: _Findings) -> None:
    gold = truth.get("gold_extracted_fields") or {}
    extras = set(gold.keys()) - _EXTRACTOR_FIELDS
    if extras:
        findings.err(
            f"gold_extracted_fields has unrecognised keys: {sorted(extras)} "
            f"(extractor only knows about {sorted(_EXTRACTOR_FIELDS)})"
        )


def _check_loader_round_trips(
    item_dir: Path, findings: _Findings
) -> dict[str, Any] | None:
    """Run the production loader; capture its TruthSchemaError as a finding.

    Returns the parsed truth.json on success, or None when the loader
    rejected it. `None` lets the caller skip the dependent checks
    cleanly without crashing on missing fields.
    """
    try:
        item = load_item(item_dir)
    except TruthSchemaError as exc:
        findings.err(f"loader rejected truth.json: {exc}")
        return None
    except FileNotFoundError as exc:
        findings.err(f"missing required file: {exc}")
        return None
    # The loader has the parsed truth on its `RealCorpusItem` only via
    # individual fields; the simplest way to give the rest of the linter
    # the dict is to re-parse from disk.
    return json.loads((item_dir / "truth.json").read_text())


# ---------------------------------------------------------------------------
# Per-item driver
# ---------------------------------------------------------------------------


def lint_item(item_dir: Path) -> _Findings:
    findings = _Findings(item_dir.name)

    truth_path = item_dir / "truth.json"
    if not truth_path.exists():
        findings.err("missing truth.json")
        return findings

    truth = _check_loader_round_trips(item_dir, findings)
    if truth is None:
        return findings

    # Re-assert schema_version + rule_id coverage even though the loader
    # checks both. Linter output is shown to humans; explicit beats
    # implicit-via-loader.
    if truth.get("schema_version") != SCHEMA_VERSION:
        findings.err(
            f"schema_version is {truth.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    bev = truth.get("beverage_type")
    if bev in RULE_IDS_BY_BEVERAGE:
        gt = truth.get("ground_truth") or {}
        missing = RULE_IDS_BY_BEVERAGE[bev] - set(gt.keys())
        if missing:
            findings.err(
                f"ground_truth missing rule_ids for {bev}: {sorted(missing)}"
            )

    _check_image_resolution(item_dir, findings)
    _check_application_present_for_spirits(truth, findings)
    _check_country_of_origin_consistency(truth, findings)
    _check_health_warning_consistency(truth, findings)
    _check_pii(truth, findings)
    _check_gold_field_keys(truth, findings)

    return findings


# ---------------------------------------------------------------------------
# Driver / CLI
# ---------------------------------------------------------------------------


def lint_root(root: Path) -> tuple[int, int, int]:
    """Lint every item directory under root. Returns (items, errors, warnings)."""
    if not root.exists():
        print(f"no directory at {root}", file=sys.stderr)
        return 0, 0, 0
    items = 0
    err_total = 0
    warn_total = 0
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("lbl-"):
            continue
        if not (entry / "truth.json").exists():
            continue
        items += 1
        findings = lint_item(entry)
        findings.emit()
        err_total += len(findings.errors)
        warn_total += len(findings.warnings)
    return items, err_total, warn_total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint truth.json files in validation/real_labels/."
    )
    parser.add_argument(
        "items",
        nargs="*",
        type=Path,
        help="Optional list of item directories to lint. If omitted, walks --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help=f"Corpus root to walk when no items are given (default: {_DEFAULT_ROOT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.items:
        err_total = 0
        warn_total = 0
        for item_dir in args.items:
            findings = lint_item(item_dir)
            findings.emit()
            err_total += len(findings.errors)
            warn_total += len(findings.warnings)
        items = len(args.items)
    else:
        items, err_total, warn_total = lint_root(args.root)
    print(
        f"lint summary: items={items} errors={err_total} warnings={warn_total}"
    )
    return 1 if err_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
