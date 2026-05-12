"""Loader for the hand-labeled real-photo corpus.

Walks `validation/real_labels/<id>/`, reads:

  * `truth.json`                  schema v2 ground truth + provenance
  * `front.jpg` / `back.jpg`      capture (or COLA artwork composite)
  * `recorded_extraction.json`    optional replay payload — when present,
                                  the harness can score the rule engine
                                  against the recording without ever
                                  calling a vision model

…and emits `RealCorpusItem` records. Items are duck-type-compatible with
`validation.corpus.CorpusItem` so `validation.measure.measure(...)` consumes
them unchanged.

Filtering helpers (`load_corpus(split=..., beverage_type=..., source_kind=...)`)
let callers run the harness against just one slice — e.g. CI runs the
`test` split, dev iteration runs `train + dev`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# v2 truth.json's `schema_version` field. Bumped intentionally; older v1
# files cause a hard parse error so they cannot be silently mis-evaluated
# (the v1 schema lacked `application` / `gold_extracted_fields` / `split`,
# which would default to None and skew measurements).
SCHEMA_VERSION = 2

# Default resolution: walk the directory next to this module.
_DEFAULT_ROOT = Path(__file__).resolve().parent / "real_labels"


# ---------------------------------------------------------------------------
# Adapter shapes
# ---------------------------------------------------------------------------


@dataclass
class RealLabelSpec:
    """Quack-typed substitute for `validation.synthesize.LabelSpec`.

    `validation.measure.measure(...)` only reads `container_size_ml` and
    `is_imported` off `item.label_spec`. Mirroring just those keeps the
    real loader independent of the synthetic synthesizer and avoids a
    circular dependency with the LabelSpec dataclass (which carries
    rendering-specific fields irrelevant to a real photo).
    """

    container_size_ml: int
    is_imported: bool
    brand: str | None = None
    class_type: str | None = None
    abv: str | None = None
    net_contents: str | None = None
    name_address: str | None = None
    health_warning_text: str | None = None
    country: str | None = None
    sulfite_declaration: str | None = None
    organic_certification: str | None = None
    age_statement: str | None = None


@dataclass
class RealCorpusItem:
    """Real-corpus analogue of `validation.corpus.CorpusItem`.

    The first six fields (`id`, `category`, `front_png`, `back_png`,
    `ground_truth`, `label_spec`) match `CorpusItem` so this drops into
    `measure.measure(...)` with no adapter. The remaining fields are
    real-corpus extras the synthetic harness doesn't need.
    """

    id: str
    category: str
    front_png: bytes
    back_png: bytes
    ground_truth: dict[str, str]
    label_spec: RealLabelSpec
    # Real-corpus-only metadata.
    beverage_type: str
    source_kind: str
    split: str
    application: dict[str, Any] = field(default_factory=dict)
    gold_extracted_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    capture_conditions: dict[str, Any] = field(default_factory=dict)
    # Replay payload (parsed JSON). `None` when the directory has no
    # `recorded_extraction.json` yet — the item still loads, but a
    # replay-mode run will skip it. Recording lands in a later phase
    # (day 3-4) via `validation/scripts/record_extraction.py`.
    recorded_extraction: dict[str, Any] | None = None
    # Path to the item's directory — handy for downstream tools that
    # need to read sibling files (augmented/, recorded_health_warning.json,
    # etc.) without re-deriving the path from `id`.
    root: Path | None = None


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


# Rule_id sets per beverage type. Mirror today's rule packs exactly so
# what's annotated is what's actually measurable — annotations on
# unmeasured rule_ids produce noise, not signal. When a rule pack grows
# (the wine pack still owes the seven shared base rules from beer), bump
# `validation/real_corpus.SCHEMA_VERSION` and add a migration that
# back-fills the new keys; SCHEMA.md tracks both the current set and
# the planned additions so annotators see the trajectory.
RULE_IDS_BY_BEVERAGE: dict[str, frozenset[str]] = {
    "beer": frozenset(
        {
            "beer.brand_name.presence",
            "beer.class_type.presence",
            "beer.alcohol_content.format",
            "beer.net_contents.presence",
            "beer.name_address.presence",
            "beer.country_of_origin.presence_if_imported",
            "beer.health_warning.exact_text",
            "beer.health_warning.size",
        }
    ),
    # Wine pack is intentionally narrow today — only the two
    # wine-specific checks exist in `app/rules/definitions/wine.yaml`.
    # The seven shared base rules (brand_name presence, etc.) will land
    # there as the wine pack grows; until then, annotators record only
    # what can be measured.
    "wine": frozenset(
        {
            "wine.sulfite.presence",
            "wine.organic.format",
        }
    ),
    "spirits": frozenset(
        {
            "spirits.brand_name.matches_application",
            "spirits.class_type.matches_application",
            "spirits.alcohol_content.format",
            "spirits.alcohol_content.matches_application",
            "spirits.net_contents.matches_application",
            "spirits.name_address.presence",
            "spirits.country_of_origin.presence_if_imported",
            "spirits.health_warning.compliance",
            "spirits.age_statement.format",
        }
    ),
}

_VALID_SPLITS = frozenset({"train", "dev", "test"})
_VALID_SOURCE_KINDS = frozenset(
    {"wikimedia_commons", "wikimedia_synth", "cola_artwork"}
)
_VALID_VERDICTS = frozenset({"pass", "fail", "advisory", "na"})


class TruthSchemaError(ValueError):
    """Raised when a truth.json file fails schema validation."""


def _validate_truth(truth: dict[str, Any], item_id: str) -> None:
    """Hard-fail when a truth.json doesn't match v2.

    The harness is a measurement gate; loading silently-malformed truth
    would skew precision/recall numbers in ways that are hard to debug
    later. Better to fail loud at load time.
    """
    schema = truth.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise TruthSchemaError(
            f"{item_id}: schema_version is {schema!r}, expected {SCHEMA_VERSION}"
        )

    bev = truth.get("beverage_type")
    if bev not in RULE_IDS_BY_BEVERAGE:
        raise TruthSchemaError(
            f"{item_id}: beverage_type {bev!r} not one of "
            f"{sorted(RULE_IDS_BY_BEVERAGE)}"
        )

    split = truth.get("split")
    if split not in _VALID_SPLITS:
        raise TruthSchemaError(
            f"{item_id}: split {split!r} not one of {sorted(_VALID_SPLITS)}"
        )

    source_kind = truth.get("source_kind")
    if source_kind not in _VALID_SOURCE_KINDS:
        raise TruthSchemaError(
            f"{item_id}: source_kind {source_kind!r} not one of "
            f"{sorted(_VALID_SOURCE_KINDS)}"
        )

    gt = truth.get("ground_truth")
    if not isinstance(gt, dict):
        raise TruthSchemaError(f"{item_id}: ground_truth must be an object")
    expected_rule_ids = RULE_IDS_BY_BEVERAGE[bev]
    actual_rule_ids = frozenset(gt.keys())
    missing = expected_rule_ids - actual_rule_ids
    extra = actual_rule_ids - expected_rule_ids
    if missing:
        raise TruthSchemaError(
            f"{item_id}: ground_truth missing rule_ids for {bev}: "
            f"{sorted(missing)}"
        )
    if extra:
        raise TruthSchemaError(
            f"{item_id}: ground_truth has off-type rule_ids: {sorted(extra)}"
        )
    bad_verdicts = {
        rid: v for rid, v in gt.items() if v not in _VALID_VERDICTS
    }
    if bad_verdicts:
        raise TruthSchemaError(
            f"{item_id}: ground_truth has invalid verdicts: {bad_verdicts}"
        )

    # Spirits require an `application` payload — every spirits rule cross-
    # references it, so an empty application turns the corpus into noise.
    if bev == "spirits" and not truth.get("application"):
        raise TruthSchemaError(
            f"{item_id}: spirits items must include a non-empty `application`"
        )


# ---------------------------------------------------------------------------
# Image read
# ---------------------------------------------------------------------------


def _read_image(path: Path, item_id: str, label: str) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"{item_id}: missing {label} image at {path}")
    return path.read_bytes()


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_item(item_dir: Path) -> RealCorpusItem:
    """Read one `lbl-XXXX/` directory into a RealCorpusItem.

    Raises `TruthSchemaError` if the truth.json fails validation.
    Raises `FileNotFoundError` if `front.jpg` or `back.jpg` is missing.
    """
    truth_path = item_dir / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"missing truth.json in {item_dir}")
    truth = json.loads(truth_path.read_text())

    item_id = truth.get("id") or item_dir.name
    _validate_truth(truth, item_id)

    front = _read_image(item_dir / "front.jpg", item_id, "front")
    back = _read_image(item_dir / "back.jpg", item_id, "back")

    spec_dict = truth.get("label_spec", {}) or {}
    label_spec = RealLabelSpec(
        container_size_ml=int(truth["container_size_ml"]),
        is_imported=bool(truth.get("is_imported", False)),
        brand=spec_dict.get("brand"),
        class_type=spec_dict.get("class_type"),
        abv=spec_dict.get("abv"),
        net_contents=spec_dict.get("net_contents"),
        name_address=spec_dict.get("name_address"),
        health_warning_text=spec_dict.get("health_warning_text"),
        country=spec_dict.get("country"),
        sulfite_declaration=spec_dict.get("sulfite_declaration"),
        organic_certification=spec_dict.get("organic_certification"),
        age_statement=spec_dict.get("age_statement"),
    )

    recorded = _read_optional_json(item_dir / "recorded_extraction.json")

    return RealCorpusItem(
        id=item_id,
        # Use source_kind as `category` so the existing
        # `corpus_summary` reporting in `measure.py` (which buckets by
        # `category`) prints the source mix without further changes.
        category=truth["source_kind"],
        front_png=front,
        back_png=back,
        ground_truth=dict(truth["ground_truth"]),
        label_spec=label_spec,
        beverage_type=truth["beverage_type"],
        source_kind=truth["source_kind"],
        split=truth["split"],
        application=dict(truth.get("application") or {}),
        gold_extracted_fields=dict(truth.get("gold_extracted_fields") or {}),
        capture_conditions=dict(truth.get("capture_conditions") or {}),
        recorded_extraction=recorded,
        root=item_dir,
    )


def load_corpus(
    root: Path | None = None,
    *,
    split: str | tuple[str, ...] | None = None,
    beverage_type: str | tuple[str, ...] | None = None,
    source_kind: str | tuple[str, ...] | None = None,
    require_recording: bool = False,
) -> list[RealCorpusItem]:
    """Walk `root` and return matching `RealCorpusItem`s.

    Filters compose with AND. None means "any value". `require_recording`
    excludes items whose `recorded_extraction.json` is missing — useful
    for replay-mode harness runs where an item without a recording cannot
    be measured.

    Items are returned sorted by `id` so test ordering is deterministic.
    """
    root = root or _DEFAULT_ROOT
    if not root.exists():
        return []

    split_set = _normalise_filter(split)
    bev_set = _normalise_filter(beverage_type)
    src_set = _normalise_filter(source_kind)

    items: list[RealCorpusItem] = []
    for item_dir in sorted(root.iterdir()):
        if not item_dir.is_dir():
            continue
        # Skip non-item directories that may grow inside real_labels/
        # over time (e.g. an `_inbox/` for not-yet-annotated photos).
        if not item_dir.name.startswith("lbl-"):
            continue
        if not (item_dir / "truth.json").exists():
            continue
        item = load_item(item_dir)
        if split_set is not None and item.split not in split_set:
            continue
        if bev_set is not None and item.beverage_type not in bev_set:
            continue
        if src_set is not None and item.source_kind not in src_set:
            continue
        if require_recording and item.recorded_extraction is None:
            continue
        items.append(item)
    return items


def _normalise_filter(
    value: str | tuple[str, ...] | None,
) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(value)


def corpus_summary(items: list[RealCorpusItem]) -> dict[str, int]:
    """Tally items per source_kind — slot-in for `validation.corpus.corpus_summary`."""
    summary: dict[str, int] = {}
    for item in items:
        summary[item.source_kind] = summary.get(item.source_kind, 0) + 1
    summary["total"] = len(items)
    return summary
