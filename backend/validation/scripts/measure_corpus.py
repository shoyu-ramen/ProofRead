"""Corpus-wide aggregator and reporter.

Runs the harness in replay mode against every corpus item that has a
recording, broken down by beverage_type / source_kind / split, and emits
a combined report.

Output:

  * Markdown by default — written to stdout (`-`) or a file with `--out`.
    The repo convention is `validation/real_labels/MEASUREMENTS.md` as
    the most recent baseline; subsequent runs diff against it.

  * JSON with `--json` — for CI dashboards or regression diffing.

Replay mode means zero recurring cost. The aggregator is safe to wire
into a pre-merge step or a daily local cron.

Run from `backend/`:

    # Print the markdown report
    python -m validation.scripts.measure_corpus

    # Snapshot to disk
    python -m validation.scripts.measure_corpus --out validation/real_labels/MEASUREMENTS.md

    # JSON for dashboards
    python -m validation.scripts.measure_corpus --json > out.json

    # Restrict to a beverage / source / split combo
    python -m validation.scripts.measure_corpus --beverage beer --split test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from validation.measure import HarnessReport, RuleScore, measure
from validation.real_corpus import (
    RULE_IDS_BY_BEVERAGE,
    RealCorpusItem,
    load_corpus,
)
from validation.replay_extractor import ReplayVisionExtractor

_VALIDATION_ROOT = Path(__file__).resolve().parents[1]
_REAL_LABELS = _VALIDATION_ROOT / "real_labels"
_FIXTURES = _VALIDATION_ROOT / "tests_fixtures"

# Surface descriptors (from `capture_conditions.surface`) where the
# pre-capture gate's container_type fallback to "bottle" is documented
# behaviour, not a regression. For flat label artwork the prompt
# instructs the model: "pick 'bottle' if it's just a flat label graphic
# and you can't tell". Scoring container_type on those frames would
# punish the model for following its own contract.
_FLAT_ARTWORK_SURFACES = frozenset({"flat_artwork"})

# Brand-name substring match floor. Containment-based fuzzy matching
# treats "CALVERT BREWING" inside "Calvert Brewing Company" as a hit
# (the model appended "Company"), but only when the shorter side is
# at least this many characters — short brands like "Bud" or "DG"
# would otherwise match nearly any longer string by coincidence.
_BRAND_SUBSTRING_MIN = 5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _replay_factory(item: RealCorpusItem):
    return ReplayVisionExtractor.from_payload(
        item.recorded_extraction, source=item.id
    )


def _measure_subset(items: list[RealCorpusItem]) -> HarnessReport | None:
    if not items:
        return None
    return measure(
        items,
        vision_extractor_factory=_replay_factory,
        skip_capture_quality=True,
    )


@dataclass
class CorpusBreakdown:
    """Per-beverage measurement plus a roll-up across all beverages."""

    by_beverage: dict[str, HarnessReport]
    overall_rule_scores: dict[str, RuleScore]
    overall_precision: float
    overall_recall: float
    overall_f1: float
    items: list[RealCorpusItem]


def aggregate(items: Iterable[RealCorpusItem]) -> CorpusBreakdown:
    items = list(items)
    by_beverage: dict[str, HarnessReport] = {}
    for bev in ("beer", "wine", "spirits"):
        slice_ = [i for i in items if i.beverage_type == bev]
        report = _measure_subset(slice_)
        if report is not None:
            by_beverage[bev] = report

    # Stitch per-beverage rule scores into one overall map. Rule_ids are
    # disjoint across beverages so the union is simply a merge of dicts.
    overall: dict[str, RuleScore] = {}
    for report in by_beverage.values():
        overall.update(report.rule_scores)

    total_tp = sum(s.tp for s in overall.values())
    total_fp = sum(s.fp for s in overall.values())
    total_fn = sum(s.fn for s in overall.values())
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return CorpusBreakdown(
        by_beverage=by_beverage,
        overall_rule_scores=overall,
        overall_precision=p,
        overall_recall=r,
        overall_f1=f1,
        items=items,
    )


# ---------------------------------------------------------------------------
# Detect-container scoring
# ---------------------------------------------------------------------------


# Maps the truth `capture_conditions.container` string to the
# `ContainerDetection.container_type` literal the model returns. Loose
# substring match — truth strings carry sizes ("16oz_can", "750ml_wine")
# while the model returns the bare type. Anything that doesn't match
# one of these prefixes is treated as "unknown" and scored only on
# detected/brand/net_contents (container_type is skipped).
_CONTAINER_TYPE_PREFIXES: dict[str, tuple[str, ...]] = {
    "can": ("can",),
    "bottle": ("bottle", "wine"),  # 750ml_wine → bottle
    "box": ("box",),
}


def _expected_container_type(truth: dict[str, Any]) -> str | None:
    container = (
        (truth.get("capture_conditions") or {}).get("container") or ""
    ).lower()
    if not container:
        return None
    for ctype, prefixes in _CONTAINER_TYPE_PREFIXES.items():
        if any(p in container for p in prefixes):
            return ctype
    return None


_BRAND_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm_brand(value: str | None) -> str:
    """Aggressive normalisation for brand-name comparison.

    The model's transcription of all-caps wordmark art can differ from
    the printed string ("BARKABOOM" → "Barka Boom"). For the corpus
    measurement we treat both as the same brand — the goal is to grade
    whether the model recognised the brand, not whether it preserved
    the printed case/spacing.
    """
    if not value:
        return ""
    return _BRAND_NORM_RE.sub("", value.lower())


def _brand_match(actual: str | None, expected: str | None) -> bool:
    """Containment-based fuzzy match for brand names.

    Returns True when either normalised form contains the other AND
    the shorter side is at least `_BRAND_SUBSTRING_MIN` characters.
    This catches "CALVERT BREWING" ↔ "Calvert Brewing Company"
    (model appended "Company") without letting "Bud" match
    "Anheuser-Busch Budweiser Light Lager".
    """
    a = _norm_brand(actual)
    e = _norm_brand(expected)
    if not a or not e:
        return False
    if a == e:
        return True
    shorter, longer = (a, e) if len(a) <= len(e) else (e, a)
    if len(shorter) < _BRAND_SUBSTRING_MIN:
        return False
    return shorter in longer


def _norm_net_contents(value: str | None) -> str:
    """Loose normalisation for net-contents comparison.

    Truth strings ("16 FL OZ") and model output ("16 FL OZ", "16 FL.
    OZ.") differ on punctuation; this strips everything that isn't
    alphanumeric so both forms collapse to "16floz".
    """
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


@dataclass
class DetectContainerScore:
    """Per-item + aggregate scoring of the pre-capture detect-container gate."""

    items_evaluated: int = 0
    detected_true: int = 0
    container_type_evaluated: int = 0
    container_type_correct: int = 0
    brand_evaluated: int = 0
    brand_match: int = 0
    net_contents_evaluated: int = 0
    net_contents_match: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)
    skipped_no_recording: list[str] = field(default_factory=list)
    skipped_placeholder: list[str] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        if not self.items_evaluated:
            return 1.0
        return self.detected_true / self.items_evaluated

    @property
    def container_type_accuracy(self) -> float:
        if not self.container_type_evaluated:
            return 1.0
        return self.container_type_correct / self.container_type_evaluated

    @property
    def brand_match_rate(self) -> float:
        if not self.brand_evaluated:
            return 1.0
        return self.brand_match / self.brand_evaluated

    @property
    def net_contents_match_rate(self) -> float:
        if not self.net_contents_evaluated:
            return 1.0
        return self.net_contents_match / self.net_contents_evaluated


def score_detect_container(items: Iterable[RealCorpusItem]) -> DetectContainerScore:
    """Grade detect-container output against truth for each corpus item.

    Items are skipped (with a note in the score) when:
      * The item directory has no `recorded_detect_container.json` —
        the recorder hasn't been run for it yet.
      * The item's `capture_conditions.notes` flags the image as a
        placeholder (the `tests_fixtures/` plumbing items) — running
        the gate on a blank JPG just confirms the model rejects it,
        which is informative but not what this scorer is measuring.

    Scoring is permissive: brand/net_contents comparisons normalise
    case + whitespace + punctuation so "BARKABOOM" vs "Barka Boom"
    counts as a hit; the goal is to gate model regressions, not to
    grade transcription fidelity.
    """
    score = DetectContainerScore()
    for item in items:
        if item.root is None:
            score.skipped_no_recording.append(item.id)
            continue
        # tests_fixtures/ holds plumbing-only items whose front.jpg is a
        # blank placeholder. Detect-container correctly returns
        # detected=False on those frames, but that's a vacuous "model
        # rejects a blank image" signal — not what this scorer measures.
        # Skip by path so the rule is impossible to break by editing
        # `capture_conditions.notes`.
        if "tests_fixtures" in item.root.parts:
            score.skipped_placeholder.append(item.id)
            continue
        recording_path = item.root / "recorded_detect_container.json"
        if not recording_path.exists():
            score.skipped_no_recording.append(item.id)
            continue

        recording = json.loads(recording_path.read_text())
        detection = recording.get("detection") or {}
        detected = bool(detection.get("detected"))
        score.items_evaluated += 1
        if detected:
            score.detected_true += 1

        truth_payload = {
            "capture_conditions": item.capture_conditions,
            "label_spec": item.application,  # not used; kept for future
        }
        surface = (
            (item.capture_conditions or {}).get("surface") or ""
        ).lower()
        expected_type = _expected_container_type(
            {"capture_conditions": item.capture_conditions}
        )
        type_ok: bool | None = None
        # Skip container_type scoring on flat artwork — the model's
        # documented fallback there is "bottle" regardless of the
        # physical container the label was destined for.
        if (
            expected_type is not None
            and detected
            and surface not in _FLAT_ARTWORK_SURFACES
        ):
            score.container_type_evaluated += 1
            actual_type = detection.get("container_type")
            if actual_type == expected_type:
                score.container_type_correct += 1
                type_ok = True
            else:
                type_ok = False

        # Brand: only score when truth carries a brand and the model
        # returned something (a null brand_name on a detected=True
        # response just means the wordmark wasn't legible — not a
        # regression).
        truth_brand = (
            (item.gold_extracted_fields.get("brand_name") or {}).get("value")
        )
        actual_brand = detection.get("brand_name")
        brand_ok: bool | None = None
        if truth_brand:
            score.brand_evaluated += 1
            if _brand_match(actual_brand, truth_brand):
                score.brand_match += 1
                brand_ok = True
            else:
                brand_ok = False

        truth_net = (
            (item.gold_extracted_fields.get("net_contents") or {}).get("value")
        )
        actual_net = detection.get("net_contents")
        net_ok: bool | None = None
        if truth_net:
            score.net_contents_evaluated += 1
            if actual_net and _norm_net_contents(actual_net) == _norm_net_contents(truth_net):
                score.net_contents_match += 1
                net_ok = True
            else:
                net_ok = False

        score.per_item.append(
            {
                "id": item.id,
                "detected": detected,
                "container_type": detection.get("container_type"),
                "container_type_ok": type_ok,
                "brand_actual": actual_brand,
                "brand_expected": truth_brand,
                "brand_ok": brand_ok,
                "net_actual": actual_net,
                "net_expected": truth_net,
                "net_ok": net_ok,
            }
        )
    return score


# ---------------------------------------------------------------------------
# Composition tables
# ---------------------------------------------------------------------------


def _composition(items: list[RealCorpusItem]) -> dict[str, Any]:
    """Counts by beverage × source_kind × split — the corpus's coverage matrix.

    Returned shape uses nested dicts (rather than tuple keys) so the
    structure round-trips through `json.dumps` cleanly.
    """
    by_bev: Counter[str] = Counter()
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        by_bev[item.beverage_type] += 1
        by_source[item.beverage_type][item.source_kind] += 1
        by_split[item.beverage_type][item.split] += 1
    return {
        "total": len(items),
        "by_beverage": dict(by_bev),
        "by_source": {b: dict(d) for b, d in by_source.items()},
        "by_split": {b: dict(d) for b, d in by_split.items()},
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


_ADVISORY_RULE_IDS = {"beer.health_warning.size"}


def _format_rule_table(scores: dict[str, RuleScore]) -> list[str]:
    lines = [
        "| Rule | Support | TP | FP | FN | TN | Precision | Recall | F1 | Disagreements |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rule_id in sorted(scores):
        s = scores[rule_id]
        if rule_id in _ADVISORY_RULE_IDS:
            lines.append(
                f"| `{rule_id}` | (advisory: {s.advisory_count}) | — | — | — | — | — | — | — | — |"
            )
            continue
        dis = ", ".join(d[0] for d in s.disagreements[:3])
        if len(s.disagreements) > 3:
            dis += f", … (+{len(s.disagreements) - 3})"
        lines.append(
            f"| `{rule_id}` | {s.support} | {s.tp} | {s.fp} | {s.fn} | {s.tn} "
            f"| {s.precision:.3f} | {s.recall:.3f} | {s.f1:.3f} | {dis} |"
        )
    return lines


def _render_detect_container_section(score: DetectContainerScore) -> list[str]:
    lines: list[str] = []
    lines.append("## Label detection")
    lines.append("")
    if not score.items_evaluated:
        lines.append(
            "_No items with `recorded_detect_container.json` yet — run "
            "`validation/scripts/record_detect_container.py --all "
            "--i-know-this-costs-money` to populate._"
        )
        if score.skipped_placeholder:
            lines.append("")
            lines.append(
                f"Skipped placeholder fixtures: "
                f"`{', '.join(score.skipped_placeholder)}` (image bytes are "
                "blank plumbing fillers, not real labels)."
            )
        return lines

    lines.append(
        "Scored against `recorded_detect_container.json` (single-frame "
        "/v1/detect-container output). Replay-mode, $0 per run."
    )
    lines.append("")
    def _fmt(value: float, num: int, denom: int) -> str:
        if denom == 0:
            return "_n/a (no items eligible)_"
        return f"**{value:.3f}** ({num}/{denom})"

    lines.append(f"- Items evaluated:           **{score.items_evaluated}**")
    lines.append(
        "- Detection rate:            "
        + _fmt(score.detection_rate, score.detected_true, score.items_evaluated)
    )
    lines.append(
        "- Container-type accuracy:   "
        + _fmt(
            score.container_type_accuracy,
            score.container_type_correct,
            score.container_type_evaluated,
        )
    )
    lines.append(
        "- Brand-name match rate:     "
        + _fmt(score.brand_match_rate, score.brand_match, score.brand_evaluated)
    )
    lines.append(
        "- Net-contents match rate:   "
        + _fmt(
            score.net_contents_match_rate,
            score.net_contents_match,
            score.net_contents_evaluated,
        )
    )
    if score.skipped_placeholder:
        lines.append(
            f"- Skipped placeholder items: "
            f"`{', '.join(score.skipped_placeholder)}`"
        )
    if score.skipped_no_recording:
        lines.append(
            f"- Skipped (no recording):    "
            f"`{', '.join(score.skipped_no_recording)}`"
        )
    lines.append("")

    lines.append("| Item | Detected | Type | Brand | Net contents |")
    lines.append("|---|---|---|---|---|")
    for row in score.per_item:
        def _cell(ok: bool | None, value: Any) -> str:
            if value is None:
                return "—"
            if ok is None:
                return f"`{value}`"
            mark = "✓" if ok else "✗"
            return f"{mark} `{value}`"

        lines.append(
            f"| `{row['id']}` "
            f"| {'✓' if row['detected'] else '✗'} "
            f"| {_cell(row['container_type_ok'], row['container_type'])} "
            f"| {_cell(row['brand_ok'], row['brand_actual'])} "
            f"| {_cell(row['net_ok'], row['net_actual'])} |"
        )
    lines.append("")
    return lines


def render_markdown(breakdown: CorpusBreakdown) -> str:
    items = breakdown.items
    comp = _composition(items)
    lines: list[str] = []
    lines.append("# ProofRead corpus measurements")
    lines.append("")
    lines.append(
        f"Generated {dt.date.today().isoformat()} by "
        "`validation/scripts/measure_corpus.py`. Replay-mode against "
        "committed `recorded_extraction.json` + "
        "`recorded_detect_container.json` payloads — zero $ per run."
    )
    lines.append("")
    providers: Counter[str] = Counter()
    for it in items:
        if it.recorded_extraction:
            providers[
                it.recorded_extraction.get("model_provider", "unknown")
            ] += 1
    if providers:
        prov_str = ", ".join(f"{p}: {n}" for p, n in providers.most_common())
        lines.append(f"Extraction provenance: {prov_str}.")
        lines.append("")

    lines.append("## Composition")
    lines.append("")
    lines.append("| Beverage | Items | Sources | Splits |")
    lines.append("|---|---|---|---|")
    for bev in ("beer", "wine", "spirits"):
        n = comp["by_beverage"].get(bev, 0)
        if not n:
            continue
        sources = ", ".join(
            f"{src}: {ct}"
            for src, ct in sorted(comp["by_source"].get(bev, {}).items())
        )
        splits = ", ".join(
            f"{split}: {ct}"
            for split, ct in sorted(comp["by_split"].get(bev, {}).items())
        )
        lines.append(f"| {bev} | {n} | {sources} | {splits} |")
    lines.append(f"| **total** | **{comp['total']}** | | |")
    lines.append("")

    lines.append("## Whole-corpus rule scores")
    lines.append("")
    lines.append(
        f"- Items evaluated: **{len(items)}**"
    )
    lines.append(
        f"- Micro-averaged precision: **{breakdown.overall_precision:.3f}**"
    )
    lines.append(
        f"- Micro-averaged recall:    **{breakdown.overall_recall:.3f}**"
    )
    lines.append(
        f"- Micro-averaged F1:        **{breakdown.overall_f1:.3f}**"
    )
    lines.append("")
    lines.extend(_format_rule_table(breakdown.overall_rule_scores))
    lines.append("")

    for bev, report in breakdown.by_beverage.items():
        n = comp["by_beverage"].get(bev, 0)
        lines.append(f"## {bev.capitalize()} ({n} items)")
        lines.append("")
        lines.append(
            f"- Overall precision: **{report.overall_precision:.3f}**"
        )
        lines.append(
            f"- Overall recall:    **{report.overall_recall:.3f}**"
        )
        lines.append(
            f"- Overall F1:        **{report.overall_f1:.3f}**"
        )
        lines.append("")
        lines.extend(_format_rule_table(report.rule_scores))
        lines.append("")

    # Label-detection section (pre-capture gate). Always rendered so the
    # absence of recordings is visible; the helper inlines a "run the
    # recorder" hint when the score is empty.
    detect_score = score_detect_container(items)
    lines.extend(_render_detect_container_section(detect_score))

    # Disagreement explorer — first stop for the "what moved?" question.
    disagreements = []
    for rule_id, score in breakdown.overall_rule_scores.items():
        for item_id, predicted, expected in score.disagreements:
            disagreements.append((rule_id, item_id, predicted, expected))
    if disagreements:
        lines.append("## Disagreements")
        lines.append("")
        lines.append("| Rule | Item | Predicted | Expected |")
        lines.append("|---|---|---|---|")
        for rule_id, item_id, predicted, expected in disagreements:
            lines.append(
                f"| `{rule_id}` | `{item_id}` | {predicted} | {expected} |"
            )
        lines.append("")
    else:
        lines.append("## Disagreements")
        lines.append("")
        lines.append("*(none)*")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------


def render_json(breakdown: CorpusBreakdown) -> str:
    comp = _composition(breakdown.items)
    detect_score = score_detect_container(breakdown.items)
    return json.dumps(
        {
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "composition": comp,
            "overall": {
                "precision": breakdown.overall_precision,
                "recall": breakdown.overall_recall,
                "f1": breakdown.overall_f1,
                "items": len(breakdown.items),
            },
            "detect_container": {
                "items_evaluated": detect_score.items_evaluated,
                "detection_rate": detect_score.detection_rate,
                "container_type_accuracy": detect_score.container_type_accuracy,
                "brand_match_rate": detect_score.brand_match_rate,
                "net_contents_match_rate": detect_score.net_contents_match_rate,
                "skipped_no_recording": detect_score.skipped_no_recording,
                "skipped_placeholder": detect_score.skipped_placeholder,
                "per_item": detect_score.per_item,
            },
            "by_beverage": {
                bev: {
                    "overall": {
                        "precision": rep.overall_precision,
                        "recall": rep.overall_recall,
                        "f1": rep.overall_f1,
                    },
                    "rules": {
                        rid: _rule_score_to_dict(s)
                        for rid, s in rep.rule_scores.items()
                    },
                }
                for bev, rep in breakdown.by_beverage.items()
            },
            "rules": {
                rid: _rule_score_to_dict(s)
                for rid, s in breakdown.overall_rule_scores.items()
            },
        },
        indent=2,
    )


def _rule_score_to_dict(s: RuleScore) -> dict[str, Any]:
    return {
        "support": s.support,
        "tp": s.tp,
        "fp": s.fp,
        "fn": s.fn,
        "tn": s.tn,
        "precision": s.precision,
        "recall": s.recall,
        "f1": s.f1,
        "advisory_count": s.advisory_count,
        "na_count": s.na_count,
        "disagreements": [
            {"item": item, "predicted": pred, "expected": exp}
            for item, pred, exp in s.disagreements
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_filtered(
    *,
    roots: list[Path],
    beverage: str | None,
    split: str | None,
    source_kind: str | None,
) -> list[RealCorpusItem]:
    items: list[RealCorpusItem] = []
    for root in roots:
        if not root.exists():
            continue
        items.extend(
            load_corpus(
                root=root,
                beverage_type=beverage,
                split=split,
                source_kind=source_kind,
                require_recording=True,
            )
        )
    return items


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate harness scores across the corpus."
    )
    parser.add_argument(
        "--beverage",
        choices=["beer", "wine", "spirits"],
        default=None,
        help="Restrict to one beverage type (default: all).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default=None,
        help="Restrict to one split (default: all).",
    )
    parser.add_argument(
        "--source-kind",
        choices=["wikimedia_commons", "wikimedia_synth", "cola_artwork"],
        default=None,
        help="Restrict to one source_kind (default: all).",
    )
    parser.add_argument(
        "--include-fixtures",
        action="store_true",
        help=(
            "Include items under validation/tests_fixtures/ (off by "
            "default; the fixtures are dev-mode plumbing tests, not real "
            "corpus items)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the report to a file instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    roots = [_REAL_LABELS]
    if args.include_fixtures:
        roots.append(_FIXTURES)
    items = _load_filtered(
        roots=roots,
        beverage=args.beverage,
        split=args.split,
        source_kind=args.source_kind,
    )
    if not items:
        print(
            "no items matched the filter (and have a recording); nothing to score",
            file=sys.stderr,
        )
        return 1
    breakdown = aggregate(items)
    text = render_json(breakdown) if args.json else render_markdown(breakdown)
    if args.out:
        args.out.write_text(text + ("\n" if not text.endswith("\n") else ""))
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
