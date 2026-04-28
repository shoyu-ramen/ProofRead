"""Run a corpus through the production pipeline and score against ground truth.

Scoring is per-rule binary classification on the `pass` label:

    TP: predicted == "pass" AND ground_truth == "pass"
    FP: predicted == "pass" AND ground_truth != "pass" (and ground_truth != "na")
    FN: predicted != "pass" AND ground_truth == "pass"
    TN: predicted != "pass" AND ground_truth != "pass" (and ground_truth != "na")

`na` outcomes (rule didn't apply, e.g., country-of-origin on a domestic
label) are excluded from the per-rule denominator entirely. Including them
would inflate every score by free wins on irrelevant rules.

Advisory rules (in v1, only `beer.health_warning.size`) are excluded from
the precision/recall report — they're advisory by design and never carry
pass/fail signal. They appear only in the agreement column of the table.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.services.ocr import OCRBlock, OCRProvider, OCRResult
from app.services.pipeline import ScanInput, VisionExtractor, process_scan
from validation.corpus import CorpusItem, corpus_summary, generate_corpus

# ----------------------------------------------------------------------------
# Perfect-mock OCR provider
# ----------------------------------------------------------------------------


class PerfectMockOCRProvider:
    """Returns the synthesizer's ground-truth text as if it were OCR'd.

    The pipeline calls `process(image_bytes, hint=surface)` once per
    surface (front, back). We dispatch by `hint` against a lookup that
    must be set per-item before each `process_scan` invocation.

    Construction option `tied_text`: for ad-hoc testing, you can pass a
    single `{"front": str, "back": str}` dict that's used for every
    process() call. In normal corpus measurement, callers use the
    `for_item` instance method to swap the text per item.
    """

    def __init__(self, tied_text: dict[str, str] | None = None):
        self._text = tied_text

    def for_item(self, ocr_text: dict[str, str]) -> PerfectMockOCRProvider:
        return PerfectMockOCRProvider(tied_text=ocr_text)

    def process(self, image_bytes: bytes, hint: str | None = None) -> OCRResult:
        if self._text is None:
            raise RuntimeError(
                "PerfectMockOCRProvider has no text bound; call for_item(...) first "
                "or construct with tied_text=..."
            )
        text = self._text.get(hint or "", "")
        # One block per non-empty line. Bbox positions are arbitrary but
        # plausible; the rule engine doesn't depend on them for any
        # exact-text comparison, but the brand-name extractor uses block
        # area to pick the largest block — so we make the first line on
        # the front have the largest area, since the synthesizer always
        # renders the brand first on the front.
        lines = [ln for ln in text.split("\n") if ln.strip()]
        blocks: list[OCRBlock] = []
        for i, line in enumerate(lines):
            # First block per surface gets a large bbox so the
            # brand-name "largest block on front" extractor picks it.
            if i == 0:
                bbox = (0, 0, 700, 120)
            else:
                bbox = (0, 200 + i * 30, 400, 25)
            blocks.append(OCRBlock(text=line, bbox=bbox, confidence=0.99))

        return OCRResult(
            full_text=text,
            blocks=blocks,
            provider="perfect_mock",
            raw={},
        )


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


@dataclass
class RuleScore:
    rule_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    na_count: int = 0
    advisory_count: int = 0
    # (item_id, predicted, expected)
    disagreements: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def support(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class HarnessReport:
    rule_scores: dict[str, RuleScore]
    overall_precision: float
    overall_recall: float
    overall_f1: float
    items_evaluated: int
    corpus_summary: dict[str, int]


# Rules we treat as advisory and skip from precision/recall scoring.
_ADVISORY_RULE_IDS = {"beer.health_warning.size"}


def _score_one(
    item: CorpusItem,
    rule_results: list,
    scores: dict[str, RuleScore],
) -> None:
    by_id = {r.rule_id: r.status.value for r in rule_results}
    for rule_id, expected in item.ground_truth.items():
        predicted = by_id.get(rule_id)
        score = scores.setdefault(rule_id, RuleScore(rule_id=rule_id))

        # Advisory rule -> tally and skip from precision/recall.
        if rule_id in _ADVISORY_RULE_IDS:
            score.advisory_count += 1
            continue

        # NA outcomes don't contribute to precision/recall.
        if expected == "na" or predicted == "na":
            score.na_count += 1
            if predicted != expected:
                score.disagreements.append((item.id, str(predicted), expected))
            continue

        if predicted == "pass" and expected == "pass":
            score.tp += 1
        elif predicted == "pass" and expected != "pass":
            score.fp += 1
            score.disagreements.append((item.id, predicted, expected))
        elif predicted != "pass" and expected == "pass":
            score.fn += 1
            score.disagreements.append((item.id, str(predicted), expected))
        else:
            score.tn += 1


def measure(
    items: Iterable[CorpusItem],
    *,
    ocr_provider: OCRProvider | None = None,
    vision_extractor: VisionExtractor | None = None,
) -> HarnessReport:
    """Run each corpus item through `process_scan` and aggregate scores.

    Provider selection mirrors the production pipeline: pass either an
    `ocr_provider` or a `vision_extractor` (or both — vision is preferred
    with OCR as fallback). With neither supplied, the harness uses a
    fresh `PerfectMockOCRProvider` whose text is rebound per-item to the
    synthesizer's recorded ground-truth text.

    Args:
        items: corpus iterable.
        ocr_provider: real OCR provider (Google Vision, future Claude OCR,
            or a fixture-driven `MockOCRProvider`). When omitted and no
            vision extractor is supplied, `PerfectMockOCRProvider` is used.
        vision_extractor: a `VisionExtractor` implementing the protocol in
            `app.services.pipeline` (e.g. `ClaudeVisionExtractor`).
    """
    items = list(items)
    use_perfect = ocr_provider is None and vision_extractor is None
    perfect = PerfectMockOCRProvider() if use_perfect else None

    scores: dict[str, RuleScore] = {}
    for item in items:
        # If we're using the default perfect mock, rebind its text to the
        # synthesizer's ground-truth text for this item. Real providers
        # are passed through untouched.
        if use_perfect:
            assert perfect is not None
            ocr = perfect.for_item(item.ocr_text)
            vision = None
        else:
            ocr = ocr_provider
            vision = vision_extractor

        scan = ScanInput(
            beverage_type="beer",
            container_size_ml=item.label_spec.container_size_ml,
            images={"front": item.front_png, "back": item.back_png},
            is_imported=item.label_spec.is_imported,
        )
        report = process_scan(scan, ocr=ocr, vision=vision)
        _score_one(item, report.rule_results, scores)

    # Overall = micro-average across rules (sum TP/FP/FN/TN, then ratio).
    total_tp = sum(s.tp for s in scores.values())
    total_fp = sum(s.fp for s in scores.values())
    total_fn = sum(s.fn for s in scores.values())
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r)
        else 0.0
    )

    return HarnessReport(
        rule_scores=scores,
        overall_precision=overall_p,
        overall_recall=overall_r,
        overall_f1=overall_f1,
        items_evaluated=len(items),
        corpus_summary=corpus_summary(items),
    )


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def render_markdown(report: HarnessReport) -> str:
    """Format a HarnessReport as a markdown table for stdout / CI logs."""
    lines: list[str] = []
    lines.append("# ProofRead OCR Harness Report")
    lines.append("")
    lines.append(f"**Items evaluated:** {report.items_evaluated}")
    lines.append("")
    lines.append("## Corpus composition")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for k, v in report.corpus_summary.items():
        if k == "total":
            continue
        lines.append(f"| `{k}` | {v} |")
    lines.append(f"| **total** | **{report.corpus_summary.get('total', 0)}** |")
    lines.append("")

    lines.append("## Per-rule precision / recall / F1")
    lines.append("")
    lines.append("| Rule | Support | TP | FP | FN | TN | Precision | Recall | F1 | Disagreements |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for rule_id in sorted(report.rule_scores.keys()):
        s = report.rule_scores[rule_id]
        if rule_id in _ADVISORY_RULE_IDS:
            lines.append(
                f"| `{rule_id}` | (advisory: {s.advisory_count}) | — | — | — | — | — | — | — | — |"
            )
            continue
        disagreement_str = ", ".join(d[0] for d in s.disagreements[:3])
        if len(s.disagreements) > 3:
            disagreement_str += f", … (+{len(s.disagreements) - 3})"
        lines.append(
            f"| `{rule_id}` | {s.support} | {s.tp} | {s.fp} | {s.fn} | {s.tn} "
            f"| {s.precision:.3f} | {s.recall:.3f} | {s.f1:.3f} | {disagreement_str} |"
        )

    lines.append("")
    lines.append("## Overall (micro-averaged across rules)")
    lines.append("")
    lines.append(f"- Precision: **{report.overall_precision:.3f}**")
    lines.append(f"- Recall:    **{report.overall_recall:.3f}**")
    lines.append(f"- F1:        **{report.overall_f1:.3f}**")
    lines.append("")

    # Spotlight the spec target rule.
    target = "beer.health_warning.exact_text"
    if target in report.rule_scores:
        s = report.rule_scores[target]
        lines.append(f"## SPEC v1.3 target — `{target}`")
        lines.append("")
        lines.append("- Required: precision ≥ 0.98, recall ≥ 0.99")
        lines.append(f"- Measured: precision = {s.precision:.3f}, recall = {s.recall:.3f}")
        if s.precision >= 0.98 and s.recall >= 0.99:
            lines.append("- **Status: meets target**")
        else:
            lines.append("- **Status: BELOW target**")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProofRead OCR validation harness.")
    parser.add_argument(
        "--provider",
        choices=["perfect_mock", "google_vision", "claude_vision"],
        default="perfect_mock",
        help=(
            "Extractor to use. perfect_mock uses synthesized ground-truth text "
            "directly; google_vision hits real OCR; claude_vision goes through "
            "the Claude-vision extractor."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Corpus generation seed (deterministic).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary instead of the markdown table.",
    )
    return parser.parse_args(argv)


def _load_provider(name: str) -> tuple[OCRProvider | None, VisionExtractor | None]:
    """Resolve a CLI provider name to (ocr_provider, vision_extractor).

    Exactly one of the two will be non-None for real providers; the
    `perfect_mock` case returns (None, None) and `measure()` uses its
    default mock.
    """
    if name == "perfect_mock":
        return None, None
    if name == "google_vision":
        from app.services.ocr import GoogleVisionOCRProvider

        return GoogleVisionOCRProvider(), None
    if name == "claude_vision":
        from app.services.extractors.claude_vision import ClaudeVisionExtractor

        return None, ClaudeVisionExtractor()
    raise ValueError(f"Unknown provider: {name}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    items = generate_corpus(seed=args.seed)
    ocr_p, vision_p = _load_provider(args.provider)
    report = measure(items, ocr_provider=ocr_p, vision_extractor=vision_p)
    if args.json:
        out = {
            "items_evaluated": report.items_evaluated,
            "corpus_summary": report.corpus_summary,
            "overall": {
                "precision": report.overall_precision,
                "recall": report.overall_recall,
                "f1": report.overall_f1,
            },
            "rules": {
                rid: {
                    "support": s.support,
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1": s.f1,
                    "tp": s.tp,
                    "fp": s.fp,
                    "fn": s.fn,
                    "tn": s.tn,
                    "advisory_count": s.advisory_count,
                    "na_count": s.na_count,
                }
                for rid, s in report.rule_scores.items()
            },
        }
        print(json.dumps(out, indent=2))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
