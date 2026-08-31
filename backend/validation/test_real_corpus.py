"""Day-1 + day-2 smoke tests for the real-labels replay harness.

Exercises the full loader → replay-extractor → measure → score-table
pipeline against:

  * The seed corpus (`validation/real_labels/lbl-000{1..6}/`) — six
    TTB COLA composites for the beer path.
  * The day-2 fixtures (`validation/tests_fixtures/lbl-900{1,2}/`) —
    one wine, one spirits item for the non-beer measurement path.

These are *plumbing* tests, not precision/recall gates. Synthesised
recordings carry the truth file's values verbatim, so by construction
they score 100%. What's being tested is that the loader, validator,
replay extractor, beverage-aware routing in `measure()`, and the
`_evaluate_via_replay` path all wire together cleanly.

Real precision/recall floors land on day 8 once the Wikimedia round 1
items are recorded with the local Qwen3-VL extractor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.measure import measure
from validation.real_corpus import RULE_IDS_BY_BEVERAGE, load_corpus
from validation.replay_extractor import ReplayVisionExtractor

_FIXTURES_ROOT = Path(__file__).resolve().parent / "tests_fixtures"


@pytest.fixture(scope="module")
def seed_corpus():
    """The six COLA seed items — the frozen `test` split of the beer corpus.

    Wikimedia round-2 items land in `train`/`dev`, so the seed set is
    addressed by split rather than "all beer items with recordings".
    """
    items = load_corpus(beverage_type="beer", split="test", require_recording=True)
    if not items:
        pytest.skip(
            "no real-labels beer items with recordings found; "
            "run `python -m validation.scripts.synth_from_truth --all` first"
        )
    return items


def _replay_factory(item):
    return ReplayVisionExtractor.from_payload(
        item.recorded_extraction,
        source=item.id,
    )


@pytest.mark.real_corpus
def test_seed_corpus_loads(seed_corpus):
    """Loader walked the directory and validated every truth.json."""
    assert len(seed_corpus) == 6, (
        f"expected 6 seed COLA items, got {len(seed_corpus)}: "
        f"{[i.id for i in seed_corpus]}"
    )
    for item in seed_corpus:
        assert item.beverage_type == "beer"
        assert item.source_kind == "cola_artwork"
        assert item.split == "test"
        # Every required beer rule_id is present in ground_truth.
        assert set(item.ground_truth.keys()) == RULE_IDS_BY_BEVERAGE["beer"], (
            f"{item.id}: ground_truth keys mismatch beer rule set"
        )
        # Recording loaded.
        assert item.recorded_extraction is not None
        assert item.recorded_extraction["schema_version"] == 1


@pytest.mark.real_corpus
def test_replay_runs_end_to_end(seed_corpus):
    """measure() runs the harness in replay mode with no API calls.

    Plumbing assertions only:
      * 6 items evaluated.
      * Non-empty score table.
      * The advisory rule registers 6 advisory_counts (one per item).
      * The canonical Health Warning rule scores precision=1.0 and
        recall=1.0. Historical note: rule v1 did a case-sensitive
        match and false-negatived the ALL-CAPS renderings on
        lbl-0002/lbl-0003; rule v2 (warning_compliance) accepts them,
        so recall is asserted again.

    New corpus items may surface legitimate annotator-vs-rule-engine
    disagreements on other rules. Those are signal for the rule pack,
    not plumbing failures, so they are surfaced via stdout rather than
    failing the test (the regression gate vs the committed baseline is
    what fails CI when a previously-green rule moves).
    """
    report = measure(
        seed_corpus,
        vision_extractor_factory=_replay_factory,
        skip_capture_quality=True,
    )
    assert report.items_evaluated == 6
    assert report.rule_scores, "no rules scored — measure() returned empty"

    advisory = report.rule_scores["beer.health_warning.size"]
    assert advisory.advisory_count == 6, (
        f"expected 6 advisory hits on health_warning.size; got {advisory.advisory_count}"
    )

    hw_text = report.rule_scores["beer.health_warning.exact_text"]
    assert hw_text.precision == 1.0, (
        f"canonical Health Warning rule should never false-positive; "
        f"got precision={hw_text.precision:.3f}, "
        f"disagreements={hw_text.disagreements}"
    )
    assert hw_text.recall == 1.0, (
        f"rule v2 (warning_compliance) must accept the ALL-CAPS renderings "
        f"on lbl-0002/lbl-0003; got recall={hw_text.recall:.3f}, "
        f"disagreements={hw_text.disagreements}"
    )

    # Surface (not assert) other disagreements for visibility.
    leftovers = {
        rid: s.disagreements
        for rid, s in report.rule_scores.items()
        if s.disagreements and rid not in {"beer.health_warning.exact_text"}
    }
    if leftovers:
        print(
            "\nrule-engine vs annotator disagreements on the seed corpus "
            "(not plumbing failures — see test docstring):"
        )
        for rid, dis in leftovers.items():
            print(f"  {rid}: {dis}")


@pytest.fixture(scope="module")
def wine_fixture():
    items = load_corpus(
        root=_FIXTURES_ROOT, beverage_type="wine", require_recording=True
    )
    if not items:
        pytest.skip(
            f"no wine fixtures under {_FIXTURES_ROOT}; "
            "run synth_from_truth.py on the fixture directory"
        )
    return items


@pytest.fixture(scope="module")
def spirits_fixture():
    items = load_corpus(
        root=_FIXTURES_ROOT, beverage_type="spirits", require_recording=True
    )
    if not items:
        pytest.skip(
            f"no spirits fixtures under {_FIXTURES_ROOT}; "
            "run synth_from_truth.py on the fixture directory"
        )
    return items


@pytest.mark.real_corpus
def test_wine_replay_runs_end_to_end(wine_fixture):
    """Wine items route through `_evaluate_via_replay` (process_scan rejects them).

    The fixture's synth recording carries `sulfite_declaration: "Contains
    Sulfites"` and no organic claim. Both wine rules (`sulfite.presence`,
    `organic.format`) should produce PASS — the absence of an organic
    claim satisfies the optional-regex rule.
    """
    report = measure(
        wine_fixture,
        vision_extractor_factory=_replay_factory,
        skip_capture_quality=True,
    )
    assert report.items_evaluated == 1
    assert set(report.rule_scores.keys()) == RULE_IDS_BY_BEVERAGE["wine"]
    for rule_id, score in report.rule_scores.items():
        assert score.precision == 1.0 and score.recall == 1.0, (
            f"{rule_id}: precision={score.precision:.3f}, "
            f"recall={score.recall:.3f}, disagreements={score.disagreements}"
        )


@pytest.mark.real_corpus
def test_spirits_replay_runs_end_to_end(spirits_fixture):
    """Spirits items thread `application` into ctx.application['producer_record'].

    Without that thread the four `*.matches_application` cross-reference
    rules would all see "no producer record" and produce ADVISORY
    instead of PASS. This test is the regression gate on the threading
    in `_evaluate_via_replay`.
    """
    report = measure(
        spirits_fixture,
        vision_extractor_factory=_replay_factory,
        skip_capture_quality=True,
    )
    assert report.items_evaluated == 1
    assert set(report.rule_scores.keys()) == RULE_IDS_BY_BEVERAGE["spirits"]
    for rule_id, score in report.rule_scores.items():
        assert score.precision == 1.0 and score.recall == 1.0, (
            f"{rule_id}: precision={score.precision:.3f}, "
            f"recall={score.recall:.3f}, disagreements={score.disagreements}"
        )


# ----------------------------------------------------------------------------
# Day-4 corpus-wide gate
# ----------------------------------------------------------------------------


# Per-rule floors, at the day-0 plan's 0.85. The rule-pack gaps that
# originally forced lower floors are fixed:
#   * `beer.net_contents.presence` v3 accepts pint/quart/gallon
#     (27 CFR 7.65(b)), so lbl-0003's "1 PINT" passes.
#   * `beer.health_warning.exact_text` v2 uses the warning_compliance
#     check (caps prefix verbatim, body case-insensitive with edit
#     tolerance), so the ALL-CAPS renderings on lbl-0002/lbl-0003 pass.
#   * `infer_is_imported` no longer flips on domestic origin statements
#     ("MADE IN USA" on lbl-0002), so country-of-origin stays NA.
#
# On today's six-item corpus one new false negative on a 6-support rule
# lands at recall 0.833 and trips the gate — that is intentional: any
# single new disagreement should be root-caused, not absorbed.
PRECISION_FLOOR = 0.85
RECALL_FLOOR = 0.85


@pytest.fixture(scope="module")
def whole_corpus():
    """Every item in real_labels/ + tests_fixtures/ that has a recording.

    The fixtures are included intentionally — they're the only wine and
    spirits items today, and the gate needs to assert non-beer rules
    too. Once Wikimedia round-1 wine/spirits items land, the fixtures
    can drop out without changing the test.
    """
    real = load_corpus(require_recording=True)
    fixtures = load_corpus(root=_FIXTURES_ROOT, require_recording=True)
    items = real + fixtures
    if not items:
        pytest.skip("no corpus items with recordings; nothing to gate against")
    return items


@pytest.mark.real_corpus
def test_corpus_wide_per_rule_floors(whole_corpus):
    """Every non-advisory rule with support ≥ 1 must clear the day-0 floors.

    The gate ignores:
      * Advisory rules (`beer.health_warning.size`) — not scored by design.
      * Rules with `support == 0` — no items hit pass/fail/na so there is
        no signal to gate. Common today on
        `*.country_of_origin.presence_if_imported` since the seed corpus
        is all domestic.

    Failures dump the disagreement list so the operator immediately sees
    which item moved. The floors are deliberately permissive on day 4 —
    tighten in a separate commit when the corpus has grown enough that
    the higher number is trustworthy.
    """
    from validation.scripts.measure_corpus import _ADVISORY_RULE_IDS, aggregate

    breakdown = aggregate(whole_corpus)
    failed: list[str] = []
    for rule_id, score in breakdown.overall_rule_scores.items():
        if rule_id in _ADVISORY_RULE_IDS:
            continue
        if score.support == 0:
            continue
        if score.precision < PRECISION_FLOOR:
            failed.append(
                f"{rule_id}: precision {score.precision:.3f} < {PRECISION_FLOOR}; "
                f"disagreements={score.disagreements}"
            )
        if score.recall < RECALL_FLOOR:
            failed.append(
                f"{rule_id}: recall {score.recall:.3f} < {RECALL_FLOOR}; "
                f"disagreements={score.disagreements}"
            )
    assert not failed, "corpus-wide rule floors breached:\n  " + "\n  ".join(failed)


# ----------------------------------------------------------------------------
# Detect-container (pre-capture gate) floors
# ----------------------------------------------------------------------------


# Detection rate must be near-perfect on positive frames — the gate's job
# is to let real labels through, and a label slipping past it would
# strand the user on a "Please point at a container" screen. Brand and
# net-contents floors are looser because the model deliberately
# null-leaves those fields when text is ambiguous; we gate enough to
# catch regressions but not so tightly that a single off-by-one
# transcription on a 6-item corpus breaks CI.
DETECTION_RATE_FLOOR = 0.95
BRAND_MATCH_FLOOR = 0.50
NET_CONTENTS_MATCH_FLOOR = 0.50


@pytest.mark.real_corpus
def test_detect_container_floors(whole_corpus):
    """The pre-capture detect-container gate must clear day-0 floors.

    Skips when no item carries a `recorded_detect_container.json` — that
    just means the operator hasn't run
    `validation/scripts/record_detect_container.py` yet, not a
    regression. Test_fixtures items are filtered out by the scorer
    itself (their front.jpg is a blank placeholder).
    """
    from validation.scripts.measure_corpus import score_detect_container

    score = score_detect_container(whole_corpus)
    if score.items_evaluated == 0:
        pytest.skip(
            "no items have recorded_detect_container.json yet; "
            "run validation/scripts/record_detect_container.py --all "
            "--i-know-this-costs-money to populate"
        )

    failed: list[str] = []
    if score.detection_rate < DETECTION_RATE_FLOOR:
        failed.append(
            f"detection_rate {score.detection_rate:.3f} < "
            f"{DETECTION_RATE_FLOOR}; "
            f"detected={score.detected_true}/{score.items_evaluated}"
        )
    if score.brand_evaluated and score.brand_match_rate < BRAND_MATCH_FLOOR:
        misses = [
            f"{row['id']} (got {row['brand_actual']!r}, want {row['brand_expected']!r})"
            for row in score.per_item
            if row["brand_ok"] is False
        ]
        failed.append(
            f"brand_match_rate {score.brand_match_rate:.3f} < "
            f"{BRAND_MATCH_FLOOR}; misses={misses}"
        )
    if (
        score.net_contents_evaluated
        and score.net_contents_match_rate < NET_CONTENTS_MATCH_FLOOR
    ):
        misses = [
            f"{row['id']} (got {row['net_actual']!r}, want {row['net_expected']!r})"
            for row in score.per_item
            if row["net_ok"] is False
        ]
        failed.append(
            f"net_contents_match_rate {score.net_contents_match_rate:.3f} < "
            f"{NET_CONTENTS_MATCH_FLOOR}; misses={misses}"
        )

    assert not failed, "detect-container floors breached:\n  " + "\n  ".join(failed)


# ----------------------------------------------------------------------------
# Regression gate vs committed baseline snapshot
# ----------------------------------------------------------------------------


# Slack we allow on each metric before the gate fails. Replay-mode runs
# are deterministic so the only legitimate sub-tolerance noise is
# floating-point accumulator drift; the tolerance just keeps the gate
# from firing on cosmetic differences. A real regression (a rule pack
# change, an extraction edit, a corpus item swap) moves metrics by
# whole percentage points and will trip this comfortably.
_BASELINE_TOLERANCE = 0.005


def _baseline_path() -> Path:
    return Path(__file__).resolve().parent / "real_labels" / "measurements_baseline.json"


@pytest.mark.real_corpus
def test_corpus_does_not_regress_vs_baseline():
    """Replay-mode run must clear every metric in `measurements_baseline.json`.

    Compares:
      * Whole-corpus precision / recall / F1.
      * Per-rule precision / recall / F1 for every rule the baseline
        carries (rules new to the current run are ignored — they're an
        improvement, not a regression).
      * Detect-container detection / brand / net-contents rates.

    When a regression is intentional (rule pack edit, new corpus item,
    rerecorded extraction), regenerate the baseline with:

        python -m validation.scripts.measure_corpus --include-fixtures \\
          --json --out validation/real_labels/measurements_baseline.json

    The fixtures flag must match how the baseline was originally generated
    — leaving it off would compare a 6-item run against an 8-item baseline.
    """
    baseline_path = _baseline_path()
    if not baseline_path.exists():
        pytest.skip(
            f"no baseline at {baseline_path}; "
            "run `python -m validation.scripts.measure_corpus --include-fixtures "
            "--json --out validation/real_labels/measurements_baseline.json` to seed"
        )

    # Lazy imports — keeps top-of-module clean and lets the script's
    # private helpers move without breaking unrelated tests.
    from validation.scripts.measure_corpus import (
        _FIXTURES,
        _REAL_LABELS,
        aggregate,
        score_detect_container,
    )

    baseline = json.loads(baseline_path.read_text())
    items = load_corpus(root=_REAL_LABELS, require_recording=True) + load_corpus(
        root=_FIXTURES, require_recording=True
    )
    if not items:
        pytest.skip("no corpus items with recordings; nothing to gate against")
    breakdown = aggregate(items)

    failures: list[str] = []

    # Whole-corpus micro-averaged metrics.
    overall_b = baseline["overall"]
    for metric, current in (
        ("precision", breakdown.overall_precision),
        ("recall", breakdown.overall_recall),
        ("f1", breakdown.overall_f1),
    ):
        b = float(overall_b[metric])
        if current < b - _BASELINE_TOLERANCE:
            failures.append(
                f"overall {metric}: {current:.3f} < baseline {b:.3f} "
                f"(tolerance {_BASELINE_TOLERANCE})"
            )

    # Per-rule metrics — iterate the baseline so rules new to the
    # current run don't trip the gate (they're improvements).
    rules_b = baseline.get("rules", {})
    for rule_id, b_score in rules_b.items():
        current_score = breakdown.overall_rule_scores.get(rule_id)
        if current_score is None:
            failures.append(
                f"{rule_id}: present in baseline but missing from current run"
            )
            continue
        for metric, current_val in (
            ("precision", current_score.precision),
            ("recall", current_score.recall),
            ("f1", current_score.f1),
        ):
            b_val = float(b_score[metric])
            if current_val < b_val - _BASELINE_TOLERANCE:
                failures.append(
                    f"{rule_id} {metric}: {current_val:.3f} < baseline "
                    f"{b_val:.3f}"
                )

    # Detect-container metrics. Only gate when the baseline actually
    # carries a recorded score for the metric (denominator-zero metrics
    # default to 1.0 in the score dataclass; a baseline of 1.0 with
    # zero support is meaningless).
    detect_b = baseline.get("detect_container") or {}
    if detect_b and detect_b.get("items_evaluated", 0) > 0:
        detect_current = score_detect_container(items)
        for label, key, current_val, support_attr in (
            (
                "detection_rate",
                "detection_rate",
                detect_current.detection_rate,
                "items_evaluated",
            ),
            (
                "container_type_accuracy",
                "container_type_accuracy",
                detect_current.container_type_accuracy,
                "container_type_evaluated",
            ),
            (
                "brand_match_rate",
                "brand_match_rate",
                detect_current.brand_match_rate,
                "brand_evaluated",
            ),
            (
                "net_contents_match_rate",
                "net_contents_match_rate",
                detect_current.net_contents_match_rate,
                "net_contents_evaluated",
            ),
        ):
            # Skip metrics whose current run has zero support — that's
            # the vacuous 1.0 floor and comparing it would be noise.
            if getattr(detect_current, support_attr) == 0:
                continue
            if key not in detect_b:
                continue
            b_val = float(detect_b[key])
            if current_val < b_val - _BASELINE_TOLERANCE:
                failures.append(
                    f"detect_container.{label}: {current_val:.3f} < "
                    f"baseline {b_val:.3f}"
                )

    assert not failures, (
        "regression vs measurements_baseline.json:\n  "
        + "\n  ".join(failures)
        + "\n\nIf the regression is intentional, regenerate the baseline:\n"
        "  python -m validation.scripts.measure_corpus --include-fixtures "
        "--json --out validation/real_labels/measurements_baseline.json"
    )


@pytest.mark.real_corpus
def test_split_filter_isolates_test_items(seed_corpus):
    """The split filter is the holdout-policy control surface.

    The `test` split is the frozen six-item COLA holdout; Wikimedia
    round-2 items are stamped `train` or `dev`. The splits must be
    disjoint and the holdout must stay at exactly the six seed items —
    break this and the test split leaks into dev/train.
    """
    train_items = load_corpus(beverage_type="beer", split="train")
    dev_items = load_corpus(beverage_type="beer", split="dev")
    test_items = load_corpus(
        beverage_type="beer", split="test", require_recording=True
    )
    assert len(test_items) == 6
    assert {i.id for i in test_items} == {f"lbl-000{n}" for n in range(1, 7)}
    overlap = ({i.id for i in train_items} | {i.id for i in dev_items}) & {
        i.id for i in test_items
    }
    assert not overlap, f"split leak between train/dev and the test holdout: {overlap}"
