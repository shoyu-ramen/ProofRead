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

from pathlib import Path

import pytest

from validation.measure import measure
from validation.real_corpus import RULE_IDS_BY_BEVERAGE, load_corpus
from validation.replay_extractor import ReplayVisionExtractor

_FIXTURES_ROOT = Path(__file__).resolve().parent / "tests_fixtures"


@pytest.fixture(scope="module")
def seed_corpus():
    items = load_corpus(beverage_type="beer", require_recording=True)
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
      * The canonical Health Warning rule scores 1.0 (these are six
        TTB COLA composites; if the canonical-text rule disagrees, the
        recording / rule-engine plumbing is broken).

    Other rules may have legitimate annotator-vs-rule-engine disagreements
    (e.g. `beer.net_contents.presence` rejects "1 PINT" because the regex
    doesn't list that unit, while the annotator correctly marked PASS).
    Those are signal for the rule pack, not plumbing failures, so they
    are surfaced via stdout rather than failing the test.
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
    assert hw_text.precision == 1.0 and hw_text.recall == 1.0, (
        f"canonical Health Warning rule should be perfect on COLA composites; "
        f"got precision={hw_text.precision:.3f}, recall={hw_text.recall:.3f}, "
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


@pytest.mark.real_corpus
def test_split_filter_isolates_test_items(seed_corpus):
    """The split filter is the holdout-policy control surface.

    Loading by `split="train"` against the seed corpus must return zero
    items (everything is `split: "test"` after migration). Loading by
    `split="test"` returns all six. This is the contract the day-8 CI
    gate relies on — break it and the test split leaks into dev.
    """
    train_items = load_corpus(
        beverage_type="beer", split="train", require_recording=True
    )
    test_items = load_corpus(
        beverage_type="beer", split="test", require_recording=True
    )
    assert train_items == []
    assert len(test_items) == 6
