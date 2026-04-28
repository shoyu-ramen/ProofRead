"""Pytest assertions on SPEC v1.3 precision/recall targets.

The synthetic corpus run under perfect-mock OCR exists to validate the
*harness*. With perfect OCR, every rule should score 1.0 — that's the
baseline. If a non-Health-Warning rule drops below 1.0, that's a
synthesizer / extractor mismatch worth fixing before any real-OCR run.

The `real_ocr` mark guards the real-provider variant: skipped by default,
runs when the user explicitly opts in via:

    pytest validation/test_precision_targets.py -m real_ocr

The selected provider is read from `OCR_PROVIDER` env var (`google_vision`
is the only real provider wired today; future Claude-vision plugs in here
the same way).
"""

from __future__ import annotations

import os

import pytest

from validation.corpus import generate_corpus
from validation.measure import measure

# `real_ocr` mark registration and default-skip behaviour live in
# `validation/conftest.py` so they apply across the whole harness module.


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(seed=1234)


# ----------------------------------------------------------------------------
# Perfect-mock validations (always run)
# ----------------------------------------------------------------------------


def test_corpus_size(corpus):
    """The harness brief specifies a 50-label corpus split 20/10/5/5/5/5."""
    assert len(corpus) == 50

    by_cat: dict[str, int] = {}
    for item in corpus:
        by_cat[item.category] = by_cat.get(item.category, 0) + 1

    assert by_cat == {
        "compliant": 20,
        "hw_typo": 10,
        "missing_field": 5,
        "hw_missing": 5,
        "imported_with_country": 5,
        "imported_no_country": 5,
    }


def test_health_warning_meets_spec_v1_3_targets(corpus):
    """SPEC v1.3: precision ≥ 0.98, recall ≥ 0.99 on Health Warning exact text.

    Perfect-mock OCR produces 1.0 for both — anything lower is a harness
    bug (synthesizer vs. rule-engine drift), not a real-OCR signal.
    """
    report = measure(corpus)
    score = report.rule_scores["beer.health_warning.exact_text"]
    assert score.precision >= 0.98, (
        f"Health Warning precision {score.precision:.3f} < 0.98 (SPEC v1.3); "
        f"disagreements: {score.disagreements}"
    )
    assert score.recall >= 0.99, (
        f"Health Warning recall {score.recall:.3f} < 0.99 (SPEC v1.3); "
        f"disagreements: {score.disagreements}"
    )


def test_perfect_mock_yields_perfect_scores(corpus):
    """All non-advisory rules should score 1.0 under perfect-mock OCR.

    Failures here mean the synthesizer's spec disagrees with the rule
    engine's interpretation — a methodology bug to fix before scoring real OCR.
    """
    report = measure(corpus)
    for rule_id, score in report.rule_scores.items():
        if rule_id == "beer.health_warning.size":
            continue  # advisory, not scored
        assert score.precision == 1.0, (
            f"{rule_id}: precision {score.precision:.3f} != 1.0; "
            f"disagreements={score.disagreements}"
        )
        assert score.recall == 1.0, (
            f"{rule_id}: recall {score.recall:.3f} != 1.0; "
            f"disagreements={score.disagreements}"
        )


def test_typo_cases_actually_fail_under_perfect_ocr(corpus):
    """The 10 typo cases must produce `fail` on the Health Warning rule.

    Under perfect-mock OCR, the typo string flows verbatim from the
    synthesizer to the rule engine; if the rule fails to detect the
    Levenshtein-distance-1 substitution, the test corpus has lost its
    signal value and the harness is misconfigured.
    """
    typo_items = [item for item in corpus if item.category == "hw_typo"]
    assert len(typo_items) == 10

    report = measure(typo_items)
    score = report.rule_scores["beer.health_warning.exact_text"]
    # All 10 should be detected as failures (TN, since expected != pass).
    assert score.tn == 10, (
        f"Expected all 10 typo cases to be flagged as fail; got TP={score.tp}, "
        f"FP={score.fp}, FN={score.fn}, TN={score.tn}"
    )


def test_imported_no_country_fails_coo_rule(corpus):
    """Imported labels missing country must fail the COO rule, not be NA."""
    items = [i for i in corpus if i.category == "imported_no_country"]
    assert len(items) == 5

    report = measure(items)
    coo = report.rule_scores["beer.country_of_origin.presence_if_imported"]
    assert coo.tn == 5, (
        f"Expected COO rule to fail on all 5 imported-no-country labels; got "
        f"TP={coo.tp}, FP={coo.fp}, FN={coo.fn}, TN={coo.tn}"
    )


# ----------------------------------------------------------------------------
# Real-OCR variant — opt-in only
# ----------------------------------------------------------------------------


@pytest.mark.real_ocr
def test_real_ocr_meets_spec_v1_3_targets(corpus):
    """Same SPEC v1.3 thresholds, against a real OCR / vision provider.

    Selection via the `OCR_PROVIDER` env var:

    - `google_vision` — instantiates `GoogleVisionOCRProvider`
      (requires the `[google-vision]` extra and ADC credentials).
    - `claude_vision` — instantiates `ClaudeVisionExtractor`
      (requires `ANTHROPIC_API_KEY`).

    Missing dependencies are reported as `pytest.skip` so the run is
    interpretable when the user opts in but the environment isn't set up.
    """
    provider_name = os.environ.get("OCR_PROVIDER", "claude_vision")

    ocr_provider = None
    vision_extractor = None

    if provider_name == "google_vision":
        try:
            from app.services.ocr import GoogleVisionOCRProvider

            ocr_provider = GoogleVisionOCRProvider()
        except ModuleNotFoundError as exc:
            pytest.skip(f"google-cloud-vision not installed: {exc}")
        except Exception as exc:
            pytest.skip(f"GoogleVisionOCRProvider failed to initialize: {exc}")
    elif provider_name == "claude_vision":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set; skipping Claude-vision real_ocr test")
        try:
            from app.services.extractors.claude_vision import ClaudeVisionExtractor

            vision_extractor = ClaudeVisionExtractor()
        except Exception as exc:
            pytest.skip(f"ClaudeVisionExtractor failed to initialize: {exc}")
    else:
        raise RuntimeError(f"Unknown OCR_PROVIDER: {provider_name!r}")

    report = measure(
        corpus,
        ocr_provider=ocr_provider,
        vision_extractor=vision_extractor,
    )
    score = report.rule_scores["beer.health_warning.exact_text"]
    assert score.precision >= 0.98
    assert score.recall >= 0.99
