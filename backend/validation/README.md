# ProofRead OCR validation harness

End-to-end validation of the v1 beer compliance pipeline against a
synthetic 50-label corpus. The harness exists to:

1. Catch regressions in the rule engine and beer field extractor.
2. Provide a spec-aligned precision/recall measurement framework that
   real-OCR runs (Google Vision, Claude-vision, future TrOCR fine-tune)
   can plug into without code changes.
3. Make the SPEC v1.3 Health-Warning thresholds (precision ≥ 0.98,
   recall ≥ 0.99) checkable in CI.

## Layout

```
validation/
├── __init__.py
├── synthesize.py              Pillow-based label-image generator
├── corpus.py                  50-label corpus generator
├── measure.py                 Run pipeline, score per-rule, render markdown
├── test_precision_targets.py  Pytest assertions on SPEC v1.3 targets
├── README.md                  (this file)
└── real_labels/
    └── README.md              Format spec for hand-labeled real photos
```

## How to run

From `backend/`:

```bash
source .venv/bin/activate

# CI-style: run the pytest assertions (perfect-mock OCR by default).
pytest validation/

# Ad-hoc: print the markdown report for a corpus run.
python -m validation.measure

# JSON output (for CI dashboards / regression diffing).
python -m validation.measure --json

# Same corpus, against real OCR (requires GOOGLE_APPLICATION_CREDENTIALS).
python -m validation.measure --provider google_vision
pytest validation/ -m real_ocr
```

## Scoring model

For each `rule_id` we treat the system as a binary classifier on the
`pass` label:

| | Predicted `pass` | Predicted not-`pass` |
|---|---|---|
| **Truth `pass`** | TP | FN |
| **Truth not-`pass`** | FP | TN |

`na` outcomes are excluded from the per-rule denominator (rules that
don't apply contribute neither signal nor noise). Advisory rules
(only `beer.health_warning.size` in v1) are excluded from
precision/recall scoring entirely — they're advisory by design.

The "overall" line in the report micro-averages TP/FP/FN across all
non-advisory rules.

## What the synthetic corpus does and does not validate

**Validates:**

- The rule engine + beer field extractor produces the right outcome
  given an OCR text that exactly matches the rule engine's
  expectations. This catches regressions in extraction logic, regex
  drift, and rule-definition errors.
- The harness plumbing (`process_scan` glue, OCR provider abstraction,
  ground-truth comparison) is consistent end-to-end.
- The 50-label distribution covers each major rule's pass and fail
  branches, plus the `na` branch on country-of-origin.

**Does not validate:**

- Real OCR accuracy on photographic labels. Pillow-rendered text on a
  flat background is dramatically easier than a real bottle photo.
- Curved-surface, glare, low-light, or any of the SPEC §0.5 robustness
  conditions. Those are real-photo concerns and are out of scope here.
- The Health-Warning second-pass TrOCR fine-tune (SPEC v1.13) — that
  needs a real-photo corpus, contributed via `validation/real_labels/`.
- Brand-name extraction quality on creative labels. The synthesizer
  always renders the brand as the largest text on the front; real
  labels often hide the brand inside ornate artwork.

## Methodology decisions

A few choices in this harness materially affect the numbers it produces.
Documented here so they aren't surprises:

1. **Health Warning typos are alphabetic-only and same-case.** The
   single-character substitution preserves string length and sentence
   boundaries. `_extract_health_warning` in
   `app/services/extractors/beer.py` trims the captured snippet to the
   last `.` if one is found ≥200 characters in; punctuation
   substitutions could move that cutoff and accidentally erase the
   typo. Alphabetic-only substitution sidesteps that.

2. **Compliant labels declare `is_imported=False`.** The COO rule is
   then `na` (not applicable), per SPEC v1.11. We treat `na` as neither
   TP nor FP; including it would give every domestic label a free
   correct prediction and inflate precision.

3. **The "perfect mock" OCR is OCR-equivalent, not pixel-equivalent.**
   It returns the synthesizer's *recorded* text rather than running
   real OCR over the rendered PNG. Two reasons: (a) running real OCR
   over Pillow-rendered text is no fairer a test than the recorded
   text (the synthesizer is the ground truth either way); (b) the
   harness's job is to validate the rule pipeline, not to
   double-check Pillow.

4. **The synthesizer's brand-name rendering is the largest block on
   the front.** Otherwise the beer extractor's "biggest block on
   front" heuristic would mis-identify the brand. Real labels do not
   always honor this convention; that's a known limitation of the
   v1 extractor and out of scope for this harness.

   *Corollary:* In a "missing brand" corpus item (one of the five
   missing-field cases), the v1 extractor still picks the largest
   remaining block on the front (e.g., the class type) and reports
   it as the brand. So `beer.brand_name.presence` *passes* on those
   items — that is what the v1 system actually does, and ground
   truth reflects it. A real brand-name *recognizer* would behave
   differently; the new Claude-vision extractor is expected to
   honestly report `brand=None` when no brand is present, and the
   harness's ground truth will start to mark those `fail` once we
   wire that path in. Until then, `missing_field-brand` items test
   the *plumbing* (image bytes flow through, OCR text is read), not
   the *brand recognition*.

5. **ABV format is generated as well-formed only.** SPEC v1.11 lists
   `beer.alcohol_content.format` as required-when-present; the
   synthesizer never produces malformed ABV, so this rule always
   scores `pass`. To stress-test it, extend `LabelSpec` with an
   `abv_malformed` flag and seed a category in `corpus.py`.

6. **Health Warning size is always advisory in v1.** The rule is
   tagged `severity: advisory` and excluded from precision/recall;
   we surface its advisory count in the report so reviewers can see
   it's running.

## Switching OCR / vision providers

The harness is provider-agnostic — `measure.measure(items, ocr_provider=...)`
accepts anything implementing `OCRProvider.process(bytes, hint=str) -> OCRResult`.
Three providers exist or are planned:

- `MockOCRProvider` (fixture-driven; used in `tests/`).
- `PerfectMockOCRProvider` (defined in `measure.py`; uses the
  synthesizer's recorded text — the harness default).
- `GoogleVisionOCRProvider` (real OCR; requires the `[google-vision]`
  extra and `GOOGLE_APPLICATION_CREDENTIALS`).
- *Planned:* a Claude-vision provider (config flags
  `VISION_EXTRACTOR=claude` + `ANTHROPIC_API_KEY` already in
  `app/config.py`). Wire it into `measure._load_provider` and the
  `pytest -m real_ocr` test once it lands.

The pytest `real_ocr` mark is registered in
`test_precision_targets.py::pytest_configure` so it's always
discoverable via `pytest --markers`. Tests under it are skipped by
default; opt in with `pytest validation/ -m real_ocr`.

## Extending with real-photo labels

The synthetic corpus is the floor, not the ceiling. SPEC v1.13
identifies the Health Warning second-pass as a key risk and prescribes
a 500-label hand-labeled test set. To add real photos to this harness:

1. Drop labeled photos into `validation/real_labels/<id>/{front.jpg,back.jpg,truth.json}`.
2. The `truth.json` schema is documented in `real_labels/README.md`.
3. A future `validation/real_corpus.py` module (not built yet) will
   load the directory tree the same way `corpus.py` builds the
   synthetic items and feed them through `measure.measure(...)`.

Until that module exists, the directory is the holding area for
contributed photos. Adding it to the harness is a small follow-up
once the first batch of real labels is contributed.
