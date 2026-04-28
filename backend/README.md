# Proofread — backend

TTB alcohol-label compliance verification. FastAPI + a data-driven rule engine
+ Claude vision for field extraction. Two flow paths share the rule engine:

| Path | Purpose | Latency target | Status |
|---|---|---|---|
| `POST /v1/verify` | Single-shot UI flow: drop one image + paste application metadata → verdict | ≤ 5 s p95 | wired (this scaffold) |
| `POST /v1/scans` | Multi-image scan flow: front + back upload → finalize → report. Postgres-backed. | seconds | wired (v1 beer) |

The riskiest piece — verifying the Government Health Warning verbatim — is
covered by tests on both paths.

## Layout

```
app/
  main.py                  FastAPI entrypoint, routers, /static + /
  config.py                pydantic-settings
  db.py                    async SQLAlchemy engine
  auth.py                  auth stub (Auth0 wiring deferred)
  models.py                SQLAlchemy 2.0 ORM models for v1 tables
  api/
    scans.py               POST /v1/scans (create, upload, finalize, report)
    verify.py              POST /v1/verify (single-shot Claude-vision flow)
  rules/
    types.py               Rule/Check/Result dataclasses + WARN outcome + worse()
    canonical.py           loads canonical texts (e.g. Health Warning) from disk
    checks.py              check registry — presence, regex, exact_text,
                           advisory_note, cross_reference (string + WARN on
                           case-only diffs), cross_reference_numeric (ABV/proof
                           equivalence), cross_reference_volume (mL ↔ L ↔ fl oz),
                           warning_compliance (ALL-CAPS prefix + body tolerance)
    engine.py              rule executor with applies_if / exempt_if +
                           confidence-aware degradation (fields with low/
                           unreadable confidence downgrade required rules to
                           ADVISORY rather than guessing pass/fail)
    loader.py              YAML → Rule objects
    definitions/
      beer.yaml            v1 beer rule set (8 rules)
      spirits.yaml         spirits rule set with cross-references against
                           the producer's submitted record
  services/
    ocr.py                 OCRProvider Protocol + Mock + Google Vision (for
                           the multi-image scan flow)
    pipeline.py            scan-flow orchestrator: OCR → extract → engine
    vision.py              Claude vision extractor (Sonnet/Opus 4.x with
                           image input) + MockVisionExtractor for tests
    verify.py              single-shot orchestrator: image → vision →
                           context → engine → report
    extractors/beer.py     beer field extraction from OCR blocks
    sensor_check.py        capture-quality assessment (sharpness, glare)
    storage.py             local + S3 image storage backends
  canonical/
    health_warning.txt     canonical 27 CFR 16.21 statement
  static/
    index.html             demo UI (vanilla JS) for the verify flow
    wordmark.svg, favicon.svg
    samples/               four PNGs covering pass / warn / fail / unreadable
tests/
  test_rule_engine.py      8 v1 beer rules — direct engine tests
  test_extractor.py        beer field extraction
  test_pipeline_e2e.py     scan-flow end-to-end (mocked OCR)
  test_api.py              scan-flow HTTP lifecycle
  test_db_persistence.py   scan + report row persistence
  test_claude_vision.py    Claude vision extractor unit tests
  test_sensor_check.py     capture-quality assessment
  test_new_checks.py       warning_compliance + cross_reference_{numeric,volume}
  test_spirits_rules.py    spirits.yaml loads + rule-level scenarios
  test_verify.py           /v1/verify path with MockVisionExtractor
```

## Running locally

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env to set ANTHROPIC_API_KEY (only required for /v1/verify against real images)

# Optional — start Postgres + Redis for the scan-flow persistence tests
docker compose up -d

# Run the test suite (no DB or API key needed for unit/integration tests)
pytest -v

# Start the API + demo UI
uvicorn app.main:app --reload
# UI at http://localhost:8000/
```

## What's wired

- **Rule engine, two beverage types.** Beer (8 rules) and spirits (8 rules)
  load from YAML and evaluate against an `ExtractionContext`. Results carry
  citations, findings, fix suggestions, and bounding boxes.
- **Claude vision extractor.** `services/vision.py` calls Claude with vision
  input and a structured-JSON system prompt (cache-control'd) to read the
  seven TTB-relevant fields verbatim. Prefers `unreadable: true` over
  guessing on poor images; the engine's confidence-aware degradation then
  surfaces required rules as ADVISORY rather than spurious FAIL.
- **Smart cross-references** for the verify flow:
  - **Brand / class / address** — case-only deltas surface as **WARN**, not
    FAIL (Dave's "STONE'S THROW" vs. "Stone's Throw" example).
  - **Alcohol content** — numeric comparison with optional ABV ↔ proof
    equivalence ("90 Proof" matches an application ABV of 45.0).
  - **Net contents** — unit-aware ("750 mL" matches "0.75 L"; "16 FL OZ"
    matches "473 mL").
  - **Government warning** — ALL-CAPS prefix verbatim required (Jenny's
    title-case rejection); body matched case-insensitively with a small
    edit-distance tolerance for typos.
- **Single-shot verify endpoint.** `POST /v1/verify` accepts a multipart
  upload (image + JSON metadata + beverage_type) and returns the verdict
  synchronously. The demo UI at `/` exercises it against four bundled
  sample labels.
- **Multi-image scan flow.** `POST /v1/scans` lifecycle (create → upload
  front+back → finalize → report) with Postgres persistence and Google
  Vision OCR. Beer-only in v1.

## Test strategy

The two riskiest assertions:

1. **The rule engine correctly identifies a non-compliant Health Warning.**
   Covered for beer (`exact_text` strict mode) by
   `test_rule_engine.py::test_health_warning_*` and for spirits (relaxed
   `warning_compliance` mode that catches title-case prefix violations) by
   `test_new_checks.py::test_warning_*` and
   `test_spirits_rules.py::test_warning_with_titlecase_prefix_fails`.

2. **The verify flow doesn't auto-PASS Dave's "STONE'S THROW" case.**
   `test_verify.py::test_warn_scenario_brand_case_only` asserts overall
   verdict = `warn`, not `pass`.

Run only the verify-flow tests:

```bash
pytest tests/test_new_checks.py tests/test_spirits_rules.py tests/test_verify.py -v
```

## Citations

- Beer labeling: 27 CFR Part 7
- Spirits labeling: 27 CFR Part 5
- Health Warning: 27 CFR Part 16 (16.21 = statutory text; 16.22 = type-size)
- Net contents: 27 CFR 7.27 / 5.38
- Class/type: 27 CFR 7.24 / 5.35

The canonical TTB Health Warning text is in `app/canonical/health_warning.txt`.

## What's not yet wired

- Real Auth0 JWT validation (auth stub returns a fixed test user)
- Production Google Vision integration test (skeleton only)
- Batch upload UI and endpoint (Janet's "200–300 labels at once")
- On-device pre-check protocol with mobile
