# ProofRead

**Automated TTB label-compliance scanning for alcoholic beverages.**
Point a camera at a beer / wine / spirits label and get a pass / warn / fail report against the federal regulations the U.S. Treasury enforces — every requirement cited (27 CFR), every finding pinned to a region on the image, and the system refuses to guess when the capture isn't good enough to verify.

> **Live demo:** <https://proofread-rk-production.up.railway.app/>
> Click *Old Tom Bourbon*, *Stone's Throw Gin*, *Mountain Crest IPA*, or *Heritage Cabernet* to run a real verify in ~3 s — no signup, no install.

---

## Why TTB?

The **Alcohol and Tobacco Tax and Trade Bureau** (TTB) is a bureau of the U.S. Department of the Treasury. Before any beverage producer can ship a labeled bottle interstate, TTB has to approve the label under a process called **COLA** (Certificate of Label Approval). A reviewer reads the artwork by hand and checks it against 27 CFR Parts 5 (spirits), 7 (malt beverages), and 16 (the Government Health Warning).

That work is repetitive, error-prone, and a known queue-time problem for both TTB staff and the producers waiting on approval. ProofRead automates the rule-application step end-to-end: capture the label, extract the regulated fields with a vision model, and run a citation-bearing rule engine that returns the same verdict a reviewer would — with the underlying evidence visible.

---

## Demo

| Start | Report |
|---|---|
| ![Scan view — sample picker](scan-view.png) | ![Compliance report — Mountain Crest IPA fails warning + net contents](result-current.png) |

The demo bundles four canonical scenarios — a clean **PASS**, a case-only **WARN** (`STONE'S THROW` vs. `Stone's Throw` — a real COLA reviewer judgment call, not a hard fail), a **FAIL** with two non-compliant rules, and an **UNREAD** capture that the system refuses to verdict.

---

## How it works

The verify path is the user-visible surface and the riskiest one to get wrong. A wrong "pass" means a producer submits non-compliant artwork to TTB on the app's say-so. The pipeline is built around that risk:

1. **On-device pre-check.** Before any model call, the capture is scored for sharpness, glare, and label coverage. An unreadable frame short-circuits with `unreadable: true` — no VLM call, no guess.
2. **Field extraction (Claude Sonnet 4.6).** Reads the seven TTB-relevant fields verbatim into a typed JSON shape. The system prompt is cache-controlled; the extractor prefers `unreadable` over hallucination on poor regions.
3. **Redundant Government-Warning read (Claude Haiku 4.5).** The 27 CFR §16.21 statement runs through a *second model from a different family* in parallel — genuine independence, not just temperature variation. Disagreement between the two reads downgrades the warning rule to **ADVISORY** rather than serving a confident wrong-pass.
4. **Rule engine.** YAML-defined rules per beverage type (8 beer, 8 spirits in v1) evaluated against the extracted context. Each rule carries its CFR citation, expected vs. found values, a fix suggestion, and a bounding box keyed back to the source image. Smart cross-references handle:
   - **Brand / class / address** — case-only deltas surface as **WARN**, not FAIL.
   - **Alcohol content** — numeric comparison with ABV ↔ proof equivalence (`90 Proof` matches `45.0% ABV`).
   - **Net contents** — unit-aware (`750 mL` = `0.75 L` = `25.4 fl oz`).
   - **Government warning** — ALL-CAPS prefix verbatim required; body matched case-insensitively with edit-distance tolerance for OCR typos.
5. **Confidence-aware degradation.** Field-level confidence is capped at the surface confidence — a model can never claim higher confidence than the frame supports. Required rules over low-confidence fields downgrade to advisory rather than producing a phantom finding.
6. **In-process LRU cache** keyed on image SHA-256 + verdict-affecting inputs returns a hit in <50 ms.

The mandate is documented in [`SPEC.md` §0.5 — "fail honestly"](SPEC.md): when the system cannot confidently verify a rule, it must say so, not guess. A wrong "pass" is the worst outcome; a wrong "fail" is second worst; an honest "advisory: couldn't verify" is acceptable.

---

## Architecture highlights

**Dual-model independence for the highest-stakes rule.** The §16.21 Government Health Warning is the single rule where a wrong-pass has the biggest legal exposure for the producer. Running the redundant read through a *different model family* (Sonnet vs. Haiku) means a class of failure mode — Anthropic-specific quirk, prompt-side ambiguity — can't pass both reads silently. Cross-check disagreement is wired to downgrade, not to majority-vote.

**Cylindrical-scan capture in a single rotation pass.** The mobile app reads a curved bottle label by tracking the bottle silhouette, integrating angular position from optical flow, and painting strips into a Skia off-screen surface as the user rotates the bottle once. The unrolled-label panorama is what the backend OCR's against — flat text out of a curved capture. Architecture in [`.claude/CYLINDRICAL_SCAN_ARCHITECTURE.md`](.claude/CYLINDRICAL_SCAN_ARCHITECTURE.md).

**Citation-bearing rule engine.** Rules are YAML, not code. Each one carries its CFR citation, an `applies_if` predicate (so a rule about imported-product country-of-origin only fires when `is_imported=true`), an `exempt_if` predicate, and a `confidence_threshold`. New rules ship as data; the engine doesn't change. Beer (`backend/app/rules/definitions/beer.yaml`) and spirits (`backend/app/rules/definitions/spirits.yaml`) are wired in v1.

---

## Repo layout

```
backend/    Python 3.12 + FastAPI + uvicorn. Vision extractor chain, rule engine,
            verify + scan endpoints. Tests in backend/tests/.
mobile/     Expo SDK 51 + expo-router + react-native-vision-camera v4.
            Cylindrical-scan subsystems in src/scan/ (tracker, panorama, ui, state).
artwork/    Sample labels for tests and the demo.
.claude/    Architecture and review docs (CYLINDRICAL_SCAN_ARCHITECTURE,
            SCAN_DESIGN, SCAN_AUDIT, REVIEW_FINDINGS, MODEL_INTEGRATION_PLAN, …).
SPEC.md     v1 → v3 product + technical spec.
DEPLOY.md   Railway deployment runbook for the live demo.
Dockerfile  Single-stage backend image; auto-deploys on git push.
```

---

## Running locally

**Backend:**

```sh
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set ANTHROPIC_API_KEY for /v1/verify
pytest -v              # unit + integration; no DB or API key needed
uvicorn app.main:app --reload   # demo UI at http://localhost:8000/
```

**Mobile:**

```sh
cd mobile
npm install
npx expo prebuild
npx expo run:ios   # custom dev client — Vision Camera v4 needs native modules
```

See [`backend/README.md`](backend/README.md) and [`mobile/README.md`](mobile/README.md) for the full layout, test strategy, and stub status.

---

## Status

| Path | What it does | State |
|---|---|---|
| `POST /v1/verify` | Single-shot UI flow: image + application metadata → verdict (≤5 s p95) | Wired; live on Railway |
| `POST /v1/scans` | Multi-image scan flow with Postgres persistence | Wired (beer); needs `DATABASE_URL` |
| Beer rule set | 8 rules covering 27 CFR Part 7 + §16.21 | Wired |
| Spirits rule set | 8 rules covering 27 CFR Part 5 + §16.21 | Wired (verify path) |
| Wine rule set | 27 CFR Part 4 | Gated 422 — v2 |
| Cylindrical scan | One-rotation panorama capture on iOS | Wired (custom dev client) |

[`SPEC.md`](SPEC.md) covers v1 → v3 in full — extreme-condition robustness (§0.5), data model, rule schema, COLA cross-reference (v2), API key auth + audit log (v3).

---

## Key citations

- Beer labeling: 27 CFR Part 7
- Spirits labeling: 27 CFR Part 5
- Wine labeling: 27 CFR Part 4 *(v2)*
- Government Health Warning: 27 CFR §16.21 (text), §16.22 (type-size)
- Net contents: 27 CFR §7.27, §5.38
- Class / type: 27 CFR §7.24, §5.35

Canonical Health Warning text lives at [`backend/app/canonical/health_warning.txt`](backend/app/canonical/health_warning.txt).
