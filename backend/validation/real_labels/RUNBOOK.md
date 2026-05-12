# Annotator runbook — adding a real-labels item

End-to-end steps for moving an item from "I found a Wikimedia photo I
want" to "the harness can score it." The whole loop is ~10 min per item
once you have a candidate URL.

The corpus target (per the day-0 plan) is **100 items** distributed
50 beer / 25 wine / 25 spirits, sourced from Wikimedia Commons +
synthesised degradations. Cost stays at $0 — recordings come from a
local Qwen3-VL endpoint (or fall back to the synth stub if you don't
have one running).

---

## 0. Prereqs

* Backend venv activated: `cd backend && source .venv/bin/activate`.
* Optional but recommended: a local Qwen3-VL OpenAI-compatible server
  (Ollama / vLLM / LM Studio / llama.cpp) reachable at
  `$QWEN_VL_BASE_URL`. Without it, `record_extraction.py` falls back
  to the synth stub and you lose extractor signal.

```bash
# Example: Ollama at the default port, with Qwen3-VL pulled
export QWEN_VL_BASE_URL=http://localhost:11434/v1
export QWEN_VL_MODEL=qwen3-vl:7b
```

---

## 1. Triage a candidate on Wikimedia Commons

Browse `commons.wikimedia.org` and find a candidate photo. The hit list:

* **Beer**: `Category:Beer cans`, `Category:Beer bottles`, plus brand
  categories (`Category:Pabst Brewing Company`, etc.).
* **Wine**: `Category:Wine bottles`, plus varietal/region
  (`Category:Bordeaux wine`, …).
* **Spirits**: `Category:Whisky bottles`, `Category:Vodka bottles`,
  `Category:Tequila bottles`.

Filter heuristics — reject if any apply:
* Long edge < 1024 px.
* License is non-commercial / no-derivatives / fair-use only.
* Image is a marketing render, not a real bottle/can photograph.
* Label text is illegible at the available resolution.

A useful read-only sanity check: open the file's "Wiki page" and look
for the rendered license box. Anything CC-BY / CC-BY-SA / CC0 / PD
passes the fetcher's filter.

---

## 2. Fetch the image

```bash
# By URL
python -m validation.scripts.wikimedia_fetcher \
    "https://commons.wikimedia.org/wiki/File:Example.jpg"

# By File: title
python -m validation.scripts.wikimedia_fetcher "File:Example.jpg"

# Dry-run first to read the license / size before committing
python -m validation.scripts.wikimedia_fetcher "File:Example.jpg" --dry-run
```

The fetcher:

1. Resolves the file via the Wikimedia API.
2. Refuses anything outside the accepted-license set or below the
   1024 px floor.
3. Auto-allocates the next free `lbl-XXXX` ID (90xx is reserved for
   ad-hoc fixtures and is skipped automatically).
4. Writes `front.jpg` and a duplicate `back.jpg` (most Wikimedia
   photos show one face — note that explicitly in capture_conditions
   so an "advisory on health-warning" outcome is honest).
5. Appends a per-label entry to `SOURCES.md`.

If the file has a separate back-label photo (rare), pass
`--back "File:OtherFile.jpg"` to fetch both.

---

## 3. Annotate

```bash
python -m validation.scripts.annotate validation/real_labels/lbl-XXXX
```

The CLI walks every truth.json field with sensible defaults:

* `beverage_type`, `container_size_ml`, `is_imported` — set per the
  visible label.
* `source_kind` — `wikimedia_commons` for fetched photos,
  `wikimedia_synth` for items produced by `synth_augment.py`,
  `cola_artwork` for the seed corpus.
* `split` — pick `train`, `dev`, or `test` per the stratification
  policy. Defaults to `train`. The `test` split is the release-gate
  holdout; only stamp items as `test` after the corpus freeze on day 8.
* `label_spec.health_warning_text` — VERBATIM transcription including
  any typo. Don't auto-correct. If absent, set null.
* Verdict prompts are pre-computed from `is_imported` / `country` /
  health-warning state; press Enter to accept the default unless you
  see a reason to override.

The annotator runs `validate_truth.py` against the saved file before
exiting — schema slips fail loud, not silent.

To edit an existing truth.json, pass `--edit`.

---

## 4. Record the extraction

```bash
# Default: tries Qwen3-VL, falls back to synth on failure
python -m validation.scripts.record_extraction validation/real_labels/lbl-XXXX

# Explicit
python -m validation.scripts.record_extraction \
    validation/real_labels/lbl-XXXX --mode qwen
```

The recorder calls the live extractor on `front.jpg` and `back.jpg`,
merges the per-panel reads (highest-confidence wins per field, worse-of
per-panel image_quality), and writes `recorded_extraction.json` next to
`truth.json`. A small recording (≤2 KB JSON) per item is committed and
re-read by every harness run from then on.

The Anthropic mode (`--mode anthropic`) costs money and is gated
behind `--i-know-this-costs-money`. Don't use it for routine
re-recording; use it only for one-off comparisons against the
production extractor.

---

## 5. Smoke-check the new item

```bash
# Lint everything (incl. the new item)
python -m validation.scripts.validate_truth

# Run the harness against just the new item
python -m pytest validation/test_real_corpus.py -v
```

Linter clean + harness green = the item is wired into the regression
gate. Commit the entire `lbl-XXXX/` directory **plus** the new
`SOURCES.md` line in one commit so provenance and corpus stay in sync.

---

## 6. When something breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| `RecordingSchemaError: schema_version is None` | Old recording | Re-run `record_extraction.py --force` |
| `qwen mode failed: ExtractorUnavailable` | `$QWEN_VL_BASE_URL` not set or unreachable | Start the local model server, or add `--mode synth` for stub recordings |
| `linter: long edge XYZ px < 1024 px floor` | Source image too small | Reject the candidate; pick a higher-resolution Wikimedia file |
| `linter: country_of_origin verdict is 'fail' but is_imported=False` | Annotator slipped | `annotate.py --edit` to fix the verdict |
| `linter: spirits items must include a non-empty application` | Spirits truth without claim | Edit `application` in the truth.json or run `annotate.py --edit` |
| Harness reports rule disagreement on a "pass" item | Either the rule pack is wrong (regex doesn't accept the unit) or the annotator was wrong | Investigate; fix whichever side is wrong. Don't paper over with `advisory` — the corpus is supposed to find these. |

---

## 7. Synth augmentations (later)

Once a Wikimedia photo is in the corpus, `synth_augment.py` (lands day 7)
will derive degraded variants under `lbl-XXXX/augmented/` so the corpus
covers SPEC §0.5 conditions (glare, curvature, low-light) without
requiring physical access to bottles. Each augmentation has its own
`recorded_extraction.json` produced by the same recorder.

That step is a corpus *expansion*, not an item-add — leave it to the
day-7 pass.
