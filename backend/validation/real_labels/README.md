# Real-photo labels — contribution format

This directory holds hand-labeled real beer-label photos for
validating real OCR / vision providers against SPEC v1.3 thresholds
(Health Warning exact-text precision ≥ 0.98, recall ≥ 0.99 on a
500-label test set).

The synthetic harness in `validation/corpus.py` produces a 50-label
floor. SPEC v1.13 calls out the need for a real-photo corpus on top
— this directory is the holding area until the loader module lands.

## Directory layout

One directory per label, named with a stable opaque ID:

```
real_labels/
├── README.md           (this file)
├── lbl-0001/
│   ├── front.jpg       front-label photo (≥ 1024px on the long edge)
│   ├── back.jpg        back-label photo (same)
│   └── truth.json      ground-truth annotation, format below
├── lbl-0002/
│   ├── front.jpg
│   ├── back.jpg
│   └── truth.json
└── …
```

Photos:
- JPEG or PNG
- Long edge ≥ 1024 px so OCR has enough resolution
- Both surfaces required (front + back); for cans, "front" is the
  primary brand surface and "back" is the wrap continuation
- Capture conditions varied per SPEC §0.5 — sunlight, dim bar,
  curved bottle, condensation; document the condition in `truth.json`

## `truth.json` schema

```json
{
  "id": "lbl-0001",
  "beverage_type": "beer",
  "container_size_ml": 355,
  "is_imported": false,
  "capture_conditions": {
    "lighting": "indoor_office",
    "surface": "dry",
    "container": "12oz_can",
    "device": "iPhone 15 Pro",
    "notes": "free-form"
  },
  "ground_truth": {
    "beer.brand_name.presence": "pass",
    "beer.class_type.presence": "pass",
    "beer.alcohol_content.format": "pass",
    "beer.net_contents.presence": "pass",
    "beer.name_address.presence": "pass",
    "beer.country_of_origin.presence_if_imported": "na",
    "beer.health_warning.exact_text": "pass",
    "beer.health_warning.size": "advisory"
  },
  "label_spec": {
    "brand": "Anytown Ale",
    "class_type": "India Pale Ale",
    "abv": "5.5% ABV",
    "net_contents": "12 FL OZ",
    "name_address": "Brewed and bottled by Anytown Brewing Co., Anytown, ST 00000",
    "health_warning_text": "<verbatim transcribed text or null>",
    "country": null
  },
  "annotator": "rosskuehl@gmail.com",
  "annotated_at": "2026-04-28"
}
```

Field notes:

- `ground_truth` keys are the v1 beer rule IDs from
  `app/rules/definitions/beer.yaml`. Values are
  `pass | fail | advisory | na`.
- `label_spec` is what the label *actually says* (transcribed by a
  human). The harness uses this for diagnostics — when real OCR
  disagrees with `ground_truth`, the diff against `label_spec` shows
  whether the disagreement is OCR error or rule-engine error.
- `health_warning_text` should be the verbatim transcription if
  present (typos and all), or `null` if absent. Don't substitute
  the canonical text — the value of this corpus is in the typos.
- `capture_conditions` is free-form; aim to cover the SPEC §0.5
  robustness matrix as the corpus grows.

## Contribution checklist

- [ ] Photos at ≥ 1024 px long edge.
- [ ] Both front and back included.
- [ ] `truth.json` follows the schema above.
- [ ] All 8 v1 beer rule IDs accounted for in `ground_truth`.
- [ ] `health_warning_text` transcribed verbatim (no auto-correct).
- [ ] No commercial-product images that break vendor ToS without
      written permission. When in doubt, redact the brand and
      record `brand` as `<redacted>`.
- [ ] No PII in capture_conditions notes.

## What runs against this directory

Nothing yet. A `validation/real_corpus.py` loader is planned to walk
this tree, build `CorpusItem`s analogous to `corpus.py`, and feed
them into `validation.measure.measure(...)` against whichever
provider is configured. Until then, the directory is purely a
holding area for contributed labels.
