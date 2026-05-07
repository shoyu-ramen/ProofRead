# `truth.json` schema v2

Schema for hand-annotated entries in `validation/real_labels/`. Versioned so
the loader can refuse mismatched files cleanly when the schema changes
again.

## Top-level shape

```json
{
  "schema_version": 2,
  "id": "lbl-XXXX",
  "beverage_type": "beer | wine | spirits",
  "container_size_ml": 355,
  "is_imported": false,

  "source_kind": "wikimedia_commons | wikimedia_synth | cola_artwork",
  "split": "train | dev | test",

  "capture_conditions": { ... },
  "ground_truth":       { "<rule_id>": "pass | fail | advisory | na" },
  "label_spec":         { ... },
  "application":        { ... },
  "gold_extracted_fields": { ... },

  "annotator": "rosskuehl@gmail.com",
  "annotated_at": "YYYY-MM-DD",
  "two_annotator_check": false
}
```

## Required fields

### `schema_version`
Always `2`. Loader refuses any other value with a clear error so a stale
file can't be silently mis-parsed.

### `id`
Stable opaque ID. Matches the directory name.

### `beverage_type`
One of `beer | wine | spirits`. Drives which `ground_truth` rule_ids must
appear and which beverage-type-conditional fields are expected in
`label_spec` and `gold_extracted_fields`.

### `container_size_ml`
Container volume in mL. Used as the `container_size_ml` parameter to the
verify pipeline.

### `is_imported`
Boolean. Drives whether `*.country_of_origin.presence_if_imported` rules
are required (`pass`/`fail`) or `na`.

### `source_kind`
- `wikimedia_commons` — real photograph licensed CC-BY / CC-BY-SA / CC0 /
  PD-self. Counts toward photo-realism precision/recall floors.
- `wikimedia_synth` — degradation applied to a Wikimedia photo (glare,
  curvature, low-light tone shift, motion blur). Same license as the
  source. Counts toward the photo subset.
- `cola_artwork` — TTB COLA composite artwork (real label content, no
  photographic capture realism). Excluded from the photo-realism floors;
  counts only on the canonical-text Health Warning regression.

### `split`
- `train` (~70%) — extractor prompt iteration; humans look at it.
- `dev`   (~15%) — loop-closing on tweaks; measured locally on demand.
- `test`  (~15%) — release gate; opened only by CI via `--include-test`.
  Stratified by `beverage_type × verdict` so no rule_id is starved.

### `capture_conditions`
```json
{
  "lighting": "studio | indoor_office | bar_dim | outdoor_bright | mixed | backlit",
  "surface":  "dry | wet_condensation | foggy | dirty | torn | foil | embossed",
  "container": "12oz_can | 12oz_bottle | 750ml_wine | 750ml_spirits | 1750ml_handle | growler",
  "device":   "free-form (Wikimedia upload | iPhone 15 Pro | …)",
  "notes":    "free-form, no PII"
}
```

### `ground_truth`
Map of `rule_id` → `pass | fail | advisory | na`. The set of required keys
is beverage-type-aware:

**Beer** (8 keys):
```
beer.brand_name.presence
beer.class_type.presence
beer.alcohol_content.format
beer.net_contents.presence
beer.name_address.presence
beer.country_of_origin.presence_if_imported
beer.health_warning.exact_text
beer.health_warning.size
```

**Wine** (2 keys — the wine pack is intentionally narrow today):
```
wine.sulfite.presence
wine.organic.format
```

The seven shared base rules (`wine.brand_name.presence`,
`wine.class_type.presence`, …) are *not* in `RULE_IDS_BY_BEVERAGE["wine"]`
yet because `app/rules/definitions/wine.yaml` doesn't implement them.
Annotating them today would produce truth values nothing scores against.
When the wine pack grows, bump `validation.real_corpus.SCHEMA_VERSION`
and add a migration that back-fills the new keys — same pattern as the
v1→v2 migration.

**Spirits** (9 keys):
```
spirits.brand_name.matches_application
spirits.class_type.matches_application
spirits.alcohol_content.format
spirits.alcohol_content.matches_application
spirits.net_contents.matches_application
spirits.name_address.presence
spirits.country_of_origin.presence_if_imported
spirits.health_warning.compliance
spirits.age_statement.format
```

Wine doesn't have a beer-shaped rule pack today; the `wine.*` rule_ids
above mirror what a future wine pack must expose so the truth file is
forward-compatible. The loader validates that **every** key for the
declared `beverage_type` is present and that no off-type keys appear.

### `label_spec`
Verbatim transcription of what the label *actually says*. Used for
diagnostics — when an extractor output disagrees with `ground_truth`,
diffing against `label_spec` shows whether the disagreement is
extractor error or rule-engine error.

```json
{
  "brand": "Caymus",
  "class_type": "Cabernet Sauvignon",
  "abv": "14.6% Alc/Vol",
  "net_contents": "750 mL",
  "name_address": "Vinted and Bottled by …",
  "health_warning_text": "<verbatim or null>",
  "country": "USA",
  "sulfite_declaration": "Contains Sulfites",
  "organic_certification": null,
  "age_statement": null
}
```

Beverage-type-conditional fields (`sulfite_declaration`,
`organic_certification`, `age_statement`) may be omitted if not relevant
— the loader treats absent-as-null.

### `application`
The producer record passed to the verify endpoint as the `application`
form field. Required for spirits items (cross-reference rules score
zero signal otherwise). Optional for beer / wine.

```json
{
  "brand_name": "Caymus",
  "class_type": "Cabernet Sauvignon",
  "alcohol_content": "14.6%",
  "net_contents": "750 mL"
}
```

For deliberate-mismatch test items (a spirits label whose ABV doesn't
match the application), set the `application` field to the *claimed*
value and the `label_spec` to what the photograph shows; the
`ground_truth` for the matching rule is then `fail`.

### `gold_extracted_fields`
What a perfect extractor *should* return for this image. Enables
shadow-evaluation: the harness can score the extractor itself
(field-by-field) separately from the rule engine. `bbox` is optional
and may be omitted while a corpus is being bootstrapped.

```json
{
  "brand_name":      {"value": "Caymus",            "bbox": [120, 80, 320, 140]},
  "class_type":      {"value": "Cabernet Sauvignon"},
  "alcohol_content": {"value": "14.6% Alc/Vol"},
  "net_contents":    {"value": "750 mL"},
  "name_address":    {"value": "Vinted and Bottled by …"},
  "country_of_origin": {"value": null, "unreadable": false},
  "health_warning":  {"value": "GOVERNMENT WARNING: …"}
}
```

Set `value: null` + `unreadable: false` when the field is genuinely
absent (so the rule engine produces a real FAIL, not ADVISORY). Set
`unreadable: true` when the photograph itself doesn't carry enough
signal to read the field — the rule engine downgrades to ADVISORY.

## Provenance fields
`annotator` — email of the human who labelled it. `annotated_at` —
ISO date. `two_annotator_check` — `true` once a second annotator has
re-labelled and the labels matched (10% sample target).

## Validation

`validation/scripts/validate_truth.py` (lands later this phase) enforces:

- `schema_version == 2`.
- All required keys present for the declared beverage_type.
- No off-type rule_ids in `ground_truth`.
- Photos exist at `front.jpg` / `back.jpg` and have long edge ≥ 1024 px.
- `health_warning_text` parses with the same regex the extractor uses.
- `application` present when `beverage_type == "spirits"`.
- No PII patterns in `capture_conditions.notes`.
- `split` distribution stays within the configured stratification.
