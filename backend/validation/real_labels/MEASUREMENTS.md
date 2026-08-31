# ProofRead corpus measurements

Generated 2026-06-11 by `validation/scripts/measure_corpus.py`. Replay-mode against committed `recorded_extraction.json` + `recorded_detect_container.json` payloads — zero $ per run.

Extraction provenance: synth_from_truth: 9, claude-opus-4-7: 6.

## Composition

| Beverage | Items | Sources | Splits |
|---|---|---|---|
| beer | 9 | cola_artwork: 6, wikimedia_commons: 3 | dev: 1, test: 6, train: 2 |
| wine | 1 | cola_artwork: 1 | test: 1 |
| spirits | 5 | cola_artwork: 1, wikimedia_commons: 4 | dev: 1, test: 1, train: 3 |
| **total** | **15** | | |

## Whole-corpus rule scores

- Items evaluated: **15**
- Micro-averaged precision: **1.000**
- Micro-averaged recall:    **1.000**
- Micro-averaged F1:        **1.000**

| Rule | Support | TP | FP | FN | TN | Precision | Recall | F1 | Disagreements |
|---|---|---|---|---|---|---|---|---|---|
| `beer.alcohol_content.format` | 9 | 9 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `beer.brand_name.presence` | 9 | 9 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `beer.class_type.presence` | 9 | 8 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `beer.country_of_origin.presence_if_imported` | 1 | 0 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `beer.health_warning.exact_text` | 9 | 6 | 0 | 0 | 3 | 1.000 | 1.000 | 1.000 |  |
| `beer.health_warning.size` | (advisory: 9) | — | — | — | — | — | — | — | — |
| `beer.name_address.presence` | 9 | 8 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `beer.net_contents.presence` | 9 | 9 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `spirits.age_statement.format` | 5 | 3 | 0 | 0 | 2 | 1.000 | 1.000 | 1.000 |  |
| `spirits.alcohol_content.format` | 5 | 4 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `spirits.alcohol_content.matches_application` | 5 | 5 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `spirits.brand_name.matches_application` | 5 | 5 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `spirits.class_type.matches_application` | 5 | 5 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `spirits.country_of_origin.presence_if_imported` | 2 | 2 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `spirits.health_warning.compliance` | 5 | 1 | 0 | 0 | 4 | 1.000 | 1.000 | 1.000 |  |
| `spirits.name_address.presence` | 5 | 4 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `spirits.net_contents.matches_application` | 5 | 5 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `wine.organic.format` | 1 | 1 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `wine.sulfite.presence` | 1 | 1 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |

## Beer (9 items)

- Overall precision: **1.000**
- Overall recall:    **1.000**
- Overall F1:        **1.000**

| Rule | Support | TP | FP | FN | TN | Precision | Recall | F1 | Disagreements |
|---|---|---|---|---|---|---|---|---|---|
| `beer.alcohol_content.format` | 9 | 9 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `beer.brand_name.presence` | 9 | 9 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `beer.class_type.presence` | 9 | 8 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `beer.country_of_origin.presence_if_imported` | 1 | 0 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `beer.health_warning.exact_text` | 9 | 6 | 0 | 0 | 3 | 1.000 | 1.000 | 1.000 |  |
| `beer.health_warning.size` | (advisory: 9) | — | — | — | — | — | — | — | — |
| `beer.name_address.presence` | 9 | 8 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `beer.net_contents.presence` | 9 | 9 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |

## Wine (1 items)

- Overall precision: **1.000**
- Overall recall:    **1.000**
- Overall F1:        **1.000**

| Rule | Support | TP | FP | FN | TN | Precision | Recall | F1 | Disagreements |
|---|---|---|---|---|---|---|---|---|---|
| `wine.organic.format` | 1 | 1 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `wine.sulfite.presence` | 1 | 1 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |

## Spirits (5 items)

- Overall precision: **1.000**
- Overall recall:    **1.000**
- Overall F1:        **1.000**

| Rule | Support | TP | FP | FN | TN | Precision | Recall | F1 | Disagreements |
|---|---|---|---|---|---|---|---|---|---|
| `spirits.age_statement.format` | 5 | 3 | 0 | 0 | 2 | 1.000 | 1.000 | 1.000 |  |
| `spirits.alcohol_content.format` | 5 | 4 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `spirits.alcohol_content.matches_application` | 5 | 5 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `spirits.brand_name.matches_application` | 5 | 5 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `spirits.class_type.matches_application` | 5 | 5 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `spirits.country_of_origin.presence_if_imported` | 2 | 2 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |
| `spirits.health_warning.compliance` | 5 | 1 | 0 | 0 | 4 | 1.000 | 1.000 | 1.000 |  |
| `spirits.name_address.presence` | 5 | 4 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 |  |
| `spirits.net_contents.matches_application` | 5 | 5 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |

## Label detection

Scored against `recorded_detect_container.json` (single-frame /v1/detect-container output). Replay-mode, $0 per run.

- Items evaluated:           **6**
- Detection rate:            **1.000** (6/6)
- Container-type accuracy:   _n/a (no items eligible)_
- Brand-name match rate:     **0.667** (4/6)
- Net-contents match rate:   **0.667** (4/6)
- Skipped placeholder items: `lbl-9001, lbl-9002`
- Skipped (no recording):    `lbl-0007, lbl-0008, lbl-0009, lbl-0010, lbl-0011, lbl-0012, lbl-0013`

| Item | Detected | Type | Brand | Net contents |
|---|---|---|---|---|
| `lbl-0001` | ✓ | `bottle` | ✓ `Barka Boom` | ✓ `16 FL OZ` |
| `lbl-0002` | ✓ | `can` | ✗ `Genessee Cream Ale Ultra Rag` | ✗ `12 FL. OZ. | ALC. BY VOL. 4.8%` |
| `lbl-0003` | ✓ | `bottle` | ✓ `Calvert Brewing` | — |
| `lbl-0004` | ✓ | `bottle` | ✓ `Calvert Brewing Company` | ✓ `12 FL OZ` |
| `lbl-0005` | ✓ | `bottle` | ✗ `Calvert Brewing Company` | ✓ `1 PINT/16 FL OZ.` |
| `lbl-0006` | ✓ | `bottle` | ✓ `Calvert Brewing Company` | ✓ `12 FL OZ` |

## Per-item

Drill-down by corpus item. `Fails / Eval` is the count of rule disagreements (FP+FN+wrong-NA) out of applicable non-advisory rules for that beverage. Sorted by failure count, then `id`.

| Item | Beverage | Source | Split | Fails / Eval | Failing rules |
|---|---|---|---|---|---|
| `lbl-0001` | beer | cola_artwork | test | 0 / 7 | — |
| `lbl-0002` | beer | cola_artwork | test | 0 / 7 | — |
| `lbl-0003` | beer | cola_artwork | test | 0 / 7 | — |
| `lbl-0004` | beer | cola_artwork | test | 0 / 7 | — |
| `lbl-0005` | beer | cola_artwork | test | 0 / 7 | — |
| `lbl-0006` | beer | cola_artwork | test | 0 / 7 | — |
| `lbl-0007` | spirits | wikimedia_commons | train | 0 / 9 | — |
| `lbl-0008` | spirits | wikimedia_commons | dev | 0 / 9 | — |
| `lbl-0009` | spirits | wikimedia_commons | train | 0 / 9 | — |
| `lbl-0010` | spirits | wikimedia_commons | train | 0 / 9 | — |
| `lbl-0011` | beer | wikimedia_commons | train | 0 / 7 | — |
| `lbl-0012` | beer | wikimedia_commons | dev | 0 / 7 | — |
| `lbl-0013` | beer | wikimedia_commons | train | 0 / 7 | — |
| `lbl-9001` | wine | cola_artwork | test | 0 / 2 | — |
| `lbl-9002` | spirits | cola_artwork | test | 0 / 9 | — |

## Disagreements

*(none)*

## Coverage gaps

*(no gaps — every rule applicable to a represented beverage has at least one pass/fail data point)*
