# Stress-test report — SPEC §0.5 capture conditions

Labels: 4 · Conditions: 10 · Severities: 3 · Total runs: 120

Verdict glyph: G = good · D = degraded · U = unreadable · ! = error

## Matrix

| condition | 01-pass/L | 01-pass/M | 01-pass/H | 02-warn/L | 02-warn/M | 02-warn/H | 03-fail/L | 03-fail/M | 03-fail/H | 04-unreadable/L | 04-unreadable/M | 04-unreadable/H |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glare | G | G | D | G | G | G | G | G | D | G | D | D |
| low_light | D | U | U | D | U | U | G | D | U | D | D | U |
| motion_blur | G | G | D | G | G | D | G | G | D | D | D | U |
| defocus_blur | D | D | D | D | D | D | D | D | D | D | U | U |
| rotation | G | G | G | U | U | U | G | D | D | G | G | D |
| perspective_warp | G | G | G | G | G | G | G | G | G | G | G | G |
| condensation | G | D | D | G | D | U | G | D | D | D | U | U |
| jpeg_compression_artifacts | G | G | D | G | G | G | G | G | G | G | G | G |
| low_resolution | G | G | G | G | G | G | G | G | G | G | G | G |
| colored_lighting | D | D | D | G | D | D | D | D | D | D | D | D |

## Worst breakdowns — acceptable -> unreadable

| label | condition | severity | confidence | top issue |
|---|---|---|---|---|
| 01_pass_old_tom_distillery | low_light | heavy | 0.25 | Underexposed — likely a dim environment (mean luminance 22) |
| 01_pass_old_tom_distillery | low_light | medium | 0.25 | Almost no contrast on label region (stddev 12) — camera may be aimed away from the label |
| 02_warn_stones_throw_gin | condensation | heavy | 0.25 | Soft / mildly blurry label region (sharpness 53) |
| 02_warn_stones_throw_gin | low_light | heavy | 0.25 | Underexposed — likely a dim environment (mean luminance 23) |
| 02_warn_stones_throw_gin | low_light | medium | 0.25 | Almost no contrast on label region (stddev 11) — camera may be aimed away from the label |
| 02_warn_stones_throw_gin | rotation | heavy | 0.25 | Excessive glare on the label (46% clipped; largest blob covers 44% of label) |
| 02_warn_stones_throw_gin | rotation | light | 0.25 | Excessive glare on the label (73% clipped; largest blob covers 70% of label) |
| 02_warn_stones_throw_gin | rotation | medium | 0.25 | Excessive glare on the label (55% clipped; largest blob covers 52% of label) |
| 03_fail_mountain_crest_ipa | low_light | heavy | 0.25 | Underexposed — likely a dim environment (mean luminance 22) |
| 04_unreadable_heritage_vineyards | condensation | heavy | 0.25 | Severe motion blur on the label region (sharpness 19 < 36) |
| 04_unreadable_heritage_vineyards | condensation | medium | 0.25 | Severe motion blur on the label region (sharpness 23 < 36) |
| 04_unreadable_heritage_vineyards | defocus_blur | heavy | 0.25 | Severe motion blur on the label region (sharpness 29 < 36) |
| 04_unreadable_heritage_vineyards | defocus_blur | medium | 0.25 | Severe motion blur on the label region (sharpness 31 < 36) |
| 04_unreadable_heritage_vineyards | low_light | heavy | 0.25 | Underexposed — likely a dim environment (mean luminance 20) |
| 04_unreadable_heritage_vineyards | motion_blur | heavy | 0.25 | Severe motion blur on the label region (sharpness 29 < 36) |

## Probably-too-mild conditions

- `perspective_warp` — no severity dropped any label below `good`.
- `low_resolution` — no severity dropped any label below `good`.

## Vision-extractor sample

_Skipped — ANTHROPIC_API_KEY was not set, or sampling was disabled._
