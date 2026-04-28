# ProofRead — Complete v1 → v3 Specification

A cross-platform mobile app that scans an alcoholic beverage label, classifies the beverage, and produces a TTB-compliance report — pass/fail per requirement with the offending region highlighted on the image.

---

## 0. Cross-cutting (locked across all versions)

### Tech stack

- **Mobile:** React Native 0.74+, Expo SDK 51+ (with prebuild for native modules); `react-native-vision-camera` v4 (frame processors); `expo-ml-kit` (on-device text recognition for pre-checks); `react-native-reanimated` v3 (camera overlays); `tanstack-query` (server state); `zustand` (local state); `expo-secure-store` (auth tokens).
- **Backend:** Python 3.12, FastAPI, uvicorn, asyncpg; pydantic v2; alembic; celery + redis (async OCR from v2).
- **Storage:** Postgres 16, Redis 7, S3.
- **OCR:** Google Cloud Vision (general), PaddleOCR (curved/specialized), custom TrOCR fine-tune (Health Warning, dedicated).
- **Auth:** Auth0; JWT bearer.
- **Observability:** Sentry, OpenTelemetry → Honeycomb, Posthog.
- **CI/CD:** GitHub Actions, EAS Build (mobile), AWS ECS Fargate (backend).

### Data model (final shape — tables added per version annotated)

```
users (v1)               id, email, role, company_id, created_at
companies (v1)           id, name, ttb_basic_permit, billing_plan
scans (v1)               id, user_id, beverage_type, container_size_ml,
                         container_size_source, status, created_at, completed_at
scan_images (v1)         id, scan_id, surface[front|back|side|neck], s3_key, w, h
ocr_results (v1)         id, scan_image_id, provider, raw_json, text, confidence, ms
extracted_fields (v1)    id, scan_id, field_id, value, bbox, confidence, source_image_id
rules (v1)               id, version, beverage_types[], citation, definition_yaml,
                         effective_from, effective_to
reports (v1)             id, scan_id, overall, rule_version, rendered_pdf_s3_key (v2)
rule_results (v1)        id, report_id, rule_id, status, finding, expected,
                         citation, fix_suggestion, bbox, image_id
calibrations (v2)        id, scan_id, source[upc|reference_card|ar], mm_per_pixel, confidence
cola_records (v2)        id, ttb_id, brand_name, fanciful_name, class_type,
                         image_keys, raw_metadata, perceptual_hash, fetched_at
skus (v2)                upc, brand_name, container_height_mm, container_diameter_mm,
                         container_volume_ml
api_keys (v3)            id, company_id, key_hash, scopes[], created_at, revoked_at
audit_logs (v3)          id, actor_id, action, target, before, after, at
webhooks (v3)            id, company_id, url, secret_hash, scopes[]
```

### Security & ops baseline (v1 onward)

- TLS + HSTS; certificate pinning on mobile.
- Private S3 buckets; signed-URL upload from mobile.
- Image retention default 90 days; configurable per company up to 7 years.
- Rate limit: 60 scans/user/hour; 1,000/company/day on starter plan.
- Postgres PITR 30-day; nightly cold snapshots.

### Observability baseline

- Trace ID propagated API → OCR worker → rule engine.
- Metric: scan→report median, p95, p99.
- Metric: rule pass rate per `rule_id@version` (regression tracking).
- Metric: user-flagged false-fail rate per rule.

---

## 0.5 Robustness — performance in extreme conditions

The app must work where alcohol is actually encountered: bars, restaurants, festivals, outdoor events, breweries, warehouses, retail floors. The capture environment is rarely controlled. This section catalogs the conditions the system must handle, the mitigation for each, and the per-version scope.

### Condition catalog

**Lighting**
- Direct sunlight (festival, beach): saturation, harsh shadows
- Dim bar / restaurant / cellar (<50 lux): sensor noise, motion blur
- Mixed / colored lighting (neon, LED bar lights, candlelight): white-balance shifts
- Backlight (bottle against window): silhouette dominates frame, label region underexposed
- Strobe / disco lighting: frame-to-frame inconsistency

**Surface**
- Wet / condensed bottle (just out of ice): light scatter, droplet occlusion
- Foggy chilled bottle in warm air
- Frosted glass containers
- Smudged or dirty label
- Faded label (old stock, sun damage)
- Torn / peeling label
- Re-labeled with overlay stickers (private-label, retailer rebrands)

**Container**
- Curved bottles, multiple radii (750 mL wine, 12 oz beer, 1.75 L spirits)
- Cans (tighter cylindrical curvature; smaller text)
- Embossed labels (no ink, only relief)
- Foil-wrapped (champagne, certain spirits)
- Wax-dipped (e.g., Maker's Mark)
- Holographic / iridescent / metallic labels — appearance changes with angle
- Clear glass with label visible through the back, ghosted
- Dark glass with poor contrast
- Crowlers, growlers, kegs

**Device & user**
- Shaky hands (intoxicated user, walking, festival crowd)
- Old/low-end phones with poor camera optics
- Smudged/scratched camera lens
- Thermal throttling (hot device, sun exposure)
- Low battery (camera frame rate throttled)

**Network**
- No network (rural, basement bar, warehouse, festival cell saturation)
- Slow / unstable network
- Captive-portal Wi-Fi (hotel, conference)

**Adversarial / out-of-distribution**
- Counterfeit labels (out of scope; explicit non-goal)
- Foreign-language-only labels (handled in v3)
- Custom / private-label products with no COLA history
- Intentionally obscured labels (price stickers, security tags)

### Capture-time mitigations (mobile)

| Mitigation | Implementation | Version |
|---|---|---|
| HDR multi-exposure capture | Vision Camera frame processor + native HDR API | v1 |
| Live exposure histogram check | Reject under/over-exposed frames pre-capture | v1 |
| Live glare-region detection | Count saturated pixels per frame; warn if >20% of label region | v1 |
| Live motion detection | Gyroscope + frame-to-frame variance; defer capture during high motion | v1 |
| Low-light mode | Longer exposure + stabilization; warn if SNR estimate too low | v1 |
| Backlight detection | If frame's highest-luminance region is outside the detected label area, prompt user to reposition | v1 |
| Wet-bottle detection | Detect droplet pattern (small high-contrast blobs); prompt "Wipe and retry" | v2 |
| Lens-smudge detection | Persistent low-contrast across captures triggers "Clean lens" hint | v2 |
| Multi-angle capture | Capture 3 frames at slight tilt; pipeline picks best regions per area to recover behind specular highlights | v2 |
| Curvature-aware framing | On-device segmentation estimates bottle silhouette; overlay adapts to container | v2 |
| Multi-image panorama | 3-shot stitched panorama for tightly-curved cans | v3 |
| Foil / embossed / holographic detection | Classify label material; warn confidence will be capped advisory | v2 |

### Pipeline-time mitigations (backend)

- **White-balance + tone normalization** before OCR (v1).
- **Cylindrical unwrap** of detected bottle silhouette (v2).
- **Tiered OCR fallback:** Google Vision → PaddleOCR with curved-text mode → TrOCR fine-tune. If primary's confidence < threshold, escalate to next tier (v1).
- **Health Warning second pass:** dedicated TrOCR fine-tune on the cropped warning region runs *every* time, regardless of primary success — redundancy by design (v1).
- **Confidence-aware degradation:** if any extraction step's confidence < threshold, the affected rule downgrades from pass/fail to advisory rather than producing a wrong finding (v1). Surfaced as "couldn't verify with confidence — rescan recommended."
- **Image-quality classifier** trained on labeled good/bad captures; low-quality score escalates to "rescan required" before running rules (v2).
- **Foil / embossed regions** masked out of OCR input; rules over those regions produce advisory results with explanation (v2).
- **Multi-frame fusion** in cylindrical-unwrap stage: combine information from the 3 multi-angle captures (v2).

### Offline & degraded-network operation

| Capability | v1 | v2 | v3 |
|---|---|---|---|
| Capture works fully offline | ✓ | ✓ | ✓ |
| Queue scans for upload when network returns | ✓ | ✓ | ✓ |
| Last 50 reports cached on device for offline viewing | ✓ | ✓ | ✓ |
| On-device presence-only quick-check (pre-cloud feedback) | — | ✓ | ✓ |
| Full offline OCR + rule engine (slower; no COLA cross-ref) | — | — | ✓ |
| Captive-portal detection + clear messaging | ✓ | ✓ | ✓ |
| Background upload with exponential backoff | ✓ | ✓ | ✓ |

The offline rule engine (v3) ships compiled rule definitions in the app bundle and runs a stripped-down OCR (Apple Vision / ML Kit). Designed for warehouse and basement-bar environments with no connectivity.

### Device support matrix

- **iOS:** 15+ (covers ~95% of active iPhones). Type-size accuracy benchmarked on iPhone XS and newer; older devices report a downgraded calibration confidence. Detect device class at first launch; warn if older than minimum.
- **Android:** 12+ (v2 launch). Minimum 8 MP rear camera. OEM camera quality varies wildly — maintain a tested-device list in support docs and surface "limited support" banner on untested devices.
- All platforms: explicit calibration-confidence value in every report, so users can see when their *device* is the limiting factor rather than the *label*.

### Thermal & battery

- Vision Camera frame-processor frequency adapts to `ProcessInfo.thermalState` (iOS) / equivalent on Android.
- HDR + multi-frame fusion automatically disabled when thermal state is `serious` or `critical`; user shown an info banner.
- Capture flow blocks entirely above `critical` thermal state; user prompted to let device cool.
- Below 15% battery, multi-frame and HDR disabled; capture still works in single-frame mode.

### Adversarial inputs

- **Counterfeit detection** is explicitly out of scope. The product verifies *whether a label complies with TTB regulations*, not whether the bottle is genuine. EULA and onboarding state this clearly.
- **Foreign-language-only labels** (no English): v1/v2 detect and refuse with "Submit a label with English text or supplemental panel." v3 adds Spanish and French native support.
- **Obscured labels** (price stickers, security tags): detected as occlusion; user prompted to remove or reposition.
- **Image upload of a screenshot, photo of a photo, or rendered mockup:** detected via metadata + screen-pattern classifier; either accepted with an "image source: digital" tag in the report (v2) or refused (configurable per company plan).

### Acceptance criteria — extreme conditions (samples)

| Condition | Target | Version |
|---|---|---|
| Direct sunlight (>50,000 lux) | Health Warning exact-text precision ≥95% on 100-label test set | v1 |
| Dim bar (10–50 lux) | Capture completes; precision ≥90% | v1 |
| Wet bottle (visible condensation) | Pre-check prompts "Wipe and retry" OR scan succeeds with no precision loss | v2 |
| Backlight (window behind bottle) | Pre-check prompts reposition OR HDR captures sufficient label data | v1 |
| Curved 750 mL wine bottle | Type-size check accuracy ≥95% post-unwrap | v2 |
| 12 oz aluminum can | Type-size check accuracy ≥95% post-unwrap | v2 |
| Foil-wrapped neck (champagne) | Foil detected; rules over foil region marked advisory; non-foil regions verified normally | v2 |
| Embossed-only label | Detected; whole scan returned as advisory with explanation | v2 |
| No network during scan | Scan queued; user sees "Pending upload" badge; auto-retries with exponential backoff | v1 |
| Shaky capture (>5°/s rotation) | Pre-check rejects; user retries before submission | v1 |
| Thermal `serious` state | Capture continues with HDR disabled; banner shown | v1 |
| Holographic label | Detected; advisory-only mode with explanation | v3 |
| Spanish-only import label | OCR + rule set works end-to-end | v3 |
| 1 year-old mid-tier Android device | Type-size check completes with calibration confidence reported; rules above confidence threshold pass/fail normally | v2 |

### Design principle — fail honestly

> When the system cannot confidently verify a rule, it must say so — not guess.

A wrong "pass" is the worst outcome (user submits non-compliant artwork to TTB on the app's say-so). A wrong "fail" is the second worst (user wastes time chasing a phantom problem). An honest "advisory: couldn't verify" is acceptable.

The **confidence-aware degradation path** is the central robustness mechanism: every check has an explicit confidence threshold below which it downgrades from required to advisory, with the reason surfaced to the user. This is what makes the product trustworthy in extreme conditions — not the absence of failure, but the integrity of the failure mode.

---

## v1 — MVP (Beer-only, iOS, ~3 months)

### v1.1 Goals

Validate that automated label compliance is **trustworthy enough** that a brewer would run it before submitting a COLA. Cover beer end-to-end. Establish the rule-engine architecture and OCR pipeline that v2/v3 extend.

### v1.2 Non-goals

Wine, spirits, Android, type-size enforcement (advisory only), COLA cross-reference, multi-language, PDF export, bulk upload.

### v1.3 Success metrics

- 100 brewers onboarded
- ≥10 scans/active brewer/week
- Health Warning exact-text check: precision ≥98%, recall ≥99% on held-out 500-label test set
- p95 scan→report ≤ 25 s
- NPS ≥ 40 from active users

### v1.4 Personas

- **Brewer Quality Manager** — runs final QA on label artwork before COLA submission. Wants pass/fail with citations.
- **Compliance Consultant (Beer)** — reviews labels for many small breweries. Wants a fast first-pass.

### v1.5 Functional requirements

| ID | Requirement |
|---|---|
| F1.1 | Sign up / sign in via email+password or Google |
| F1.2 | Capture front and back of a beer label using guided camera |
| F1.3 | App pre-checks each capture for focus, glare, label coverage; rejects with reshoot prompt |
| F1.4 | Manual container-size entry in mL (default options 355, 473, 500, 650) |
| F1.5 | Upload images and metadata; user sees progress |
| F1.6 | Backend produces a report within 30 s p95 |
| F1.7 | Report screen with per-rule pass/fail/advisory and CFR citations |
| F1.8 | Tap a rule result to see the bounding box on the original image |
| F1.9 | Rescan retains scan_id for iteration history |
| F1.10 | Flag a result as "incorrect" with free-text comment |
| F1.11 | Scan history list, most recent first |
| F1.12 | Admin (internal) can view raw OCR + intermediate state for any scan |

### v1.6 Primary flow

1. **Home** → `Scan new label`
2. **Beverage type picker** — only Beer enabled
3. **Container size picker** — radio + custom mL
4. **Camera (front)** — overlay shows bottle outline; live pre-check; capture validates on-device; reshoot if bad
5. **Camera (back)** — same
6. **Review** — thumbnail strip; retake either; tap `Analyze`
7. **Processing** — progress, cancellable
8. **Report** — summary + rule list + bounding-box drawer
9. **Actions** — Rescan, Share findings (text only in v1), Flag result

### v1.7 Screens

| Screen | Components |
|---|---|
| Splash | Logo, 1s timeout |
| Sign in / up | Email + password + Google button (Auth0 hosted) |
| Home | Big CTA + recent 3 scans |
| Beverage type picker | Segmented control (Beer enabled) |
| Container size picker | Radio + custom mL input |
| Camera | Live frame, overlay, pre-check indicator, capture button, surface label |
| Capture review | Thumbnails, retake buttons, Analyze CTA |
| Processing | Progress bar, cancel |
| Report | Header (pass/fail icon), per-rule list, image overlay drawer |
| Rule detail | Bounding box, expected vs found, citation, fix suggestion |
| History | List |
| Settings | Account, sign out, image retention |

### v1.8 Backend architecture

Single deploy unit: `api` (FastAPI) with synchronous in-process OCR + rule engine. Acceptable at MVP scale; async upgrade in v2. Single region (us-east-1). Externals: Postgres + Redis (managed), S3, Google Vision, Auth0, Sentry, Honeycomb, Posthog.

### v1.9 API surface

```
POST  /v1/auth/exchange                # Auth0 token → app session
GET   /v1/me

POST  /v1/scans                        # {beverage_type, container_size_ml}
                                       # → {scan_id, upload_urls: [{surface, signed_url}]}
PUT   {signed_url}                     # direct to S3
POST  /v1/scans/:id/finalize           # uploads complete; trigger processing
GET   /v1/scans/:id                    # status + report (when ready)
GET   /v1/scans/:id/report             # full report, polling-friendly
GET   /v1/scans                        # paginated history
POST  /v1/scans/:id/rule-results/:rid/flag   # {comment}

GET   /v1/admin/scans/:id/raw          # full OCR + intermediates (admin role)
```

JSON; errors as RFC 7807 problem-details; auth via Bearer JWT.

### v1.10 OCR + rule pipeline

```
1. /scans/:id/finalize triggers process_scan(scan_id)
2. For each scan_image:
   a. Download from S3
   b. Light preprocess: orient, crop label region (rectangle detection)
   c. Google Vision OCR → store ocr_results
3. Field extraction (beer-specific):
   - Brand name: largest text in upper region of front
   - Class/type: word match against beer-class taxonomy (ale/lager/stout/ipa/...)
   - ABV: regex (\d+(?:\.\d+)?)\s*%\s*(?:abv|alc(?:\.|ohol)?)
   - Net contents: regex with units (\d+)\s*(ml|fl ?oz|fluid ounces)
   - Name/address: TTB-pattern "Brewed and bottled by"/"Brewed by"/"Imported by"
   - Health Warning: anchor on "GOVERNMENT WARNING" then capture ~280 chars
4. Rule engine:
   - Load active beer rule set
   - Run each rule's checks against extracted_fields
   - Each rule produces a rule_result (status + finding + bbox)
5. Report assembly:
   - overall = fail if any required rule fails;
              advisory if only advisory rules fail; pass otherwise
6. Mobile polls /scans/:id until status == complete
```

### v1.11 Rules (v1, beer-only)

| Rule ID | Citation | Description | Checks |
|---|---|---|---|
| `beer.brand_name.presence` | 27 CFR 7.22 | Brand name present | presence |
| `beer.class_type.presence` | 27 CFR 7.24 | Class/type designation | presence + class_taxonomy |
| `beer.alcohol_content.format` | 27 CFR 7.71 | If declared, ABV in valid format | regex |
| `beer.net_contents.presence` | 27 CFR 7.27 | Net contents present | presence + format |
| `beer.name_address.presence` | 27 CFR 7.25 | Bottler name + address | presence |
| `beer.country_of_origin.presence_if_imported` | 27 CFR 7.25(b) | Country if imported | conditional presence |
| `beer.health_warning.exact_text` | 27 CFR 16.21 | Health Warning exact text | exact_text (edit_distance=0) |
| `beer.health_warning.size` | 27 CFR 16.22 | Health Warning min size | **advisory** in v1 |

8 rules total. Each rule_id versioned (`beer.health_warning.exact_text@1`) so historical reports remain reproducible across rule changes.

### v1.12 Acceptance criteria (samples)

**F1.6 — latency:** 100 sequential scans on 1 MB images, p95 ≤ 30 s on production hardware. Cold (no cache) ≤ 30 s; warm (image hash hit) ≤ 5 s.

**Health Warning rule:**
- Exact statutory text → passes
- One substituted character → fails with `expected` showing canonical, `finding` showing actual
- Missing entirely → fails with `presence` failure mode

**Camera pre-check:**
- Laplacian variance < threshold → "Image is blurry"
- Glare region > 40% → "Reduce glare"
- Label area < 30% of frame → "Move closer"

### v1.13 Risks

- **Health Warning OCR precision** → mitigation: dedicated TrOCR fine-tune on cropped warning region; second-pass with Google Vision DOCUMENT_TEXT_DETECTION; 500-label hand-labeled test set in CI.
- **Field extraction brittleness on creative labels** → user feedback loop on flagged results; weekly miss review.
- **Backend latency under load** → rate limit + async upgrade in v2.

---

## v2 — Production (Wine + Spirits, Calibration, COLA, Android, ~3 months after v1)

### v2.1 Goals

All three TTB beverage types; replace type-size advisory with measured pass/fail; COLA cross-reference; Android; PDF export.

### v2.2 Non-goals

Auto beverage-type detection, multi-language, webhook integrations, admin UI for rules.

### v2.3 Success metrics

- 500 paid customers
- COLA short-circuit hits ≥30% of scans
- Type-size measurement: pass/fail accuracy ≥95% on hand-labeled test set
- p95 scan→report ≤ 20 s (faster despite more checks, due to async pipeline)
- Wine + spirits ≥ 40% of total scans within 60 days of launch

### v2.4 New functional requirements

| ID | Requirement |
|---|---|
| F2.1 | Pick wine or distilled spirits as beverage type |
| F2.2 | UPC scan on container-size step; on hit, container dimensions auto-fill |
| F2.3 | Calibration card flow (credit-card AR overlay) when no UPC match |
| F2.4 | Backend computes mm-per-pixel and runs type-size checks for real |
| F2.5 | Backend looks up scanned label against COLA mirror (perceptual hash + brand-name match) |
| F2.6 | Export report as PDF (signed S3 URL, 24 h expiry) |
| F2.7 | Android parity with iOS |
| F2.8 | OCR + rule engine async via Celery; mobile uses websocket for status |
| F2.9 | Wine-specific rules (sulfites, vintage, appellation, varietal, standard fills) |
| F2.10 | Spirits-specific rules (proof, age statement, class taxonomy, neutral-spirits declaration) |

### v2.5 New screens

- **UPC scan** — barcode camera; on hit shows matched SKU; on miss falls back to manual entry or calibration card
- **Calibration card** — AR overlay with credit-card-sized rectangle; user aligns flat against bottle; capture
- **PDF preview** — rendered PDF + share sheet
- **COLA match badge** — banner on report screen with TTB ID + link to TTB Public Registry

### v2.6 Architecture changes

**New services:**
- `ocr_worker` becomes a separate Celery worker (autoscaled)
- `cola_sync` — weekly job that scrapes/snapshots TTB COLA Public Registry into `cola_records`, indexed by perceptual hash
- `report_renderer` — PDF service (WeasyPrint)

**New external dependencies:**
- UPC database (Open Food Facts + commercial UPC API fallback)
- TTB COLA Public Registry (scraped, throttled, ToS-respecting)

### v2.7 New API surface

```
POST  /v2/scans                          # adds container_size_source, calibration payload
POST  /v2/upc/lookup                     # {upc} → {sku?, dimensions?}
POST  /v2/scans/:id/calibration          # {source, mm_per_pixel, confidence}
GET   /v2/scans/:id/cola-match           # {match: bool, ttb_id?, score}
GET   /v2/scans/:id/report.pdf           # signed redirect to S3
WS    /v2/scans/:id/stream               # live status (replaces polling)
```

### v2.8 Calibration math

Given reference object width `W_mm` (credit card = 85.6 mm) detected in image with bbox width `W_px`, or SKU container diameter `D_mm` and detected bottle silhouette diameter `D_px`:

```
mm_per_pixel = W_mm / W_px
```

Bottles are curved → mm-per-pixel varies across the label. Mitigation:
- Detect bottle silhouette → cylindrical unwrap (OpenCV `remap` with cylindrical model)
- Measure mm-per-pixel on the unwrapped (flat) image
- Track per-region confidence: center most accurate, edges less so
- Type-size check measures bbox height on unwrapped image, converts to mm, compares to threshold
- Below confidence threshold (0.8) → check downgrades to advisory

### v2.9 New rules (~22 added; total ~30)

**Wine**
- `wine.brand_name.presence`
- `wine.class_type.presence` (varietal / generic / semi-generic / geographic taxonomy)
- `wine.appellation.presence_if_claimed`
- `wine.vintage.format_if_claimed`
- `wine.alcohol_content.presence` (mandatory above 7% ABV) + tolerance (1.5% for ≤14% ABV, 1.0% for >14%)
- `wine.net_contents.standard_fill` (50/100/187/375/500/750/1000/1500/3000 mL)
- `wine.bottler_address.presence`
- `wine.country_of_origin.presence_if_imported`
- `wine.sulfite_declaration.presence` (mandatory if SO₂ ≥10 ppm; defaults to required unless producer affirmatively declares exempt)
- `wine.health_warning.exact_text`
- `wine.health_warning.size`
- `wine.fdc_yellow_5.declaration_if_present`

**Distilled spirits**
- `spirits.brand_name.presence`
- `spirits.class_type.presence` + class taxonomy (whisky subtypes, vodka, gin, rum, tequila — each with its own rules)
- `spirits.alcohol_content.presence` (mandatory) + format (% ABV)
- `spirits.proof.presence_optional` + format
- `spirits.net_contents.standard_fill` (50/100/200/375/750/1000/1750 mL)
- `spirits.name_address.presence`
- `spirits.country_of_origin.presence_if_imported`
- `spirits.age_statement.format_if_claimed` (rules vary by class — e.g., straight whiskey requires age if <4y)
- `spirits.neutral_spirits_declaration.presence_if_applicable`
- `spirits.health_warning.exact_text`
- `spirits.health_warning.size`

**Type-size promotion:** All `*.health_warning.size` rules upgraded from advisory to pass/fail in v2 — gated on calibration confidence ≥0.8; below threshold falls back to advisory with a notice.

### v2.10 Acceptance criteria (samples)

**Calibration:** Credit-card capture (card flat in frame) → mm-per-pixel error ≤3% on 100-capture test set. UPC-based calibration → error ≤2%.

**COLA match:** Known approved label image from TTB Registry → match in `cola_records` with score ≥0.9. Novel label → no match, no false positives over 1,000-label control set.

**Wine sulfite rule:** Label without "CONTAINS SULFITES" → fails (we cannot verify exemption). Label with the declaration in any TTB-acceptable form → passes.

### v2.11 Risks

- **COLA scraping legal risk** — respect robots.txt + ToS; throttle politely; engage TTB about a feed if usage scales.
- **Calibration UX friction** — prefer UPC; calibration card is fallback; track funnel metrics on calibration completion.
- **Rule explosion** — manageable at 30; admin UI in v3 once we project past 50.

---

## v3 — Scale (Auto-detect, Bulk, Integrations, Admin, Multi-language, ~6 months after v2)

### v3.1 Goals

Reduce per-scan friction (auto type detection); serve high-volume customers (bulk + webhooks); enable compliance ops to maintain rules without engineering; cover non-English supplemental labels; multi-region for performance + data residency.

### v3.2 Non-goals

Counterfeit detection, TTB submission automation, consumer app.

### v3.3 Success metrics

- 5,000 paid customers
- ≥50% of scans via bulk upload (power-user adoption)
- ≥10 design-tool integrations live
- Rule changes deployable by compliance ops in <1 hour, no engineering involvement
- Multi-language: English + Spanish + French at launch

### v3.4 New functional requirements

| ID | Requirement |
|---|---|
| F3.1 | Auto-detect beverage type from front-label imagery; user can override |
| F3.2 | Bulk upload of label image folder (mobile + new web app) |
| F3.3 | Web app (Next.js) for bulk upload, scan list, report viewer, admin |
| F3.4 | Admin (compliance ops role) edits rules via YAML editor with preview against test set |
| F3.5 | Companies get scoped API keys; can POST images via public API + receive webhook results |
| F3.6 | Esko / Adobe Illustrator plugin submits artwork mid-design and surfaces findings inline |
| F3.7 | OCR pipeline supports Spanish + French; rule engine supports localized class/type taxonomies |
| F3.8 | Multi-region deploy (us-east-1, us-west-2, eu-west-1) with regional data residency |
| F3.9 | State-level rules surface as advisory (CA Prop 65, NY ABV labeling, TX permit display) |
| F3.10 | Audit log of all rule changes with diff and rollback |

### v3.5 New surfaces

- **Web app (Next.js)** — auth, bulk uploader, scan list, report viewer, admin
- **Esko / Adobe plugin** — submits from designer's tool; findings as panel
- **Public API** — OpenAPI-documented at `/v3/docs`

### v3.6 Architecture changes

**New services:**
- `webhook_dispatcher` — fan-out scan-complete events to customer webhooks (HMAC signed)
- `rule_admin` — CRUD over rules; runs preview against historical-scan corpus before promoting a rule version
- `bevtype_classifier` — image classification model (ViT fine-tune) for beverage type
- `i18n_ocr` — OCR pipeline variant with language-aware preprocessing

**Multi-region:**
- Postgres per-region primary with cross-region read replicas; data tagged with `region`; scans pinned to user's home region
- Object storage: per-region S3 buckets
- Routing: latency-based DNS

### v3.7 New API surface

```
POST    /v3/bulk-scans                   # {scans: [{images, metadata}]}
GET     /v3/bulk-scans/:id               # batch status

POST    /v3/webhooks                     # {url, secret}
GET     /v3/webhooks
DELETE  /v3/webhooks/:id

POST    /v3/api-keys                     # company admin only
DELETE  /v3/api-keys/:id

# Public API (auth: API key, scoped)
POST    /v3/public/scans
GET     /v3/public/scans/:id
POST    /v3/public/scans/:id/webhook     # re-send the webhook (testing)

# Admin (compliance ops role)
GET     /v3/admin/rules
POST    /v3/admin/rules                  # {yaml, effective_from}
POST    /v3/admin/rules/:id/preview      # {test_set_id} → diff vs current
POST    /v3/admin/rules/:id/promote      # makes rule version live
GET     /v3/admin/rules/:id/history
POST    /v3/admin/rules/:id/rollback     # to a prior version
```

### v3.8 New rules (~50 added; total ~80)

- State-level (advisory) — CA Prop 65 cancer/repro warning, NY ABV format on cans, TX bottler-permit display, etc.
- Multi-language Health Warning equivalents on bilingual supplemental panels
- Distilled spirits subtype "must-bill" rules — bourbon (≥51% corn, new charred oak, ≤80% ABV distilled, ≤62.5% ABV barreled, no additives), rye, Tennessee whiskey, etc., verified against declared class/type
- Wine appellation correctness — declared AVA must exist in TTB's AVA registry; cross-check varietal labeling rules (75% rule, "Meritage" etc.)

### v3.9 Acceptance criteria (samples)

**F3.1 — auto-detect:** Top-1 classification accuracy ≥95% on held-out 1,000-label balanced set. Confidence <0.8 → app prompts user to confirm rather than auto-routing.

**F3.4 — rule admin:** Rule edit preview runs against historical scan corpus and shows pass-rate delta within 60 s. New rule cannot be promoted without preview run within last 24 h. Promotion reversible via one-click rollback.

**F3.8 — multi-region:** EU user sees data residency in eu-west-1 (verified via response header `x-region`). p95 latency in EU within 20% of US p95.

### v3.10 Risks

- **Multi-region complexity** — defer until a real EU customer commits; don't build speculatively
- **Rule admin safety** — preview-required gate, audit log, fast rollback
- **OCR quality on non-English** — language-specific fine-tunes; ship Spanish first (largest US import volume)

---

## Appendix A — Rule definition schema

```yaml
- id: string                          # e.g., wine.health_warning.exact_text
  version: int                        # bumped on any change
  beverage_types: [beer, wine, spirits]
  citation: string                    # e.g., "27 CFR 16.21"
  description: string
  applies_if: expression?             # e.g., "abv_pct > 0.5"
  exempt_if: expression?              # e.g., "container_size_ml < 50"
  severity: required | advisory       # advisory rules don't fail the scan
  state_jurisdiction: string?         # e.g., "CA" — only evaluated if scan's state == this
  fix_suggestion: string?             # surfaced in report
  checks:
    - type: presence | exact_text | regex | format | placement
            | min_height_mm | contrast | taxonomy
            | conditional_presence | cross_reference
      params: {...}                   # type-specific
```

## Appendix B — Phase summary

| Phase | Duration | Beverages | Platforms | Rules | Calibration | COLA | Bulk | Admin | Public API |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 3 mo | Beer | iOS | 8 | — | — | — | — | — |
| v2 | 3 mo | + Wine, Spirits | + Android | ~30 | ✓ | ✓ | — | — | — |
| v3 | 6 mo | All | + Web, plugins | ~80 | ✓ | ✓ | ✓ | ✓ | ✓ |

## Appendix C — Team shape

- **v1 (~7):** 2 mobile (RN/iOS), 2 backend (Python), 1 ML (OCR/CV), 1 design, 1 PM, 1 compliance SME (part-time)
- **v2 (+2):** + 1 mobile (Android lead), + 1 ML (calibration / COLA matching)
- **v3 (+3):** + 1 web frontend, + 1 platform/devops (multi-region), compliance SME → full-time
