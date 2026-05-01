/**
 * TypeScript mirrors of the pydantic schemas in
 * backend/app/api/scans.py. Keep in sync when the contract changes.
 *
 * Source of truth: /Users/ross/ProofRead/backend/app/api/scans.py
 */

// v1 only supports beer; the request schema accepts beer|wine|spirits but
// the endpoint rejects non-beer until v2.
export type BeverageType = 'beer' | 'wine' | 'spirits';

// In v1 the backend issues a single "panorama" upload — the unrolled
// label captured by the cylindrical-scan flow (ARCH §7). The legacy
// front/back/side/neck values are retired; older rule_results coming
// off the multi-panel /v1/verify path are translated by the response
// builder before they reach the mobile client.
export type Surface = 'panorama';

// Scan lifecycle states from `_ScanState.status` in scans.py.
export type ScanStatus = 'uploading' | 'processing' | 'complete' | 'failed';

// `overall` from RuleEngine.overall_status — see services/pipeline.py.
export type OverallStatus = 'pass' | 'fail' | 'advisory';

// Per-rule outcome from app.rules.types.CheckOutcome.
export type RuleStatus = 'pass' | 'fail' | 'advisory';

// ----- Requests -----

export interface CreateScanRequest {
  beverage_type: BeverageType;
  container_size_ml: number;
  is_imported?: boolean;
}

export interface FlagRuleResultRequest {
  comment: string;
}

// ----- Responses -----

export interface UploadURL {
  surface: Surface;
  signed_url: string;
}

export interface CreateScanResponse {
  scan_id: string;
  upload_urls: UploadURL[];
}

export interface ScanStatusResponse {
  scan_id: string;
  status: ScanStatus;
  overall: OverallStatus | null;
}

export type BBox = [number, number, number, number]; // (x, y, w, h)

export interface RuleResultDTO {
  rule_id: string;
  rule_version: number;
  citation: string;
  status: RuleStatus;
  finding: string | null;
  expected: string | null;
  fix_suggestion: string | null;
  bbox: BBox | null;
  // Which submission this rule's evidence came from. In v1 this is
  // always `"panorama"` for both endpoints (`/v1/verify` and
  // `/v1/scans/:id/report`); the multi-panel web path can still emit
  // `"panel_N"` but the mobile client never sees those values because
  // it always uploads a single panorama. `null` when the rule isn't
  // tied to a specific extracted field.
  surface?: string | null;
  // AI-generated one-sentence plain-language explanation tailored to
  // this scan's extracted values. Populated by the backend for failed
  // and advisory rules. Optional + nullable because the /v1/scans
  // endpoint may not emit it yet — the UI must tolerate absence.
  explanation?: string | null;
}

export interface FieldSummary {
  value: string | number | boolean | null;
  confidence: number | null;
  bbox: BBox | null;
}

// Capture-quality verdict from the report row. Ordered worst→best so
// downstream code can compare with `<` semantics if needed.
export type ImageQuality = 'poor' | 'fair' | 'good';

// Reverse-lookup match against an external regulatory source (currently
// just TTB COLA Online — `source: "ttb_cola"`). Populated by the
// backend when the perceptual-hash lookup finds a confident match.
export interface ExternalMatchDTO {
  source: string; // 'ttb_cola'
  source_id: string;
  brand: string | null;
  fanciful_name: string | null;
  class_type: string | null;
  approval_date: string | null; // ISO-8601 date
  label_image_url: string | null;
  confidence: number;
  source_url: string | null;
}

export interface ReportResponse {
  scan_id: string;
  overall: OverallStatus;
  // Capture-quality verdict produced during finalize. The pill on the
  // report header is sourced from this.
  image_quality: ImageQuality;
  // Free-form notes explaining why the verdict landed where it did
  // (e.g. "blurry highlights on the front"). Surfaced under the pill
  // when present.
  image_quality_notes: string | null;
  // Identifier of the extractor that produced the result, e.g.
  // "claude-3-5-sonnet" or "ocr-fallback". Useful for diagnostics; not
  // currently surfaced in UI.
  extractor: string;
  rule_results: RuleResultDTO[];
  // Backend returns dict with arbitrary keys per extracted field.
  // Values are FieldSummary-shaped but we type loose to match the dict.
  fields_summary: Record<string, FieldSummary | unknown>;
  // Reverse-image-lookup match (TTB COLA today). Optional + nullable
  // because the /v1/scans path doesn't emit it yet — the UI conditions
  // on truthiness so absence is a no-op.
  external_match?: ExternalMatchDTO | null;
}

// ----- Known-label recognition (Decision 4 in KNOWN_LABEL_DESIGN.md) -----

/**
 * `verdict_summary.overall` from the backend's `overall_status(...)` —
 * wider than `OverallStatus` because the recognition pipeline can
 * surface every state the rule engine knows about:
 *
 *   - `pass` / `fail` / `advisory` — the standard verdicts.
 *   - `warn` — rule engine raised a non-blocking concern.
 *   - `unreadable` — cached extraction's image_quality was too poor to
 *     render a confident verdict.
 *   - `na` — zero rules applied (rare; possible at edge cases when no
 *     rule definition matched the cached fields).
 *
 * The recognition sheet's `<StatusBadge>` only renders the three
 * `OverallStatus` values cleanly; the wider cases are mapped by
 * `coerceKnownLabelOverall` before display.
 */
export type KnownLabelOverall =
  | 'pass'
  | 'fail'
  | 'advisory'
  | 'warn'
  | 'unreadable'
  | 'na';

/**
 * Lightweight rule result echoed inside `KnownLabelVerdictSummary`.
 * Distinct from `RuleResultDTO` (which is the report-level shape with
 * `rule_version`, `bbox`, `surface`, etc.) — the recognition payload
 * only carries the fields the inline overlay or downstream report could
 * surface.
 */
export interface KnownLabelRuleResult {
  rule_id: string;
  status: RuleStatus;
  citation: string;
  finding: string | null;
  explanation: string | null;
  fix_suggestion: string | null;
  // Echoed by the backend for parity with `RuleResultDTO`. The
  // recognition sheet doesn't render either today, but a future
  // expanded recognition detail view (or a pass-through into report
  // composition) will want the rule version + expected value. Optional
  // so existing fixtures in tests don't have to enumerate them.
  rule_version?: number;
  expected?: string | null;
}

/**
 * Pre-computed verdict for a recognized label. The backend re-runs the
 * rule engine against the cached extraction with the user's actual
 * `container_size_ml` + `is_imported` (Decision 4 §"Verdict summary
 * construction"), so the `overall` reflects the live inputs, not a
 * frozen verdict.
 */
export interface KnownLabelVerdictSummary {
  overall: KnownLabelOverall;
  rule_results: KnownLabelRuleResult[];
  /** field_id → { value, confidence }. Loose-typed because the field set varies per beverage_type. */
  extracted: Record<string, { value: unknown; confidence: number }>;
  image_quality: ImageQuality;
}

/**
 * Map a `KnownLabelOverall` onto the narrower `OverallStatus` the
 * existing `<StatusBadge>` palette supports.
 *
 *   - `warn` → `advisory` (same severity tier: amber palette).
 *   - `unreadable` → `fail` (user can't act on it without a reshoot).
 *   - `na` → `advisory` (informational, not blocking — same tier as
 *     warn since "no rules applied" is something the user should see
 *     but can't otherwise act on).
 */
export function coerceKnownLabelOverall(
  overall: KnownLabelOverall,
): OverallStatus {
  switch (overall) {
    case 'pass':
    case 'fail':
    case 'advisory':
      return overall;
    case 'warn':
    case 'na':
      return 'advisory';
    case 'unreadable':
      return 'fail';
  }
}

/**
 * `DetectContainerResponse.known_label` — populated when the backend
 * matches the captured frame to a previously-scanned label by brand
 * name or first-frame perceptual hash. Present only when the recognition
 * lookup succeeds; `null` on miss.
 */
export interface KnownLabelPayload {
  /** UUID of the matching `LabelCacheEntry` row — passed back to /v1/scans/from-cache. */
  entry_id: string;
  beverage_type: BeverageType;
  /** Derived from `net_contents` (or fallback to extraction value). */
  container_size_ml: number;
  is_imported: boolean;
  brand_name: string | null;
  fanciful_name: string | null;
  verdict_summary: KnownLabelVerdictSummary;
  /** Which side of the recognition lookup matched. */
  source: 'brand' | 'first_frame' | 'both';
}

// ----- /v1/scans/from-cache (Decision 5) -----

export interface FromCacheRequest {
  entry_id: string;
  beverage_type: BeverageType;
  container_size_ml: number;
  is_imported: boolean;
}

export interface FromCacheResponse {
  scan_id: string;
  status: ScanStatus;
  overall: OverallStatus;
  image_quality: ImageQuality;
}

// History list item — one row in GET /v1/scans (SPEC §v1.9). The
// backend route isn't implemented yet; this shape is what the mobile
// client expects when the endpoint lands. Marked TODO at the call site.
export interface HistoryItem {
  scan_id: string;
  label: string;
  overall: OverallStatus;
  scanned_at: string;
}

export interface HistoryResponse {
  items: HistoryItem[];
}

// RFC 7807 problem-details (per SPEC §v1.9: errors as RFC 7807).
export interface ProblemDetails {
  type?: string;
  title?: string;
  status?: number;
  detail?: string;
  instance?: string;
  // FastAPI default error shape uses `detail` only.
  [key: string]: unknown;
}

export class ApiError extends Error {
  readonly status: number;
  readonly problem: ProblemDetails | null;

  constructor(message: string, status: number, problem: ProblemDetails | null = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.problem = problem;
  }
}
