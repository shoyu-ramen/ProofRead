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
}

export interface FieldSummary {
  value: string | number | boolean | null;
  confidence: number | null;
  bbox: BBox | null;
}

export interface ReportResponse {
  scan_id: string;
  overall: OverallStatus;
  rule_results: RuleResultDTO[];
  // Backend returns dict with arbitrary keys per extracted field.
  // Values are FieldSummary-shaped but we type loose to match the dict.
  fields_summary: Record<string, FieldSummary | unknown>;
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
