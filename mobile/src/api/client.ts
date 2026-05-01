/**
 * Typed API client for the ProofRead backend.
 *
 * Each method maps to one endpoint in backend/app/api/scans.py.
 *
 * Auth: bearer JWT (SPEC §v1.9). Today the backend's get_current_user
 * returns a fixed test user, so the token is effectively ignored — but
 * the client still attaches Authorization when configured so the wiring
 * works the moment the backend enforces it.
 */

import Constants from 'expo-constants';
import { getAuthToken } from '@src/state/auth';
import {
  ApiError,
  CreateScanRequest,
  CreateScanResponse,
  FlagRuleResultRequest,
  FromCacheRequest,
  FromCacheResponse,
  HistoryResponse,
  KnownLabelPayload,
  ProblemDetails,
  ReportResponse,
  ScanStatusResponse,
} from './types';

const DEFAULT_BASE_URL = 'http://localhost:8000';

function resolveBaseUrl(): string {
  const fromEnv = process.env.EXPO_PUBLIC_API_BASE_URL;
  if (typeof fromEnv === 'string' && fromEnv.length > 0) {
    return fromEnv.replace(/\/$/, '');
  }
  const fromExtra = Constants.expoConfig?.extra?.apiBaseUrl;
  if (typeof fromExtra === 'string' && fromExtra.length > 0) {
    return fromExtra.replace(/\/$/, '');
  }
  return DEFAULT_BASE_URL;
}

export interface ApiClientConfig {
  baseUrl?: string;
  // Function so the client can pull a fresh token per request without
  // tight-coupling to the auth store.
  getToken?: () => string | null | Promise<string | null>;
}

export class ApiClient {
  private readonly baseUrl: string;
  private readonly getToken: () => string | null | Promise<string | null>;

  constructor(config: ApiClientConfig = {}) {
    this.baseUrl = (config.baseUrl ?? resolveBaseUrl()).replace(/\/$/, '');
    this.getToken = config.getToken ?? (() => null);
  }

  private async authHeaders(): Promise<Record<string, string>> {
    const token = await this.getToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  private async request<T>(
    path: string,
    init: RequestInit & { parseJson?: boolean } = {}
  ): Promise<T> {
    const { parseJson = true, headers, ...rest } = init;
    const auth = await this.authHeaders();
    const res = await fetch(`${this.baseUrl}${path}`, {
      ...rest,
      headers: {
        Accept: 'application/json',
        ...auth,
        ...(headers ?? {}),
      },
    });
    if (!res.ok) {
      let problem: ProblemDetails | null = null;
      try {
        problem = (await res.json()) as ProblemDetails;
      } catch {
        // body wasn't JSON; fall back to status text.
      }
      // The thrown message is the raw server detail — kept here so
      // logs / Sentry breadcrumbs carry the diagnostic string. Call
      // sites must NOT show this to end users; use describeError()
      // from ./errors.ts to render plain-language copy instead.
      const message =
        (problem && (problem.detail as string | undefined)) ||
        problem?.title ||
        res.statusText ||
        `HTTP ${res.status}`;
      throw new ApiError(message, res.status, problem);
    }
    if (!parseJson || res.status === 204) {
      return undefined as T;
    }
    return (await res.json()) as T;
  }

  // POST /v1/scans
  createScan(req: CreateScanRequest): Promise<CreateScanResponse> {
    return this.request<CreateScanResponse>('/v1/scans', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  }

  /**
   * Upload an image to a signed URL returned by createScan().
   *
   * The signed_url is opaque to the client — in production this points
   * at S3 with no auth header, today it points at the backend's own
   * PUT /v1/scans/:id/upload/:surface route. Either way: do not attach
   * the bearer token, and do not pass through fetch's CORS-affecting
   * defaults.
   */
  async uploadImage(
    signedUrl: string,
    body: ArrayBuffer | Blob | Uint8Array,
    contentType: string = 'image/jpeg'
  ): Promise<void> {
    const res = await fetch(signedUrl, {
      method: 'PUT',
      headers: { 'Content-Type': contentType },
      // React Native's fetch accepts these body types. Cast to BodyInit
      // because RN's lib types are narrower than DOM's.
      body: body as unknown as BodyInit,
    });
    if (!res.ok) {
      throw new ApiError(`upload failed: ${res.statusText}`, res.status, null);
    }
  }

  // POST /v1/scans/:id/finalize
  //
  // `firstFrameSignatureHex` (when present) is forwarded as a multipart
  // form field so enrichment can stamp it on the cache entry — feeds
  // future first-frame recognition lookups (KNOWN_LABEL_DESIGN.md
  // Decision 6).
  finalizeScan(
    scanId: string,
    opts: { firstFrameSignatureHex?: string | null } = {},
  ): Promise<ScanStatusResponse> {
    const hex = opts.firstFrameSignatureHex;
    if (hex) {
      const form = new FormData();
      form.append('first_frame_signature_hex', hex);
      return this.request<ScanStatusResponse>(`/v1/scans/${scanId}/finalize`, {
        method: 'POST',
        // Do NOT set Content-Type: the runtime fills the multipart
        // boundary. Same pattern as detectContainer().
        body: form as unknown as BodyInit,
      });
    }
    return this.request<ScanStatusResponse>(`/v1/scans/${scanId}/finalize`, {
      method: 'POST',
    });
  }

  /**
   * POST /v1/scans/from-cache (KNOWN_LABEL_DESIGN.md Decision 5).
   *
   * Called when the user accepts a known-label recognition from the
   * confirming{detected} sheet. The backend re-runs the rule engine
   * against the cached extraction with the user's actual
   * container_size_ml + is_imported and returns a fully-formed scan_id
   * the user can navigate straight to /report on.
   */
  createScanFromCache(req: FromCacheRequest): Promise<FromCacheResponse> {
    return this.request<FromCacheResponse>('/v1/scans/from-cache', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  }

  // GET /v1/scans/:id
  getScan(scanId: string): Promise<ScanStatusResponse> {
    return this.request<ScanStatusResponse>(`/v1/scans/${scanId}`, {
      method: 'GET',
    });
  }

  // GET /v1/scans/:id/report
  getReport(scanId: string): Promise<ReportResponse> {
    return this.request<ReportResponse>(`/v1/scans/${scanId}/report`, {
      method: 'GET',
    });
  }

  /**
   * GET /v1/scans (history).
   *
   * Returns up to the most recent 50 scans for the current user,
   * ordered by `Scan.created_at DESC` server-side. Each item carries
   * `{ scan_id, label, overall, scanned_at }` — see scans.py
   * `HistoryResponse`. The mobile history screen + home rail consume
   * this directly; no client-side sorting required.
   */
  getHistory(): Promise<HistoryResponse> {
    return this.request<HistoryResponse>('/v1/scans', { method: 'GET' });
  }

  /**
   * Alias for `getHistory()`. Matches the SPEC §v1.9 endpoint name
   * literally (`GET /v1/scans` → "list scans") so call sites that
   * think in terms of "list" instead of "history" read naturally.
   */
  listScans(): Promise<HistoryResponse> {
    return this.getHistory();
  }

  /**
   * POST /v1/scans/:id/rule-results/:rid/flag
   * NOTE: backend route not yet implemented per current scans.py — kept
   * here so the client surface matches SPEC §v1.9.
   */
  flagRuleResult(
    scanId: string,
    ruleId: string,
    payload: FlagRuleResultRequest
  ): Promise<void> {
    return this.request<void>(
      `/v1/scans/${scanId}/rule-results/${encodeURIComponent(ruleId)}/flag`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        parseJson: false,
      }
    );
  }

  /**
   * POST /v1/detect-container — pre-capture confirmation gate.
   *
   * Snapshots a single frame from the scan flow and asks the backend to
   * locate a bottle / can / box. Used by the unwrap screen between
   * `ready` and `scanning` so the user gets to confirm what the system
   * sees before committing to a 8–15s panorama capture.
   *
   * Multipart upload — `image` form field, named `frame.jpg`. Auth
   * header is attached via `request<T>()` like all other endpoints.
   *
   * The returned `bbox` is normalized to [0..1] in [x0, y0, x1, y1]
   * order; the caller is responsible for converting to its own pixel
   * (or DP) coordinate system.
   */
  async detectContainer(imageUri: string): Promise<DetectContainerResponse> {
    const form = new FormData();
    // RN's FormData accepts the `{uri, name, type}` object literal for
    // file parts — same pattern Expo Image Picker / vision-camera docs
    // recommend. Cast through `unknown` because the DOM lib types only
    // know about Blob/string and complain about the RN file shape.
    form.append('image', {
      uri: imageUri,
      name: 'frame.jpg',
      type: 'image/jpeg',
    } as unknown as Blob);
    return this.request<DetectContainerResponse>('/v1/detect-container', {
      method: 'POST',
      // Do NOT set Content-Type: the runtime needs to fill in the
      // multipart boundary. RN's fetch handles this automatically when
      // the body is a FormData and Content-Type is absent.
      body: form as unknown as BodyInit,
    });
  }
}

/**
 * Response shape for `POST /v1/detect-container`. `bbox` is normalized
 * to 0..1 in [x0, y0, x1, y1] order. When `detected` is false the
 * `reason` string is a one-sentence user-facing hint (or null if the
 * backend didn't supply one).
 *
 * `brand_name` / `net_contents` are best-effort recognition signals
 * (KNOWN_LABEL_DESIGN.md Decision 2). `image_dhash` is the 64-bit
 * perceptual hash the backend computed from the upload bytes, hex-
 * encoded; the mobile app threads it into the finalize call so
 * enrichment can stamp it on the cache entry. `known_label` carries the
 * recognition payload when the lookup matched — null on miss.
 */
export interface DetectContainerResponse {
  detected: boolean;
  container_type: 'bottle' | 'can' | 'box' | null;
  bbox: [number, number, number, number] | null;
  confidence: number;
  reason: string | null;
  brand_name?: string | null;
  net_contents?: string | null;
  image_dhash?: string | null;
  known_label?: KnownLabelPayload | null;
}

// Default singleton; modules can also instantiate their own with
// alternate base URLs (e.g. tests). Token is read from the auth store
// at request time so signed-in state is reflected immediately.
export const apiClient = new ApiClient({ getToken: () => getAuthToken() });
