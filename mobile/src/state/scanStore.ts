/**
 * In-progress scan state.
 *
 * Holds everything the user has selected for the current scan flow
 * before it's been submitted to POST /v1/scans, plus the unrolled
 * panorama once captured, plus the scan_id once the backend has issued
 * one. Resets on completion or explicit cancel.
 *
 * Per ARCH §6, v1 captures a single unrolled-label panorama (replacing
 * the old front+back surface model). The flow that reads this store
 * (per SPEC §v1.6 + ARCH §1):
 *   home → setup → unwrap (cylindrical scan)
 *        → review → processing/[id] → report/[id]
 */

import { create } from 'zustand';
import type { BeverageType, KnownLabelPayload } from '@src/api/types';

/**
 * One raw frame captured during the rotation. Stored for diagnostics
 * and for the optional "raw frames" debug viewer; the upload payload
 * is the stitched panorama, not the individual frames.
 */
export interface ScanFrame {
  /** file:// URI on device. */
  uri: string;
  /** Coverage at the moment of capture, 0..1. */
  coverage: number;
  capturedAt: number;
}

/**
 * The composed unrolled label produced at scan completion. This is
 * the canonical type — the panorama subsystem re-exports it from
 * `mobile/src/scan/panorama/types.ts` for self-containment.
 *
 * Shape matches `BBoxOverlay`'s `image` prop so the rule-detail screen
 * can pass it through directly.
 */
export interface UnrolledPanorama {
  /** file:// (or data:) URI of the encoded JPEG. */
  uri: string;
  width: number;
  height: number;
  /** Number of strip checkpoints that contributed to the panorama. */
  frameCount: number;
  /** Wall-clock duration of the scan, in milliseconds. */
  durationMs: number;
}

/**
 * Cached panorama keyed by backend scan_id. Populated when a scan
 * completes (review→processing transition, where we have both ids and
 * pixels in hand). Persists across `reset()` so the home recent-scans
 * rail and the report screen can display the local panorama for any
 * scan we've completed in this session without re-downloading from the
 * server.
 *
 * Pure in-memory cache (not persisted to AsyncStorage in v1) — that
 * keeps the implementation tight and matches the user expectation that
 * a fresh app launch will fetch from the server. The map only grows
 * during a session; we accept that since the scan flow is short-lived.
 */
export interface CachedPanorama {
  uri: string;
  width: number;
  height: number;
  capturedAt: number;
}

export interface ScanDraft {
  beverageType: BeverageType | null;
  containerSizeMl: number | null;
  // Whether the user marked the product as imported on the
  // setup step. Drives the country-of-origin rule.
  isImported: boolean;
  /** Final stitched panorama for upload + review. Null until complete. */
  panorama: UnrolledPanorama | null;
  /** Raw frames captured during the rotation, in capture order. */
  frames: ScanFrame[];
  // scan_id from POST /v1/scans, set after we hit the backend at the
  // start of upload.
  scanId: string | null;
  // dhash hex of the first detect-container frame, as returned by the
  // backend in DetectContainerResponse.image_dhash. Threaded into the
  // finalize call so enrichment can stamp it on the L3 cache row for
  // future first-frame recognition lookups.
  firstFrameSignatureHex: string | null;
  // Recognition payload from DetectContainerResponse.known_label when
  // the backend matches the captured frame to a previously-scanned
  // label. The unwrap screen renders the recognition sheet inside the
  // confirming{detected} sub-phase when this is non-null.
  knownLabel: KnownLabelPayload | null;
}

interface ScanStoreState extends ScanDraft {
  /**
   * Per-scan-id panorama cache. Lives outside `ScanDraft` so `reset()`
   * doesn't wipe panoramas the user has already completed in this
   * session — the home recent-scans rail relies on it to render
   * thumbnails without a re-fetch.
   */
  recentPanoramas: Record<string, CachedPanorama>;
  setBeverageType: (t: BeverageType) => void;
  setContainerSize: (ml: number) => void;
  setIsImported: (imported: boolean) => void;
  setPanorama: (p: UnrolledPanorama | null) => void;
  appendFrame: (f: ScanFrame) => void;
  clearScanCaptures: () => void;
  setScanId: (id: string | null) => void;
  setFirstFrameSignatureHex: (hex: string | null) => void;
  setKnownLabel: (payload: KnownLabelPayload | null) => void;
  /**
   * Insert a freshly-captured panorama into the per-scan-id cache.
   * Idempotent: re-recording the same scan_id (e.g. user re-tries the
   * upload) overwrites the entry. Untouched by `reset()`.
   */
  rememberPanorama: (scanId: string, panorama: UnrolledPanorama) => void;
  reset: () => void;
}

const EMPTY: ScanDraft = {
  beverageType: null,
  containerSizeMl: null,
  isImported: false,
  panorama: null,
  frames: [],
  scanId: null,
  firstFrameSignatureHex: null,
  knownLabel: null,
};

export const useScanStore = create<ScanStoreState>((set) => ({
  ...EMPTY,
  recentPanoramas: {},
  setBeverageType: (t) => set({ beverageType: t }),
  setContainerSize: (ml) => set({ containerSizeMl: ml }),
  setIsImported: (imported) => set({ isImported: imported }),
  setPanorama: (p) => set({ panorama: p }),
  appendFrame: (f) => set((s) => ({ frames: [...s.frames, f] })),
  clearScanCaptures: () => set({ panorama: null, frames: [] }),
  // When we associate a scan_id with the in-flight draft (post-create,
  // pre-finalize), opportunistically snapshot the panorama into the
  // recents cache. That way the rail and the report can render the
  // local pixels even after `reset()` clears the draft on Done.
  setScanId: (id) =>
    set((s) => {
      if (id && s.panorama) {
        return {
          scanId: id,
          recentPanoramas: {
            ...s.recentPanoramas,
            [id]: {
              uri: s.panorama.uri,
              width: s.panorama.width,
              height: s.panorama.height,
              capturedAt: Date.now(),
            },
          },
        };
      }
      return { scanId: id };
    }),
  rememberPanorama: (scanId, panorama) =>
    set((s) => ({
      recentPanoramas: {
        ...s.recentPanoramas,
        [scanId]: {
          uri: panorama.uri,
          width: panorama.width,
          height: panorama.height,
          capturedAt: Date.now(),
        },
      },
    })),
  setFirstFrameSignatureHex: (hex) => set({ firstFrameSignatureHex: hex }),
  setKnownLabel: (payload) => set({ knownLabel: payload }),
  // NB: leaves `recentPanoramas` intact on purpose — completed scans
  // need to remain visible in the home rail across the Done button.
  reset: () => set({ ...EMPTY, frames: [] }),
}));

// Default container sizes (mL) per SPEC §v1.5 F1.4.
export const DEFAULT_CONTAINER_SIZES: ReadonlyArray<{ label: string; ml: number }> = [
  { label: '12 oz can', ml: 355 },
  { label: '16 oz can', ml: 473 },
  { label: '500 mL bottle', ml: 500 },
  { label: '22 oz bomber', ml: 650 },
];
