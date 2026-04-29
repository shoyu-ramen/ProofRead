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
 *   home → beverage-type → container-size → unwrap (cylindrical scan)
 *        → review → processing/[id] → report/[id]
 */

import { create } from 'zustand';
import type { BeverageType } from '@src/api/types';

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

export interface ScanDraft {
  beverageType: BeverageType | null;
  containerSizeMl: number | null;
  // Whether the user marked the product as imported on the
  // beverage-type / container step. Drives the country-of-origin rule.
  isImported: boolean;
  /** Final stitched panorama for upload + review. Null until complete. */
  panorama: UnrolledPanorama | null;
  /** Raw frames captured during the rotation, in capture order. */
  frames: ScanFrame[];
  // scan_id from POST /v1/scans, set after we hit the backend at the
  // start of upload.
  scanId: string | null;
}

interface ScanStoreState extends ScanDraft {
  setBeverageType: (t: BeverageType) => void;
  setContainerSize: (ml: number) => void;
  setIsImported: (imported: boolean) => void;
  setPanorama: (p: UnrolledPanorama | null) => void;
  appendFrame: (f: ScanFrame) => void;
  clearScanCaptures: () => void;
  setScanId: (id: string | null) => void;
  reset: () => void;
}

const EMPTY: ScanDraft = {
  beverageType: null,
  containerSizeMl: null,
  isImported: false,
  panorama: null,
  frames: [],
  scanId: null,
};

export const useScanStore = create<ScanStoreState>((set) => ({
  ...EMPTY,
  setBeverageType: (t) => set({ beverageType: t }),
  setContainerSize: (ml) => set({ containerSizeMl: ml }),
  setIsImported: (imported) => set({ isImported: imported }),
  setPanorama: (p) => set({ panorama: p }),
  appendFrame: (f) => set((s) => ({ frames: [...s.frames, f] })),
  clearScanCaptures: () => set({ panorama: null, frames: [] }),
  setScanId: (id) => set({ scanId: id }),
  reset: () => set({ ...EMPTY, frames: [] }),
}));

// Default container sizes (mL) per SPEC §v1.5 F1.4.
export const DEFAULT_CONTAINER_SIZES: ReadonlyArray<{ label: string; ml: number }> = [
  { label: '12 oz can', ml: 355 },
  { label: '16 oz can', ml: 473 },
  { label: '500 mL bottle', ml: 500 },
  { label: '22 oz bomber', ml: 650 },
];
