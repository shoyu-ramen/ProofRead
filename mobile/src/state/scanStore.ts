/**
 * In-progress scan state.
 *
 * Holds everything the user has selected for the current scan flow
 * before it's been submitted to POST /v1/scans, plus the captured
 * images per surface, plus the scan_id once the backend has issued
 * one. Resets on completion or explicit cancel.
 *
 * The flow that reads this store (per SPEC §v1.6):
 *   home → beverage-type → container-size → camera/front
 *        → camera/back → review → processing/[id] → report/[id]
 */

import { create } from 'zustand';
import type { BeverageType, Surface } from '@src/api/types';

export interface CapturedImage {
  // file:// URI on device. Loaded into bytes only at upload time.
  uri: string;
  width: number;
  height: number;
  capturedAt: number;
}

export interface ScanDraft {
  beverageType: BeverageType | null;
  containerSizeMl: number | null;
  // Whether the user marked the product as imported on the
  // beverage-type / container step. Drives the country-of-origin rule.
  isImported: boolean;
  // Per-surface captures. Front + back required in v1 (SPEC §v1.6 step 4–5).
  captures: Partial<Record<Surface, CapturedImage>>;
  // scan_id from POST /v1/scans, set after we hit the backend at the
  // start of upload.
  scanId: string | null;
}

interface ScanStoreState extends ScanDraft {
  setBeverageType: (t: BeverageType) => void;
  setContainerSize: (ml: number) => void;
  setIsImported: (imported: boolean) => void;
  setCapture: (surface: Surface, capture: CapturedImage) => void;
  clearCapture: (surface: Surface) => void;
  setScanId: (id: string | null) => void;
  reset: () => void;
  hasRequiredCaptures: () => boolean;
}

const EMPTY: ScanDraft = {
  beverageType: null,
  containerSizeMl: null,
  isImported: false,
  captures: {},
  scanId: null,
};

const REQUIRED_SURFACES: Surface[] = ['front', 'back'];

export const useScanStore = create<ScanStoreState>((set, get) => ({
  ...EMPTY,
  setBeverageType: (t) => set({ beverageType: t }),
  setContainerSize: (ml) => set({ containerSizeMl: ml }),
  setIsImported: (imported) => set({ isImported: imported }),
  setCapture: (surface, capture) =>
    set((s) => ({ captures: { ...s.captures, [surface]: capture } })),
  clearCapture: (surface) =>
    set((s) => {
      const next = { ...s.captures };
      delete next[surface];
      return { captures: next };
    }),
  setScanId: (id) => set({ scanId: id }),
  reset: () => set({ ...EMPTY, captures: {} }),
  hasRequiredCaptures: () => {
    const c = get().captures;
    return REQUIRED_SURFACES.every((s) => Boolean(c[s]));
  },
}));

export const REQUIRED_CAPTURE_SURFACES: ReadonlyArray<Surface> = REQUIRED_SURFACES;

// Default container sizes (mL) per SPEC §v1.5 F1.4.
export const DEFAULT_CONTAINER_SIZES: ReadonlyArray<{ label: string; ml: number }> = [
  { label: '12 oz can', ml: 355 },
  { label: '16 oz can', ml: 473 },
  { label: '500 mL bottle', ml: 500 },
  { label: '22 oz bomber', ml: 650 },
];
