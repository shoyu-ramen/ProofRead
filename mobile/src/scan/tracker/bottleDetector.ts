/**
 * Bottle silhouette detector (ARCH §4.2).
 *
 * Pure worklet: given the luma plane of the resized frame, find the
 * bottle's left and right vertical edges in a central horizontal band.
 * Heuristic CV — no ML model — adequate for v1 because cylindrical
 * bottles produce strong vertical Sobel responses against typical
 * backgrounds. The caller (frameProcessor) is responsible for EMA-
 * smoothing the steadiness score across frames.
 */

import type { BottleSilhouette } from './types';

// Vertical band of the frame we sample for left/right edges. Top and
// bottom rows are skipped because cap/base often pinch the silhouette
// in a way that throws off the median.
const BAND_TOP_FRAC = 0.25;
const BAND_BOT_FRAC = 0.85;

// Sobel response below this is noise. Tuned against the dev iPhone's
// 64×96 pre-check buffer — at 160×240 this is conservative.
const SOBEL_MIN = 32;

// Reject silhouettes outside these width fractions per spec.
const MIN_WIDTH_FRAC = 0.3;
const MAX_WIDTH_FRAC = 0.9;

// Reject if either edge column std-dev (across rows) exceeds this many
// pixels. A cylindrical bottle's vertical edges are nearly straight in
// the resized frame; a high std-dev means we latched onto label text.
const MAX_EDGE_STDDEV_PX = 6;

// Empty silhouette returned when detection fails. Stable shape so the
// shared value's writes stay JIT-friendly.
const EMPTY: BottleSilhouette = {
  detected: false,
  edgeLeftX: 0,
  edgeRightX: 0,
  centerX: 0,
  widthPx: 0,
  steadinessScore: 0,
};

/**
 * Scan the luma plane and return the detected silhouette. The function
 * is pure: caller threads steadiness across frames.
 */
export function detectBottle(
  luma: Uint8Array,
  w: number,
  h: number,
): BottleSilhouette {
  'worklet';

  const yTop = Math.max(1, Math.floor(h * BAND_TOP_FRAC));
  const yBot = Math.min(h - 1, Math.floor(h * BAND_BOT_FRAC));
  const cx = w >> 1;

  // Per-row left and right edge columns. Allocated fresh per call —
  // band height is ~140 entries at 240px, so the cost is dominated by
  // the resize+luma pass that's already happened.
  const leftCols: number[] = [];
  const rightCols: number[] = [];

  for (let y = yTop; y < yBot; y++) {
    const row = y * w;
    let bestLeft = -1;
    let bestLeftMag = SOBEL_MIN;
    let bestRight = -1;
    let bestRightMag = SOBEL_MIN;

    // 3-tap horizontal Sobel (vertical-edge response). We scan the
    // row once, splitting at cx so each side keeps the strongest
    // response independently.
    for (let x = 1; x < w - 1; x++) {
      const left = luma[row + x - 1];
      const right = luma[row + x + 1];
      const mag = right > left ? right - left : left - right;
      if (mag <= bestLeftMag && mag <= bestRightMag) continue;
      if (x < cx) {
        if (mag > bestLeftMag) {
          bestLeftMag = mag;
          bestLeft = x;
        }
      } else if (mag > bestRightMag) {
        bestRightMag = mag;
        bestRight = x;
      }
    }

    if (bestLeft >= 0) leftCols.push(bestLeft);
    if (bestRight >= 0) rightCols.push(bestRight);
  }

  // Need a quorum on both sides — sparse rows mean the bottle is
  // probably not in frame, or we're looking at a busy background.
  const minRows = Math.floor((yBot - yTop) * 0.5);
  if (leftCols.length < minRows || rightCols.length < minRows) return EMPTY;

  const leftMedian = median(leftCols);
  const rightMedian = median(rightCols);
  if (rightMedian <= leftMedian) return EMPTY;

  const widthPx = rightMedian - leftMedian;
  const widthFrac = widthPx / w;
  if (widthFrac < MIN_WIDTH_FRAC || widthFrac > MAX_WIDTH_FRAC) return EMPTY;

  // Std-dev against the medians; rejects label-edge noise that beat
  // the silhouette edge in stray rows.
  const leftStd = stddev(leftCols, leftMedian);
  const rightStd = stddev(rightCols, rightMedian);
  if (leftStd > MAX_EDGE_STDDEV_PX || rightStd > MAX_EDGE_STDDEV_PX) return EMPTY;

  // Steadiness here is per-frame "edge crispness"; the caller folds it
  // into a multi-frame EMA.
  const tightness =
    1 - (leftStd + rightStd) / (2 * MAX_EDGE_STDDEV_PX);

  return {
    detected: true,
    edgeLeftX: leftMedian,
    edgeRightX: rightMedian,
    centerX: (leftMedian + rightMedian) * 0.5,
    widthPx,
    steadinessScore: clamp01(tightness),
  };
}

function median(xs: number[]): number {
  'worklet';
  // Quickselect would be faster but the array is ~140 entries; a sort
  // is fine and avoids the worklet-mutability concerns of in-place
  // partition.
  const sorted = xs.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) * 0.5
    : sorted[mid];
}

function stddev(xs: number[], mean: number): number {
  'worklet';
  let acc = 0;
  for (let i = 0; i < xs.length; i++) {
    const d = xs[i] - mean;
    acc += d * d;
  }
  return Math.sqrt(acc / xs.length);
}

function clamp01(x: number): number {
  'worklet';
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
