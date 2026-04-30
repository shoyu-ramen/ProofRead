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

// Width-fraction floor for detection. Below this, paired vertical
// edges are likely background features (table edges, frame seams) and
// not a bottle. The valid scan-distance band is 30–90% of frame width;
// detection no longer hard-rejects between this floor and the upper
// limit so the frame processor can classify too_far / too_close.
const MIN_WIDTH_DETECT_FRAC = 0.15;

// Reject if either edge column std-dev (across rows) exceeds this many
// pixels. A cylindrical bottle's vertical edges are nearly straight in
// the resized frame; a high std-dev means we latched onto label text.
const MAX_EDGE_STDDEV_PX = 6;

// Minimum ratio between the chosen edge response and the next-best
// candidate in the same row half. Audit finding: a vertical tile
// grout / window edge / door frame in the background can pass the
// stddev test if it sits at a roughly stable column. Requiring the
// chosen edge to be 1.5× stronger than its nearest competitor in
// the same half-row weeds out those ties without rejecting genuine
// (single-vertical) bottle edges.
const MIN_EDGE_DOMINANCE = 1.5;

// If more than this fraction of rows are "ambiguous" (chosen edge
// not dominant over the next-best), treat the silhouette as unstable
// and fail detection. 0.30 = up to ~3 ambiguous rows in a typical
// 10-row band before we reject.
const MAX_AMBIGUOUS_ROW_FRAC = 0.3;

// How many rows beyond the band we may walk before giving up on
// finding the body's top/bottom. Bounds the tail scan even when the
// bottle truly extends edge-to-edge (the loop also stops at the frame
// border in that case).
const VERT_SCAN_MAX_ROWS = 200;

// How many consecutive missed rows we tolerate before declaring the
// edge has ended. A small slack handles cap glints, label seams, and
// background clutter that briefly knock out the per-row Sobel hit.
const VERT_SCAN_MISS_TOL = 2;

// Per-row column tolerance for the top/bottom scan, expressed as a
// fraction of body width. Bottles taper near the cap; widen the
// search a touch so the cap shoulder stays inside the window. Floored
// so tiny silhouettes don't collapse the search to a single column.
const VERT_SCAN_TOL_FRAC = 0.15;
const VERT_SCAN_TOL_FLOOR_PX = 12;

// Empty silhouette returned when detection fails. Stable shape so the
// shared value's writes stay JIT-friendly.
const EMPTY: BottleSilhouette = {
  detected: false,
  edgeLeftX: 0,
  edgeRightX: 0,
  edgeTopY: 0,
  edgeBottomY: 0,
  centerX: 0,
  widthPx: 0,
  heightPx: 0,
  steadinessScore: 0,
  class: null,
  classConfidence: 0,
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

  // Per-row dominance flag: true if either edge in this row is
  // ambiguous (chosen response not >= MIN_EDGE_DOMINANCE × next-best).
  // Aggregated below to reject silhouettes contaminated by background
  // verticals (audit finding: tile grout, window edges, door frames).
  let scannedRows = 0;
  let ambiguousRows = 0;

  for (let y = yTop; y < yBot; y++) {
    const row = y * w;
    let bestLeft = -1;
    let bestLeftMag = SOBEL_MIN;
    // Second-best magnitudes per side. Tracked separately from the
    // best so we can compute an edge-dominance ratio per row without
    // breaking the existing single-pass scan.
    let secondLeftMag = 0;
    let bestRight = -1;
    let bestRightMag = SOBEL_MIN;
    let secondRightMag = 0;

    // 3-tap horizontal Sobel (vertical-edge response). We scan the
    // row once, splitting at cx so each side keeps the strongest
    // response independently.
    for (let x = 1; x < w - 1; x++) {
      const left = luma[row + x - 1];
      const right = luma[row + x + 1];
      const mag = right > left ? right - left : left - right;
      if (x < cx) {
        if (mag > bestLeftMag) {
          // Demote the previous best to second-best before we
          // overwrite it — keeps the dominance check honest.
          secondLeftMag = bestLeftMag;
          bestLeftMag = mag;
          bestLeft = x;
        } else if (mag > secondLeftMag) {
          secondLeftMag = mag;
        }
      } else {
        if (mag > bestRightMag) {
          secondRightMag = bestRightMag;
          bestRightMag = mag;
          bestRight = x;
        } else if (mag > secondRightMag) {
          secondRightMag = mag;
        }
      }
    }

    if (bestLeft >= 0) leftCols.push(bestLeft);
    if (bestRight >= 0) rightCols.push(bestRight);

    if (bestLeft >= 0 && bestRight >= 0) {
      scannedRows++;
      // SOBEL_MIN initial seed for the second-best comparator means
      // a row with literally one edge passes (denominator floor of
      // SOBEL_MIN). If a real second peak exists, the ratio reflects
      // it. Edge is ambiguous if EITHER side fails dominance.
      const leftDominance = bestLeftMag / Math.max(SOBEL_MIN, secondLeftMag);
      const rightDominance = bestRightMag / Math.max(SOBEL_MIN, secondRightMag);
      if (
        leftDominance < MIN_EDGE_DOMINANCE ||
        rightDominance < MIN_EDGE_DOMINANCE
      ) {
        ambiguousRows++;
      }
    }
  }

  // Need a quorum on both sides — sparse rows mean the bottle is
  // probably not in frame, or we're looking at a busy background.
  const minRows = Math.floor((yBot - yTop) * 0.5);
  if (leftCols.length < minRows || rightCols.length < minRows) return EMPTY;

  // Background-vertical guard. If too many rows had a competitor
  // edge close to the chosen one, the silhouette is sitting on top
  // of (or next to) a background vertical and we can't trust the
  // median to track the actual bottle (audit finding).
  if (
    scannedRows > 0 &&
    ambiguousRows / scannedRows > MAX_AMBIGUOUS_ROW_FRAC
  ) {
    return EMPTY;
  }

  const leftMedian = median(leftCols);
  const rightMedian = median(rightCols);
  if (rightMedian <= leftMedian) return EMPTY;

  const widthPx = rightMedian - leftMedian;
  const widthFrac = widthPx / w;
  // Reject only widths so small they can't be told from background
  // edge noise. The valid-scan-distance band (30–90%) is now classified
  // by the frame processor as too_far / too_close, not silently dropped.
  if (widthFrac < MIN_WIDTH_DETECT_FRAC) return EMPTY;

  // Std-dev against the medians; rejects label-edge noise that beat
  // the silhouette edge in stray rows.
  const leftStd = stddev(leftCols, leftMedian);
  const rightStd = stddev(rightCols, rightMedian);
  if (leftStd > MAX_EDGE_STDDEV_PX || rightStd > MAX_EDGE_STDDEV_PX) return EMPTY;

  // Steadiness here is per-frame "edge crispness"; the caller folds it
  // into a multi-frame EMA.
  const tightness =
    1 - (leftStd + rightStd) / (2 * MAX_EDGE_STDDEV_PX);

  // Vertical extent. Walk up from the band's top row and down from its
  // bottom, looking for the rows where strong vertical edges still
  // sit near both medians. Caller scales these into photo-pixel space
  // for cropping and into screen-px for the silhouette overlay.
  const colTol = Math.max(
    VERT_SCAN_TOL_FLOOR_PX,
    Math.round(widthPx * VERT_SCAN_TOL_FRAC),
  );
  const edgeTopY = scanForTopEdge(luma, w, h, yTop, leftMedian, rightMedian, colTol);
  const edgeBottomY = scanForBottomEdge(luma, w, h, yBot, leftMedian, rightMedian, colTol);
  const heightPx = edgeBottomY > edgeTopY ? edgeBottomY - edgeTopY : 0;

  return {
    detected: true,
    edgeLeftX: leftMedian,
    edgeRightX: rightMedian,
    edgeTopY,
    edgeBottomY,
    centerX: (leftMedian + rightMedian) * 0.5,
    widthPx,
    heightPx,
    steadinessScore: clamp01(tightness),
    // Heuristic detector can't classify; Phase 2's TFLite path will
    // populate these. Keeping the field present keeps the shape stable
    // so the worklet's writes to trackerStateSv don't deopt.
    class: null,
    classConfidence: 0,
  };
}

/**
 * Walk up from `yTop` until vertical edges no longer hit near the
 * left/right medians. Returns the highest row where both edges still
 * register; falls back to `yTop` if no extension is found.
 */
function scanForTopEdge(
  luma: Uint8Array,
  w: number,
  h: number,
  yTop: number,
  leftMedian: number,
  rightMedian: number,
  colTol: number,
): number {
  'worklet';
  let edge = yTop;
  let misses = 0;
  const yMin = Math.max(0, yTop - VERT_SCAN_MAX_ROWS);
  for (let y = yTop - 1; y >= yMin; y--) {
    if (
      rowHasEdgeNear(luma, w, y, leftMedian, colTol) &&
      rowHasEdgeNear(luma, w, y, rightMedian, colTol)
    ) {
      edge = y;
      misses = 0;
    } else {
      misses++;
      if (misses > VERT_SCAN_MISS_TOL) break;
    }
  }
  // h is unused but kept in the signature so symmetry with the
  // bottom-edge scan is obvious; suppress the unused-arg warning.
  void h;
  return edge;
}

/**
 * Mirror of `scanForTopEdge`, walking downward from `yBot`.
 */
function scanForBottomEdge(
  luma: Uint8Array,
  w: number,
  h: number,
  yBot: number,
  leftMedian: number,
  rightMedian: number,
  colTol: number,
): number {
  'worklet';
  let edge = yBot - 1;
  let misses = 0;
  const yMax = Math.min(h - 1, yBot + VERT_SCAN_MAX_ROWS - 1);
  for (let y = yBot; y <= yMax; y++) {
    if (
      rowHasEdgeNear(luma, w, y, leftMedian, colTol) &&
      rowHasEdgeNear(luma, w, y, rightMedian, colTol)
    ) {
      edge = y;
      misses = 0;
    } else {
      misses++;
      if (misses > VERT_SCAN_MISS_TOL) break;
    }
  }
  return edge;
}

/**
 * True if any column within ±`tol` of `targetX` has a vertical-edge
 * Sobel response ≥ SOBEL_MIN in row `y`. Worklet-safe; bounded by
 * 2·tol pixels per call.
 */
function rowHasEdgeNear(
  luma: Uint8Array,
  w: number,
  y: number,
  targetX: number,
  tol: number,
): boolean {
  'worklet';
  const xMin = Math.max(1, targetX - tol);
  const xMax = Math.min(w - 2, targetX + tol);
  const row = y * w;
  for (let x = xMin; x <= xMax; x++) {
    const left = luma[row + x - 1];
    const right = luma[row + x + 1];
    const mag = right > left ? right - left : left - right;
    if (mag >= SOBEL_MIN) return true;
  }
  return false;
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
