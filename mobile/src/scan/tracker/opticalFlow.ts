/**
 * Sparse template-match optical flow (ARCH §4.3).
 *
 * Pure worklet: given the previous and current luma planes plus the
 * detected silhouette, return the median horizontal pixel motion of
 * the bottle's surface. A full LK solve is overkill here — we only
 * care about horizontal motion (rotation about the vertical axis), the
 * label is well-textured, and we constrain search to inside the
 * silhouette so background motion can't pollute the median.
 */

import type { BottleSilhouette, FlowMeasurement } from './types';

// Template grid. 12 templates in a vertical stack, each 8 wide × 16
// tall. The grid sits in a thin vertical strip just off the bottle's
// centerline (least foreshortened). Costs ~12 × 41 SADs per frame
// (search ±20 px), well inside the per-frame budget.
const TEMPLATE_COUNT = 12;
const TEMPLATE_W = 8;
const TEMPLATE_H = 16;
const SEARCH_RADIUS = 20;

// Templates are laid out from the top of the silhouette band to the
// bottom, with these vertical insets so they don't straddle the cap
// or base.
const GRID_TOP_FRAC = 0.25;
const GRID_BOT_FRAC = 0.85;

// Fraction of the silhouette width to use as the template-strip
// half-extent. 0.10 lands a 2×8=16-pixel strip on a 160-px-wide bottle
// — still in the "facing camera" zone.
const STRIP_HALF_FRAC = 0.1;

// Ignore matches with SAD-per-pixel above this — flat or near-flat
// templates produce ambiguous matches and shouldn't influence the
// median.
const MAX_SAD_PER_PIXEL = 50;

// Templates whose motion falls outside MAD * MAD_REJECT of the median
// are voted out. MAD (median absolute deviation) is robust to a small
// number of bad matches without needing a real RANSAC pass.
const MAD_REJECT = 2.5;

const EMPTY: FlowMeasurement = {
  dxPx: 0,
  inliers: 0,
  confidence: 0,
};

/**
 * Measure horizontal flow between two consecutive luma frames inside
 * the bottle silhouette. Returns the empty measurement if either luma
 * is invalid, the silhouette is undetected, or the templates couldn't
 * agree.
 */
export function measureFlow(
  prevLuma: Uint8Array,
  currLuma: Uint8Array,
  w: number,
  h: number,
  silhouette: BottleSilhouette,
): FlowMeasurement {
  'worklet';

  if (!silhouette.detected) return EMPTY;
  if (prevLuma.length !== w * h || currLuma.length !== w * h) return EMPTY;

  const halfStrip = silhouette.widthPx * STRIP_HALF_FRAC;
  const stripLeft = silhouette.centerX - halfStrip;

  // Templates need SEARCH_RADIUS of horizontal slack on the current
  // frame. Pull the strip in if it'd overflow.
  const earliestX = SEARCH_RADIUS + 1;
  const latestX = w - TEMPLATE_W - SEARCH_RADIUS - 1;
  const baseX = Math.max(earliestX, Math.min(latestX, Math.floor(stripLeft)));

  const yTop = Math.floor(h * GRID_TOP_FRAC);
  const yBot = Math.floor(h * GRID_BOT_FRAC) - TEMPLATE_H;
  if (yBot <= yTop) return EMPTY;
  const yStep = (yBot - yTop) / Math.max(1, TEMPLATE_COUNT - 1);

  // Per-template best displacement. We push valid ones into a flat
  // array for the median pass.
  const dxs: number[] = [];

  for (let i = 0; i < TEMPLATE_COUNT; i++) {
    const ty = yTop + Math.round(yStep * i);
    let bestSad = Infinity;
    let bestDx = 0;

    for (let dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++) {
      let sad = 0;
      // Inner loop is the hot path — keep it branchless and indexed.
      for (let ry = 0; ry < TEMPLATE_H; ry++) {
        const prevRow = (ty + ry) * w + baseX;
        const currRow = (ty + ry) * w + baseX + dx;
        for (let rx = 0; rx < TEMPLATE_W; rx++) {
          const a = prevLuma[prevRow + rx];
          const b = currLuma[currRow + rx];
          sad += a > b ? a - b : b - a;
        }
        // Early-out: if we've already exceeded the running best for
        // this template, no point finishing the SAD.
        if (sad >= bestSad) break;
      }
      if (sad < bestSad) {
        bestSad = sad;
        bestDx = dx;
      }
    }

    const sadPerPixel = bestSad / (TEMPLATE_W * TEMPLATE_H);
    if (sadPerPixel <= MAX_SAD_PER_PIXEL) dxs.push(bestDx);
  }

  if (dxs.length < TEMPLATE_COUNT * 0.5) return EMPTY;

  const med = median(dxs);
  const mad = medianAbsDeviation(dxs, med);
  // Reject zero-mad outlier filtering would discard everything if
  // every template agreed exactly — fall back to a 1-pixel band.
  const tol = Math.max(1, MAD_REJECT * mad);

  let inliers = 0;
  let inlierSum = 0;
  for (let i = 0; i < dxs.length; i++) {
    if (Math.abs(dxs[i] - med) <= tol) {
      inliers++;
      inlierSum += dxs[i];
    }
  }
  if (inliers === 0) return EMPTY;

  const dxPx = inlierSum / inliers;
  // Confidence ≈ inlier ratio × tightness of the MAD band, both
  // clamped to [0,1]. A tight, well-supported median scores ~1.
  const inlierRatio = inliers / TEMPLATE_COUNT;
  const tightness = 1 - Math.min(1, mad / SEARCH_RADIUS);
  const confidence = clamp01(inlierRatio * tightness);

  return { dxPx, inliers, confidence };
}

function median(xs: number[]): number {
  'worklet';
  const sorted = xs.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) * 0.5
    : sorted[mid];
}

function medianAbsDeviation(xs: number[], med: number): number {
  'worklet';
  const dev: number[] = [];
  for (let i = 0; i < xs.length; i++) dev.push(Math.abs(xs[i] - med));
  return median(dev);
}

function clamp01(x: number): number {
  'worklet';
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
