/**
 * Unit tests for `detectBottle` (bottleDetector.ts) — the silhouette
 * geometry path (lines ~115-256), distinct from the already-tested
 * `classifyContainer` aspect bands.
 *
 * The detector is a pure worklet over a luma plane → BottleSilhouette.
 * We drive it with synthetic luma frames built so the heuristics
 * (per-row Sobel, edge dominance, median, std-dev, vertical extent)
 * resolve deterministically:
 *
 *   - background luma 10
 *   - a vertical bottle band [L..R] of luma 220
 *   - single-column ramps (115) just inside each edge so each row has
 *     exactly one dominant Sobel peak per side (clears MIN_EDGE_DOMINANCE)
 *
 * This pins detection on a clean band and rejection on the degenerate
 * frames the worklet must shrug off (empty, too-narrow).
 */

import { detectBottle } from '../bottleDetector';

const W = 160;
const H = 240;
const BG = 10;
const FILL = 220;
const RAMP = 115;

/**
 * Build a luma plane with a full-height vertical band from column L to
 * column R. The ramp columns at L and R give a single dominant Sobel
 * peak at exactly x=L and x=R (see the test header rationale).
 */
function bandFrame(L: number, R: number, w = W, h = H): Uint8Array {
  const luma = new Uint8Array(w * h).fill(BG);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = L; x <= R; x++) {
      luma[row + x] = x === L || x === R ? RAMP : FILL;
    }
  }
  return luma;
}

describe('detectBottle — clean vertical band', () => {
  test('detects the band and locates its edges + center', () => {
    const s = detectBottle(bandFrame(50, 110), W, H);
    expect(s.detected).toBe(true);
    expect(s.edgeLeftX).toBe(50);
    expect(s.edgeRightX).toBe(110);
    expect(s.widthPx).toBe(60);
    expect(s.centerX).toBeCloseTo(80, 5);
  });

  test('a full-height band yields a tall heightPx and classifies bottle', () => {
    const s = detectBottle(bandFrame(50, 110), W, H);
    // Vertical scan extends to the frame edges → heightPx near full H.
    expect(s.heightPx).toBeGreaterThan(H * 0.7);
    expect(s.class).toBe('bottle');
    expect(s.containerConfidence).toBeGreaterThan(0.7);
  });

  test('crisp straight edges yield a high steadiness score', () => {
    // std-dev of both edge columns is 0 (every row lands the same
    // column), so tightness should be at its max.
    const s = detectBottle(bandFrame(50, 110), W, H);
    expect(s.steadinessScore).toBeCloseTo(1, 5);
  });

  test('a narrower band still detects and centers correctly', () => {
    // 60..100 = width 40 (frac 0.25), comfortably above the 0.15 floor.
    const s = detectBottle(bandFrame(60, 100), W, H);
    expect(s.detected).toBe(true);
    expect(s.widthPx).toBe(40);
    expect(s.centerX).toBeCloseTo(80, 5);
  });
});

describe('detectBottle — rejection paths', () => {
  test('returns an undetected silhouette for a flat (no-edge) frame', () => {
    const flat = new Uint8Array(W * H).fill(128);
    const s = detectBottle(flat, W, H);
    expect(s.detected).toBe(false);
    expect(s.widthPx).toBe(0);
    expect(s.class).toBeNull();
  });

  test('rejects a band narrower than the detection floor (frac < 0.15)', () => {
    // 75..85 = width 10, frac 0.0625 — below MIN_WIDTH_DETECT_FRAC.
    const s = detectBottle(bandFrame(75, 85), W, H);
    expect(s.detected).toBe(false);
  });

  test('returns an undetected silhouette for an all-zero frame', () => {
    const zero = new Uint8Array(W * H);
    const s = detectBottle(zero, W, H);
    expect(s.detected).toBe(false);
  });
});
