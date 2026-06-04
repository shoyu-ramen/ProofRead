/**
 * Unit tests for `measureFlow` (opticalFlow.ts).
 *
 * The matcher is a pure worklet: (prevLuma, currLuma, w, h, silhouette)
 * → median horizontal flow inside the silhouette. We build a textured
 * column pattern, shift it by a known number of pixels between frames,
 * and assert the recovered dxPx matches. The guard cases pin the audit
 * fixes: undetected silhouette, mismatched buffers, and the zero-
 * variance (clear-glass) gate that must return uncertain rather than a
 * confident dx=0.
 */

import { measureFlow } from '../opticalFlow';
import type { BottleSilhouette } from '../types';

const W = 160;
const H = 240;

function silhouette(overrides: Partial<BottleSilhouette> = {}): BottleSilhouette {
  return {
    detected: true,
    edgeLeftX: 50,
    edgeRightX: 110,
    edgeTopY: 0,
    edgeBottomY: H,
    centerX: 80,
    widthPx: 60,
    heightPx: H,
    steadinessScore: 0.9,
    class: 'bottle',
    classConfidence: 0.9,
    containerConfidence: 0.9,
    ...overrides,
  };
}

// Deterministic, aperiodic per-column intensities (mulberry32). An
// aperiodic pattern guarantees a unique SAD minimum across the ±20
// search window — a periodic stripe pattern would alias.
function columnPattern(w: number, seed = 0x9e3779b9): number[] {
  let s = seed >>> 0;
  const col: number[] = [];
  for (let x = 0; x < w; x++) {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    col.push((((t ^ (t >>> 14)) >>> 0) % 200) + 20); // 20..219
  }
  return col;
}

/** Fill every row with the same column pattern (vertical texture). */
function frameFromColumns(col: number[], w = W, h = H): Uint8Array {
  const luma = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) luma[row + x] = col[x];
  }
  return luma;
}

/** Shift a column pattern right by `shift` px (clamping at the edge). */
function shiftColumns(col: number[], shift: number): number[] {
  return col.map((_, x) => col[Math.max(0, Math.min(col.length - 1, x - shift))]);
}

describe('measureFlow — recovers horizontal motion', () => {
  test('recovers a rightward shift of +4 px', () => {
    const base = columnPattern(W);
    const prev = frameFromColumns(base);
    const curr = frameFromColumns(shiftColumns(base, 4));
    const m = measureFlow(prev, curr, W, H, silhouette());
    expect(m.dxPx).toBeCloseTo(4, 0);
    expect(m.confidence).toBeGreaterThan(0.5);
    expect(m.inliers).toBeGreaterThan(0);
  });

  test('recovers a leftward shift of -6 px', () => {
    const base = columnPattern(W, 0x12345678);
    const prev = frameFromColumns(base);
    const curr = frameFromColumns(shiftColumns(base, -6));
    const m = measureFlow(prev, curr, W, H, silhouette());
    expect(m.dxPx).toBeCloseTo(-6, 0);
    expect(m.confidence).toBeGreaterThan(0.5);
  });

  test('reports near-zero flow for an identical frame pair', () => {
    const base = columnPattern(W);
    const frame = frameFromColumns(base);
    const m = measureFlow(frame, frame, W, H, silhouette());
    expect(Math.abs(m.dxPx)).toBeLessThanOrEqual(1);
  });
});

describe('measureFlow — guard cases', () => {
  test('returns the empty measurement when the silhouette is undetected', () => {
    const base = columnPattern(W);
    const prev = frameFromColumns(base);
    const curr = frameFromColumns(shiftColumns(base, 4));
    const m = measureFlow(prev, curr, W, H, silhouette({ detected: false }));
    expect(m).toEqual({ dxPx: 0, inliers: 0, confidence: 0 });
  });

  test('returns empty when a luma buffer length disagrees with w·h', () => {
    const base = columnPattern(W);
    const prev = frameFromColumns(base);
    const wrong = new Uint8Array(W * H - 1);
    const m = measureFlow(prev, wrong, W, H, silhouette());
    expect(m).toEqual({ dxPx: 0, inliers: 0, confidence: 0 });
  });

  test('returns uncertain (not confident dx=0) on a flat clear-glass body', () => {
    // Zero-variance templates must be gated out so the integrator never
    // reads a fake stationary bottle (the too_slow audit finding).
    const flat = new Uint8Array(W * H).fill(128);
    const m = measureFlow(flat, flat, W, H, silhouette());
    expect(m.confidence).toBe(0);
    expect(m.inliers).toBe(0);
  });
});
