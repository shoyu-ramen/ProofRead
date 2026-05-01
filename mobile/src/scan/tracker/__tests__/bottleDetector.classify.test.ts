/**
 * Unit tests for `classifyContainer` (bottleDetector.ts).
 *
 * The classifier is a pure function over (widthPx, heightPx,
 * steadinessScore) → { class, classConfidence, containerConfidence }.
 * These tests pin the aspect-ratio bands brief #1 specifies so the
 * auto-trigger gate downstream gets predictable inputs:
 *
 *   - bottle:  aspectRatio (heightPx / widthPx) > 1.8
 *   - can:     1.0 ≤ aspectRatio ≤ 1.8
 *   - reject:  aspectRatio < 1.0  → class === null
 *
 * Confidence is a composite of edge tightness (`steadinessScore`) and
 * aspect-ratio fit; tests assert that high-steady + on-band silhouettes
 * clear the 0.7 auto-trigger gate while low-steady or off-band ones
 * stay below.
 */

import { classifyContainer } from '../bottleDetector';

describe('classifyContainer — aspect ratio bands', () => {
  test('rejects silhouettes wider than tall (aspect < 1.0)', () => {
    // 100x80: aspect 0.8 — clearly not a vertical container.
    const result = classifyContainer(100, 80, 1.0);
    expect(result.class).toBe(null);
    expect(result.containerConfidence).toBe(0);
    expect(result.classConfidence).toBe(0);
  });

  test('rejects silhouettes that are square (aspect < 1.0)', () => {
    // Edge: aspect exactly 1.0 should land in the can band, not be
    // rejected. Test the "below 1.0" rejection separately.
    const below = classifyContainer(100, 99, 1.0);
    expect(below.class).toBe(null);
    expect(below.containerConfidence).toBe(0);
  });

  test('classifies 12oz can shape as can (aspect ~1.5)', () => {
    // Real-world 12oz can: 4.83in tall × 2.6in wide → aspect ~1.86.
    // Heuristic detector under-measures the rim → land closer to 1.5
    // in tracker pixels. 80x120 = aspect 1.5.
    const result = classifyContainer(80, 120, 1.0);
    expect(result.class).toBe('can');
    expect(result.containerConfidence).toBeGreaterThan(0);
  });

  test('classifies 16oz can shape near band edge as can (aspect ~1.7)', () => {
    // Larger can: 60x100 = aspect ~1.67 → still "can" under the 1.8 cap.
    const result = classifyContainer(60, 100, 1.0);
    expect(result.class).toBe('can');
  });

  test('classifies wine bottle shape as bottle (aspect ~3.0)', () => {
    // Wine bottle: 50x150 = aspect 3.0 → solidly in the bottle band.
    const result = classifyContainer(50, 150, 1.0);
    expect(result.class).toBe('bottle');
    expect(result.containerConfidence).toBeGreaterThan(0.7);
  });

  test('classifies tall wine bottle as bottle (aspect ~3.5)', () => {
    // 50x175 = aspect 3.5 → at the confidence peak.
    const result = classifyContainer(50, 175, 1.0);
    expect(result.class).toBe('bottle');
    // At the peak with steadiness=1.0 the composite should max out.
    expect(result.containerConfidence).toBeGreaterThanOrEqual(0.95);
  });

  test('classifies just-above-can-cap as bottle (aspect 1.81)', () => {
    // 100x181 = aspect 1.81 — first pixel above the bottle threshold.
    const result = classifyContainer(100, 181, 1.0);
    expect(result.class).toBe('bottle');
  });

  test('classifies just-below-bottle-min as can (aspect 1.79)', () => {
    // 100x179 = aspect 1.79 — last pixel below the bottle threshold.
    const result = classifyContainer(100, 179, 1.0);
    expect(result.class).toBe('can');
  });

  test('returns null for zero dimensions', () => {
    expect(classifyContainer(0, 100, 1.0).class).toBe(null);
    expect(classifyContainer(100, 0, 1.0).class).toBe(null);
    expect(classifyContainer(0, 0, 1.0).class).toBe(null);
  });
});

describe('classifyContainer — confidence blending', () => {
  test('high steadiness + on-band aspect clears the 0.7 auto-trigger gate', () => {
    // Centered can shape with crisp edges should be confidently
    // classified — the auto-trigger gate (>0.7) needs to fire here.
    const aspect14 = classifyContainer(70, 98, 1.0); // aspect 1.4 (mid-can)
    expect(aspect14.containerConfidence).toBeGreaterThan(0.7);

    const wineBottle = classifyContainer(50, 150, 1.0); // aspect 3.0 (bottle)
    expect(wineBottle.containerConfidence).toBeGreaterThan(0.7);
  });

  test('low steadiness keeps confidence below the auto-trigger gate', () => {
    // Same on-band aspect, but the user is shaky — confidence has to
    // be pulled down by the steadiness factor so the gate doesn't fire
    // until they hold still.
    const result = classifyContainer(50, 150, 0.3);
    expect(result.class).toBe('bottle');
    expect(result.containerConfidence).toBeLessThan(0.7);
  });

  test('confidence scales monotonically with steadiness', () => {
    const low = classifyContainer(50, 150, 0.3).containerConfidence;
    const mid = classifyContainer(50, 150, 0.6).containerConfidence;
    const high = classifyContainer(50, 150, 0.9).containerConfidence;
    expect(low).toBeLessThan(mid);
    expect(mid).toBeLessThan(high);
  });

  test('confidence stays in 0..1', () => {
    // A handful of cases at the extremes.
    const cases = [
      { w: 50, h: 200, s: 1.0 },
      { w: 100, h: 110, s: 0.0 },
      { w: 80, h: 200, s: 0.5 },
    ];
    for (const { w, h, s } of cases) {
      const r = classifyContainer(w, h, s);
      expect(r.containerConfidence).toBeGreaterThanOrEqual(0);
      expect(r.containerConfidence).toBeLessThanOrEqual(1);
      expect(r.classConfidence).toBeGreaterThanOrEqual(0);
      expect(r.classConfidence).toBeLessThanOrEqual(1);
    }
  });

  test('classConfidence equals containerConfidence in v1', () => {
    // Phase 2 will split these (real per-class probabilities); v1
    // shares the heuristic composite. Pin the contract so a future
    // refactor doesn't silently desync them without a test fail.
    const r = classifyContainer(80, 120, 0.7);
    expect(r.classConfidence).toBe(r.containerConfidence);
  });

  test('rejected aspects have zero confidence', () => {
    // Below band → class null → confidence zero.
    const r = classifyContainer(100, 80, 1.0);
    expect(r.class).toBe(null);
    expect(r.classConfidence).toBe(0);
    expect(r.containerConfidence).toBe(0);
  });
});
