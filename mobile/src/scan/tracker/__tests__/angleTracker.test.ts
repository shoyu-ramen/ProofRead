/**
 * Unit tests for `computeAngularProgress` (angleTracker.ts).
 *
 * The integrator is a pure function over (priorState, flow, silhouette)
 * → next coverage / velocity / direction / flowQuality. It's the single
 * most safety-relevant piece of the scan engine — a drift bug here
 * silently corrupts the progress meter on every scan — yet it had no
 * direct coverage. These tests pin the cylinder model and the guard
 * rails documented in the module header (ARCH §4.4):
 *
 *   - revolution fraction = dxPx / (π · widthPx)
 *   - per-frame contribution capped at MAX_PER_FRAME (0.05)
 *   - direction commits after ~5° cumulative, then opposite motion is
 *     half-weighted (BACKTRACK_WEIGHT)
 *   - flowQuality is an EMA that decays even when the bottle is lost
 *   - coverage is floored at 0 and ceiled at 1
 */

import { computeAngularProgress } from '../angleTracker';
import type { AngleTrackerInputs } from '../angleTracker';
import type { BottleSilhouette, FlowMeasurement } from '../types';

// A detected silhouette wide enough that one frame of flow lands a
// small, sub-cap coverage delta. widthPx=100 → circumference π·100 ≈
// 314 px, so a 31.4 px flow is ~0.1 rev.
function silhouette(overrides: Partial<BottleSilhouette> = {}): BottleSilhouette {
  return {
    detected: true,
    edgeLeftX: 30,
    edgeRightX: 130,
    edgeTopY: 0,
    edgeBottomY: 240,
    centerX: 80,
    widthPx: 100,
    heightPx: 240,
    steadinessScore: 0.9,
    class: 'bottle',
    classConfidence: 0.9,
    containerConfidence: 0.9,
    ...overrides,
  };
}

function flow(overrides: Partial<FlowMeasurement> = {}): FlowMeasurement {
  return { dxPx: 5, inliers: 10, confidence: 0.9, ...overrides };
}

function inputs(overrides: Partial<AngleTrackerInputs> = {}): AngleTrackerInputs {
  return {
    coverage: 0,
    rotationDirection: null,
    angularVelocity: 0,
    flowQuality: 0,
    dtSec: 1 / 30,
    ...overrides,
  };
}

describe('computeAngularProgress — cylinder model', () => {
  test('advances coverage by dxPx / (π · widthPx)', () => {
    // dxPx=31.4159, widthPx=100 → exactly 0.1 rev. But that exceeds the
    // 0.05 per-frame cap, so use a smaller flow that stays under the cap.
    const out = computeAngularProgress(
      inputs(),
      flow({ dxPx: Math.PI * 100 * 0.02 }), // 0.02 rev
      silhouette(),
    );
    expect(out.coverage).toBeCloseTo(0.02, 5);
  });

  test('caps a single frame contribution at MAX_PER_FRAME (0.05)', () => {
    // A huge flow (hand bump / label glint) must not jump the meter.
    const out = computeAngularProgress(
      inputs(),
      flow({ dxPx: Math.PI * 100 * 0.5 }), // would be 0.5 rev uncapped
      silhouette(),
    );
    expect(out.coverage).toBeCloseTo(0.05, 5);
  });

  test('coverage is monotonic across repeated forward frames', () => {
    let st = inputs();
    let prev = 0;
    for (let i = 0; i < 10; i++) {
      const out = computeAngularProgress(
        st,
        flow({ dxPx: Math.PI * 100 * 0.02 }),
        silhouette(),
      );
      expect(out.coverage).toBeGreaterThanOrEqual(prev);
      prev = out.coverage;
      st = { ...st, coverage: out.coverage, rotationDirection: out.rotationDirection };
    }
    expect(prev).toBeCloseTo(0.2, 4);
  });

  test('coverage never exceeds 1.0', () => {
    const out = computeAngularProgress(
      inputs({ coverage: 0.99, rotationDirection: 'cw' }),
      flow({ dxPx: Math.PI * 100 * 0.04 }),
      silhouette(),
    );
    expect(out.coverage).toBeLessThanOrEqual(1);
  });

  test('coverage never goes negative even on sustained backtrack', () => {
    const out = computeAngularProgress(
      inputs({ coverage: 0.0, rotationDirection: 'cw' }),
      flow({ dxPx: -Math.PI * 100 * 0.04 }), // reverse motion
      silhouette(),
    );
    expect(out.coverage).toBeGreaterThanOrEqual(0);
  });
});

describe('computeAngularProgress — direction commitment', () => {
  test('stays uncommitted below the ~5° threshold', () => {
    // First tiny forward frame: cumulative coverage stays under the
    // 5/360 commit threshold, so direction should remain null.
    const out = computeAngularProgress(
      inputs(),
      flow({ dxPx: Math.PI * 100 * 0.005 }), // 0.005 rev < 0.0139
      silhouette(),
    );
    expect(out.rotationDirection).toBeNull();
  });

  test('commits clockwise once cumulative motion passes the threshold', () => {
    const out = computeAngularProgress(
      inputs(),
      flow({ dxPx: Math.PI * 100 * 0.02 }), // 0.02 rev > 0.0139
      silhouette(),
    );
    expect(out.rotationDirection).toBe('cw');
  });

  test('commits counter-clockwise for negative flow past the threshold', () => {
    const out = computeAngularProgress(
      inputs(),
      flow({ dxPx: -Math.PI * 100 * 0.02 }),
      silhouette(),
    );
    expect(out.rotationDirection).toBe('ccw');
  });

  test('half-weights motion against the committed direction', () => {
    // Committed cw, then a reverse frame. The cylinder delta is 0.04
    // rev; backtrack weight halves it to 0.02 before the cap, so
    // coverage drops by 0.02 from 0.50.
    const out = computeAngularProgress(
      inputs({ coverage: 0.5, rotationDirection: 'cw' }),
      flow({ dxPx: -Math.PI * 100 * 0.04 }),
      silhouette(),
    );
    expect(out.coverage).toBeCloseTo(0.48, 5);
    // Direction commitment is sticky — a single reverse frame doesn't
    // flip it.
    expect(out.rotationDirection).toBe('cw');
  });
});

describe('computeAngularProgress — low-signal handling', () => {
  test('holds coverage steady when the silhouette is lost', () => {
    const out = computeAngularProgress(
      inputs({ coverage: 0.42, rotationDirection: 'cw', angularVelocity: 0.3 }),
      flow({ dxPx: 50 }),
      silhouette({ detected: false }),
    );
    expect(out.coverage).toBe(0.42);
    // Velocity decays toward 0 for a smooth UI handoff.
    expect(out.angularVelocity).toBeLessThan(0.3);
    expect(out.angularVelocity).toBeGreaterThan(0);
  });

  test('holds coverage steady when flow confidence is below MIN_CONFIDENCE', () => {
    const out = computeAngularProgress(
      inputs({ coverage: 0.42 }),
      flow({ dxPx: 50, confidence: 0.1 }), // < 0.25
      silhouette(),
    );
    expect(out.coverage).toBe(0.42);
  });

  test('holds coverage steady when widthPx is non-positive', () => {
    const out = computeAngularProgress(
      inputs({ coverage: 0.3 }),
      flow({ dxPx: 50 }),
      silhouette({ widthPx: 0 }),
    );
    expect(out.coverage).toBe(0.3);
  });
});

describe('computeAngularProgress — flowQuality EMA', () => {
  test('rolls a high-confidence sample up from zero', () => {
    const out = computeAngularProgress(inputs({ flowQuality: 0 }), flow({ confidence: 1 }), silhouette());
    // EMA with alpha 0.2 from 0 toward 1: 0.2.
    expect(out.flowQuality).toBeCloseTo(0.2, 5);
  });

  test('decays toward zero when the bottle is lost (untrackable signal)', () => {
    // flowQuality must keep decaying even on a dropped frame so the
    // state machine can fault into untrackable_surface.
    const out = computeAngularProgress(
      inputs({ flowQuality: 0.8 }),
      flow({ confidence: 0 }),
      silhouette({ detected: false }),
    );
    expect(out.flowQuality).toBeCloseTo(0.64, 5); // 0.8 * (1 - 0.2)
    expect(out.flowQuality).toBeLessThan(0.8);
  });
});

describe('computeAngularProgress — angular velocity', () => {
  test('ignores an absurd dt and keeps the prior estimate as the instant term', () => {
    // dt >= 1s is rejected → instantaneous velocity falls back to the
    // prior, so the EMA stays put.
    const out = computeAngularProgress(
      inputs({ angularVelocity: 0.5, dtSec: 5 }),
      flow({ dxPx: Math.PI * 100 * 0.02 }),
      silhouette(),
    );
    expect(out.angularVelocity).toBeCloseTo(0.5, 5);
  });

  test('produces a positive velocity for forward motion with a sane dt', () => {
    const out = computeAngularProgress(
      inputs({ angularVelocity: 0, dtSec: 1 / 30 }),
      flow({ dxPx: Math.PI * 100 * 0.02 }),
      silhouette(),
    );
    expect(out.angularVelocity).toBeGreaterThan(0);
  });
});
