/**
 * Angular-progress integrator (ARCH §4.4).
 *
 * Pure helper: turn a per-frame horizontal-flow measurement into a
 * monotonic coverage delta plus an angular-velocity estimate. The
 * model is the simple cylinder one — visible width-in-pixels equals
 * diameter, so circumference is π × widthPx and one frame's coverage
 * delta is dxPx / (π × widthPx) revolutions.
 *
 * The integrator commits to a rotation direction after the first
 * ~5° of cumulative motion; thereafter, motion in the committed
 * direction adds full weight, motion opposite is half-weight (so
 * users can correct minor over-rotation without gaming the meter).
 */

import type {
  BottleSilhouette,
  FlowMeasurement,
  TrackerState,
} from './types';

// One frame can't add more than this. Caps spurious flows from
// label glints, hand bumps, or detector jumps. 0.05 ≈ 18°.
const MAX_PER_FRAME = 0.05;

// Direction stays uncommitted until cumulative absolute coverage
// exceeds this. 5° / 360° ≈ 0.0139.
const DIRECTION_COMMIT_THRESHOLD = 5 / 360;

// Penalty applied to motion against the committed direction.
const BACKTRACK_WEIGHT = 0.5;

// EMA factor for angular-velocity smoothing.
const VELOCITY_ALPHA = 0.3;

// Minimum confidence to accept a flow measurement at all. Below this
// the bottle is probably stationary or the templates couldn't agree.
const MIN_CONFIDENCE = 0.25;

// Exposed input to keep this helper independent of the worklet's
// shared-value plumbing. The frameProcessor builds it from its
// previous TrackerState plus the current dt.
export interface AngleTrackerInputs {
  coverage: number;
  rotationDirection: TrackerState['rotationDirection'];
  angularVelocity: number;
  dtSec: number;
}

export interface AngleTrackerOutputs {
  coverage: number;
  angularVelocity: number;
  rotationDirection: TrackerState['rotationDirection'];
}

/**
 * Update the angular-progress accumulator. Called once per accepted
 * frame in the worklet. The function is pure and worklet-safe.
 */
export function computeAngularProgress(
  state: AngleTrackerInputs,
  flow: FlowMeasurement,
  silhouette: BottleSilhouette,
): AngleTrackerOutputs {
  'worklet';

  // No silhouette or no confident flow → coverage holds steady.
  // Velocity decays toward zero so the UI gets a smooth handoff.
  if (
    !silhouette.detected ||
    silhouette.widthPx <= 0 ||
    flow.confidence < MIN_CONFIDENCE
  ) {
    return {
      coverage: state.coverage,
      angularVelocity: state.angularVelocity * (1 - VELOCITY_ALPHA),
      rotationDirection: state.rotationDirection,
    };
  }

  // Per-frame revolution fraction, signed.
  let dRev = flow.dxPx / (Math.PI * silhouette.widthPx);

  // Direction commitment. Until the user has rotated past the
  // threshold, we read both directions; once committed, opposite-sign
  // motion is back-tracking.
  let direction = state.rotationDirection;
  if (direction === null) {
    if (state.coverage + Math.abs(dRev) >= DIRECTION_COMMIT_THRESHOLD) {
      direction = dRev >= 0 ? 'cw' : 'ccw';
    }
  }

  let signed = dRev;
  if (direction !== null) {
    const expectSign = direction === 'cw' ? 1 : -1;
    if (Math.sign(dRev) !== expectSign) signed = dRev * BACKTRACK_WEIGHT;
  }

  // Cap per-frame contribution. Sign-preserving so back-tracking can
  // walk coverage down a bit.
  const capped = clamp(signed, -MAX_PER_FRAME, MAX_PER_FRAME);

  // Coverage is monotonic-by-default but allowed to bleed back a bit
  // when the user back-tracks. Floor at 0 so we never go negative.
  const nextCoverage = Math.max(
    0,
    Math.min(1, state.coverage + Math.abs(capped) * Math.sign(signed)),
  );

  // Angular velocity in revs/sec. dt comes from the worklet clock;
  // if it's missing or absurd we keep the prior estimate.
  const safeDt = state.dtSec > 0 && state.dtSec < 1 ? state.dtSec : 0;
  const instantaneousVel = safeDt > 0 ? dRev / safeDt : state.angularVelocity;
  const nextVel =
    state.angularVelocity * (1 - VELOCITY_ALPHA) +
    instantaneousVel * VELOCITY_ALPHA;

  return {
    coverage: nextCoverage,
    angularVelocity: nextVel,
    rotationDirection: direction,
  };
}

function clamp(x: number, lo: number, hi: number): number {
  'worklet';
  return x < lo ? lo : x > hi ? hi : x;
}
