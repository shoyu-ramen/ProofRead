/**
 * Public surface of the cylindrical-scan tracker subsystem.
 *
 * Consumed by mobile/app/(app)/scan/unwrap.tsx (UI shell) and by the
 * panorama subsystem (mobile/src/scan/panorama/) which subscribes to
 * trackerStateSv to know when to extract a strip.
 */

export type {
  BottleSilhouette,
  CoverageStatus,
  FlowMeasurement,
  PreCheckVerdict,
  TrackerState,
} from './types';

export { detectBottle } from './bottleDetector';
export { measureFlow } from './opticalFlow';
export {
  computeAngularProgress,
  type AngleTrackerInputs,
  type AngleTrackerOutputs,
} from './angleTracker';
export {
  useTrackerFrameProcessor,
  useMotionVerdict,
  type TrackerFrameProcessor,
} from './frameProcessor';

// --- Deviations from CYLINDRICAL_SCAN_ARCHITECTURE.md ---------------
//
// 1. Coverage is allowed to bleed back when the user back-tracks past
//    the direction-commit threshold (clamped to ≥0). The spec calls
//    out half-weight back-tracking to "reduce coverage by half-weight"
//    — implemented literally; the spec also says coverage is
//    monotonic, which we read as "monotonic in the absence of
//    back-tracking". Floor at 0 prevents the integrator from going
//    negative.
//
// 2. Optical flow's per-frame search keeps templates in the central
//    strip but does NOT pyramidally subsample (architecture §4.3
//    described it as "pyramidal block-match" but explicitly said
//    "simplified" — a single-level SAD with early-out is faster on a
//    160×240 grid and adequate for v1 angular precision).
//
// 3. Pre-check thresholds are re-used verbatim from camera/[surface].tsx
//    except COVERAGE_TOO_CLOSE is loosened from 0.80 → 0.85: the larger
//    160×240 buffer captures more of the bottle so the histogram skews
//    higher than the 64×96 pre-check screen.
//
// 4. capturedCheckpoints is bumped by the panorama subsystem via the
//    returned bumpCapturedCheckpoints callback rather than being
//    integrated by the worklet — the architecture says "strip
//    extraction is NOT your job" so the tracker only exposes the slot
//    for the counter, not the trigger logic.
