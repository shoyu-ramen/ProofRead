/**
 * useScanStateMachine — wires the tracker's shared values to the
 * `scanReducer` (ARCH §3).
 *
 * One `useAnimatedReaction` watches frameTick + tracker fields,
 * derives the JS-side `ScanMachineInputs`, and dispatches a `tick`.
 * The reducer's pure transitions decide the next discrete `ScanState`,
 * which the unwrap screen consumes for the silhouette / ring / dial /
 * instruction overlays.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';
import {
  runOnJS,
  useAnimatedReaction,
  type SharedValue,
} from 'react-native-reanimated';

import type { TrackerState } from '@src/scan/tracker';
import {
  INITIAL_SCAN_STATE,
  scanReducer,
  type FailReason,
  type PauseReason,
  type ScanAction,
  type ScanMachineInputs,
  type ScanState,
} from '@src/scan/state/scanMachine';

// Tunables — see ARCH §3 transitions.
const STEADY_MS = 500;
const STEADY_SCORE_MIN = 0.6;
// Below this angular velocity (revs/sec) for ≥STALL_MS while in
// `scanning` triggers `too_slow`.
const MIN_REV_PER_SEC = 0.05; // ~18°/sec; <2° in 100ms ≈ stalled
const STALL_MS = 2000;
// Audit finding: on ready→scanning transition, the first frame can
// have velocity < MIN_REV_PER_SEC (direction not yet committed). On
// a loaded JS thread where frames arrive 2s apart, `too_slow` would
// fire immediately. Suppress stall detection for this grace window
// after entering `scanning`.
const STALL_GRACE_MS = 600;
// Above this we treat as "too fast" — cap from ARCH §3.
const MAX_REV_PER_SEC = 1.2; // ~432°/sec; protects optical-flow precision

export interface UseScanStateMachineResult {
  state: ScanState;
  /** Imperative escape hatches for the parent screen. */
  fail: (reason: FailReason) => void;
  complete: (panoramaUri: string) => void;
  cancel: () => void;
  reset: () => void;
}

/**
 * Hook.
 *
 * @param trackerStateSv  the tracker's TrackerState shared value
 * @param frameTickSv     monotonic frame counter — change-driver for the reaction
 */
export function useScanStateMachine(
  trackerStateSv: SharedValue<TrackerState>,
  frameTickSv: SharedValue<number>,
): UseScanStateMachineResult {
  const [state, dispatch] = useReducer(scanReducer, INITIAL_SCAN_STATE);

  // We track "bottle steady since" on the JS side so the worklet
  // doesn't need to plumb an integrator. Reset to null whenever the
  // detector loses lock.
  const steadySinceRef = useRef<number | null>(null);
  // For `too_slow`: when did we drop below MIN_REV_PER_SEC? null when
  // we're rotating fast enough.
  const stalledSinceRef = useRef<number | null>(null);
  // Wall-clock ms when `scanning` was last entered. Audit finding:
  // suppress `too_slow` for STALL_GRACE_MS after the transition so a
  // first-frame velocity dip doesn't immediately pause the scan.
  // null whenever we're not in `scanning`.
  const scanningEnteredAtRef = useRef<number | null>(null);

  const dispatchTick = useCallback(
    (inputs: ScanMachineInputs) => {
      dispatch({ type: 'tick', inputs });
    },
    [],
  );

  const applyInputs = useCallback(
    (raw: {
      steadyNow: boolean;
      coverage: number;
      velocity: number;
      pauseReason: PauseReason | null;
    }) => {
      // Read the timestamp here, after the runOnJS hop. Reading it in
      // the worklet would let the value drift by 50–100ms under JS
      // back-pressure, throwing off the STEADY_MS / STALL_MS windows.
      const nowMs = Date.now();

      // Steadiness integrator on the JS side: bottleSteady fires only
      // after STEADY_MS of continuous detection.
      if (raw.steadyNow) {
        if (steadySinceRef.current === null) steadySinceRef.current = nowMs;
      } else {
        steadySinceRef.current = null;
      }
      const bottleSteady =
        steadySinceRef.current !== null &&
        nowMs - steadySinceRef.current >= STEADY_MS;

      // Slow-rotation integrator: only emit `too_slow` after the
      // STALL_MS window. Reset on first accept-rate frame.
      let pauseReason = raw.pauseReason;
      const rotatingNow = raw.velocity >= MIN_REV_PER_SEC;
      // Stall-detection grace window: skip too_slow for STALL_GRACE_MS
      // after entering `scanning` so first-frame zero-velocity doesn't
      // immediately fault the scan (audit finding).
      const enteredAt = scanningEnteredAtRef.current;
      const inGrace =
        enteredAt !== null && nowMs - enteredAt < STALL_GRACE_MS;
      if (raw.coverage > 0 && !rotatingNow && pauseReason === null) {
        if (stalledSinceRef.current === null) stalledSinceRef.current = nowMs;
        if (
          !inGrace &&
          nowMs - stalledSinceRef.current >= STALL_MS
        ) {
          pauseReason = 'too_slow';
        }
      } else {
        stalledSinceRef.current = null;
      }

      dispatchTick({
        bottleSteady,
        coverage: raw.coverage,
        rotating: rotatingNow,
        pauseReason,
      });
    },
    [dispatchTick],
  );

  /**
   * The reaction observes frameTick (monotonic) and rebuilds the
   * inputs from the tracker state on every accepted frame. We never
   * dispatch from the worklet directly; everything is `runOnJS`.
   */
  useAnimatedReaction(
    () => frameTickSv.value,
    (tick, prev) => {
      'worklet';
      if (tick === 0 || tick === prev) return;
      const ts = trackerStateSv.value;

      // Steadiness gate: detected + above-threshold steadiness.
      const steadyNow =
        ts.silhouette.detected &&
        ts.silhouette.steadinessScore >= STEADY_SCORE_MIN;

      // Pre-check verdict → pause reason. `motion` and the
      // visual-quality reasons all funnel into pause.
      let pauseReason: PauseReason | null = null;
      if (ts.preCheck.kind === 'warn') {
        if (
          ts.preCheck.reason === 'blur' ||
          ts.preCheck.reason === 'glare' ||
          ts.preCheck.reason === 'motion'
        ) {
          pauseReason = ts.preCheck.reason;
        }
      }
      // Distance feedback (too_far / too_close) — the tracker only
      // emits these when the silhouette is locked, so we don't need a
      // detected gate here. They run *before* the lost-bottle override
      // because they're a more specific story when we still have a
      // lock.
      if (ts.coverageStatus !== null) {
        pauseReason = ts.coverageStatus;
      }
      if (!ts.silhouette.detected && ts.coverage > 0) {
        // Lost-bottle takes precedence over a pre-check chip.
        pauseReason = 'lost_bottle';
      }
      // Rotation-rate sanity. Slow-rotation hysteresis lives on the
      // JS side; the worklet only emits the immediate "too_fast".
      const v = Math.abs(ts.angularVelocity);
      if (v > MAX_REV_PER_SEC) pauseReason = 'too_fast';

      runOnJS(applyInputs)({
        steadyNow,
        coverage: ts.coverage,
        velocity: v,
        pauseReason,
      });
    },
    // Empty deps — the reaction body captures the SharedValue refs
    // (which are stable across renders); listing them in deps would
    // trigger Reanimated's "_value from JS" guard via Object.is.
    [],
  );

  // Stable imperative escape hatches.
  const fail = useCallback((reason: FailReason) => {
    dispatch({ type: 'fail', reason } satisfies ScanAction);
  }, []);
  const complete = useCallback((panoramaUri: string) => {
    dispatch({ type: 'complete', panoramaUri } satisfies ScanAction);
  }, []);
  const cancel = useCallback(() => {
    dispatch({ type: 'cancel' } satisfies ScanAction);
  }, []);
  const reset = useCallback(() => {
    dispatch({ type: 'reset' } satisfies ScanAction);
    steadySinceRef.current = null;
    stalledSinceRef.current = null;
    scanningEnteredAtRef.current = null;
  }, []);

  // Track `scanning` entry/exit so applyInputs can apply the
  // STALL_GRACE_MS suppression window. Stamp on entry; clear on every
  // transition away from `scanning` (so a paused→scanning bounce
  // restarts the grace window, which is the right behavior — the
  // user is effectively "starting" again).
  useEffect(() => {
    if (state.kind === 'scanning') {
      if (scanningEnteredAtRef.current === null) {
        scanningEnteredAtRef.current = Date.now();
      }
    } else {
      scanningEnteredAtRef.current = null;
    }
  }, [state.kind]);

  // Cleanup integrator state on unmount so a re-mounted scan doesn't
  // inherit a stale "steady since" timestamp.
  useEffect(() => {
    return () => {
      steadySinceRef.current = null;
      stalledSinceRef.current = null;
      scanningEnteredAtRef.current = null;
    };
  }, []);

  return useMemo(
    () => ({ state, fail, complete, cancel, reset }),
    [state, fail, complete, cancel, reset],
  );
}
