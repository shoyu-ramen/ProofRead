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
      nowMs: number;
    }) => {
      // Steadiness integrator on the JS side: bottleSteady fires only
      // after STEADY_MS of continuous detection.
      if (raw.steadyNow) {
        if (steadySinceRef.current === null) steadySinceRef.current = raw.nowMs;
      } else {
        steadySinceRef.current = null;
      }
      const bottleSteady =
        steadySinceRef.current !== null &&
        raw.nowMs - steadySinceRef.current >= STEADY_MS;

      // Slow-rotation integrator: only emit `too_slow` after the
      // STALL_MS window. Reset on first accept-rate frame.
      let pauseReason = raw.pauseReason;
      const rotatingNow = raw.velocity >= MIN_REV_PER_SEC;
      if (raw.coverage > 0 && !rotatingNow && pauseReason === null) {
        if (stalledSinceRef.current === null) stalledSinceRef.current = raw.nowMs;
        if (raw.nowMs - stalledSinceRef.current >= STALL_MS) {
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
        nowMs: Date.now(),
      });
    },
    [trackerStateSv, frameTickSv],
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
  }, []);

  // Cleanup integrator state on unmount so a re-mounted scan doesn't
  // inherit a stale "steady since" timestamp.
  useEffect(() => {
    return () => {
      steadySinceRef.current = null;
      stalledSinceRef.current = null;
    };
  }, []);

  return useMemo(
    () => ({ state, fail, complete, cancel, reset }),
    [state, fail, complete, cancel, reset],
  );
}
