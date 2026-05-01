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

import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react';
import {
  runOnJS,
  useAnimatedReaction,
  type SharedValue,
} from 'react-native-reanimated';

import type { TrackerState } from '@src/scan/tracker';
import {
  evaluateAutoCaptureTick,
  INITIAL_AUTO_CAPTURE_TIMER,
  INITIAL_SCAN_STATE,
  scanReducer,
  type AutoCaptureTimerState,
  type DetectedContainerType,
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

// Untrackable-surface hysteresis: if the EMA of optical-flow confidence
// stays below UNTRACKABLE_THRESHOLD for UNTRACKABLE_MS while the user
// is mid-scan (coverage > 0), surface a pause that asks for a labeled
// container. Threshold sits below the angle integrator's per-frame
// MIN_CONFIDENCE (0.25) so a few transient bad frames don't trip it.
const UNTRACKABLE_THRESHOLD = 0.18;
const UNTRACKABLE_MS = 2500;

// Auto-capture gate constants live in scanMachine.ts so tests can
// exercise them against the same numbers the hook uses. See
// `evaluateAutoCaptureTick` for the pure timer math.

export interface UseScanStateMachineResult {
  state: ScanState;
  /** Imperative escape hatches for the parent screen. */
  fail: (reason: FailReason) => void;
  complete: (panoramaUri: string) => void;
  cancel: () => void;
  reset: () => void;
  /**
   * True iff the auto-capture predicate is currently true (preCheck
   * ready, container & grip confidence both above the gate). Goes
   * true the moment the predicate trips; flips back as soon as any
   * leg drops. Use this to render the "Hold steady — starting in
   * 1.5s" countdown pill — it visualizes the live gate state, not
   * the latched verdict that drives the transition.
   */
  autoCaptureCandidate: boolean;
  /**
   * True after MANUAL_OVERRIDE_AFTER_MS of being in `ready` without
   * the auto-capture predicate sustaining. Drives the "Tap to start
   * manually" button — flips back to false on any state transition
   * away from `ready`.
   */
  manualOverrideAvailable: boolean;
  /**
   * True iff the auto-capture timer has latched (the predicate held
   * for ≥ AUTO_CAPTURE_HOLD_MS). The unwrap screen consumes this to
   * decide when to snapshot the frame and POST it to the
   * /v1/detect-container gate — it's the *trigger* signal, distinct
   * from `autoCaptureCandidate` which only flips per-frame.
   */
  autoCaptureLatched: boolean;
  /** Dispatch a manual-start action (consumes the override button). */
  manualStart: () => void;
  /**
   * Dispatch a `requestConfirmation` action with a captured snapshot
   * URI — drives `ready → confirming{phase:'detecting'}`. Called by
   * the unwrap screen once `cameraRef.current.takePhoto()` resolves.
   */
  requestConfirmation: (snapshotUri: string) => void;
  /**
   * Dispatch the backend response — drives `confirming{detecting} →
   * confirming{detected|failed}`. The unwrap screen calls this after
   * `apiClient.detectContainer()` resolves (or rejects).
   */
  detectionResolved: (
    result:
      | {
          detected: true;
          bbox: [number, number, number, number];
          containerType: DetectedContainerType;
        }
      | { detected: false; reason: string | null },
  ) => void;
  /** User tapped "Start scan" while in `confirming{detected}`. */
  confirmStart: () => void;
  /** User tapped "Reshoot" / "Retry" while in any `confirming` sub-phase. */
  confirmRetry: () => void;
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
  // For `untrackable_surface`: when did flowQuality first drop below
  // UNTRACKABLE_THRESHOLD while the user was already mid-scan? null
  // when flow is healthy.
  const untrackableSinceRef = useRef<number | null>(null);
  // Wall-clock ms when `scanning` was last entered. Audit finding:
  // suppress `too_slow` for STALL_GRACE_MS after the transition so a
  // first-frame velocity dip doesn't immediately pause the scan.
  // null whenever we're not in `scanning`.
  const scanningEnteredAtRef = useRef<number | null>(null);
  // Auto-capture timer state — held in a ref so it persists across
  // ticks without driving a re-render every frame. The pure logic
  // lives in `evaluateAutoCaptureTick`; this ref is just where we
  // stash the previous tick's `next` so the test contract and the
  // hook agree on the maths.
  const autoCaptureTimerRef = useRef<AutoCaptureTimerState>(
    INITIAL_AUTO_CAPTURE_TIMER,
  );
  // Wall-clock ms when `ready` was last entered. Drives the manual-
  // override fallback timer. null whenever we're not in `ready`.
  const readyEnteredAtRef = useRef<number | null>(null);
  // We need the FSM `kind` inside `applyInputs` to decide whether the
  // manual-override timer is active. `state` is captured inside the
  // closure but the runOnJS callback may carry stale closure state on
  // hot paths; a ref keeps the read O(1) and current.
  const stateKindRef = useRef<ScanState['kind']>(INITIAL_SCAN_STATE.kind);
  // Live JS-side mirror of the auto-capture predicate (per-tick value
  // before the 1500ms hold completes). Mirrored as React state so the
  // unwrap screen can render the "starting in 1.5s" countdown pill
  // without sampling the trackerStateSv shared value directly.
  const [autoCaptureCandidate, setAutoCaptureCandidate] = useState(false);
  // Latched manual-override flag. Once set, stays set until the user
  // leaves `ready` (manualStart, lost-bottle drop to aligning, fail).
  const [manualOverrideAvailable, setManualOverrideAvailable] = useState(false);
  // Latched auto-capture flag — flips true the tick the 1.5s hold
  // completes and stays true until the FSM leaves `ready`. The unwrap
  // screen reads this to know when to snapshot a frame and POST to
  // `/v1/detect-container`. Distinct from `autoCaptureCandidate`,
  // which is per-frame.
  const [autoCaptureLatched, setAutoCaptureLatched] = useState(false);

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
      flowQuality: number;
      pauseReason: PauseReason | null;
      preCheckReady: boolean;
      containerConfidence: number;
      gripSteadiness: number;
      containerClassPresent: boolean;
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

      // Untrackable-surface integrator: only fault when the EMA stays
      // below UNTRACKABLE_THRESHOLD for UNTRACKABLE_MS *and* the user
      // is mid-scan (coverage > 0). Like too_slow, lower priority than
      // any worklet-side reason — a transient blur / lost_bottle has a
      // more specific fix and should win the copy slot.
      if (
        raw.coverage > 0 &&
        raw.flowQuality < UNTRACKABLE_THRESHOLD &&
        pauseReason === null
      ) {
        if (untrackableSinceRef.current === null) {
          untrackableSinceRef.current = nowMs;
        }
        if (nowMs - untrackableSinceRef.current >= UNTRACKABLE_MS) {
          pauseReason = 'untrackable_surface';
        }
      } else {
        untrackableSinceRef.current = null;
      }

      // Auto-capture + manual-override timing — pure tick that the
      // unit tests exercise directly. We keep the persistent state in
      // a ref so the next tick sees the previous candidateSince /
      // latched flag.
      const autoTick = evaluateAutoCaptureTick(autoCaptureTimerRef.current, {
        nowMs,
        preCheckReady: raw.preCheckReady,
        containerConfidence: raw.containerConfidence,
        gripSteadiness: raw.gripSteadiness,
        containerClassPresent: raw.containerClassPresent,
        inReady: stateKindRef.current === 'ready',
        readyEnteredAtMs: readyEnteredAtRef.current,
      });
      autoCaptureTimerRef.current = autoTick.next;

      // Mirror the live candidate flag into React state for the
      // countdown pill. Wrap in a setState that no-ops on equality so
      // we don't re-render at frame rate.
      setAutoCaptureCandidate((prev) =>
        prev === autoTick.candidate ? prev : autoTick.candidate,
      );
      // Latch the manual-override flag in React state so the UI re-
      // renders on the transition. The pure helper already dedupes
      // (latched stays latched) but we still gate the setState behind
      // an equality check to avoid a no-op render.
      setManualOverrideAvailable((prev) =>
        prev === autoTick.manualOverrideAvailable
          ? prev
          : autoTick.manualOverrideAvailable,
      );
      // Same for the latched auto-capture flag — drives the unwrap
      // screen's snapshot-and-detect trigger. Pre-capture gate: we no
      // longer pass `autoTick.ready` straight through to the reducer
      // as `autoCaptureReady`; the reducer's `ready → scanning` edge
      // would skip the confirmation gate. Instead the unwrap screen
      // watches `autoCaptureLatched` and dispatches
      // `requestConfirmation` (with a snapshot URI) when it flips.
      setAutoCaptureLatched((prev) =>
        prev === autoTick.ready ? prev : autoTick.ready,
      );

      dispatchTick({
        bottleSteady,
        coverage: raw.coverage,
        rotating: rotatingNow,
        pauseReason,
        // Pre-capture gate: autoCaptureReady is force-set to false in
        // the reducer's tick path so the FSM never auto-skips the
        // `confirming` state. The unwrap screen reads
        // `autoCaptureLatched` (above) to drive the snapshot+detect
        // pipeline instead.
        autoCaptureReady: false,
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

      // Auto-capture inputs — fed through to the JS-side timer in
      // applyInputs. preCheckReady is the simple kind===ready check;
      // the worklet doesn't need to know about the 0.85 confidence
      // floor (that lives in JS so it's tunable without a worklet
      // recompile).
      const preCheckReady = ts.preCheck.kind === 'ready';
      const containerConfidence = ts.silhouette.containerConfidence;
      const gripSteadiness = ts.gripSteadiness;
      // Defense in depth alongside the 0.85 gate: a silhouette that
      // doesn't classify as bottle/can must not fire the snapshot.
      const containerClassPresent = ts.silhouette.class !== null;

      runOnJS(applyInputs)({
        steadyNow,
        coverage: ts.coverage,
        velocity: v,
        flowQuality: ts.flowQuality,
        pauseReason,
        preCheckReady,
        containerConfidence,
        gripSteadiness,
        containerClassPresent,
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
    untrackableSinceRef.current = null;
    scanningEnteredAtRef.current = null;
    autoCaptureTimerRef.current = INITIAL_AUTO_CAPTURE_TIMER;
    readyEnteredAtRef.current = null;
    setAutoCaptureCandidate(false);
    setManualOverrideAvailable(false);
    setAutoCaptureLatched(false);
  }, []);
  const manualStart = useCallback(() => {
    // Pre-capture gate: manualStart is no longer the trigger that
    // drives `ready → scanning`. The unwrap screen handles the manual
    // tap by snapshotting the frame and dispatching
    // `requestConfirmation` itself; this dispatch is now a no-op
    // safety net for callers that wired up the legacy contract. The
    // timer reset below stays — when the user hits the override they
    // shouldn't inherit stale window state.
    dispatch({ type: 'manualStart' } satisfies ScanAction);
    autoCaptureTimerRef.current = INITIAL_AUTO_CAPTURE_TIMER;
    readyEnteredAtRef.current = null;
    setAutoCaptureCandidate(false);
    setManualOverrideAvailable(false);
    setAutoCaptureLatched(false);
  }, []);
  const requestConfirmation = useCallback((snapshotUri: string) => {
    dispatch({
      type: 'requestConfirmation',
      snapshotUri,
    } satisfies ScanAction);
    // The user is now mid-confirmation; clear the auto-capture timers
    // so a `confirmRetry` (back to `aligning`) gives a clean re-arm
    // window when they next reach `ready`.
    autoCaptureTimerRef.current = INITIAL_AUTO_CAPTURE_TIMER;
    setAutoCaptureCandidate(false);
    setAutoCaptureLatched(false);
  }, []);
  const detectionResolved = useCallback(
    (
      result:
        | {
            detected: true;
            bbox: [number, number, number, number];
            containerType: DetectedContainerType;
          }
        | { detected: false; reason: string | null },
    ) => {
      dispatch({ type: 'detectionResolved', result } satisfies ScanAction);
    },
    [],
  );
  const confirmStart = useCallback(() => {
    dispatch({ type: 'confirmStart' } satisfies ScanAction);
  }, []);
  const confirmRetry = useCallback(() => {
    dispatch({ type: 'confirmRetry' } satisfies ScanAction);
    // Mirrors `manualStart` / `reset`: dropping back to `aligning`
    // resets the integrators so the next `ready` window starts clean.
    steadySinceRef.current = null;
    autoCaptureTimerRef.current = INITIAL_AUTO_CAPTURE_TIMER;
    readyEnteredAtRef.current = null;
    setAutoCaptureCandidate(false);
    setManualOverrideAvailable(false);
    setAutoCaptureLatched(false);
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

  // Track `ready` entry/exit for the auto-capture + manual-override
  // timers. Stamp on entry; clear on exit. Manual-override flag is
  // tied to this lifecycle — once we leave `ready` it should
  // disappear (the button is no longer relevant), and on re-entry
  // (e.g. paused → ready isn't a thing in the current FSM, but
  // aligning → ready will be) we want a fresh 8s window.
  useEffect(() => {
    stateKindRef.current = state.kind;
    if (state.kind === 'ready') {
      readyEnteredAtRef.current = Date.now();
      // Clear the latched manual-override flag — a fresh ready window
      // gets a fresh 8s timer.
      autoCaptureTimerRef.current = {
        ...autoCaptureTimerRef.current,
        manualOverrideLatched: false,
      };
      setManualOverrideAvailable(false);
    } else {
      readyEnteredAtRef.current = null;
      // On exit from `ready`, clear the live candidate so the UI
      // pill doesn't briefly show a "starting in 1.5s" cue while we
      // animate into `scanning`.
      setAutoCaptureCandidate(false);
      setManualOverrideAvailable(false);
      // Clear the latched-trigger flag too — once we've left `ready`
      // (into `confirming`, `scanning`, or back to `aligning` from a
      // lost-bottle drop) the snapshot trigger should not re-fire on
      // re-entry until the predicate sustains again.
      setAutoCaptureLatched(false);
      // Reset the auto-capture sustain timer so the next entry to
      // `ready` (after a lost-bottle drop, say) doesn't inherit a
      // partial hold or a stale latched override.
      autoCaptureTimerRef.current = INITIAL_AUTO_CAPTURE_TIMER;
    }
  }, [state.kind]);

  // Cleanup integrator state on unmount so a re-mounted scan doesn't
  // inherit a stale "steady since" timestamp.
  useEffect(() => {
    return () => {
      steadySinceRef.current = null;
      stalledSinceRef.current = null;
      untrackableSinceRef.current = null;
      scanningEnteredAtRef.current = null;
      autoCaptureTimerRef.current = INITIAL_AUTO_CAPTURE_TIMER;
      readyEnteredAtRef.current = null;
    };
  }, []);

  return useMemo(
    () => ({
      state,
      fail,
      complete,
      cancel,
      reset,
      autoCaptureCandidate,
      manualOverrideAvailable,
      autoCaptureLatched,
      manualStart,
      requestConfirmation,
      detectionResolved,
      confirmStart,
      confirmRetry,
    }),
    [
      state,
      fail,
      complete,
      cancel,
      reset,
      autoCaptureCandidate,
      manualOverrideAvailable,
      autoCaptureLatched,
      manualStart,
      requestConfirmation,
      detectionResolved,
      confirmStart,
      confirmRetry,
    ],
  );
}
