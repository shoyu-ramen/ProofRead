/**
 * Scan state machine (ARCH §3) — the heart of unwrap.tsx.
 *
 * Local to the unwrap screen. Owns the discriminated `ScanState` and
 * the transition logic; consumed via `useScanStateMachine` to drive
 * UI choices (silhouette color, instruction copy, capture trigger).
 *
 * Transitions are derived from tracker shared values via a
 * `useAnimatedReaction` in `useScanStateMachine`; the machine itself is
 * pure JS (`useReducer`) so consumers always read the most recent
 * committed transition, not a worklet-side mutation that hasn't
 * reached the renderer.
 */

/**
 * Terminal failure reasons. Canonical here so the reducer's discriminated
 * union and the UI's instruction copy share a single source of truth —
 * previously this lived in `scan/ui/ScanInstructions.tsx`, which created
 * an import cycle (state ← ui ← state) and broke `node:test` runs that
 * try to load the reducer in isolation.
 */
export type FailReason =
  | 'permission_denied'
  | 'no_camera'
  | 'capture_error'
  | 'cancelled';

export type ScanStateKind =
  | 'aligning'
  | 'ready'
  | 'confirming'
  | 'scanning'
  | 'paused'
  | 'complete'
  | 'failed';

/**
 * Sub-phase for the `confirming` state — the pre-capture gate inserted
 * between `ready` and `scanning`. The unwrap screen takes a snapshot,
 * POSTs `/v1/detect-container`, and renders the resulting bbox so the
 * user gets to confirm what the system sees before committing to a
 * 8–15s panorama capture.
 *
 *   - `detecting`: the snapshot has been taken (or is about to be);
 *     waiting on the backend response.
 *   - `detected`: backend located a container; render bbox + Start/Reshoot.
 *   - `failed`: backend couldn't find a container; show ErrorState +
 *     route the user back to `aligning` via `confirmRetry`.
 */
export type ConfirmingPhase = 'detecting' | 'detected' | 'failed';

/**
 * Container class echoed back from `POST /v1/detect-container`. Keep
 * this in sync with `DetectContainerResponse.container_type` in
 * `mobile/src/api/client.ts` — same union, copied here so the reducer
 * doesn't have to import from `@src/api`.
 */
export type DetectedContainerType = 'bottle' | 'can' | 'box';

/**
 * Pause-reason union — canonical here. ARCH §3 enumerated the original
 * six (`too_fast`, `too_slow`, `lost_bottle`, `blur`, `glare`,
 * `motion`); Stream 3 added `too_far` and `too_close` so the user gets
 * actionable copy when the bottle is at the wrong scan distance, even
 * before rotation starts. Phase 1 of the embodied-scan refresh added
 * `untrackable_surface` for sustained low optical-flow confidence
 * (clean glass, blank can — surfaces our heuristic flow can't latch
 * onto). `ScanInstructions` renders copy for all nine.
 */
export type PauseReason =
  | 'too_fast'
  | 'too_slow'
  | 'lost_bottle'
  | 'blur'
  | 'glare'
  | 'motion'
  | 'too_far'
  | 'too_close'
  | 'untrackable_surface';

export type ScanState =
  | { kind: 'aligning' }
  | { kind: 'ready' }
  | {
      kind: 'confirming';
      phase: ConfirmingPhase;
      /** File URI of the snapshot uploaded for detection. */
      snapshotUri?: string;
      /** Normalized [x0, y0, x1, y1] in 0..1 — present when phase==='detected'. */
      bbox?: [number, number, number, number];
      /** Container class returned by the backend; present when phase==='detected'. */
      containerType?: DetectedContainerType;
      /** Backend `reason` string for `phase==='failed'`. Used as ErrorState description. */
      failureReason?: string;
    }
  | { kind: 'scanning'; coverage: number }
  | { kind: 'paused'; coverage: number; reason: PauseReason }
  | { kind: 'complete'; panoramaUri: string }
  | { kind: 'failed'; reason: FailReason };

/**
 * Inputs threaded into the reducer on every tracker update. The
 * caller derives these from `trackerStateSv` once per UI tick (we
 * don't wake the reducer at worklet rate — the reaction throttles
 * itself by waiting on `frameTickSv`).
 */
export interface ScanMachineInputs {
  /** True once the silhouette has been detected stable for ≥500ms. */
  bottleSteady: boolean;
  /** Coverage from the tracker's monotonic integrator. */
  coverage: number;
  /** True iff angular velocity has been > MIN_RATE for a while. */
  rotating: boolean;
  /** Pause reason from rate / pre-check evaluation; null when fine. */
  pauseReason: PauseReason | null;
  /**
   * True once the auto-capture predicate (preCheck=ready,
   * containerConfidence>0.7, gripSteadiness>0.7) has been sustained
   * for ≥AUTO_CAPTURE_HOLD_MS. Drives the `ready → scanning`
   * transition without requiring the user to manually rotate first
   * (the original `rotating` trigger still works in parallel).
   *
   * Hysteresis lives in the hook (`useScanStateMachine`); the reducer
   * just consumes the latched verdict.
   */
  autoCaptureReady: boolean;
}

export type ScanAction =
  | { type: 'tick'; inputs: ScanMachineInputs }
  | { type: 'fail'; reason: FailReason }
  | { type: 'complete'; panoramaUri: string }
  | { type: 'cancel' }
  | { type: 'reset' }
  /**
   * User tap on the manual override button (exposed once
   * `manualOverrideAvailable` flips true in `useScanStateMachine`).
   * Forces a `ready → confirming` transition so the user can start
   * capture even when the auto-capture predicate never sustains —
   * dim labels, harsh backgrounds, anything that keeps the heuristic
   * confidence below the 0.85 gate.
   */
  | { type: 'manualStart' }
  /**
   * Pre-capture gate — the unwrap screen has snapshotted a frame and
   * is about to POST it to `/v1/detect-container`. Drives
   * `ready → confirming{phase: 'detecting'}`. The auto-trigger path
   * (1.5s sustain) and the manual-override path both funnel through
   * this action.
   */
  | { type: 'requestConfirmation'; snapshotUri: string }
  /**
   * Backend response landed (success or failure). Drives
   * `confirming{phase:'detecting'} → confirming{phase:'detected'|'failed'}`.
   */
  | {
      type: 'detectionResolved';
      result:
        | {
            detected: true;
            bbox: [number, number, number, number];
            containerType: DetectedContainerType;
          }
        | { detected: false; reason: string | null };
    }
  /**
   * User tapped "Start scan" while in `confirming{phase:'detected'}`.
   * Drives `confirming → scanning{coverage: 0}`.
   */
  | { type: 'confirmStart' }
  /**
   * User tapped "Reshoot" / "Retry" — either after a successful
   * detection (changed their mind) or after a failed detection.
   * Drives `confirming → aligning` and clears the snapshot so the
   * next entry into `confirming` starts fresh.
   */
  | { type: 'confirmRetry' };

export const INITIAL_SCAN_STATE: ScanState = { kind: 'aligning' };

/**
 * Reducer — pure transition table. Reads only the previous state and
 * the action; never the tracker shared values directly.
 *
 * The `complete` and `failed` states are terminal until an explicit
 * reset; a stray `tick` after completion does not back-step the UI.
 */
export function scanReducer(state: ScanState, action: ScanAction): ScanState {
  switch (action.type) {
    case 'reset':
      return INITIAL_SCAN_STATE;
    case 'fail':
      return { kind: 'failed', reason: action.reason };
    case 'complete':
      // Once complete, ignore further ticks.
      if (state.kind === 'complete') return state;
      return { kind: 'complete', panoramaUri: action.panoramaUri };
    case 'cancel':
      return { kind: 'failed', reason: 'cancelled' };
    case 'manualStart': {
      // Only meaningful from `ready` — anywhere else it would either
      // be a no-op (already scanning / paused / complete) or a regress
      // (aligning, no detection yet). The hook only surfaces the
      // button while in `ready` so this is a defensive guard.
      //
      // Pre-capture gate: manualStart no longer drops us straight into
      // `scanning`. It funnels through the same `confirming` path as
      // the auto-trigger so the user always sees the bbox confirm
      // before capture begins. The hook is responsible for dispatching
      // `requestConfirmation` once the snapshot URI is in hand; this
      // case here is the no-op fallback if the hook ever dispatches a
      // bare manualStart.
      if (state.kind !== 'ready') return state;
      return state;
    }
    case 'requestConfirmation': {
      // Only meaningful from `ready`. Anywhere else (aligning, already
      // confirming, scanning, paused, complete, failed) is a no-op so
      // a stale dispatch can't reorder the user's flow.
      if (state.kind !== 'ready') return state;
      return {
        kind: 'confirming',
        phase: 'detecting',
        snapshotUri: action.snapshotUri,
      };
    }
    case 'detectionResolved': {
      // Only consume the result while we're in `confirming{detecting}`.
      // A late response that lands after the user has retried (we'd
      // already be in `aligning` or `confirming{detecting}` with a new
      // snapshotUri) should not stomp the new state — the hook also
      // guards via an inFlight ref but the reducer is defensive.
      if (state.kind !== 'confirming') return state;
      if (state.phase !== 'detecting') return state;
      if (action.result.detected) {
        return {
          kind: 'confirming',
          phase: 'detected',
          snapshotUri: state.snapshotUri,
          bbox: action.result.bbox,
          containerType: action.result.containerType,
        };
      }
      return {
        kind: 'confirming',
        phase: 'failed',
        snapshotUri: state.snapshotUri,
        failureReason: action.result.reason ?? undefined,
      };
    }
    case 'confirmStart': {
      // Only meaningful from `confirming{detected}` — we need a
      // confirmed bbox before we commit to the capture pipeline.
      if (state.kind !== 'confirming') return state;
      if (state.phase !== 'detected') return state;
      return { kind: 'scanning', coverage: 0 };
    }
    case 'confirmRetry': {
      // From any confirming sub-phase, drop back to `aligning` and let
      // the steadiness integrator re-detect + re-arm. Drops the
      // snapshot so the next entry into `confirming` starts fresh.
      if (state.kind !== 'confirming') return state;
      return { kind: 'aligning' };
    }
    case 'tick': {
      const {
        bottleSteady,
        coverage,
        rotating,
        pauseReason,
        autoCaptureReady,
      } = action.inputs;

      // Terminal states stop accepting ticks (other than reset).
      if (state.kind === 'complete' || state.kind === 'failed') return state;

      // Hitting full coverage always wins. The caller dispatches
      // `complete` as a separate action once the panorama is stitched;
      // we stay in `scanning` (or `paused`) here so the UI can hold the
      // last visual frame while the parent runs the stitcher.
      if (coverage >= 1.0) {
        if (state.kind === 'scanning' || state.kind === 'paused') {
          return { kind: 'scanning', coverage: 1.0 };
        }
      }

      switch (state.kind) {
        case 'aligning':
          // Surface distance feedback before the user has even started
          // rotating — too_far / too_close are derived from the locked
          // silhouette width, so they're meaningful in `aligning` (and
          // only there does the user have something to act on).
          if (
            pauseReason === 'too_far' ||
            pauseReason === 'too_close'
          ) {
            return { kind: 'paused', coverage: 0, reason: pauseReason };
          }
          if (!bottleSteady) return state;
          return { kind: 'ready' };

        case 'ready':
          // Drop back to aligning if we lose the bottle in the ready
          // hold-window. Pre-capture gate: the auto-trigger
          // (autoCaptureReady) and rotation paths no longer transition
          // straight to `scanning` — the hook intercepts both and
          // dispatches `requestConfirmation` to drop into the
          // `confirming` gate. The reducer leaves the `ready → scanning`
          // edge intact for legacy callers / tests; the hook simply
          // never sets `rotating || coverage > 0 || autoCaptureReady`
          // before it fires `requestConfirmation`.
          if (!bottleSteady) return { kind: 'aligning' };
          if (rotating || coverage > 0 || autoCaptureReady) {
            return { kind: 'scanning', coverage };
          }
          if (pauseReason) {
            return { kind: 'paused', coverage: 0, reason: pauseReason };
          }
          return state;

        case 'confirming':
          // While the user is mid-confirmation, ticks must NOT boot
          // them back to `aligning` — the snapshot is on its way to
          // the backend or already showing a bbox the user is about
          // to act on. lose-bottle / pauseReason are deliberately
          // ignored here; coverage / rotation can't possibly advance
          // because the capture pipeline isn't running. The state is
          // driven by the explicit `confirmStart` / `confirmRetry` /
          // `detectionResolved` actions instead.
          return state;

        case 'scanning':
          if (pauseReason) {
            return { kind: 'paused', coverage, reason: pauseReason };
          }
          if (coverage === state.coverage) return state;
          return { kind: 'scanning', coverage };

        case 'paused':
          if (!pauseReason) {
            // Audit finding: previously, clearing a too_far/too_close
            // pause always routed back to `aligning`, throwing away any
            // coverage already captured. The user reads that as "I
            // restarted" — bad UX when they were halfway through and
            // just adjusted distance. If we still have a confident lock
            // (bottleSteady) AND non-trivial coverage, jump straight
            // back into `scanning` and preserve progress. The 0.05
            // threshold filters out the just-started case where
            // `aligning` is the more honest visual.
            //
            // For too_slow / lost_bottle / blur / glare / motion /
            // too_fast, the original aligning-recovery is correct —
            // those imply real tracking-state loss and the user
            // benefits from a brief realign window.
            if (state.reason === 'too_far' || state.reason === 'too_close') {
              if (bottleSteady && state.coverage > 0.05) {
                return { kind: 'scanning', coverage };
              }
              return { kind: 'aligning' };
            }
            return { kind: 'scanning', coverage };
          }
          // Pause-reason rotation while still paused: surface the
          // newest reason (debounced upstream).
          if (pauseReason !== state.reason || coverage !== state.coverage) {
            return { kind: 'paused', coverage, reason: pauseReason };
          }
          return state;
      }
      return state;
    }
  }
}

/**
 * Helper: extract the coverage value from any `ScanState` for callers
 * that don't want to discriminate (e.g. the cancel-confirm guard).
 */
export function coverageOf(state: ScanState): number {
  switch (state.kind) {
    case 'scanning':
    case 'paused':
      return state.coverage;
    case 'complete':
      return 1.0;
    case 'confirming':
    case 'aligning':
    case 'ready':
    case 'failed':
    default:
      return 0;
  }
}

// --- Auto-capture timer (brief #2) -----------------------------------

/**
 * Threshold for both `containerConfidence` and `gripSteadiness` before
 * the auto-capture predicate trips. Exported so the `useScanStateMachine`
 * hook can apply the same gate the tests assert on.
 *
 * Tightened from 0.7 → 0.85 alongside the pre-capture confirmation gate
 * (the new `confirming` state). The thinking is defense-in-depth: the
 * server-side `/v1/detect-container` is the authoritative gate, and
 * this on-device threshold is a cheap pre-filter that keeps us from
 * spamming snapshots up to the backend on low-confidence detections.
 * Pair this with the `silhouette.class !== null` requirement enforced
 * at the auto-capture predicate site (`useScanStateMachine`) so a
 * silhouette that doesn't classify as bottle/can/box doesn't even
 * trigger the snapshot.
 */
export const AUTO_CAPTURE_CONFIDENCE_GATE = 0.85;
/** Time the predicate must hold before `autoCaptureReady` latches. */
export const AUTO_CAPTURE_HOLD_MS = 1500;
/**
 * Time after entering `ready` without auto-capture latching at which
 * the manual-override affordance becomes available.
 */
export const MANUAL_OVERRIDE_AFTER_MS = 8000;

/**
 * Per-tick inputs to the auto-capture timer. `nowMs` is wall-clock
 * (ms); the timer is purely time-based so a frozen clock means the
 * timer never advances — handy for deterministic tests.
 */
export interface AutoCaptureTickInputs {
  /** Wall-clock ms (typically `Date.now()`). */
  nowMs: number;
  /** True iff `preCheck.kind === 'ready'`. */
  preCheckReady: boolean;
  /** Live container-confidence from the silhouette (0..1). */
  containerConfidence: number;
  /** Live grip-steadiness from the tracker (0..1). */
  gripSteadiness: number;
  /**
   * True iff `silhouette.class !== null` — the heuristic detector has
   * classified the silhouette as a bottle or can. Defense-in-depth
   * alongside the 0.85 confidence gate so an unclassified silhouette
   * never fires a snapshot at the backend.
   */
  containerClassPresent: boolean;
  /**
   * Whether the state machine is currently in `ready`. Drives the
   * manual-override timer — outside `ready` the timer doesn't run.
   */
  inReady: boolean;
  /**
   * Wall-clock ms at which the FSM most recently entered `ready`.
   * `null` if we're not in `ready` (or never have been). The auto-
   * capture timer itself is independent (it can pre-warm in
   * `aligning`), but `manualOverrideAvailable` only counts time
   * spent in `ready`.
   */
  readyEnteredAtMs: number | null;
}

/**
 * Persistent state for the auto-capture timer — the bits the hook
 * keeps in refs across ticks. Pure data so tests can construct, mutate,
 * and assert on it without React or shared values.
 */
export interface AutoCaptureTimerState {
  /**
   * Wall-clock ms when the auto-capture predicate first turned true
   * in the current run. `null` whenever any leg has dropped (so a
   * flicker resets the 1500ms hold instead of extending it).
   */
  candidateSinceMs: number | null;
  /**
   * Latched manual-override flag. Once flipped to `true` it stays
   * true until the FSM leaves `ready` (the hook clears it in the
   * state-transition effect).
   */
  manualOverrideLatched: boolean;
}

export const INITIAL_AUTO_CAPTURE_TIMER: AutoCaptureTimerState = {
  candidateSinceMs: null,
  manualOverrideLatched: false,
};

/**
 * Per-tick output from the auto-capture timer. `next` is the new
 * persistent state to write back; the booleans are what the hook
 * forwards to React state and the reducer.
 */
export interface AutoCaptureTickResult {
  next: AutoCaptureTimerState;
  /** True iff the predicate is currently true (drives the UI pill). */
  candidate: boolean;
  /** True iff the predicate has held for ≥AUTO_CAPTURE_HOLD_MS. */
  ready: boolean;
  /**
   * True iff we've been in `ready` for ≥MANUAL_OVERRIDE_AFTER_MS
   * without the auto-capture latching. Once true, stays true for the
   * rest of the `ready` window.
   */
  manualOverrideAvailable: boolean;
}

/**
 * Pure tick function for the auto-capture timer. Given the previous
 * persistent state and the current inputs, returns the next state and
 * the derived booleans. Exposed for unit tests; the hook uses refs to
 * carry `AutoCaptureTimerState` across ticks but the math is the same.
 *
 * Test contract:
 *   (a) sustained-good signals trigger after AUTO_CAPTURE_HOLD_MS
 *   (b) flicker (predicate transiently false) resets the timer
 *   (c) MANUAL_OVERRIDE_AFTER_MS in `ready` without trigger latches
 *       `manualOverrideAvailable`
 */
export function evaluateAutoCaptureTick(
  prev: AutoCaptureTimerState,
  inputs: AutoCaptureTickInputs,
): AutoCaptureTickResult {
  const candidate =
    inputs.preCheckReady &&
    inputs.containerClassPresent &&
    inputs.containerConfidence > AUTO_CAPTURE_CONFIDENCE_GATE &&
    inputs.gripSteadiness > AUTO_CAPTURE_CONFIDENCE_GATE;

  const candidateSinceMs = candidate
    ? prev.candidateSinceMs ?? inputs.nowMs
    : null;

  const ready =
    candidateSinceMs !== null &&
    inputs.nowMs - candidateSinceMs >= AUTO_CAPTURE_HOLD_MS;

  // Manual-override: only counts wall-clock time in `ready`. The
  // latch persists once set; the hook is responsible for clearing it
  // when the FSM transitions away from `ready`.
  let manualOverrideLatched = prev.manualOverrideLatched;
  if (
    !manualOverrideLatched &&
    inputs.inReady &&
    inputs.readyEnteredAtMs !== null &&
    !ready &&
    inputs.nowMs - inputs.readyEnteredAtMs >= MANUAL_OVERRIDE_AFTER_MS
  ) {
    manualOverrideLatched = true;
  }

  return {
    next: { candidateSinceMs, manualOverrideLatched },
    candidate,
    ready,
    manualOverrideAvailable: manualOverrideLatched,
  };
}
