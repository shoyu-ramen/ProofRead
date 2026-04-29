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

import type { PauseReason, FailReason } from '@src/scan/ui';

export type ScanStateKind =
  | 'aligning'
  | 'ready'
  | 'scanning'
  | 'paused'
  | 'complete'
  | 'failed';

export type { PauseReason, FailReason };

export type ScanState =
  | { kind: 'aligning' }
  | { kind: 'ready' }
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
}

export type ScanAction =
  | { type: 'tick'; inputs: ScanMachineInputs }
  | { type: 'fail'; reason: FailReason }
  | { type: 'complete'; panoramaUri: string }
  | { type: 'cancel' }
  | { type: 'reset' };

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
    case 'tick': {
      const { bottleSteady, coverage, rotating, pauseReason } = action.inputs;

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
          if (!bottleSteady) return state;
          return { kind: 'ready' };

        case 'ready':
          // Drop back to aligning if we lose the bottle in the ready
          // hold-window; otherwise fall into scanning the moment the
          // user starts rotating.
          if (!bottleSteady) return { kind: 'aligning' };
          if (rotating || coverage > 0) {
            return { kind: 'scanning', coverage };
          }
          if (pauseReason) {
            return { kind: 'paused', coverage: 0, reason: pauseReason };
          }
          return state;

        case 'scanning':
          if (pauseReason) {
            return { kind: 'paused', coverage, reason: pauseReason };
          }
          if (coverage === state.coverage) return state;
          return { kind: 'scanning', coverage };

        case 'paused':
          if (!pauseReason) {
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
    default:
      return 0;
  }
}
