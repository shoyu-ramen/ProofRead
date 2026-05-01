/**
 * Unit tests for the auto-trigger gate in `scanMachine.ts`.
 *
 * Two surfaces under test:
 *   1. `evaluateAutoCaptureTick` — pure timer that decides when the
 *      auto-capture predicate has been sustained long enough to latch
 *      `ready` (drives the reducer transition) and when the manual
 *      override should appear.
 *   2. `scanReducer` — the discrete transition table; verifies that
 *      `autoCaptureReady=true` actually drives `ready → scanning`
 *      and that `manualStart` is the second supported entry path.
 *
 * Brief contract (3 sub-cases for the timer):
 *   (a) sustained-good signals trigger after AUTO_CAPTURE_HOLD_MS
 *   (b) flicker (predicate transiently false) resets the timer
 *   (c) MANUAL_OVERRIDE_AFTER_MS in `ready` without trigger sets
 *       `manualOverrideAvailable`
 *
 * Wall-clock is mocked by passing explicit `nowMs` values.
 */

import {
  AUTO_CAPTURE_CONFIDENCE_GATE,
  AUTO_CAPTURE_HOLD_MS,
  evaluateAutoCaptureTick,
  INITIAL_AUTO_CAPTURE_TIMER,
  INITIAL_SCAN_STATE,
  MANUAL_OVERRIDE_AFTER_MS,
  scanReducer,
  type AutoCaptureTimerState,
  type ScanMachineInputs,
  type ScanState,
} from '../scanMachine';

// Helpers -----------------------------------------------------------

interface SignalLevels {
  preCheckReady: boolean;
  containerConfidence: number;
  gripSteadiness: number;
}

const GOOD_SIGNALS: SignalLevels = {
  preCheckReady: true,
  containerConfidence: 0.9,
  gripSteadiness: 0.9,
};

const BAD_SIGNALS: SignalLevels = {
  preCheckReady: false,
  containerConfidence: 0.0,
  gripSteadiness: 0.0,
};

function tick(
  prev: AutoCaptureTimerState,
  nowMs: number,
  signals: SignalLevels,
  opts: { inReady?: boolean; readyEnteredAtMs?: number | null } = {},
) {
  return evaluateAutoCaptureTick(prev, {
    nowMs,
    preCheckReady: signals.preCheckReady,
    containerConfidence: signals.containerConfidence,
    gripSteadiness: signals.gripSteadiness,
    inReady: opts.inReady ?? true,
    readyEnteredAtMs:
      opts.readyEnteredAtMs ?? (opts.inReady === false ? null : 0),
  });
}

function defaultInputs(
  overrides: Partial<ScanMachineInputs> = {},
): ScanMachineInputs {
  return {
    bottleSteady: false,
    coverage: 0,
    rotating: false,
    pauseReason: null,
    autoCaptureReady: false,
    ...overrides,
  };
}

// (a) Sustained-good signals trigger after AUTO_CAPTURE_HOLD_MS ---

describe('evaluateAutoCaptureTick — (a) sustained-good signals', () => {
  test('does not trigger immediately on first good tick', () => {
    const result = tick(INITIAL_AUTO_CAPTURE_TIMER, 0, GOOD_SIGNALS);
    expect(result.candidate).toBe(true);
    expect(result.ready).toBe(false);
    expect(result.next.candidateSinceMs).toBe(0);
  });

  test('triggers exactly at AUTO_CAPTURE_HOLD_MS', () => {
    let state: AutoCaptureTimerState = INITIAL_AUTO_CAPTURE_TIMER;
    const t0 = 1_000_000; // arbitrary base time
    let result = tick(state, t0, GOOD_SIGNALS);
    state = result.next;
    expect(result.ready).toBe(false);

    // Halfway through the hold — still not ready.
    result = tick(state, t0 + 750, GOOD_SIGNALS);
    state = result.next;
    expect(result.ready).toBe(false);

    // Just before the threshold — still not ready.
    result = tick(state, t0 + AUTO_CAPTURE_HOLD_MS - 1, GOOD_SIGNALS);
    state = result.next;
    expect(result.ready).toBe(false);

    // At the threshold — now latched.
    result = tick(state, t0 + AUTO_CAPTURE_HOLD_MS, GOOD_SIGNALS);
    state = result.next;
    expect(result.ready).toBe(true);
  });

  test('stays ready while predicate remains true', () => {
    let state: AutoCaptureTimerState = INITIAL_AUTO_CAPTURE_TIMER;
    state = tick(state, 0, GOOD_SIGNALS).next;
    state = tick(state, AUTO_CAPTURE_HOLD_MS, GOOD_SIGNALS).next;
    const longAfter = tick(state, AUTO_CAPTURE_HOLD_MS + 5000, GOOD_SIGNALS);
    expect(longAfter.ready).toBe(true);
    expect(longAfter.candidate).toBe(true);
  });
});

// (b) Flicker resets the timer ---

describe('evaluateAutoCaptureTick — (b) flicker resets timer', () => {
  test('a single bad tick mid-hold resets candidateSince', () => {
    let state: AutoCaptureTimerState = INITIAL_AUTO_CAPTURE_TIMER;
    const t0 = 1_000_000;

    // 750ms of sustained-good (halfway through the hold).
    state = tick(state, t0, GOOD_SIGNALS).next;
    state = tick(state, t0 + 750, GOOD_SIGNALS).next;
    expect(state.candidateSinceMs).toBe(t0);

    // Flicker at t0+750: one bad tick — predicate fails for an instant.
    state = tick(state, t0 + 750, BAD_SIGNALS).next;
    expect(state.candidateSinceMs).toBe(null);

    // Resume good — but the candidateSince anchors to the resume time,
    // not the original t0.
    state = tick(state, t0 + 800, GOOD_SIGNALS).next;
    expect(state.candidateSinceMs).toBe(t0 + 800);
  });

  test('readiness only fires AUTO_CAPTURE_HOLD_MS after the post-flicker resume', () => {
    let state: AutoCaptureTimerState = INITIAL_AUTO_CAPTURE_TIMER;
    const t0 = 1_000_000;

    // Sustain to 1499ms (one ms shy of latching).
    state = tick(state, t0, GOOD_SIGNALS).next;
    state = tick(state, t0 + AUTO_CAPTURE_HOLD_MS - 1, GOOD_SIGNALS).next;

    // Flicker — predicate drops.
    state = tick(state, t0 + AUTO_CAPTURE_HOLD_MS, BAD_SIGNALS).next;

    // Resume immediately. We've spent 1.5s in the original window but
    // the timer was reset, so we still need the FULL hold from the
    // resume time.
    state = tick(state, t0 + AUTO_CAPTURE_HOLD_MS + 1, GOOD_SIGNALS).next;
    let r = tick(state, t0 + AUTO_CAPTURE_HOLD_MS + 100, GOOD_SIGNALS);
    expect(r.ready).toBe(false);

    // Now wait the full hold from the resume.
    r = tick(
      r.next,
      t0 + AUTO_CAPTURE_HOLD_MS + 1 + AUTO_CAPTURE_HOLD_MS,
      GOOD_SIGNALS,
    );
    expect(r.ready).toBe(true);
  });

  test('predicate falls when any single leg drops below the gate', () => {
    // Container confidence drops below the 0.7 gate.
    const r1 = tick(INITIAL_AUTO_CAPTURE_TIMER, 0, {
      preCheckReady: true,
      containerConfidence: AUTO_CAPTURE_CONFIDENCE_GATE,
      gripSteadiness: 1.0,
    });
    expect(r1.candidate).toBe(false);

    // Grip steadiness drops below the 0.7 gate.
    const r2 = tick(INITIAL_AUTO_CAPTURE_TIMER, 0, {
      preCheckReady: true,
      containerConfidence: 1.0,
      gripSteadiness: AUTO_CAPTURE_CONFIDENCE_GATE,
    });
    expect(r2.candidate).toBe(false);

    // PreCheck not ready.
    const r3 = tick(INITIAL_AUTO_CAPTURE_TIMER, 0, {
      preCheckReady: false,
      containerConfidence: 1.0,
      gripSteadiness: 1.0,
    });
    expect(r3.candidate).toBe(false);
  });
});

// (c) Manual override after MANUAL_OVERRIDE_AFTER_MS ---

describe('evaluateAutoCaptureTick — (c) manual override fallback', () => {
  test('does not surface override before the timer elapses', () => {
    let state: AutoCaptureTimerState = INITIAL_AUTO_CAPTURE_TIMER;
    const t0 = 1_000_000;
    // 7s in `ready` with bad signals — short of the 8s threshold.
    state = tick(state, t0, BAD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    }).next;
    const r = tick(state, t0 + 7000, BAD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    });
    expect(r.manualOverrideAvailable).toBe(false);
  });

  test('latches override at exactly MANUAL_OVERRIDE_AFTER_MS in ready', () => {
    let state: AutoCaptureTimerState = INITIAL_AUTO_CAPTURE_TIMER;
    const t0 = 1_000_000;
    state = tick(state, t0, BAD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    }).next;

    // Just shy of the threshold.
    let r = tick(state, t0 + MANUAL_OVERRIDE_AFTER_MS - 1, BAD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    });
    expect(r.manualOverrideAvailable).toBe(false);
    state = r.next;

    // At the threshold — latch flips on.
    r = tick(state, t0 + MANUAL_OVERRIDE_AFTER_MS, BAD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    });
    expect(r.manualOverrideAvailable).toBe(true);
  });

  test('does not surface override outside `ready`', () => {
    let state: AutoCaptureTimerState = INITIAL_AUTO_CAPTURE_TIMER;
    const t0 = 1_000_000;
    // Sit in `aligning` for 10s — manual override should never fire.
    state = tick(state, t0, BAD_SIGNALS, {
      inReady: false,
      readyEnteredAtMs: null,
    }).next;
    const r = tick(state, t0 + 10_000, BAD_SIGNALS, {
      inReady: false,
      readyEnteredAtMs: null,
    });
    expect(r.manualOverrideAvailable).toBe(false);
  });

  test('override is suppressed if auto-capture has already latched', () => {
    let state: AutoCaptureTimerState = INITIAL_AUTO_CAPTURE_TIMER;
    const t0 = 1_000_000;
    // Auto-capture latches at t0 + AUTO_CAPTURE_HOLD_MS (1.5s) with
    // good signals; the manual-override timer should *not* count time
    // after the latch since the user already got their auto-trigger.
    state = tick(state, t0, GOOD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    }).next;
    state = tick(state, t0 + AUTO_CAPTURE_HOLD_MS, GOOD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    }).next;
    // Wall-clock 9 seconds in — past the 8s manual-override timer —
    // but we're already ready, so override should NOT latch.
    const r = tick(state, t0 + 9000, GOOD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    });
    expect(r.ready).toBe(true);
    expect(r.manualOverrideAvailable).toBe(false);
  });

  test('override stays latched once set, even if signals improve', () => {
    let state: AutoCaptureTimerState = INITIAL_AUTO_CAPTURE_TIMER;
    const t0 = 1_000_000;
    // 8 seconds of bad signals → override latches.
    state = tick(state, t0, BAD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    }).next;
    state = tick(state, t0 + MANUAL_OVERRIDE_AFTER_MS, BAD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    }).next;
    expect(state.manualOverrideLatched).toBe(true);

    // Signals improve — but the latch stays.
    const r = tick(state, t0 + MANUAL_OVERRIDE_AFTER_MS + 1000, GOOD_SIGNALS, {
      inReady: true,
      readyEnteredAtMs: t0,
    });
    expect(r.manualOverrideAvailable).toBe(true);
  });
});

// scanReducer — auto-trigger transition path ---

describe('scanReducer — auto-trigger transition', () => {
  test('autoCaptureReady drives `ready → scanning` without rotation', () => {
    // Get into `ready`.
    let state: ScanState = INITIAL_SCAN_STATE;
    state = scanReducer(state, {
      type: 'tick',
      inputs: defaultInputs({ bottleSteady: true }),
    });
    expect(state.kind).toBe('ready');

    // Now feed autoCaptureReady=true while NOT rotating and NOT
    // accumulating coverage. Pre-auto-trigger this would have stayed
    // in `ready`; with the gate we should slide into `scanning`.
    state = scanReducer(state, {
      type: 'tick',
      inputs: defaultInputs({
        bottleSteady: true,
        autoCaptureReady: true,
      }),
    });
    expect(state.kind).toBe('scanning');
    expect((state as { coverage: number }).coverage).toBe(0);
  });

  test('rotation path still works in parallel with auto-trigger', () => {
    let state: ScanState = INITIAL_SCAN_STATE;
    state = scanReducer(state, {
      type: 'tick',
      inputs: defaultInputs({ bottleSteady: true }),
    });
    state = scanReducer(state, {
      type: 'tick',
      inputs: defaultInputs({
        bottleSteady: true,
        rotating: true,
        autoCaptureReady: false,
      }),
    });
    expect(state.kind).toBe('scanning');
  });

  test('manualStart drives `ready → scanning`', () => {
    let state: ScanState = INITIAL_SCAN_STATE;
    state = scanReducer(state, {
      type: 'tick',
      inputs: defaultInputs({ bottleSteady: true }),
    });
    expect(state.kind).toBe('ready');
    state = scanReducer(state, { type: 'manualStart' });
    expect(state.kind).toBe('scanning');
    expect((state as { coverage: number }).coverage).toBe(0);
  });

  test('manualStart from non-ready states is a no-op', () => {
    // From `aligning`.
    const aligning: ScanState = { kind: 'aligning' };
    expect(scanReducer(aligning, { type: 'manualStart' })).toEqual(aligning);

    // From `scanning` already.
    const scanning: ScanState = { kind: 'scanning', coverage: 0.4 };
    expect(scanReducer(scanning, { type: 'manualStart' })).toEqual(scanning);

    // From `complete`.
    const complete: ScanState = {
      kind: 'complete',
      panoramaUri: 'file://x.jpg',
    };
    expect(scanReducer(complete, { type: 'manualStart' })).toEqual(complete);
  });

  test('autoCaptureReady ignored unless we are in `ready`', () => {
    // From `aligning` without bottleSteady, autoCaptureReady doesn't
    // help — we still need a steady detection first. (Defense in
    // depth: the hook only emits autoCaptureReady=true when in `ready`,
    // but the reducer shouldn't get tricked by a stale flag.)
    let state: ScanState = INITIAL_SCAN_STATE;
    state = scanReducer(state, {
      type: 'tick',
      inputs: defaultInputs({
        bottleSteady: false,
        autoCaptureReady: true,
      }),
    });
    expect(state.kind).toBe('aligning');
  });
});
