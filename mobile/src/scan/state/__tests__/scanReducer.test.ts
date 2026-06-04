/**
 * Unit tests for `scanReducer` + `coverageOf` (scanMachine.ts).
 *
 * The reducer is the pure transition table at the heart of unwrap.tsx
 * (ARCH §3). The existing suite covers the auto-capture timer
 * (`evaluateAutoCaptureTick`); this one pins the discrete state machine
 * itself — every action, the terminal-state guards, the confirming
 * pre-capture gate, and the paused→scanning progress-preservation logic
 * the audit added.
 */

import {
  INITIAL_SCAN_STATE,
  coverageOf,
  scanReducer,
} from '../scanMachine';
import type {
  ScanMachineInputs,
  ScanState,
} from '../scanMachine';

function inputs(overrides: Partial<ScanMachineInputs> = {}): ScanMachineInputs {
  return {
    bottleSteady: false,
    coverage: 0,
    rotating: false,
    pauseReason: null,
    autoCaptureReady: false,
    ...overrides,
  };
}

const tick = (inp: Partial<ScanMachineInputs> = {}) =>
  ({ type: 'tick', inputs: inputs(inp) }) as const;

describe('scanReducer — top-level actions', () => {
  test('reset returns the initial aligning state', () => {
    expect(scanReducer({ kind: 'scanning', coverage: 0.5 }, { type: 'reset' })).toEqual(
      INITIAL_SCAN_STATE,
    );
  });

  test('fail transitions to failed with the given reason', () => {
    expect(scanReducer({ kind: 'aligning' }, { type: 'fail', reason: 'no_camera' })).toEqual({
      kind: 'failed',
      reason: 'no_camera',
    });
  });

  test('cancel transitions to failed{cancelled}', () => {
    expect(scanReducer({ kind: 'scanning', coverage: 0.3 }, { type: 'cancel' })).toEqual({
      kind: 'failed',
      reason: 'cancelled',
    });
  });

  test('manualStart is a no-op (the hook funnels through requestConfirmation)', () => {
    // From ready it returns ready unchanged; from anywhere else, unchanged.
    const ready: ScanState = { kind: 'ready' };
    expect(scanReducer(ready, { type: 'manualStart' })).toBe(ready);
    const aligning: ScanState = { kind: 'aligning' };
    expect(scanReducer(aligning, { type: 'manualStart' })).toBe(aligning);
  });

  test('complete transitions from a live state and is then idempotent', () => {
    const done = scanReducer(
      { kind: 'scanning', coverage: 1 },
      { type: 'complete', panoramaUri: 'file://pano.jpg' },
    );
    expect(done).toEqual({ kind: 'complete', panoramaUri: 'file://pano.jpg' });
    // A second complete must not stomp the existing panoramaUri.
    const again = scanReducer(done, { type: 'complete', panoramaUri: 'file://other.jpg' });
    expect(again).toBe(done);
  });
});

describe('scanReducer — confirming pre-capture gate', () => {
  test('requestConfirmation only fires from ready', () => {
    expect(
      scanReducer({ kind: 'ready' }, { type: 'requestConfirmation', snapshotUri: 'file://s.jpg' }),
    ).toEqual({ kind: 'confirming', phase: 'detecting', snapshotUri: 'file://s.jpg' });
    // From aligning it's a no-op.
    const aligning: ScanState = { kind: 'aligning' };
    expect(
      scanReducer(aligning, { type: 'requestConfirmation', snapshotUri: 'file://s.jpg' }),
    ).toBe(aligning);
  });

  test('detectionResolved(detected) carries bbox + containerType', () => {
    const detecting: ScanState = { kind: 'confirming', phase: 'detecting', snapshotUri: 'file://s.jpg' };
    const out = scanReducer(detecting, {
      type: 'detectionResolved',
      result: { detected: true, bbox: [0.1, 0.2, 0.8, 0.9], containerType: 'bottle' },
    });
    expect(out).toEqual({
      kind: 'confirming',
      phase: 'detected',
      snapshotUri: 'file://s.jpg',
      bbox: [0.1, 0.2, 0.8, 0.9],
      containerType: 'bottle',
    });
  });

  test('detectionResolved(not detected) surfaces the failure reason', () => {
    const detecting: ScanState = { kind: 'confirming', phase: 'detecting', snapshotUri: 'file://s.jpg' };
    const out = scanReducer(detecting, {
      type: 'detectionResolved',
      result: { detected: false, reason: 'no container found' },
    });
    expect(out).toEqual({
      kind: 'confirming',
      phase: 'failed',
      snapshotUri: 'file://s.jpg',
      failureReason: 'no container found',
    });
  });

  test('a late detectionResolved is ignored outside confirming{detecting}', () => {
    const detected: ScanState = {
      kind: 'confirming',
      phase: 'detected',
      snapshotUri: 'file://s.jpg',
      bbox: [0, 0, 1, 1],
      containerType: 'can',
    };
    expect(
      scanReducer(detected, {
        type: 'detectionResolved',
        result: { detected: false, reason: 'stale' },
      }),
    ).toBe(detected);
  });

  test('confirmStart commits to scanning only from confirming{detected}', () => {
    const detected: ScanState = {
      kind: 'confirming',
      phase: 'detected',
      snapshotUri: 'file://s.jpg',
      bbox: [0, 0, 1, 1],
      containerType: 'bottle',
    };
    expect(scanReducer(detected, { type: 'confirmStart' })).toEqual({
      kind: 'scanning',
      coverage: 0,
    });
    // From the failed sub-phase it's a no-op.
    const failed: ScanState = { kind: 'confirming', phase: 'failed', snapshotUri: 'file://s.jpg' };
    expect(scanReducer(failed, { type: 'confirmStart' })).toBe(failed);
  });

  test('confirmRetry drops back to aligning from any confirming phase', () => {
    const failed: ScanState = { kind: 'confirming', phase: 'failed', snapshotUri: 'file://s.jpg' };
    expect(scanReducer(failed, { type: 'confirmRetry' })).toEqual({ kind: 'aligning' });
  });
});

describe('scanReducer — tick transitions', () => {
  test('terminal states ignore ticks', () => {
    const complete: ScanState = { kind: 'complete', panoramaUri: 'file://p.jpg' };
    expect(scanReducer(complete, tick({ coverage: 0.2 }))).toBe(complete);
    const failed: ScanState = { kind: 'failed', reason: 'capture_error' };
    expect(scanReducer(failed, tick())).toBe(failed);
  });

  test('full coverage snaps scanning/paused to scanning{1.0}', () => {
    expect(scanReducer({ kind: 'scanning', coverage: 0.9 }, tick({ coverage: 1.0 }))).toEqual({
      kind: 'scanning',
      coverage: 1.0,
    });
    expect(
      scanReducer({ kind: 'paused', coverage: 0.9, reason: 'blur' }, tick({ coverage: 1.0 })),
    ).toEqual({ kind: 'scanning', coverage: 1.0 });
  });

  describe('from aligning', () => {
    test('too_far / too_close pause even before rotation starts', () => {
      expect(scanReducer({ kind: 'aligning' }, tick({ pauseReason: 'too_far' }))).toEqual({
        kind: 'paused',
        coverage: 0,
        reason: 'too_far',
      });
    });

    test('stays aligning until the bottle is steady', () => {
      const s: ScanState = { kind: 'aligning' };
      expect(scanReducer(s, tick({ bottleSteady: false }))).toBe(s);
    });

    test('advances to ready once steady', () => {
      expect(scanReducer({ kind: 'aligning' }, tick({ bottleSteady: true }))).toEqual({
        kind: 'ready',
      });
    });
  });

  describe('from ready', () => {
    test('losing the bottle drops back to aligning', () => {
      expect(scanReducer({ kind: 'ready' }, tick({ bottleSteady: false }))).toEqual({
        kind: 'aligning',
      });
    });

    test('rotation / coverage / auto-capture each start scanning', () => {
      for (const inp of [{ rotating: true }, { coverage: 0.1 }, { autoCaptureReady: true }]) {
        expect(scanReducer({ kind: 'ready' }, tick({ bottleSteady: true, ...inp }))).toMatchObject({
          kind: 'scanning',
        });
      }
    });

    test('a pause reason while steady-but-idle routes to paused', () => {
      expect(
        scanReducer({ kind: 'ready' }, tick({ bottleSteady: true, pauseReason: 'glare' })),
      ).toEqual({ kind: 'paused', coverage: 0, reason: 'glare' });
    });

    test('steady and idle with no pause holds ready', () => {
      const s: ScanState = { kind: 'ready' };
      expect(scanReducer(s, tick({ bottleSteady: true }))).toBe(s);
    });
  });

  test('confirming ignores ticks entirely', () => {
    const s: ScanState = { kind: 'confirming', phase: 'detecting', snapshotUri: 'file://s.jpg' };
    expect(scanReducer(s, tick({ pauseReason: 'lost_bottle', coverage: 0.5 }))).toBe(s);
  });

  describe('from scanning', () => {
    test('a pause reason pauses while preserving coverage', () => {
      expect(
        scanReducer({ kind: 'scanning', coverage: 0.4 }, tick({ coverage: 0.4, pauseReason: 'too_fast' })),
      ).toEqual({ kind: 'paused', coverage: 0.4, reason: 'too_fast' });
    });

    test('unchanged coverage is a no-op; advancing coverage updates', () => {
      const s: ScanState = { kind: 'scanning', coverage: 0.4 };
      expect(scanReducer(s, tick({ coverage: 0.4 }))).toBe(s);
      expect(scanReducer(s, tick({ coverage: 0.6 }))).toEqual({ kind: 'scanning', coverage: 0.6 });
    });
  });

  describe('from paused', () => {
    test('clearing a distance pause with progress resumes scanning', () => {
      expect(
        scanReducer(
          { kind: 'paused', coverage: 0.4, reason: 'too_far' },
          tick({ coverage: 0.4, bottleSteady: true }),
        ),
      ).toEqual({ kind: 'scanning', coverage: 0.4 });
    });

    test('clearing a distance pause without a steady lock realigns', () => {
      expect(
        scanReducer(
          { kind: 'paused', coverage: 0.4, reason: 'too_close' },
          tick({ coverage: 0.4, bottleSteady: false }),
        ),
      ).toEqual({ kind: 'aligning' });
    });

    test('clearing a tracking-loss pause resumes scanning directly', () => {
      expect(
        scanReducer({ kind: 'paused', coverage: 0.4, reason: 'blur' }, tick({ coverage: 0.4 })),
      ).toEqual({ kind: 'scanning', coverage: 0.4 });
    });

    test('a new pause reason while paused surfaces the newest reason', () => {
      expect(
        scanReducer(
          { kind: 'paused', coverage: 0.4, reason: 'blur' },
          tick({ coverage: 0.4, pauseReason: 'glare' }),
        ),
      ).toEqual({ kind: 'paused', coverage: 0.4, reason: 'glare' });
    });

    test('the same pause reason + coverage is a no-op', () => {
      const s: ScanState = { kind: 'paused', coverage: 0.4, reason: 'glare' };
      expect(scanReducer(s, tick({ coverage: 0.4, pauseReason: 'glare' }))).toBe(s);
    });
  });
});

describe('coverageOf', () => {
  test('reads coverage from scanning and paused', () => {
    expect(coverageOf({ kind: 'scanning', coverage: 0.7 })).toBe(0.7);
    expect(coverageOf({ kind: 'paused', coverage: 0.3, reason: 'blur' })).toBe(0.3);
  });

  test('reports 1.0 for complete and 0 for the pre-capture states', () => {
    expect(coverageOf({ kind: 'complete', panoramaUri: 'file://p.jpg' })).toBe(1.0);
    expect(coverageOf({ kind: 'aligning' })).toBe(0);
    expect(coverageOf({ kind: 'ready' })).toBe(0);
    expect(coverageOf({ kind: 'failed', reason: 'cancelled' })).toBe(0);
  });
});
