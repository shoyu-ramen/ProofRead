/**
 * InScanWarningBanner predicate test — pure heuristic logic. The
 * full RN-Animated render path is exercised at integration time on
 * device; here we just lock in the band boundaries.
 */

import {
  shouldShowInScanWarning,
  IN_SCAN_WARNING_LOWER,
  IN_SCAN_WARNING_UPPER,
} from '../InScanWarningBanner';

describe('shouldShowInScanWarning', () => {
  test('hidden during aligning / ready / paused / complete / failed', () => {
    for (const state of [
      'aligning',
      'ready',
      'paused',
      'complete',
      'failed',
    ] as const) {
      expect(shouldShowInScanWarning(state, 0.7)).toBe(false);
    }
  });

  test('hidden during scanning when coverage is below the lower bound', () => {
    expect(shouldShowInScanWarning('scanning', 0.5)).toBe(false);
    expect(
      shouldShowInScanWarning('scanning', IN_SCAN_WARNING_LOWER - 0.001),
    ).toBe(false);
  });

  test('visible during scanning when coverage is in the [240°, 300°) band', () => {
    expect(shouldShowInScanWarning('scanning', IN_SCAN_WARNING_LOWER)).toBe(
      true,
    );
    expect(shouldShowInScanWarning('scanning', 0.75)).toBe(true);
    expect(
      shouldShowInScanWarning('scanning', IN_SCAN_WARNING_UPPER - 0.001),
    ).toBe(true);
  });

  test('hidden during scanning when coverage clears the upper bound', () => {
    expect(shouldShowInScanWarning('scanning', IN_SCAN_WARNING_UPPER)).toBe(
      false,
    );
    expect(shouldShowInScanWarning('scanning', 0.95)).toBe(false);
    expect(shouldShowInScanWarning('scanning', 1.0)).toBe(false);
  });

  test('boundary values match the expected angular thresholds', () => {
    expect(IN_SCAN_WARNING_LOWER).toBeCloseTo(240 / 360, 6);
    expect(IN_SCAN_WARNING_UPPER).toBeCloseTo(300 / 360, 6);
  });
});
