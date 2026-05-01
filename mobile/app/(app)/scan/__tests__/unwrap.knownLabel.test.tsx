/**
 * Tests for the known-label recognition sheet inside
 * unwrap.tsx's `ConfirmingOverlay` (KNOWN_LABEL_DESIGN.md Decision 7).
 *
 * These exercise the `KnownLabelSheet` sub-component directly — the
 * full screen is too heavy to render in jest (Camera, Skia, Reanimated,
 * frame processors) and the recognition UI logic is fully owned by
 * this component. The unwrap screen itself is just a thin wrapper that
 * threads the `knownLabel` payload + clearKnownLabel handler.
 *
 * Surfaces under test:
 *   - Renders brand name + StatusBadge from verdict_summary.overall.
 *   - "View results" → createScanFromCache → router.replace(report).
 *   - "Scan anyway" clears knownLabel + dispatches confirmStart.
 *   - "Reshoot" clears knownLabel + dispatches confirmRetry.
 *   - createScanFromCache failure → inline ErrorState (no navigation).
 */

import React from 'react';
import { fireEvent, render, waitFor } from '@testing-library/react-native';

import { KnownLabelSheet } from '../KnownLabelSheet';
import type { KnownLabelPayload } from '@src/api/types';

// expo-router: stub `router.replace` so we can assert navigation. We
// import after the mock so the unwrap module picks up the stub.
// Variables referenced inside jest.mock factories must be prefixed with
// `mock` (case-insensitive) so the babel-jest hoist guard accepts them.
const mockReplace = jest.fn();
jest.mock('expo-router', () => ({
  router: { replace: (...args: unknown[]) => mockReplace(...args) },
}));

// apiClient: stub `createScanFromCache`. The unwrap module imports
// `apiClient` from `@src/api/client`.
const mockCreateScanFromCache = jest.fn();
jest.mock('@src/api/client', () => ({
  apiClient: {
    createScanFromCache: (...args: unknown[]) =>
      mockCreateScanFromCache(...args),
  },
}));

// react-native-safe-area-context: stub useSafeAreaInsets so the
// component renders without a provider in tests.
jest.mock('react-native-safe-area-context', () => ({
  useSafeAreaInsets: () => ({ top: 0, bottom: 0, left: 0, right: 0 }),
}));

function makeKnownLabel(
  overrides: Partial<KnownLabelPayload> = {},
): KnownLabelPayload {
  return {
    entry_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    beverage_type: 'beer',
    container_size_ml: 355,
    is_imported: false,
    brand_name: 'Sierra Nevada',
    fanciful_name: 'Pale Ale',
    verdict_summary: {
      overall: 'pass',
      rule_results: [],
      extracted: {},
      image_quality: 'good',
    },
    source: 'brand',
    ...overrides,
  };
}

describe('KnownLabelSheet', () => {
  beforeEach(() => {
    mockReplace.mockReset();
    mockCreateScanFromCache.mockReset();
  });

  test('renders brand name + StatusBadge of verdict_summary.overall', () => {
    const knownLabel = makeKnownLabel();
    const { getByText, getByLabelText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={jest.fn()}
        onReshoot={jest.fn()}
      />,
    );

    expect(getByText('Sierra Nevada')).toBeTruthy();
    expect(getByText('Pale Ale')).toBeTruthy();
    expect(getByText('Recognized from a previous scan.')).toBeTruthy();
    // StatusBadge exposes `accessibilityLabel` of the form
    // `Status: <status>` — reach it by label rather than the upper-cased
    // text so we couple to the underlying contract.
    expect(getByLabelText('Status: pass')).toBeTruthy();
  });

  test('coerces verdict_summary.overall="warn" to advisory on the badge', () => {
    // Backend can emit `warn` or `unreadable` (KnownLabelOverall is
    // wider than OverallStatus). The sheet collapses both onto the
    // existing palette so the badge always renders.
    const knownLabel = makeKnownLabel();
    knownLabel.verdict_summary.overall = 'warn';
    const { getByLabelText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={jest.fn()}
        onReshoot={jest.fn()}
      />,
    );
    expect(getByLabelText('Status: advisory')).toBeTruthy();
  });

  test('coerces verdict_summary.overall="unreadable" to fail on the badge', () => {
    const knownLabel = makeKnownLabel();
    knownLabel.verdict_summary.overall = 'unreadable';
    const { getByLabelText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={jest.fn()}
        onReshoot={jest.fn()}
      />,
    );
    expect(getByLabelText('Status: fail')).toBeTruthy();
  });

  test('coerces verdict_summary.overall="na" to advisory on the badge', () => {
    // Rare edge case: zero rules applied to the cached extraction.
    // Treated as informational (advisory tier) since there's nothing
    // for the user to act on.
    const knownLabel = makeKnownLabel();
    knownLabel.verdict_summary.overall = 'na';
    const { getByLabelText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={jest.fn()}
        onReshoot={jest.fn()}
      />,
    );
    expect(getByLabelText('Status: advisory')).toBeTruthy();
  });

  test('falls back to a placeholder headline when brand_name is null', () => {
    const knownLabel = makeKnownLabel({ brand_name: null });
    const { getByText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={jest.fn()}
        onReshoot={jest.fn()}
      />,
    );

    expect(getByText('Recognized label')).toBeTruthy();
  });

  test('omits fanciful_name row when null', () => {
    const knownLabel = makeKnownLabel({ fanciful_name: null });
    const { queryByText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={jest.fn()}
        onReshoot={jest.fn()}
      />,
    );

    expect(queryByText('Pale Ale')).toBeNull();
  });

  test('"View results" calls createScanFromCache and navigates to /report', async () => {
    const knownLabel = makeKnownLabel();
    mockCreateScanFromCache.mockResolvedValueOnce({
      scan_id: 'scan-from-cache-1',
      status: 'complete',
      overall: 'pass',
      image_quality: 'good',
    });

    const { getByText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={jest.fn()}
        onReshoot={jest.fn()}
      />,
    );

    fireEvent.press(getByText('View results'));

    await waitFor(() => {
      expect(mockCreateScanFromCache).toHaveBeenCalledWith({
        entry_id: knownLabel.entry_id,
        beverage_type: 'beer',
        container_size_ml: 355,
        is_imported: false,
      });
    });

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith(
        '/(app)/scan/report/scan-from-cache-1',
      );
    });
  });

  test('"Scan anyway" fires onScanAnyway (which clears knownLabel + confirmStart)', () => {
    const knownLabel = makeKnownLabel();
    const onScanAnyway = jest.fn();

    const { getByText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={onScanAnyway}
        onReshoot={jest.fn()}
      />,
    );

    fireEvent.press(getByText('Scan anyway'));
    expect(onScanAnyway).toHaveBeenCalledTimes(1);
    // Should NOT call createScanFromCache or navigate.
    expect(mockCreateScanFromCache).not.toHaveBeenCalled();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  test('"Reshoot" fires onReshoot (which clears knownLabel + confirmRetry)', () => {
    const knownLabel = makeKnownLabel();
    const onReshoot = jest.fn();

    const { getByText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={jest.fn()}
        onReshoot={onReshoot}
      />,
    );

    fireEvent.press(getByText('Reshoot'));
    expect(onReshoot).toHaveBeenCalledTimes(1);
    expect(mockCreateScanFromCache).not.toHaveBeenCalled();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  test('createScanFromCache failure renders inline ErrorState (no navigation)', async () => {
    const knownLabel = makeKnownLabel();
    const error = Object.assign(new Error('cache miss'), { status: 404 });
    mockCreateScanFromCache.mockRejectedValueOnce(error);

    const { getByText, queryByText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={jest.fn()}
        onReshoot={jest.fn()}
      />,
    );

    fireEvent.press(getByText('View results'));

    // Wait for the ErrorState title to appear.
    await waitFor(() => {
      expect(getByText("Couldn't load the saved verdict.")).toBeTruthy();
    });
    // The error description surfaces the underlying message so the user
    // sees something concrete.
    expect(getByText('cache miss')).toBeTruthy();
    // Sanity: the recognition headline is no longer on screen — we've
    // swapped to the error surface.
    expect(queryByText('Sierra Nevada')).toBeNull();
    // No navigation on failure.
    expect(mockReplace).not.toHaveBeenCalled();
  });

  test('error surface offers a secondary "Scan anyway" affordance', async () => {
    const knownLabel = makeKnownLabel();
    mockCreateScanFromCache.mockRejectedValueOnce(new Error('boom'));
    const onScanAnyway = jest.fn();

    const { getByText } = render(
      <KnownLabelSheet
        knownLabel={knownLabel}
        onScanAnyway={onScanAnyway}
        onReshoot={jest.fn()}
      />,
    );

    fireEvent.press(getByText('View results'));
    await waitFor(() => {
      expect(getByText("Couldn't load the saved verdict.")).toBeTruthy();
    });

    fireEvent.press(getByText('Scan anyway'));
    expect(onScanAnyway).toHaveBeenCalledTimes(1);
  });
});
