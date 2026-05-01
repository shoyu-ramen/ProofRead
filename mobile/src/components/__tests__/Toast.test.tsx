/**
 * Toast subsystem tests.
 *
 * Variants render the right semantic accent, durationMs auto-dismisses
 * the entry, and the queue caps at maxVisible (3 by default) — the
 * fourth show() flips the oldest entry to dismissing.
 */

import React from 'react';
import { Text } from 'react-native';
import { render, act, fireEvent } from '@testing-library/react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { Toast } from '../Toast';
import { ToastProvider } from '../ToastProvider';
import { useToast } from '@src/hooks/useToast';
import { colors } from '@src/theme';

beforeEach(() => {
  jest.useFakeTimers();
});
afterEach(() => {
  jest.useRealTimers();
});

function withProviders(node: React.ReactElement): React.ReactElement {
  return (
    <SafeAreaProvider
      initialMetrics={{
        frame: { x: 0, y: 0, width: 320, height: 640 },
        insets: { top: 0, left: 0, right: 0, bottom: 0 },
      }}
    >
      <ToastProvider>{node}</ToastProvider>
    </SafeAreaProvider>
  );
}

function flattenStyle(
  style: unknown,
): Record<string, unknown> {
  // Testing-library renders style as either a single object or an
  // array; flatten arrays so test assertions can read fields directly.
  if (!style) return {};
  if (Array.isArray(style)) {
    return style.reduce<Record<string, unknown>>(
      (acc, s) => Object.assign(acc, flattenStyle(s)),
      {},
    );
  }
  return style as Record<string, unknown>;
}

describe('Toast view (presentational)', () => {
  test.each([
    ['info', colors.primary],
    ['success', colors.pass],
    ['warning', colors.advisory],
    ['error', colors.fail],
  ] as const)(
    'variant=%s renders with the correct accent color',
    (variant, expectedAccent) => {
      const { getByRole } = render(
        <Toast
          id={`t-${variant}`}
          variant={variant}
          message={`hello ${variant}`}
          dismissing={false}
          onDismiss={() => {}}
        />,
      );
      const card = getByRole('alert');
      const flat = flattenStyle(card.props.style);
      expect(flat.borderLeftColor).toBe(expectedAccent);
    },
  );

  test('tap fires onDismiss with the id', () => {
    const onDismiss = jest.fn();
    const { getByRole } = render(
      <Toast
        id="t-tap"
        variant="info"
        message="tap me"
        dismissing={false}
        onDismiss={onDismiss}
      />,
    );
    const card = getByRole('alert');
    fireEvent.press(card);
    expect(onDismiss).toHaveBeenCalledWith('t-tap');
  });
});

describe('ToastProvider queue behaviour', () => {
  function HostHarness({
    onReady,
  }: {
    onReady: (api: ReturnType<typeof useToast>) => void;
  }) {
    const api = useToast();
    React.useEffect(() => {
      onReady(api);
    });
    return <Text>host</Text>;
  }

  test('auto-dismiss after durationMs elapses', () => {
    let api: ReturnType<typeof useToast> | null = null;
    const { queryByText } = render(
      withProviders(<HostHarness onReady={(a) => (api = a)} />),
    );
    act(() => {
      api!.show({ variant: 'info', message: 'temporary', durationMs: 1000 });
    });
    expect(queryByText('temporary')).toBeTruthy();
    act(() => {
      // Advance just past the durationMs — auto timer fires, marks
      // dismissing.
      jest.advanceTimersByTime(1001);
    });
    act(() => {
      // Now flush the exit-reap window so the entry is removed.
      jest.advanceTimersByTime(400);
    });
    expect(queryByText('temporary')).toBeNull();
  });

  test('queue caps at 3: the 4th show() drops the oldest', () => {
    let api: ReturnType<typeof useToast> | null = null;
    const { queryByText } = render(
      withProviders(<HostHarness onReady={(a) => (api = a)} />),
    );
    act(() => {
      api!.show({ variant: 'info', message: 'first' });
      api!.show({ variant: 'info', message: 'second' });
      api!.show({ variant: 'info', message: 'third' });
    });
    expect(queryByText('first')).toBeTruthy();
    expect(queryByText('second')).toBeTruthy();
    expect(queryByText('third')).toBeTruthy();
    act(() => {
      api!.show({ variant: 'info', message: 'fourth' });
    });
    // The fourth toast pushes the oldest into dismissing — it'll still
    // be in the DOM until the exit-reap window flushes. Advance the
    // timers and verify "first" is gone.
    act(() => {
      jest.advanceTimersByTime(400);
    });
    expect(queryByText('first')).toBeNull();
    expect(queryByText('second')).toBeTruthy();
    expect(queryByText('third')).toBeTruthy();
    expect(queryByText('fourth')).toBeTruthy();
  });

  test('respects maxVisible override (cap = 1)', () => {
    let api: ReturnType<typeof useToast> | null = null;
    const { queryByText } = render(
      <SafeAreaProvider
        initialMetrics={{
          frame: { x: 0, y: 0, width: 320, height: 640 },
          insets: { top: 0, left: 0, right: 0, bottom: 0 },
        }}
      >
        <ToastProvider maxVisible={1}>
          <HostHarness onReady={(a) => (api = a)} />
        </ToastProvider>
      </SafeAreaProvider>,
    );
    act(() => {
      api!.show({ variant: 'info', message: 'alpha' });
      api!.show({ variant: 'info', message: 'beta' });
    });
    act(() => {
      jest.advanceTimersByTime(400);
    });
    // 'alpha' was bumped into dismissing on the second show().
    expect(queryByText('alpha')).toBeNull();
    expect(queryByText('beta')).toBeTruthy();
  });
});
