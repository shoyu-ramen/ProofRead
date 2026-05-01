/**
 * Hook-level tests for `useToast`. Validates that the hook adds and
 * removes entries from the queue exposed by `ToastProvider`, that
 * `dismiss` is a no-op for already-aged-out ids, and that the
 * out-of-provider fallback returns a sentinel function rather than
 * throwing (so a forgotten provider in a test setup degrades to a
 * console warning).
 */

import React from 'react';
import { Text } from 'react-native';
import { render, act } from '@testing-library/react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { useToast } from '../useToast';
import { ToastProvider } from '@src/components/ToastProvider';

beforeEach(() => {
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
});

function ConsumerHarness({
  onApi,
}: {
  onApi: (api: ReturnType<typeof useToast>) => void;
}) {
  const api = useToast();
  // Stash the api into the ref the test passes in, on every render.
  React.useEffect(() => {
    onApi(api);
  });
  return <Text>consumer</Text>;
}

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

test('show() returns a string id and queues the toast', () => {
  let api: ReturnType<typeof useToast> | null = null;
  const onApi = (a: ReturnType<typeof useToast>) => {
    api = a;
  };
  const { getByText } = render(
    withProviders(<ConsumerHarness onApi={onApi} />),
  );
  expect(getByText('consumer')).toBeTruthy();
  let id: string | null = null;
  act(() => {
    id = api!.show({ variant: 'info', message: 'hello' });
  });
  expect(typeof id).toBe('string');
  expect((id as unknown as string).length).toBeGreaterThan(0);
});

test('dismiss() removes the toast from the queue (eventually)', () => {
  let api: ReturnType<typeof useToast> | null = null;
  const onApi = (a: ReturnType<typeof useToast>) => {
    api = a;
  };
  const { queryByText } = render(
    withProviders(<ConsumerHarness onApi={onApi} />),
  );
  let id: string | null = null;
  act(() => {
    id = api!.show({ variant: 'success', message: 'flagged' });
  });
  expect(queryByText('flagged')).toBeTruthy();
  act(() => {
    api!.dismiss(id as unknown as string);
    // Advance past the exit-reap window.
    jest.advanceTimersByTime(400);
  });
  expect(queryByText('flagged')).toBeNull();
});

test('clear() drops every active toast', () => {
  let api: ReturnType<typeof useToast> | null = null;
  const onApi = (a: ReturnType<typeof useToast>) => {
    api = a;
  };
  const { queryByText } = render(
    withProviders(<ConsumerHarness onApi={onApi} />),
  );
  act(() => {
    api!.show({ variant: 'info', message: 'one' });
    api!.show({ variant: 'info', message: 'two' });
  });
  expect(queryByText('one')).toBeTruthy();
  expect(queryByText('two')).toBeTruthy();
  act(() => {
    api!.clear();
    jest.advanceTimersByTime(400);
  });
  expect(queryByText('one')).toBeNull();
  expect(queryByText('two')).toBeNull();
});

test('show() outside <ToastProvider> falls back to a logged no-op', () => {
  // Mount the consumer without the provider — the hook should hand back
  // a sentinel `show` that returns ''. We capture console.warn so the
  // fallback's own logging doesn't pollute the test runner.
  const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  let api: ReturnType<typeof useToast> | null = null;
  const onApi = (a: ReturnType<typeof useToast>) => {
    api = a;
  };
  render(<ConsumerHarness onApi={onApi} />);
  let id: string | null = null;
  act(() => {
    id = api!.show({ variant: 'error', message: 'no provider' });
  });
  expect(id).toBe('');
  expect(warnSpy).toHaveBeenCalledWith(
    expect.stringContaining('called outside <ToastProvider>'),
    'no provider',
  );
  warnSpy.mockRestore();
});
