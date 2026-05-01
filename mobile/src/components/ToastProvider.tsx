/**
 * ToastProvider — owns the queue + render position for app-wide toasts.
 *
 * Mounted once at the root layout (`mobile/app/_layout.tsx`) so any
 * screen can call `useToast().show(...)` without prop drilling. The
 * provider:
 *
 *   - Maintains an ordered queue of active toasts. Newest renders at
 *     the bottom of the visible stack so it draws the eye without
 *     pushing older messages off-screen abruptly.
 *   - Caps visible toasts at `toastMotion.maxVisible` (3). Overflow
 *     triggers an immediate dismiss-fade on the oldest entry. We do
 *     NOT queue silently for later — once the maximum is hit, older
 *     messages drop. This matches the brief "max 3 visible. Older
 *     overflow drops".
 *   - Auto-dismisses each toast after `durationMs` (default 4000ms).
 *   - Tapping a toast triggers an immediate dismiss via the same
 *     fade-out path the auto-dismiss uses.
 *   - Sits at `top = safeAreaInsets.top + 12` so notch / status-bar
 *     areas don't clip the top of the chip.
 *
 * The provider's render output is the children + an absolutely-positioned
 * stack overlay rendered last, so toasts always paint above sibling
 * surfaces (camera, modals are app-modal so they stack via the OS).
 */

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { spacing, toastMotion } from '@src/theme';
import { Toast } from './Toast';
import {
  ToastContext,
  type ToastContextValue,
  type ToastShowOptions,
  type ToastVariant,
} from './ToastContext';

export interface ToastProviderProps {
  children: React.ReactNode;
  /**
   * Override for testing — when provided, this is used instead of the
   * default duration token. Per-call `show({ durationMs })` still wins.
   */
  defaultDurationMs?: number;
  /** Override for testing — caps visible toasts. */
  maxVisible?: number;
}

interface ToastEntry {
  id: string;
  variant: ToastVariant;
  message: string;
  durationMs: number;
  /** Wall-clock when the entry was pushed. */
  createdAt: number;
  /**
   * Whether the toast is currently animating out. Once true the
   * autoshown timer is cleared and the entry is removed shortly after
   * the exit animation completes (timer fallback so a missed callback
   * doesn't leak the entry forever).
   */
  dismissing: boolean;
}

let toastIdCounter = 0;
function nextToastId(): string {
  toastIdCounter += 1;
  return `toast-${Date.now()}-${toastIdCounter}`;
}

// How long we keep an entry in the queue after `dismissing` flips to
// true. Pinned to the toast's exit-animation duration plus a little
// padding — covers the edge case where the on-completion callback
// doesn't fire (component unmounted mid-exit).
const EXIT_REAP_MS = toastMotion.fastEase.duration + 80;

export function ToastProvider({
  children,
  defaultDurationMs = toastMotion.defaultDurationMs,
  maxVisible = toastMotion.maxVisible,
}: ToastProviderProps): React.ReactElement {
  const insets = useSafeAreaInsets();
  const [entries, setEntries] = useState<ToastEntry[]>([]);

  // Active timeouts keyed by toast id — cleared on dismiss or unmount
  // so we never callback into a stale state.
  const autoTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map(),
  );
  const reapTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map(),
  );

  const clearAutoTimer = useCallback((id: string) => {
    const map = autoTimersRef.current;
    const t = map.get(id);
    if (t !== undefined) {
      clearTimeout(t);
      map.delete(id);
    }
  }, []);

  const clearReapTimer = useCallback((id: string) => {
    const map = reapTimersRef.current;
    const t = map.get(id);
    if (t !== undefined) {
      clearTimeout(t);
      map.delete(id);
    }
  }, []);

  // Hard remove — splice the entry and cancel all its timers.
  const removeEntry = useCallback(
    (id: string) => {
      clearAutoTimer(id);
      clearReapTimer(id);
      setEntries((prev) => prev.filter((e) => e.id !== id));
    },
    [clearAutoTimer, clearReapTimer],
  );

  // Mark the entry as dismissing — kicks off the exit animation in the
  // child Toast view, then schedules a hard remove after the exit
  // window. Idempotent: re-calling on an already-dismissing entry is
  // a no-op (we do NOT re-arm the reap timer because the original is
  // still in flight).
  const beginDismiss = useCallback(
    (id: string) => {
      setEntries((prev) => {
        const idx = prev.findIndex((e) => e.id === id);
        if (idx === -1) return prev;
        if (prev[idx].dismissing) return prev;
        const next = prev.slice();
        next[idx] = { ...prev[idx], dismissing: true };
        return next;
      });
      clearAutoTimer(id);
      // Schedule the hard reap so a missed animation callback can't
      // leak the entry.
      const reapMap = reapTimersRef.current;
      if (!reapMap.has(id)) {
        const timer = setTimeout(() => {
          reapMap.delete(id);
          removeEntry(id);
        }, EXIT_REAP_MS);
        reapMap.set(id, timer);
      }
    },
    [clearAutoTimer, removeEntry],
  );

  const show = useCallback(
    (options: ToastShowOptions): string => {
      const id = nextToastId();
      const durationMs =
        typeof options.durationMs === 'number' && options.durationMs > 0
          ? options.durationMs
          : defaultDurationMs;
      setEntries((prev) => {
        const next = prev.slice();
        next.push({
          id,
          variant: options.variant,
          message: options.message,
          durationMs,
          createdAt: Date.now(),
          dismissing: false,
        });
        // Overflow handling: if more than maxVisible are present (and
        // not yet dismissing), kick off dismissal on the oldest non-
        // dismissing entries until we're back at the cap. We mutate
        // the entries inline by flagging dismissing rather than splicing
        // — that way the older toast still plays an exit animation
        // instead of pop-disappearing.
        const aliveCount = next.filter((e) => !e.dismissing).length;
        if (aliveCount > maxVisible) {
          const overflow = aliveCount - maxVisible;
          let removed = 0;
          for (let i = 0; i < next.length && removed < overflow; i += 1) {
            if (!next[i].dismissing) {
              next[i] = { ...next[i], dismissing: true };
              removed += 1;
            }
          }
        }
        return next;
      });
      // Schedule auto-dismiss. The timer fires on the JS thread so the
      // dismiss path is identical to a tap dismiss.
      const timer = setTimeout(() => {
        autoTimersRef.current.delete(id);
        beginDismiss(id);
      }, durationMs);
      autoTimersRef.current.set(id, timer);
      return id;
    },
    [beginDismiss, defaultDurationMs, maxVisible],
  );

  const dismiss = useCallback(
    (id: string) => {
      beginDismiss(id);
    },
    [beginDismiss],
  );

  const clear = useCallback(() => {
    // Snapshot active ids first so we don't mutate while iterating.
    const ids = entries.map((e) => e.id);
    for (const id of ids) beginDismiss(id);
  }, [beginDismiss, entries]);

  // For overflow-driven dismissals we also need to schedule the reap
  // timer (the show() path only sets the auto-dismiss timer for the
  // newly-added entry). React's strict-mode-safe pattern: run a single
  // effect that ensures every dismissing entry has an in-flight reap.
  useEffect(() => {
    const reapMap = reapTimersRef.current;
    for (const e of entries) {
      if (e.dismissing && !reapMap.has(e.id)) {
        const id = e.id;
        const timer = setTimeout(() => {
          reapMap.delete(id);
          removeEntry(id);
        }, EXIT_REAP_MS);
        reapMap.set(id, timer);
      }
    }
  }, [entries, removeEntry]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      for (const t of autoTimersRef.current.values()) clearTimeout(t);
      for (const t of reapTimersRef.current.values()) clearTimeout(t);
      autoTimersRef.current.clear();
      reapTimersRef.current.clear();
    };
  }, []);

  const handleAnimationComplete = useCallback(
    (id: string) => {
      // The Toast view fires this when its exit animation finishes.
      // Remove the entry now rather than waiting for the reap timer —
      // saves up to EXIT_REAP_MS of stale render budget.
      removeEntry(id);
    },
    [removeEntry],
  );

  // Memo the context value so unrelated re-renders of children don't
  // re-trigger their `useEffect`s that depend on the api object.
  const ctxValue = useMemo<ToastContextValue>(
    () => ({ show, dismiss, clear }),
    [show, dismiss, clear],
  );

  return (
    <ToastContext.Provider value={ctxValue}>
      {children}
      {/* Stack overlay — sits above all children, never blocks input
          outside the toast cards themselves. */}
      <View
        pointerEvents="box-none"
        style={[
          styles.stack,
          { top: insets.top + spacing.sm, paddingHorizontal: spacing.md },
        ]}
      >
        {entries.map((entry) => (
          <View
            key={entry.id}
            style={{ marginBottom: toastMotion.stackGapPx }}
            pointerEvents="auto"
          >
            <Toast
              id={entry.id}
              variant={entry.variant}
              message={entry.message}
              dismissing={entry.dismissing}
              onDismiss={dismiss}
              onAnimationComplete={handleAnimationComplete}
            />
          </View>
        ))}
      </View>
    </ToastContext.Provider>
  );
}

const styles = StyleSheet.create({
  stack: {
    position: 'absolute',
    left: 0,
    right: 0,
    // pointerEvents on the wrapper is `box-none` so taps land on the
    // underlying screen unless they hit a Toast card directly.
    alignItems: 'stretch',
  },
});
