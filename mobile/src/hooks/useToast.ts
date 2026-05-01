/**
 * useToast — public hook for the in-app toast subsystem.
 *
 * Returns the imperative `show()` API and a `dismiss()` helper. The
 * actual queue + render lives in `ToastProvider` (mounted once at the
 * app root). The hook just consumes the React Context the provider
 * exposes; if it's called outside the provider tree (e.g. in a unit
 * test that forgot to wrap the component), `show` is a logged no-op
 * so a missing provider never crashes a screen — just silently swallows
 * the alert. Tests that care that the toast fired should wrap with
 * `<ToastProvider>` rather than rely on a thrown error.
 */

import { useContext } from 'react';
import { ToastContext, type ToastShowOptions, type ToastVariant } from '../components/ToastContext';

export type { ToastShowOptions, ToastVariant };

export interface UseToastApi {
  /**
   * Surface a toast. Returns the toast id so the caller can dismiss
   * it programmatically (e.g. once an in-flight request settles).
   */
  show: (options: ToastShowOptions) => string;
  /** Dismiss a specific toast by id. No-op if the id has already aged out. */
  dismiss: (id: string) => void;
  /** Drop everything currently visible or queued. */
  clear: () => void;
}

export function useToast(): UseToastApi {
  const ctx = useContext(ToastContext);
  if (ctx) return ctx;
  // Soft fallback — see file header. We log once per call rather than
  // throw so a forgotten <ToastProvider> in a test or storybook host
  // doesn't take down the surface under inspection.
  return {
    show: (options) => {
      console.warn(
        '[useToast] called outside <ToastProvider>; toast not shown:',
        options.message,
      );
      return '';
    },
    dismiss: () => {
      // no-op
    },
    clear: () => {
      // no-op
    },
  };
}
