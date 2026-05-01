/**
 * ToastContext — context plumbing shared between `ToastProvider`
 * (the implementation), `useToast` (the consumer hook), and the
 * `Toast` view (which only renders, never reaches into context).
 *
 * Keeping the context object in its own module avoids the circular
 * import that arises if `useToast` imports from `ToastProvider` and
 * `ToastProvider` re-exports the hook for convenience.
 */

import { createContext } from 'react';

export type ToastVariant = 'info' | 'success' | 'warning' | 'error';

export interface ToastShowOptions {
  variant: ToastVariant;
  message: string;
  /**
   * How long the toast remains visible after slide-in completes.
   * Defaults to `toastMotion.defaultDurationMs` (4000ms).
   */
  durationMs?: number;
}

export interface ToastContextValue {
  show: (options: ToastShowOptions) => string;
  dismiss: (id: string) => void;
  clear: () => void;
}

export const ToastContext = createContext<ToastContextValue | null>(null);
