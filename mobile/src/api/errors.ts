/**
 * Plain-language error rendering for ApiError + unknown failures.
 *
 * Backend errors come back as RFC 7807 problem-details (per SPEC §v1.9)
 * which are great for support and useless for end-users — the raw
 * `detail` string is often a stack-trace fragment or a config key name
 * (see /Users/ross/ProofRead/result-error.png and verify-error-503.png
 * for the kind of strings users currently see). This module keeps that
 * detail available for support workflows but classifies failures into
 * coarse user-facing copy at the call site.
 *
 * Usage at a call site:
 *
 *   try { await apiClient.createScan(req); }
 *   catch (err) { showErrorAlert(err); }
 *
 * showErrorAlert renders the plain title+message with an optional
 * "Show details" button that opens a second Alert containing the raw
 * server string. Two-tap reveal so support has the diagnostic without
 * exposing it as the primary message.
 */

import { Alert } from 'react-native';

import { ApiError } from './types';

export type ErrorClass =
  | 'network' // fetch threw — DNS, offline, captive portal, TLS
  | 'auth' // 401 / 403
  | 'not_found' // 404
  | 'rate_limited' // 429
  | 'client' // other 4xx
  | 'server' // 5xx
  | 'unknown';

export interface RenderedError {
  /** One-line title for an Alert / banner. */
  title: string;
  /** Plain-language body. Never includes raw server detail. */
  message: string;
  /** The classifier's verdict, useful for tests + analytics. */
  kind: ErrorClass;
  /**
   * The raw underlying string we hid from the user. Surface behind a
   * "Show details" / "Copy details" affordance for support, never as
   * the primary message.
   */
  technical: string | null;
}

export function describeError(err: unknown): RenderedError {
  if (err instanceof ApiError) {
    const kind = classifyStatus(err.status);
    return {
      kind,
      title: titleFor(kind),
      message: messageFor(kind),
      technical: technicalDetail(err),
    };
  }
  // Anything that isn't an ApiError reached us before fetch returned —
  // typically a TypeError from `fetch()` (offline, DNS failure, TLS).
  return {
    kind: 'network',
    title: titleFor('network'),
    message: messageFor('network'),
    technical: err instanceof Error ? err.message : String(err),
  };
}

function classifyStatus(status: number): ErrorClass {
  if (status === 0) return 'network';
  if (status === 401 || status === 403) return 'auth';
  if (status === 404) return 'not_found';
  if (status === 429) return 'rate_limited';
  if (status >= 400 && status < 500) return 'client';
  if (status >= 500) return 'server';
  return 'unknown';
}

function titleFor(kind: ErrorClass): string {
  switch (kind) {
    case 'network':
      return "Can't reach the server";
    case 'auth':
      return 'Sign-in needed';
    case 'not_found':
      return 'Not found';
    case 'rate_limited':
      return 'Too many requests';
    case 'client':
      return "Couldn't process this request";
    case 'server':
      return 'Server problem';
    case 'unknown':
      return 'Something went wrong';
  }
}

function messageFor(kind: ErrorClass): string {
  switch (kind) {
    case 'network':
      return "Check your internet connection and try again. If you're on hotel or conference Wi-Fi, you may need to sign in to the network first.";
    case 'auth':
      return 'Your session expired. Sign in again to continue.';
    case 'not_found':
      return "We couldn't find what you were looking for.";
    case 'rate_limited':
      return "You've made a lot of requests in a short time. Wait a minute and try again.";
    case 'client':
      return 'The request was rejected. Double-check the inputs and try again.';
    case 'server':
      return "The server hit an error. We're probably already on it — try again in a moment.";
    case 'unknown':
      return 'An unexpected error occurred. Try again, or reach out to support if it keeps happening.';
  }
}

/**
 * Best-effort technical string for support workflows. We prefer the
 * RFC 7807 `detail` (which usually carries the FastAPI exception
 * message), then fall back to the title or HTTP status line so the
 * support surface is never empty.
 */
function technicalDetail(err: ApiError): string {
  const parts: string[] = [`HTTP ${err.status}`];
  const detail = err.problem?.detail;
  if (typeof detail === 'string' && detail.length > 0) {
    parts.push(detail);
  } else if (err.problem?.title) {
    parts.push(String(err.problem.title));
  } else if (err.message) {
    parts.push(err.message);
  }
  return parts.join(' — ');
}

export interface ShowErrorAlertOptions {
  /** Override the auto-derived title (e.g. "Couldn't submit scan"). */
  title?: string;
  /** Called after the user dismisses the alert (either button). */
  onDismiss?: () => void;
}

/**
 * Render an error as a two-tap Alert: plain message first, then an
 * optional "Show details" button that reveals the raw server string.
 *
 * `Alert.alert` is intentional here — it's zero-dep, native-feel, and
 * keeps every screen's error path identical. A richer in-app surface
 * (toast, banner) is a Phase-2 design conversation.
 */
export function showErrorAlert(
  err: unknown,
  opts: ShowErrorAlertOptions = {},
): void {
  const v = describeError(err);
  const title = opts.title ?? v.title;

  const buttons: { text: string; onPress?: () => void; style?: 'default' | 'cancel' }[] = [
    { text: 'OK', onPress: opts.onDismiss, style: 'cancel' },
  ];
  if (v.technical) {
    buttons.unshift({
      text: 'Show details',
      onPress: () => {
        Alert.alert('Technical details', v.technical ?? '', [
          { text: 'Close', onPress: opts.onDismiss, style: 'cancel' },
        ]);
      },
    });
  }

  Alert.alert(title, v.message, buttons);
}
