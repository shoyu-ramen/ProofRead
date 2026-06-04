/**
 * Unit tests for the plain-language error renderer (errors.ts).
 *
 * `describeError` is the pure classifier the whole app routes failures
 * through (SPEC §v1.9: RFC 7807 problem-details → user-facing copy).
 * It had no coverage, yet it decides every error string the user sees.
 * These tests pin the status → ErrorClass mapping, the "raw detail is
 * never the primary message but is preserved for support" contract, and
 * the two-tap `showErrorAlert` reveal.
 */

import { Alert } from 'react-native';

import { describeError, showErrorAlert } from '../errors';
import { ApiError } from '../types';

describe('describeError — status classification', () => {
  const cases: { status: number; kind: string }[] = [
    { status: 401, kind: 'auth' },
    { status: 403, kind: 'auth' },
    { status: 404, kind: 'not_found' },
    { status: 429, kind: 'rate_limited' },
    { status: 400, kind: 'client' },
    { status: 422, kind: 'client' },
    { status: 500, kind: 'server' },
    { status: 503, kind: 'server' },
    { status: 0, kind: 'network' },
  ];

  test.each(cases)('HTTP $status → $kind', ({ status, kind }) => {
    const v = describeError(new ApiError('boom', status));
    expect(v.kind).toBe(kind);
    expect(v.title.length).toBeGreaterThan(0);
    expect(v.message.length).toBeGreaterThan(0);
  });

  test('an out-of-range status falls through to unknown', () => {
    // 399 is neither <400 handled-case nor >=400; the classifier guards
    // only the 4xx/5xx ranges, so this lands on unknown.
    const v = describeError(new ApiError('weird', 399));
    expect(v.kind).toBe('unknown');
  });
});

describe('describeError — non-ApiError failures', () => {
  test('a thrown TypeError (offline / DNS) classifies as network', () => {
    const v = describeError(new TypeError('Network request failed'));
    expect(v.kind).toBe('network');
    // The raw message is preserved for support, not surfaced as the body.
    expect(v.technical).toBe('Network request failed');
    expect(v.message).not.toContain('Network request failed');
  });

  test('a non-Error throwable is stringified into technical', () => {
    const v = describeError('something odd');
    expect(v.kind).toBe('network');
    expect(v.technical).toBe('something odd');
  });
});

describe('describeError — technical detail (support reveal)', () => {
  test('prefers the RFC 7807 detail field', () => {
    const err = new ApiError('fallback msg', 500, {
      title: 'Internal Server Error',
      detail: 'OPENAI_API_KEY is not configured',
    });
    const v = describeError(err);
    expect(v.technical).toBe('HTTP 500 — OPENAI_API_KEY is not configured');
    // ...but the developer-facing string never leaks into the body.
    expect(v.message).not.toContain('OPENAI_API_KEY');
  });

  test('falls back to problem.title when detail is absent', () => {
    const err = new ApiError('msg', 502, { title: 'Bad Gateway' });
    expect(describeError(err).technical).toBe('HTTP 502 — Bad Gateway');
  });

  test('falls back to the Error message when problem is null', () => {
    const err = new ApiError('raw transport message', 503);
    expect(describeError(err).technical).toBe('HTTP 503 — raw transport message');
  });

  test('still yields a non-empty technical string with no detail/title/message', () => {
    const err = new ApiError('', 500, {});
    expect(describeError(err).technical).toBe('HTTP 500');
  });
});

describe('showErrorAlert', () => {
  let alertSpy: jest.SpyInstance;

  beforeEach(() => {
    alertSpy = jest.spyOn(Alert, 'alert').mockImplementation(() => {});
  });
  afterEach(() => {
    alertSpy.mockRestore();
  });

  test('renders title + plain message and offers a "Show details" button', () => {
    showErrorAlert(new ApiError('msg', 500, { detail: 'stack trace fragment' }));
    expect(alertSpy).toHaveBeenCalledTimes(1);
    const [title, message, buttons] = alertSpy.mock.calls[0];
    expect(title).toBe('Server problem');
    expect(message).not.toContain('stack trace fragment');
    const labels = (buttons as { text: string }[]).map((b) => b.text);
    expect(labels).toEqual(['Show details', 'OK']);
  });

  test('omits the details button when there is no technical string', () => {
    // describeError always builds at least "HTTP <status>" for an
    // ApiError, so drive the no-technical path via a plain object that
    // stringifies — actually network errors always carry technical too.
    // Use an ApiError but assert the button set is the OK-only shape
    // only when technical is falsy: force it by stubbing describe? The
    // simplest real no-technical case is an empty-string non-error.
    showErrorAlert('');
    const [, , buttons] = alertSpy.mock.calls[0];
    const labels = (buttons as { text: string }[]).map((b) => b.text);
    // '' stringifies to '' → technical is '' (falsy) → OK only.
    expect(labels).toEqual(['OK']);
  });

  test('honors a caller-supplied title override', () => {
    showErrorAlert(new ApiError('msg', 422, { detail: 'bad field' }), {
      title: "Couldn't submit scan",
    });
    const [title] = alertSpy.mock.calls[0];
    expect(title).toBe("Couldn't submit scan");
  });

  test('tapping "Show details" opens a second alert with the raw string', () => {
    showErrorAlert(new ApiError('msg', 500, { detail: 'CONFIG_KEY missing' }));
    const [, , buttons] = alertSpy.mock.calls[0];
    const details = (buttons as { text: string; onPress?: () => void }[]).find(
      (b) => b.text === 'Show details',
    )!;
    details.onPress!();
    expect(alertSpy).toHaveBeenCalledTimes(2);
    const [secondTitle, secondBody] = alertSpy.mock.calls[1];
    expect(secondTitle).toBe('Technical details');
    expect(secondBody).toContain('CONFIG_KEY missing');
  });

  test('invokes onDismiss from the OK button', () => {
    const onDismiss = jest.fn();
    showErrorAlert(new ApiError('msg', 404), { onDismiss });
    const [, , buttons] = alertSpy.mock.calls[0];
    const ok = (buttons as { text: string; onPress?: () => void }[]).find(
      (b) => b.text === 'OK',
    )!;
    ok.onPress!();
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });
});
