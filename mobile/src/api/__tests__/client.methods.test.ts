/**
 * Coverage for the ApiClient methods the existing client.test.ts didn't
 * touch: createScan, uploadImage (signed-URL PUT), getScan, getReport,
 * getHistory/listScans, flagRuleResult, plus the shared `request<T>`
 * concerns — bearer-token attachment, the 204/no-body path, and the
 * error message fallback chain when the error body isn't JSON.
 *
 * fetch is mocked so each request's URL / method / headers / body can be
 * inspected without a live backend.
 */

import { ApiClient } from '../client';
import { ApiError } from '../types';

function ok(body: unknown, status = 200) {
  return {
    ok: true,
    status,
    statusText: 'OK',
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

function err(status: number, body: unknown, jsonThrows = false) {
  return {
    ok: false,
    status,
    statusText: `HTTP ${status}`,
    json: () =>
      jsonThrows ? Promise.reject(new Error('not json')) : Promise.resolve(body),
  } as unknown as Response;
}

let originalFetch: typeof fetch;
beforeEach(() => {
  originalFetch = global.fetch;
});
afterEach(() => {
  global.fetch = originalFetch;
});

function withFetch(impl: jest.Mock) {
  global.fetch = impl as unknown as typeof fetch;
  return impl;
}

describe('ApiClient — createScan', () => {
  test('POSTs JSON to /v1/scans and returns the response', async () => {
    const fetchMock = withFetch(
      jest.fn(() => Promise.resolve(ok({ scan_id: 's1', upload_urls: [] }))),
    );
    const client = new ApiClient({ baseUrl: 'http://test.local' });

    const res = await client.createScan({
      beverage_type: 'beer',
      container_size_ml: 355,
      is_imported: false,
    });

    expect(res.scan_id).toBe('s1');
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('http://test.local/v1/scans');
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>)['Content-Type']).toBe(
      'application/json',
    );
    expect(JSON.parse(init.body as string)).toMatchObject({ beverage_type: 'beer' });
  });
});

describe('ApiClient — uploadImage', () => {
  test('PUTs the body to the signed URL verbatim without an auth header', async () => {
    const fetchMock = withFetch(jest.fn(() => Promise.resolve(ok(undefined, 204))));
    // A client *with* a token — uploadImage must NOT attach it (signed
    // URLs may point at S3 where an Authorization header breaks the sig).
    const client = new ApiClient({
      baseUrl: 'http://test.local',
      getToken: () => 'secret-token',
    });

    const body = new Uint8Array([1, 2, 3]);
    await client.uploadImage('http://signed.example/put', body, 'image/png');

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('http://signed.example/put');
    expect(init.method).toBe('PUT');
    const headers = init.headers as Record<string, string>;
    expect(headers['Content-Type']).toBe('image/png');
    expect(headers.Authorization).toBeUndefined();
    expect(init.body).toBe(body);
  });

  test('defaults the content type to image/jpeg', async () => {
    const fetchMock = withFetch(jest.fn(() => Promise.resolve(ok(undefined, 204))));
    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await client.uploadImage('http://signed.example/put', new Uint8Array([0]));
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('image/jpeg');
  });

  test('throws ApiError carrying the status when the PUT fails', async () => {
    withFetch(jest.fn(() => Promise.resolve(err(413, null))));
    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await expect(
      client.uploadImage('http://signed.example/put', new Uint8Array([0])),
    ).rejects.toMatchObject({ status: 413, name: 'ApiError' });
  });
});

describe('ApiClient — read endpoints', () => {
  test('getScan GETs /v1/scans/:id', async () => {
    const fetchMock = withFetch(
      jest.fn(() => Promise.resolve(ok({ scan_id: 's1', status: 'complete' }))),
    );
    const client = new ApiClient({ baseUrl: 'http://test.local' });
    const res = await client.getScan('s1');
    expect(res.status).toBe('complete');
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('http://test.local/v1/scans/s1');
    expect(init.method).toBe('GET');
  });

  test('getReport GETs /v1/scans/:id/report', async () => {
    const fetchMock = withFetch(
      jest.fn(() =>
        Promise.resolve(ok({ scan_id: 's1', overall: 'pass', rule_results: [] })),
      ),
    );
    const client = new ApiClient({ baseUrl: 'http://test.local' });
    const res = await client.getReport('s1');
    expect(res.overall).toBe('pass');
    expect((fetchMock.mock.calls[0] as unknown as [string])[0]).toBe(
      'http://test.local/v1/scans/s1/report',
    );
  });

  test('getHistory and listScans both hit GET /v1/scans', async () => {
    const fetchMock = withFetch(jest.fn(() => Promise.resolve(ok({ items: [] }))));
    const client = new ApiClient({ baseUrl: 'http://test.local' });

    await client.getHistory();
    await client.listScans();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    for (const call of fetchMock.mock.calls) {
      const [url, init] = call as unknown as [string, RequestInit];
      expect(url).toBe('http://test.local/v1/scans');
      expect(init.method).toBe('GET');
    }
  });
});

describe('ApiClient — flagRuleResult', () => {
  test('POSTs JSON, URL-encodes the rule id, and resolves void on 204', async () => {
    const fetchMock = withFetch(jest.fn(() => Promise.resolve(ok(undefined, 204))));
    const client = new ApiClient({ baseUrl: 'http://test.local' });

    const result = await client.flagRuleResult('s1', 'beer.health_warning.exact_text', {
      comment: 'looks fine',
    });

    expect(result).toBeUndefined(); // parseJson:false + 204 → no body parse
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(
      'http://test.local/v1/scans/s1/rule-results/beer.health_warning.exact_text/flag',
    );
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ comment: 'looks fine' });
  });

  test('encodes rule ids that contain URL-special characters', async () => {
    const fetchMock = withFetch(jest.fn(() => Promise.resolve(ok(undefined, 204))));
    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await client.flagRuleResult('s1', 'rule/with space', { comment: 'x' });
    const [url] = fetchMock.mock.calls[0] as unknown as [string];
    expect(url).toContain('rule%2Fwith%20space');
  });
});

describe('ApiClient — request<T> shared behavior', () => {
  test('attaches the bearer token from getToken (sync and async)', async () => {
    const fetchMock = withFetch(jest.fn(() => Promise.resolve(ok({ items: [] }))));
    const client = new ApiClient({
      baseUrl: 'http://test.local',
      getToken: () => Promise.resolve('jwt-abc'),
    });
    await client.getHistory();
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer jwt-abc');
  });

  test('omits Authorization when no token is configured', async () => {
    const fetchMock = withFetch(jest.fn(() => Promise.resolve(ok({ items: [] }))));
    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await client.getHistory();
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect((init.headers as Record<string, string>).Authorization).toBeUndefined();
  });

  test('surfaces the RFC 7807 detail as the thrown ApiError message', async () => {
    withFetch(jest.fn(() => Promise.resolve(err(422, { detail: 'bad field x' }))));
    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await expect(client.getScan('s1')).rejects.toThrow('bad field x');
  });

  test('falls back to status text when the error body is not JSON', async () => {
    withFetch(jest.fn(() => Promise.resolve(err(500, null, /* jsonThrows */ true))));
    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await expect(client.getScan('s1')).rejects.toMatchObject({
      status: 500,
      message: 'HTTP 500',
    });
  });

  test('strips a trailing slash from the configured base URL', async () => {
    const fetchMock = withFetch(jest.fn(() => Promise.resolve(ok({ items: [] }))));
    const client = new ApiClient({ baseUrl: 'http://test.local/' });
    await client.getHistory();
    expect((fetchMock.mock.calls[0] as unknown as [string])[0]).toBe(
      'http://test.local/v1/scans',
    );
  });
});
