/**
 * Tests for the known-label additions to ApiClient (Decision 4 +
 * Decision 5 in KNOWN_LABEL_DESIGN.md):
 *
 *   - detectContainer parses image_dhash + known_label off the response
 *   - createScanFromCache calls POST /v1/scans/from-cache with the
 *     correct body
 *   - finalizeScan forwards first_frame_signature_hex as a multipart
 *     form field when provided
 *
 * fetch is mocked globally so the request shape can be inspected.
 */

import { ApiClient, type DetectContainerResponse } from '../client';
import type { KnownLabelPayload } from '../types';

function mockFetchResponse(
  body: unknown,
  init: { ok?: boolean; status?: number } = {},
) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: 'OK',
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

function makeKnownLabel(): KnownLabelPayload {
  return {
    entry_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    beverage_type: 'beer',
    container_size_ml: 355,
    is_imported: false,
    brand_name: 'Sierra Nevada',
    fanciful_name: 'Pale Ale',
    verdict_summary: {
      overall: 'pass',
      rule_results: [
        {
          rule_id: 'beer.brand_name.presence',
          status: 'pass',
          citation: 'TTB §7.22(a)',
          finding: null,
          explanation: null,
          fix_suggestion: null,
        },
      ],
      extracted: {
        brand_name: { value: 'Sierra Nevada', confidence: 0.95 },
      },
      image_quality: 'good',
    },
    source: 'brand',
  };
}

describe('ApiClient — detectContainer', () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  test('parses image_dhash + known_label when present', async () => {
    const knownLabel = makeKnownLabel();
    const responseBody: DetectContainerResponse = {
      detected: true,
      container_type: 'bottle',
      bbox: [0.1, 0.2, 0.8, 0.9],
      confidence: 0.92,
      reason: null,
      brand_name: 'Sierra Nevada',
      net_contents: '12 FL OZ (355 mL)',
      image_dhash: 'deadbeefcafebabe',
      known_label: knownLabel,
    };
    global.fetch = jest.fn(() =>
      Promise.resolve(mockFetchResponse(responseBody)),
    ) as unknown as typeof fetch;

    const client = new ApiClient({ baseUrl: 'http://test.local' });
    const result = await client.detectContainer('file:///tmp/snap.jpg');

    expect(result.image_dhash).toBe('deadbeefcafebabe');
    expect(result.known_label).toEqual(knownLabel);
    expect(result.brand_name).toBe('Sierra Nevada');
    expect(result.net_contents).toBe('12 FL OZ (355 mL)');
  });

  test('tolerates absent image_dhash + known_label (older backend)', async () => {
    const responseBody = {
      detected: true,
      container_type: 'bottle',
      bbox: [0.1, 0.2, 0.8, 0.9],
      confidence: 0.92,
      reason: null,
    };
    global.fetch = jest.fn(() =>
      Promise.resolve(mockFetchResponse(responseBody)),
    ) as unknown as typeof fetch;

    const client = new ApiClient({ baseUrl: 'http://test.local' });
    const result = await client.detectContainer('file:///tmp/snap.jpg');

    expect(result.image_dhash).toBeUndefined();
    expect(result.known_label).toBeUndefined();
    expect(result.detected).toBe(true);
  });

  test('parses known_label=null on miss', async () => {
    const responseBody: DetectContainerResponse = {
      detected: true,
      container_type: 'can',
      bbox: [0.1, 0.2, 0.8, 0.9],
      confidence: 0.88,
      reason: null,
      image_dhash: '0123456789abcdef',
      known_label: null,
    };
    global.fetch = jest.fn(() =>
      Promise.resolve(mockFetchResponse(responseBody)),
    ) as unknown as typeof fetch;

    const client = new ApiClient({ baseUrl: 'http://test.local' });
    const result = await client.detectContainer('file:///tmp/snap.jpg');

    expect(result.known_label).toBeNull();
    expect(result.image_dhash).toBe('0123456789abcdef');
  });
});

describe('ApiClient — createScanFromCache', () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  test('POSTs to /v1/scans/from-cache with the request body', async () => {
    const responseBody = {
      scan_id: 'scan-123',
      status: 'complete',
      overall: 'pass',
      image_quality: 'good',
    };
    const fetchMock = jest.fn(() =>
      Promise.resolve(mockFetchResponse(responseBody)),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const client = new ApiClient({ baseUrl: 'http://test.local' });
    const result = await client.createScanFromCache({
      entry_id: 'entry-uuid',
      beverage_type: 'beer',
      container_size_ml: 355,
      is_imported: false,
    });

    expect(result.scan_id).toBe('scan-123');
    expect(result.status).toBe('complete');
    expect(result.overall).toBe('pass');
    expect(result.image_quality).toBe('good');

    // Inspect the request shape — URL, method, body.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = (fetchMock.mock.calls[0] ?? []) as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe('http://test.local/v1/scans/from-cache');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      entry_id: 'entry-uuid',
      beverage_type: 'beer',
      container_size_ml: 355,
      is_imported: false,
    });
    const headers = init.headers as Record<string, string>;
    expect(headers['Content-Type']).toBe('application/json');
  });

  test('throws ApiError on non-2xx response', async () => {
    const fetchMock = jest.fn(() =>
      Promise.resolve(
        mockFetchResponse(
          { detail: 'unknown entry_id' },
          { ok: false, status: 404 },
        ),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await expect(
      client.createScanFromCache({
        entry_id: 'missing',
        beverage_type: 'beer',
        container_size_ml: 355,
        is_imported: false,
      }),
    ).rejects.toMatchObject({ status: 404 });
  });
});

describe('ApiClient — finalizeScan', () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  test('POSTs without a body when no first_frame_signature_hex provided', async () => {
    const fetchMock = jest.fn(() =>
      Promise.resolve(
        mockFetchResponse({
          scan_id: 'scan-1',
          status: 'complete',
          overall: 'pass',
        }),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await client.finalizeScan('scan-1');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = (fetchMock.mock.calls[0] ?? []) as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe('http://test.local/v1/scans/scan-1/finalize');
    expect(init.method).toBe('POST');
    expect(init.body).toBeUndefined();
  });

  test('attaches first_frame_signature_hex as a multipart form field', async () => {
    const fetchMock = jest.fn(() =>
      Promise.resolve(
        mockFetchResponse({
          scan_id: 'scan-1',
          status: 'complete',
          overall: 'pass',
        }),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await client.finalizeScan('scan-1', {
      firstFrameSignatureHex: 'deadbeefcafebabe',
    });

    const [, init] = (fetchMock.mock.calls[0] ?? []) as unknown as [
      string,
      RequestInit,
    ];
    // FormData body — assert it is a FormData and contains the field.
    expect(init.body).toBeInstanceOf(FormData);
    const fd = init.body as FormData;
    expect(fd.get('first_frame_signature_hex')).toBe('deadbeefcafebabe');
  });

  test('falls back to plain POST when firstFrameSignatureHex is null', async () => {
    const fetchMock = jest.fn(() =>
      Promise.resolve(
        mockFetchResponse({
          scan_id: 'scan-1',
          status: 'complete',
          overall: 'pass',
        }),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const client = new ApiClient({ baseUrl: 'http://test.local' });
    await client.finalizeScan('scan-1', { firstFrameSignatureHex: null });

    const [, init] = (fetchMock.mock.calls[0] ?? []) as unknown as [
      string,
      RequestInit,
    ];
    expect(init.body).toBeUndefined();
  });
});
