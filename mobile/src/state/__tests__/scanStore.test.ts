/**
 * Tests for the known-label additions to scanStore (Decision 7 +
 * Decision 8 in KNOWN_LABEL_DESIGN.md):
 *
 *   - setFirstFrameSignatureHex / setKnownLabel write the new fields
 *   - reset() clears both back to null
 */

import type { KnownLabelPayload } from '@src/api/types';
import { useScanStore } from '../scanStore';

function makeKnownLabel(): KnownLabelPayload {
  return {
    entry_id: '00000000-0000-0000-0000-000000000001',
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
  };
}

function resetStore() {
  useScanStore.getState().reset();
  useScanStore.getState().setKnownLabel(null);
  useScanStore.getState().setFirstFrameSignatureHex(null);
}

describe('scanStore — known-label fields', () => {
  beforeEach(() => {
    resetStore();
  });

  test('initial state has both new fields as null', () => {
    const s = useScanStore.getState();
    expect(s.firstFrameSignatureHex).toBeNull();
    expect(s.knownLabel).toBeNull();
  });

  test('setFirstFrameSignatureHex writes the hex to the store', () => {
    useScanStore.getState().setFirstFrameSignatureHex('deadbeefcafebabe');
    expect(useScanStore.getState().firstFrameSignatureHex).toBe(
      'deadbeefcafebabe',
    );
  });

  test('setFirstFrameSignatureHex(null) clears the hex', () => {
    useScanStore.getState().setFirstFrameSignatureHex('deadbeefcafebabe');
    useScanStore.getState().setFirstFrameSignatureHex(null);
    expect(useScanStore.getState().firstFrameSignatureHex).toBeNull();
  });

  test('setKnownLabel writes the payload to the store', () => {
    const payload = makeKnownLabel();
    useScanStore.getState().setKnownLabel(payload);
    expect(useScanStore.getState().knownLabel).toEqual(payload);
  });

  test('setKnownLabel(null) clears the payload', () => {
    useScanStore.getState().setKnownLabel(makeKnownLabel());
    useScanStore.getState().setKnownLabel(null);
    expect(useScanStore.getState().knownLabel).toBeNull();
  });

  test('reset() clears both firstFrameSignatureHex and knownLabel', () => {
    useScanStore.getState().setFirstFrameSignatureHex('deadbeefcafebabe');
    useScanStore.getState().setKnownLabel(makeKnownLabel());
    expect(useScanStore.getState().firstFrameSignatureHex).not.toBeNull();
    expect(useScanStore.getState().knownLabel).not.toBeNull();

    useScanStore.getState().reset();

    expect(useScanStore.getState().firstFrameSignatureHex).toBeNull();
    expect(useScanStore.getState().knownLabel).toBeNull();
  });

  test('reset() preserves the recentPanoramas cache', () => {
    // Sanity: existing reset() contract is unchanged — we don't want
    // the new fields to accidentally clobber the per-scan-id panorama
    // cache that the home rail depends on.
    useScanStore.getState().rememberPanorama('scan-1', {
      uri: 'file:///tmp/p.jpg',
      width: 1920,
      height: 480,
      frameCount: 12,
      durationMs: 8000,
    });
    useScanStore.getState().setKnownLabel(makeKnownLabel());
    useScanStore.getState().reset();
    expect(
      useScanStore.getState().recentPanoramas['scan-1']?.uri,
    ).toBe('file:///tmp/p.jpg');
  });
});
