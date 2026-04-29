/**
 * Final panorama stitcher (ARCH §5.3).
 *
 * Encodes the live PanoramaCanvas snapshot to a JPEG and writes it to
 * the cache directory. Runs once on `scanning → complete`. The caller
 * passes the SkImage that PanoramaCanvas has been keeping current with
 * every strip arrival, so we never allocate a second 12 MB off-screen
 * surface or repaint the strips a second time — that doubled the
 * panorama-related peak memory and could fail the second `MakeOffscreen`
 * on low-memory iPhones.
 *
 * The caller retains ownership of the snapshot (PanoramaCanvas disposes
 * it on the next strip or on unmount); we only read.
 *
 * Returns an `UnrolledPanorama` matching the contract in
 * `mobile/src/state/scanStore.ts`.
 */

import { ImageFormat } from '@shopify/react-native-skia';
import type { SkImage } from '@shopify/react-native-skia';
import * as FileSystem from 'expo-file-system';

import type { PanoramaState, UnrolledPanorama } from './types';

const JPEG_QUALITY = 0.92;

export interface StitchOptions {
  /**
   * When the scan started, in ms since epoch. The scan store records
   * the first strip's `capturedAt`; the flow agent passes that in here
   * so we can compute `durationMs` end-to-end.
   */
  startedAt?: number;
}

/**
 * Encode the live panorama snapshot into a JPEG file.
 *
 * Caller contract:
 *   - `liveSnapshot` is the current `snapshotSv.value` from the parent
 *     of `<PanoramaCanvas>`. It must be non-null — null indicates the
 *     PanoramaCanvas surface failed to allocate (memory pressure), at
 *     which point allocating another surface here would fail too.
 *   - Returns a `file://` URI; safe to upload via the existing
 *     multipart machinery.
 */
export async function stitchPanorama(
  liveSnapshot: SkImage,
  state: Pick<PanoramaState, 'width' | 'height' | 'strips'>,
  options: StitchOptions = {},
): Promise<UnrolledPanorama> {
  const base64 = liveSnapshot.encodeToBase64(ImageFormat.JPEG, JPEG_QUALITY);
  if (!base64) {
    throw new Error('stitchPanorama: encodeToBase64 returned empty string');
  }
  const uri = await writeJpeg(base64);

  const startedAt = options.startedAt ?? Date.now();
  const durationMs = Math.max(0, Date.now() - startedAt);

  return {
    uri,
    width: Math.max(1, state.width),
    height: Math.max(1, state.height),
    frameCount: state.strips.length,
    durationMs,
  };
}

/**
 * Write a base64 JPEG into the app cache directory and return its
 * `file://` URI. Falls back to a `data:` URI if the cache directory
 * is unavailable (e.g. on constrained simulators) — both formats are
 * accepted by the upload pipeline (`fetch().arrayBuffer()`).
 */
async function writeJpeg(base64: string): Promise<string> {
  if (!FileSystem.cacheDirectory) {
    return `data:image/jpeg;base64,${base64}`;
  }
  const filename = `panorama-${Date.now()}-${Math.floor(Math.random() * 1e6)}.jpg`;
  const uri = `${FileSystem.cacheDirectory}${filename}`;
  await FileSystem.writeAsStringAsync(uri, base64, {
    encoding: FileSystem.EncodingType.Base64,
  });
  return uri;
}
