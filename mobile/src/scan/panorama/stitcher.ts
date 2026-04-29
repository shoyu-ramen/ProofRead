/**
 * Final panorama stitcher (ARCH §5.3).
 *
 * Composes the accumulated StripCheckpoints into a single JPEG and
 * writes it to the cache directory. Runs once on `scanning → complete`,
 * not per-frame, so we prioritise output quality over speed: a fresh
 * off-screen surface is allocated at the canonical panorama dimensions,
 * every strip is repainted in coverage order, the result is encoded at
 * quality 0.92, and the bytes are persisted as a `file://` JPEG.
 *
 * Returns an `UnrolledPanorama` matching the contract in
 * `mobile/src/state/scanStore.ts` (the type is re-exported from
 * `./types` so the panorama module is self-contained for the flow
 * agent — they will move it back into the scan store later).
 */

import { AlphaType, ColorType, Skia, ImageFormat } from '@shopify/react-native-skia';
import type { SkCanvas } from '@shopify/react-native-skia';
import * as FileSystem from 'expo-file-system';

import {
  coverageToStripX,
  type PanoramaState,
  type StripCheckpoint,
  type UnrolledPanorama,
} from './types';

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
 * Stitch a completed PanoramaState into a JPEG file.
 *
 * Caller contract:
 *   - `state.strips` must be ordered by capture time (== coverage order
 *     under monotonic-coverage invariant).
 *   - Returns a `file://` URI; safe to upload via the existing
 *     multipart machinery.
 */
export async function stitchPanorama(
  state: PanoramaState,
  options: StitchOptions = {},
): Promise<UnrolledPanorama> {
  const width = Math.max(1, state.width);
  const height = Math.max(1, state.height);

  const surface = Skia.Surface.MakeOffscreen(width, height);
  if (!surface) {
    throw new Error('stitchPanorama: Skia.Surface.MakeOffscreen failed');
  }

  let uri: string;
  try {
    const canvas = surface.getCanvas();
    canvas.clear(Skia.Color('black'));

    for (const strip of state.strips) {
      paintStrip(canvas, strip, width);
    }
    surface.flush();

    const snapshot = surface.makeImageSnapshot();
    try {
      const base64 = snapshot.encodeToBase64(ImageFormat.JPEG, JPEG_QUALITY);
      if (!base64) {
        throw new Error('stitchPanorama: encodeToBase64 returned empty string');
      }
      uri = await writeJpeg(base64);
    } finally {
      if (typeof snapshot.dispose === 'function') snapshot.dispose();
    }
  } finally {
    if (typeof surface.dispose === 'function') surface.dispose();
  }

  const startedAt = options.startedAt ?? Date.now();
  const durationMs = Math.max(0, Date.now() - startedAt);

  return {
    uri,
    width,
    height,
    frameCount: state.strips.length,
    durationMs,
  };
}

function paintStrip(
  canvas: SkCanvas,
  strip: StripCheckpoint,
  panoramaWidth: number,
): void {
  const image = makeImageFromRgb(strip.imageData, strip.width, strip.height);
  if (!image) return;
  try {
    const x = coverageToStripX(strip.coverage, panoramaWidth, strip.width);
    const dst = Skia.XYWHRect(x, 0, strip.width, strip.height);
    const src = Skia.XYWHRect(0, 0, strip.width, strip.height);
    const paint = Skia.Paint();
    paint.setAntiAlias(true);
    canvas.drawImageRect(image, src, dst, paint);
  } finally {
    if (typeof image.dispose === 'function') image.dispose();
  }
}

function makeImageFromRgb(
  rgb: Uint8Array,
  width: number,
  height: number,
): ReturnType<typeof Skia.Image.MakeImage> | null {
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    rgba[j] = rgb[i];
    rgba[j + 1] = rgb[i + 1];
    rgba[j + 2] = rgb[i + 2];
    rgba[j + 3] = 255;
  }
  const data = Skia.Data.fromBytes(rgba);
  try {
    return Skia.Image.MakeImage(
      {
        width,
        height,
        colorType: ColorType.RGBA_8888,
        alphaType: AlphaType.Opaque,
      },
      data,
      width * 4,
    );
  } finally {
    if (typeof data.dispose === 'function') data.dispose();
  }
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
