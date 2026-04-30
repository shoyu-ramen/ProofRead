/**
 * Strip extraction (ARCH §5.1).
 *
 * Runs on the JS thread after `Camera.takePhoto()` returns: decodes the
 * captured photo with Skia, locates the bottle silhouette in the high-
 * res frame, crops a vertical strip centred on the silhouette, resamples
 * to STRIP_WIDTH × panoramaHeight, and returns the strip as a
 * `StripCheckpoint` (RGB-uint8 pixels). The Skia source image and any
 * intermediate surfaces are released before returning so per-checkpoint
 * memory stays bounded.
 *
 * Cylindrical correction is gated on `APPLY_CYLINDRICAL_CORRECTION`
 * (see types.ts) — disabled for v1.
 */

import { Skia } from '@shopify/react-native-skia';
import type { SkCanvas, SkImage, SkRect } from '@shopify/react-native-skia';

import type { BottleSilhouette } from '@src/scan/tracker/types';
import {
  APPLY_CYLINDRICAL_CORRECTION,
  STRIP_WIDTH as DEFAULT_STRIP_WIDTH,
  type StripCheckpoint,
} from './types';

/**
 * Tunables for the strip extractor. Defaults match ARCH §5.1.
 */
export interface ExtractStripOptions {
  /** Output strip width in px (default 80, i.e. panoramaWidth/N=36). */
  stripWidth?: number;
  /** Output strip height in px (default 1024). */
  stripHeight?: number;
  /**
   * Width of the source crop as a fraction of bottle diameter. Default
   * 1/12 (~8% — the part of the bottle facing the camera most directly,
   * least foreshortened).
   */
  cropDiameterFraction?: number;
  /** Coverage value to stamp on the returned StripCheckpoint. */
  coverage: number;
  /**
   * Optional silhouette from the tracker. When provided we trust it
   * (scaled into the high-res frame); when absent we re-detect on the
   * full-resolution image.
   */
  silhouette?: BottleSilhouette | null;
  /**
   * Width of the frame in which `silhouette` was measured. Required if
   * `silhouette` is non-null so we can scale into the photo's coords.
   */
  silhouetteSourceWidth?: number;
  /** Height of the frame in which `silhouette` was measured. */
  silhouetteSourceHeight?: number;
}

const DEFAULT_STRIP_HEIGHT = 1024;
const DEFAULT_DIAMETER_FRACTION = 1 / 12;

/**
 * Extract the central strip of `photoUri` and return a StripCheckpoint.
 *
 * Steps (ARCH §5.1):
 *   1. Decode the photo via Skia.
 *   2. Locate the silhouette (use the supplied one if any).
 *   3. Compute the source crop rect: a vertical column of width
 *      `widthPx / 12` centred on the silhouette.
 *   4. Resample into a `stripWidth × stripHeight` off-screen surface.
 *   5. Read the RGBA pixels back, convert to RGB-uint8.
 */
export async function extractStrip(
  photoUri: string,
  silhouette: BottleSilhouette | null,
  options: ExtractStripOptions,
): Promise<StripCheckpoint> {
  const stripWidth = options.stripWidth ?? DEFAULT_STRIP_WIDTH;
  const stripHeight = options.stripHeight ?? DEFAULT_STRIP_HEIGHT;
  const diameterFrac = options.cropDiameterFraction ?? DEFAULT_DIAMETER_FRACTION;

  const sourceImage = await loadImageFromUri(photoUri);
  try {
    const photoW = sourceImage.width();
    const photoH = sourceImage.height();

    const sil = options.silhouette ?? silhouette ?? null;
    const crop = computeCropRect({
      photoW,
      photoH,
      silhouette: sil,
      silhouetteSourceWidth: options.silhouetteSourceWidth ?? photoW,
      silhouetteSourceHeight: options.silhouetteSourceHeight ?? photoH,
      diameterFraction: diameterFrac,
    });

    const imageData = drawStripPixels(
      sourceImage,
      crop,
      stripWidth,
      stripHeight,
      diameterFrac,
    );

    return {
      coverage: clamp01(options.coverage),
      imageData,
      width: stripWidth,
      height: stripHeight,
    };
  } finally {
    // SkImage is reference-counted; dispose explicitly so the underlying
    // GPU/CPU pixel buffer is freed before the next checkpoint loads.
    if (typeof sourceImage.dispose === 'function') sourceImage.dispose();
  }
}

interface CropInput {
  photoW: number;
  photoH: number;
  silhouette: BottleSilhouette | null;
  silhouetteSourceWidth: number;
  silhouetteSourceHeight: number;
  diameterFraction: number;
}

interface CropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Compute the source-image crop that becomes the strip. Falls back to
 * a centred crop when no silhouette is available — keeps the pipeline
 * resilient to a single failed detection without dropping the photo.
 */
function computeCropRect(input: CropInput): CropRect {
  const { photoW, photoH, silhouette, diameterFraction } = input;

  if (!silhouette || !silhouette.detected) {
    const cropW = Math.max(1, Math.round(photoW * diameterFraction));
    const cropX = Math.round((photoW - cropW) * 0.5);
    return { x: cropX, y: 0, width: cropW, height: photoH };
  }

  // Silhouette coords are in the resized frame the tracker ran against;
  // scale them into the photo's pixel space.
  const sx = photoW / Math.max(1, input.silhouetteSourceWidth);
  const sy = photoH / Math.max(1, input.silhouetteSourceHeight);
  const centerX = silhouette.centerX * sx;
  const widthPx = silhouette.widthPx * sx;
  const cropW = Math.max(1, Math.round(widthPx * diameterFraction));
  let cropX = Math.round(centerX - cropW * 0.5);
  cropX = clamp(cropX, 0, photoW - cropW);

  // Vertical extent (ARCH §5.1 step 3). When the tracker surfaces a
  // measured top/bottom we crop to it; otherwise we fall back to the
  // full photo height. Falling back rather than throwing keeps a
  // borderline silhouette (cap shape escapes the column-tolerance
  // window) from dropping the photo — the strip is just slightly
  // looser than ideal.
  let yTop = 0;
  let yHeight = photoH;
  if (silhouette.heightPx > 0) {
    const measuredTop = clamp(Math.round(silhouette.edgeTopY * sy), 0, photoH - 1);
    const measuredBot = clamp(
      Math.round(silhouette.edgeBottomY * sy),
      measuredTop + 1,
      photoH,
    );
    yTop = measuredTop;
    yHeight = measuredBot - measuredTop;
  }
  return { x: cropX, y: yTop, width: cropW, height: yHeight };
}

/**
 * Render `crop` of `sourceImage` into an offscreen surface sized
 * stripWidth × stripHeight, then read back RGB-uint8 pixels.
 *
 * `diameterFraction` is the crop's width as a fraction of bottle
 * diameter (i.e. sin(θ_max) for the crop's angular extent); it's only
 * read when APPLY_CYLINDRICAL_CORRECTION is on.
 */
function drawStripPixels(
  sourceImage: SkImage,
  crop: CropRect,
  stripWidth: number,
  stripHeight: number,
  diameterFraction: number,
): Uint8Array {
  const surface = Skia.Surface.MakeOffscreen(stripWidth, stripHeight);
  if (!surface) {
    throw new Error('Skia.Surface.MakeOffscreen returned null');
  }
  try {
    const canvas = surface.getCanvas();
    canvas.clear(Skia.Color('transparent'));

    if (APPLY_CYLINDRICAL_CORRECTION) {
      drawCylindricallyCorrected(
        canvas,
        sourceImage,
        crop,
        stripWidth,
        stripHeight,
        diameterFraction,
      );
    } else {
      const src = Skia.XYWHRect(crop.x, crop.y, crop.width, crop.height);
      drawWholeStrip(canvas, sourceImage, src, stripWidth, stripHeight);
    }

    surface.flush();

    const snapshot = surface.makeImageSnapshot();
    try {
      return readRgbPixels(snapshot, stripWidth, stripHeight);
    } finally {
      if (typeof snapshot.dispose === 'function') snapshot.dispose();
    }
  } finally {
    if (typeof surface.dispose === 'function') surface.dispose();
  }
}

function drawWholeStrip(
  canvas: SkCanvas,
  image: SkImage,
  src: SkRect,
  stripWidth: number,
  stripHeight: number,
): void {
  const dst = Skia.XYWHRect(0, 0, stripWidth, stripHeight);
  const paint = Skia.Paint();
  // High-quality resample — we're going from a wide source band down to
  // an 80px strip, so a mitchell-style sampling beats nearest-neighbour
  // blockiness without breaking the budget.
  paint.setAntiAlias(true);
  // 1.x signature: drawImageRectOptions(image, src, dst, sampling, paint).
  // We use drawImageRect with a Paint for max compat across 1.5–1.12;
  // Skia falls back to its default linear sampler.
  canvas.drawImageRect(image, src, dst, paint);
}

/**
 * N-sub-strip cylindrical correction (ARCH §5.1 step 4).
 *
 * The source crop sits on the part of the bottle facing the camera most
 * directly, but it's still a slice of a curved surface — pixels at the
 * crop's center (θ ≈ 0) cover a denser angular sweep per source pixel
 * than pixels at its edges (|θ| ≈ θ_max). Without correction, the
 * unrolled panorama over-samples the bottle's center and under-samples
 * its tangents, subtly stretching label artwork at the seams between
 * strips.
 *
 * Geometry: the crop spans angles [-θ_max, +θ_max] with sin(θ_max) =
 * diameterFraction (since cropWidth = 2R · diameterFraction by the way
 * computeCropRect builds the crop). For each output sub-strip
 * i ∈ [0, N_SUB), we map a uniform-in-angle slice [θ_i, θ_{i+1}] back
 * to source-x range [centerX + R sin(θ_i), centerX + R sin(θ_{i+1})].
 * The local source-to-dest scale falls off as cos(θ) at the edges,
 * which is the "scale x-axis by cos(θ)" the architecture spec asks for.
 *
 * N_SUB = 8 keeps the per-strip draw cost low (8 drawImageRect calls vs
 * 1 in the linear path) while stepping the sin curve finely enough that
 * within-sub-strip linear sampling stays sub-pixel-accurate.
 */
function drawCylindricallyCorrected(
  canvas: SkCanvas,
  image: SkImage,
  crop: CropRect,
  stripWidth: number,
  stripHeight: number,
  diameterFraction: number,
): void {
  const N_SUB = 8;
  // sin(θ_max) = diameterFraction by construction. Clamp into a
  // numerically-safe range so degenerate inputs (zero-width crop, or
  // a fallback heuristic with diameterFraction outside (0, 1)) don't
  // blow up the asin / division.
  const sinThetaMax = Math.max(0.001, Math.min(0.99, diameterFraction));
  const thetaMax = Math.asin(sinThetaMax);
  const R = crop.width / (2 * sinThetaMax);
  const cropCenterX = crop.x + crop.width / 2;
  const cropLeft = crop.x;
  const cropRight = crop.x + crop.width;

  const paint = Skia.Paint();
  paint.setAntiAlias(true);

  for (let i = 0; i < N_SUB; i++) {
    const thetaStart = -thetaMax + (i / N_SUB) * 2 * thetaMax;
    const thetaEnd = -thetaMax + ((i + 1) / N_SUB) * 2 * thetaMax;
    let sxStart = cropCenterX + R * Math.sin(thetaStart);
    let sxEnd = cropCenterX + R * Math.sin(thetaEnd);
    // Defensive: FP drift at the band ends can push sxStart/sxEnd a
    // hair past the crop bounds. Clamp before constructing the rect so
    // we never sample outside the silhouette-centred crop.
    if (sxStart < cropLeft) sxStart = cropLeft;
    if (sxEnd > cropRight) sxEnd = cropRight;
    const srcW = sxEnd - sxStart;
    if (srcW <= 0) continue;

    const dxStart = (i / N_SUB) * stripWidth;
    const dstW = stripWidth / N_SUB;

    const src = Skia.XYWHRect(sxStart, crop.y, srcW, crop.height);
    const dst = Skia.XYWHRect(dxStart, 0, dstW, stripHeight);
    canvas.drawImageRect(image, src, dst, paint);
  }
}

/**
 * Snapshot pixels → tightly-packed RGB-uint8 bytes. Skia returns RGBA
 * by default; we strip the alpha channel because the panorama is fully
 * opaque and the downstream surface upload doesn't need it.
 */
function readRgbPixels(image: SkImage, w: number, h: number): Uint8Array {
  const rgba = image.readPixels();
  if (!rgba) {
    // Fallback: encode → re-decode is heavy; in practice readPixels is
    // implemented on every backend in 1.x, so this is a true error path.
    throw new Error('SkImage.readPixels returned null');
  }
  // readPixels can return Uint8Array or Float32Array depending on color
  // type. We requested defaults → Uint8Array RGBA. Defensive cast.
  const src = rgba instanceof Uint8Array ? rgba : new Uint8Array(rgba.buffer);
  const out = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < src.length; i += 4, j += 3) {
    out[j] = src[i];
    out[j + 1] = src[i + 1];
    out[j + 2] = src[i + 2];
  }
  return out;
}

/**
 * Decode a `file://` (or `data:`) photo URI into an SkImage. Uses the
 * same encode-base64 path the docs recommend; React Native's `fetch`
 * handles both `file://` and `data:` URIs uniformly.
 */
async function loadImageFromUri(uri: string): Promise<SkImage> {
  const response = await fetch(uri);
  const arrayBuffer = await response.arrayBuffer();
  const data = Skia.Data.fromBytes(new Uint8Array(arrayBuffer));
  try {
    const image = Skia.Image.MakeImageFromEncoded(data);
    if (!image) throw new Error('Skia.Image.MakeImageFromEncoded returned null');
    return image;
  } finally {
    if (typeof data.dispose === 'function') data.dispose();
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
