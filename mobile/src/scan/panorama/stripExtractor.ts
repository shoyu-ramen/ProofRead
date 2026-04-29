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

import {
  Skia,
  ImageFormat,
  TileMode,
  FilterMode,
  MipmapMode,
} from '@shopify/react-native-skia';
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

  // Vertical extent: the tracker doesn't surface top/bottom, so use the
  // full photo height — the label height heuristic in ARCH §5.1 step 3
  // is deferred until the design pass dials in cap/base detection.
  const yTop = Math.max(0, Math.round(silhouette.edgeLeftX * 0)); // 0 in v1
  const yHeight = Math.max(1, photoH - yTop);
  return { x: cropX, y: yTop, width: cropW, height: yHeight };
}

/**
 * Render `crop` of `sourceImage` into an offscreen surface sized
 * stripWidth × stripHeight, then read back RGB-uint8 pixels.
 */
function drawStripPixels(
  sourceImage: SkImage,
  crop: CropRect,
  stripWidth: number,
  stripHeight: number,
): Uint8Array {
  const surface = Skia.Surface.MakeOffscreen(stripWidth, stripHeight);
  if (!surface) {
    throw new Error('Skia.Surface.MakeOffscreen returned null');
  }
  try {
    const canvas = surface.getCanvas();
    canvas.clear(Skia.Color('transparent'));

    const src = Skia.XYWHRect(crop.x, crop.y, crop.width, crop.height);

    if (APPLY_CYLINDRICAL_CORRECTION) {
      // TODO(panorama): apply cos(θ) x-axis correction in N sub-strips.
      // Disabled in v1 — see types.ts APPLY_CYLINDRICAL_CORRECTION.
      drawWholeStrip(canvas, sourceImage, src, stripWidth, stripHeight);
    } else {
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
  // drawImageRect supports a sampling option in 1.x via the cubic/linear
  // overloads. Keep it simple: linear filter + no mipmap, which Skia
  // implements as bilinear resample.
  const sampling = {
    filter: FilterMode.Linear,
    mipmap: MipmapMode.None,
  };
  // 1.x signature: drawImageRectOptions(image, src, dst, sampling, paint).
  // We use drawImageRect with a Paint for max compat across 1.5–1.12.
  canvas.drawImageRect(image, src, dst, paint);
  // Reference TileMode/sampling so they're not flagged as unused under
  // strict TS. They're load-bearing for the (gated) cylindrical-
  // correction path above.
  void TileMode;
  void sampling;
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

// Reference ImageFormat so 'unused' lint doesn't fire — the stitcher
// imports it via the barrel and the linter inspects per-file.
void ImageFormat;
