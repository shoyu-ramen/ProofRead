/**
 * Panorama subsystem cross-team contract types (ARCH §9).
 *
 * These shapes are the public surface between the tracker (which feeds
 * captured photos in via `extractStrip`), the panorama renderer (which
 * paints `StripCheckpoint`s into a Skia surface), and the flow agent
 * (which calls `stitchPanorama` on completion). Keep them aligned with
 * §9 of CYLINDRICAL_SCAN_ARCHITECTURE.md — any change requires a
 * tech-lead amendment to the doc.
 */

/**
 * One column-strip extracted from a single high-res photo.
 *
 * `imageData` is RGB-uint8 pixels (3 bytes per pixel, row-major). The
 * Skia canvas decodes these into an `SkImage` once and draws into the
 * off-screen panorama surface; the source bytes are dropped immediately
 * after the draw so we never accumulate per-strip pixel buffers.
 */
export interface StripCheckpoint {
  /** Angular coverage at the moment of capture, 0..1 (0 = scan start). */
  coverage: number;
  /** RGB-uint8 pixel data, length === width * height * 3. */
  imageData: Uint8Array;
  /** Pixel width of the strip (matches PanoramaState.stripWidth). */
  width: number;
  /** Pixel height of the strip (matches PanoramaState.height). */
  height: number;
}

/**
 * Live panorama state — owned by the scan store, read by PanoramaCanvas.
 *
 * `strips` accumulates as the user rotates; the canvas component reacts
 * to its length growing and draws each new entry exactly once. `width`
 * and `height` are fixed for the duration of a scan (they're chosen at
 * `aligning → ready`). `isComplete` is set by the state machine when
 * coverage hits 1.0 — the canvas uses it to switch from "writing edge"
 * mode to a final-pass polish.
 */
export interface PanoramaState {
  strips: StripCheckpoint[];
  /** Total panorama width in px (~stripWidth × N_CHECKPOINTS). */
  width: number;
  /** Panorama height in px (matches captured strip height). */
  height: number;
  /** True after the state machine commits the final strip. */
  isComplete: boolean;
}

/**
 * The composed panorama returned by `stitchPanorama`.
 *
 * The canonical declaration lives in `mobile/src/state/scanStore.ts`
 * (per ARCH §6); this module re-exports it so callers that import via
 * the panorama barrel see the same type. Hand-off completed by the
 * flow integration agent.
 */
export type { UnrolledPanorama } from '@src/state/scanStore';

/** Number of angular checkpoints per full revolution (ARCH §4.5, §5.1). */
export const N_CHECKPOINTS = 36;

/** Default panorama dimensions in panorama-pixel space. */
export const DEFAULT_PANORAMA_WIDTH = 2880;
export const DEFAULT_PANORAMA_HEIGHT = 1024;

/** Width of one strip in the panorama (panorama-px). */
export const STRIP_WIDTH = DEFAULT_PANORAMA_WIDTH / N_CHECKPOINTS; // 80

/**
 * Cross-strip cylindrical correction (ARCH §5.1 step 4). Disabled in v1
 * because the residual foreshortening within an 8%-of-diameter strip is
 * subtle relative to the unmodelled per-frame jitter; flip this to true
 * once the panorama-quality pass dials in the math.
 */
export const APPLY_CYLINDRICAL_CORRECTION = false;

/** Build an empty PanoramaState with default dims. */
export function createEmptyPanoramaState(
  width: number = DEFAULT_PANORAMA_WIDTH,
  height: number = DEFAULT_PANORAMA_HEIGHT,
): PanoramaState {
  return { strips: [], width, height, isComplete: false };
}

/**
 * Map a 0..1 coverage value to the strip's left-edge x-offset (in
 * panorama pixels). Centralised so the extractor, the renderer, and
 * the stitcher agree on placement.
 */
export function coverageToStripX(
  coverage: number,
  panoramaWidth: number,
  stripWidth: number,
): number {
  const clamped = coverage < 0 ? 0 : coverage > 1 ? 1 : coverage;
  // Round to integer so successive strips abut without sub-pixel seams.
  return Math.round(clamped * (panoramaWidth - stripWidth));
}
