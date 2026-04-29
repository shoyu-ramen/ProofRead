/**
 * Panorama subsystem barrel (ARCH §5).
 *
 * Public surface:
 *   - StripCheckpoint / PanoramaState / UnrolledPanorama types
 *   - extractStrip(photoUri, silhouette, options)
 *   - PanoramaCanvas (live Skia component)
 *   - stitchPanorama(state, options)
 *
 * The flow integration agent (Wave 3) wires these into `unwrap.tsx`
 * and the scan store. The tracker agent's frame processor calls
 * `extractStrip` on each `Camera.takePhoto()` resolution.
 */

export {
  APPLY_CYLINDRICAL_CORRECTION,
  DEFAULT_PANORAMA_HEIGHT,
  DEFAULT_PANORAMA_WIDTH,
  N_CHECKPOINTS,
  STRIP_WIDTH,
  coverageToStripX,
  createEmptyPanoramaState,
} from './types';
export type {
  PanoramaState,
  StripCheckpoint,
  UnrolledPanorama,
} from './types';

export { extractStrip } from './stripExtractor';
export type { ExtractStripOptions } from './stripExtractor';

export { PanoramaCanvas } from './PanoramaCanvas';

export { stitchPanorama } from './stitcher';
export type { StitchOptions } from './stitcher';

// ---------------------------------------------------------------------
// Deviations from CYLINDRICAL_SCAN_ARCHITECTURE.md
// ---------------------------------------------------------------------
//
// 1. APPLY_CYLINDRICAL_CORRECTION (ARCH §5.1 step 4) is gated to
//    `false` for v1. The strip is rendered with a single linear
//    resample. Spec calls this out as acceptable; flip the constant in
//    types.ts when the design pass dials in the cos(θ) sub-strip math.
//
// 2. UnrolledPanorama is exported from this module instead of from
//    `mobile/src/state/scanStore.ts`. The architecture spec §6 puts
//    the canonical type in scanStore; the panorama module ships it
//    locally so it has no scan-store coupling. The flow integration
//    agent (Wave 3) is expected to move the canonical declaration into
//    scanStore and re-export from here.
//
// 3. The strip extractor accepts an optional pre-detected silhouette
//    from the tracker. ARCH §5.1 step 1 says "re-run the same detector
//    on the high-res frame"; we accept the tracker's silhouette as a
//    fast-path (scaled into photo coordinates) and fall back to a
//    centred crop when none is supplied. A future revision can re-run
//    the worklet detector on the JS-decoded luma plane if the
//    silhouette accuracy in field captures isn't good enough.
//
// 4. Strip vertical cropping (ARCH §5.1 step 3 — "crop top/bottom to
//    the silhouette's vertical extent") uses the full photo height in
//    v1 because the tracker's BottleSilhouette doesn't surface
//    top/bottom edges. Once the tracker exposes them (or once a
//    design-driven label-height heuristic lands), the extractor will
//    crop vertically too.
//
// 5. PanoramaCanvas's writing-edge cursor estimates the next-strip x
//    by dividing panorama width by the current strip count. When the
//    flow agent lands the live coverage shared value, swap the divisor
//    for the actual coverage shared-value reading.
