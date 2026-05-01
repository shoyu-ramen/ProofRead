/**
 * Public surface of the scan UI overlay module.
 *
 * Composed by `mobile/app/(app)/scan/unwrap.tsx` (the live scan
 * screen). All components consume `SharedValue<...>` props for
 * worklet-driven animation; none of them carry state of the broader
 * scan machine — that's the Flow Integration agent's job.
 */

export { CancelButton, type CancelButtonProps } from './CancelButton';
export {
  ScanInstructions,
  type ScanInstructionsProps,
  type ScanStateKind,
  type PauseReason,
  type FailReason,
} from './ScanInstructions';
export {
  BottleSilhouetteOverlay,
  type BottleSilhouetteOverlayProps,
  type SilhouetteFrame,
} from './BottleSilhouetteOverlay';
export {
  RotationGuideRing,
  type RotationGuideRingProps,
} from './RotationGuideRing';
export { ProgressDial, type ProgressDialProps } from './ProgressDial';
export { QualityChip, type QualityChipProps } from './QualityChip';
export {
  CompletionReveal,
  type CompletionRevealProps,
} from './CompletionReveal';
export {
  InScanWarningBanner,
  shouldShowInScanWarning,
  IN_SCAN_WARNING_LOWER,
  IN_SCAN_WARNING_UPPER,
  type InScanWarningBannerProps,
} from './InScanWarningBanner';

// --- Deviations from SCAN_DESIGN.md ---------------------------------
//
// 1. Rendering library: chose `react-native-svg` (^15.x) over Skia for
//    the chrome (silhouette, ring, dial). Skia is being added by the
//    panorama agent for the live unrolled-strip canvas; the chrome's
//    primitives (rounded-rect outlines, stroked arcs, ticks, simple
//    sparkle dots) are well-served by SVG and avoid us racing the
//    panorama agent on the Skia install. The completion sparkle is
//    rendered with plain RN Views + Reanimated transforms because the
//    composition (head + tail particles) is already cheap at ~9 nodes
//    and doesn't benefit from a Skia surface.
//
// 2. Rotation guide ring is rendered as an averaged-radius circle
//    rather than a true ellipse (SCAN_DESIGN §3.4 says "ellipse
//    concentric with the bottle silhouette's center"). Reason: SVG's
//    stroke-dasharray for an animated arc requires a fixed perimeter,
//    and `<Ellipse>` geometry doesn't expose a clean dash-friendly
//    perimeter under react-native-svg. For typical bottles the width
//    and height of the silhouette are within ~30% of each other and
//    the visible difference reads as "ring around bottle" either way.
//    If a curvature audit flags this, swap to a Skia path with
//    PathMeasure for true elliptical arc.
//
// 3. Milestone tick color transitions are evaluated per-frame inside
//    a worklet branch (flash > passed > track), instead of an
//    interpolateColor on the stroke prop. Reason: react-native-svg
//    v15's animatedProps does not yet support color interpolation on
//    `stroke`, so we feed string-valued color tokens directly from the
//    worklet via a branch. The visual difference (snap vs.
//    interpolated tween) is sub-perceptual for 240ms flashes.
//
// 4. Bottle-silhouette drop-shadow uses the iOS `shadow*` style props
//    only. Android's `elevation` doesn't render a shaped shadow
//    behind a stroked-rectangle SVG; if Android shadow becomes
//    important, render an additional offset-rect under the primary
//    stroke as a faux shadow.
//
// 5. The completion reveal's strip-lift transform is the panorama
//    agent's responsibility (PanoramaCanvas owns the strip element);
//    this module delivers only the choreography overlay (sparkle +
//    hero copy + scrim). The §8 storyboard's stages 0/1 (seam close,
//    ring → green flash) are also delivered by the parent screen
//    via the `state` prop the ring + silhouette already consume.
