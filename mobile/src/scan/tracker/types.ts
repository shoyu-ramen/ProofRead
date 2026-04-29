/**
 * Cross-team contract types for the cylindrical-scan tracker subsystem.
 *
 * These shapes are pinned by ARCH §9 — every other module (panorama,
 * scan-machine, UI overlays) imports from here. Update only via a
 * tech-lead amendment to CYLINDRICAL_SCAN_ARCHITECTURE.md.
 */

export interface BottleSilhouette {
  detected: boolean;
  edgeLeftX: number;
  edgeRightX: number;
  centerX: number;
  widthPx: number;
  steadinessScore: number;
  /**
   * Top COCO class among the beverage subset. `null` until Phase 2
   * lands a real classifier; Phase 1 leaves this unpopulated.
   */
  class: 'bottle' | 'wine_glass' | 'cup' | null;
  /** 0..1 probability of `class`; 0 when class is null. */
  classConfidence: number;
}

export interface HandBox {
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
}

export interface FlowMeasurement {
  dxPx: number;
  inliers: number;
  confidence: number;
}

export type PreCheckVerdict =
  | { kind: 'unknown' }
  | { kind: 'ready' }
  | { kind: 'warn'; reason: 'blur' | 'glare' | 'coverage' | 'motion' };

/**
 * Coverage-based scan-distance signal (Stream 3, ARCH §3 amendment).
 * `too_far` and `too_close` are emitted only when the silhouette is
 * detected — the frame processor measures width as a fraction of the
 * resized frame and classifies. `null` = width is in the valid-scan
 * band, or the silhouette isn't detected.
 */
export type CoverageStatus = 'too_far' | 'too_close' | null;

export interface TrackerState {
  silhouette: BottleSilhouette;
  flow: FlowMeasurement;
  coverage: number;
  angularVelocity: number;
  rotationDirection: 'cw' | 'ccw' | null;
  preCheck: PreCheckVerdict;
  /** Distance signal — see CoverageStatus. */
  coverageStatus: CoverageStatus;
  capturedCheckpoints: number;
  frameTick: number;
  /**
   * Detected hand bbox (Phase 2 — palm detector). `null` while no
   * palm is detected or before Phase 2 ships.
   */
  handBox: HandBox | null;
  /**
   * Composite "is the user's grip steady" signal in 0..1, blending the
   * silhouette steadiness EMA with accelerometer motion magnitude.
   * Drives the silhouette stroke-width tightening cue.
   */
  gripSteadiness: number;
  /** EMA of FlowMeasurement.confidence, 0..1. Drives the untrackable-surface signal. */
  flowQuality: number;
}
