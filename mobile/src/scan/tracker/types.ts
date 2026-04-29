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
}
