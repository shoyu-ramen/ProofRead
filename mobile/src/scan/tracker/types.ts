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

export interface TrackerState {
  silhouette: BottleSilhouette;
  flow: FlowMeasurement;
  coverage: number;
  angularVelocity: number;
  rotationDirection: 'cw' | 'ccw' | null;
  preCheck: PreCheckVerdict;
  capturedCheckpoints: number;
  frameTick: number;
}
