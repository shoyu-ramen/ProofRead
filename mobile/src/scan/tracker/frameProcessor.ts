/**
 * useTrackerFrameProcessor — wires the bottle detector, optical-flow
 * estimator, and angle integrator into a single Vision Camera v4
 * frame processor (ARCH §4.1).
 *
 * Responsibilities:
 *   - Resize each frame to 160×240 RGB; convert to luma in one pass
 *     while computing the pre-check signals (blur / glare / coverage).
 *   - Detect the bottle silhouette and measure horizontal optical
 *     flow against the previous frame.
 *   - Update the angular-progress integrator and write everything
 *     into trackerStateSv for UI consumption.
 *   - Adapt frame stride under thermal load by tracking the EMA of
 *     worklet latency, mirroring the existing camera-screen logic.
 *
 * The accelerometer-driven motion verdict lives in useMotionVerdict
 * here so the UI consumes a single TrackerState.preCheck field.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useSharedValue } from 'react-native-reanimated';
import {
  type Frame,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { Accelerometer, type AccelerometerMeasurement } from 'expo-sensors';

import { detectBottle } from './bottleDetector';
import { measureFlow } from './opticalFlow';
import { computeAngularProgress } from './angleTracker';
import type {
  BottleSilhouette,
  CoverageStatus,
  FlowMeasurement,
  PreCheckVerdict,
  TrackerState,
} from './types';
import type { SharedValue } from 'react-native-reanimated';

// Resize target. 160×240 is the silhouette-detection sweet spot from
// ARCH §4.1: still hits ~10–15 Hz on iPhone 12+ with the full pipeline.
const RESIZE_W = 160;
const RESIZE_H = 240;
const TOTAL_PX = RESIZE_W * RESIZE_H;

// Adaptive frame stride. Mirrors the pre-check screen: process every
// Nth frame, double N if we're consistently over budget.
const FRAME_STRIDE_FAST = 3;
const FRAME_STRIDE_SLOW = 6;
const FRAME_BUDGET_MS = 25;
const LATENCY_EMA_ALPHA = 0.2;

// Pre-check thresholds (carryover from camera/[surface].tsx with the
// Laplacian threshold scaled for the 160×240 grid — bigger ROI means
// more pixels contribute, so the variance band shifts up slightly).
const BLUR_THRESHOLD = 100;
const GLARE_LUMA_CUTOFF = 250;
const GLARE_THRESHOLD = 0.2;
const COVERAGE_DARK_CUTOFF = 80;
const COVERAGE_BRIGHT_CUTOFF = 230;
const COVERAGE_TOO_FAR = 0.3;
const COVERAGE_TOO_CLOSE = 0.85;

// Silhouette-width band (fraction of frame width) the scan considers
// usable. Outside the band we emit `coverageStatus = 'too_far' |
// 'too_close'` so the state machine can pause with actionable copy
// even before rotation starts.
const SILHOUETTE_WIDTH_TOO_FAR = 0.3;
const SILHOUETTE_WIDTH_TOO_CLOSE = 0.9;

const ROI_FRAC_X0 = 0.15;
const ROI_FRAC_X1 = 0.85;
const ROI_FRAC_Y0 = 0.15;
const ROI_FRAC_Y1 = 0.85;

// Steadiness EMA — the per-frame edge tightness from the detector
// rolls into this so the UI gets a smooth value.
const STEADINESS_ALPHA = 0.3;

// Accelerometer config (carryover from camera/[surface].tsx).
const ACCEL_INTERVAL_MS = 100;
const ACCEL_WINDOW = 10;
const MOTION_VARIANCE_THRESHOLD = 0.015;

// JS-side staleness watchdog for the verdict — if no fresh frame in
// this many ms, we fall back to PreCheckVerdict.unknown.
const VERDICT_STALE_MS = 1000;

const EMPTY_SILHOUETTE: BottleSilhouette = {
  detected: false,
  edgeLeftX: 0,
  edgeRightX: 0,
  centerX: 0,
  widthPx: 0,
  steadinessScore: 0,
  class: null,
  classConfidence: 0,
};

const EMPTY_FLOW: FlowMeasurement = {
  dxPx: 0,
  inliers: 0,
  confidence: 0,
};

const INITIAL_TRACKER_STATE: TrackerState = {
  silhouette: EMPTY_SILHOUETTE,
  flow: EMPTY_FLOW,
  coverage: 0,
  angularVelocity: 0,
  rotationDirection: null,
  preCheck: { kind: 'unknown' },
  coverageStatus: null,
  capturedCheckpoints: 0,
  frameTick: 0,
  handBox: null,
  gripSteadiness: 0,
  flowQuality: 0,
};

export interface TrackerFrameProcessor {
  /** Pass to the <Camera /> frameProcessor prop. */
  frameProcessor: ReturnType<typeof useFrameProcessor>;
  /** Live tracker state — read via .value or useAnimatedReaction. */
  trackerStateSv: ReturnType<typeof useSharedValue<TrackerState>>;
  /** Monotonic worklet frame counter; useful as a reaction trigger. */
  frameTickSv: ReturnType<typeof useSharedValue<number>>;
  /** EMA of worklet wall-clock latency (ms). */
  latencyEmaSv: ReturnType<typeof useSharedValue<number>>;
  /** Current adaptive stride; surfaced for diagnostics. */
  frameStrideSv: ReturnType<typeof useSharedValue<number>>;
  /**
   * Caller-driven setter for capturedCheckpoints. Strip extraction
   * lives outside this module (panorama subsystem); when a checkpoint
   * is captured, the panorama hook calls this so the next checkpoint
   * threshold advances.
   */
  bumpCapturedCheckpoints: () => void;
}

export function useTrackerFrameProcessor(): TrackerFrameProcessor {
  const { resize } = useResizePlugin();

  const trackerStateSv = useSharedValue<TrackerState>(INITIAL_TRACKER_STATE);
  const frameTickSv = useSharedValue<number>(0);
  const latencyEmaSv = useSharedValue<number>(0);
  const frameStrideSv = useSharedValue<number>(FRAME_STRIDE_FAST);
  const frameIdxSv = useSharedValue<number>(0);

  // Worklet-owned mutable buffers carried across frames. Holding them
  // as shared-value Uint8Arrays keeps allocation cost flat — we only
  // pay for the pair of TOTAL_PX-sized buffers once.
  const lumaA = useSharedValue<Uint8Array>(new Uint8Array(TOTAL_PX));
  const lumaB = useSharedValue<Uint8Array>(new Uint8Array(TOTAL_PX));
  // Which buffer holds the previous frame? Worklet flips this each
  // accepted frame; we always write into the "current" slot.
  const lumaSlotSv = useSharedValue<0 | 1>(0);
  // Frame timestamp (ms, performance.now()) of the last accepted
  // frame — feeds the angular-velocity dt.
  const lastFrameAtSv = useSharedValue<number>(0);
  // True once we've accepted at least one frame (so prevLuma is valid).
  const hasPrevSv = useSharedValue<boolean>(false);
  // Multi-frame steadiness EMA — survives across frames here so the
  // angle integrator's state stays small.
  const steadinessEmaSv = useSharedValue<number>(0);

  // JS-side accelerometer verdict feeds back into the worklet via a
  // shared value so the worklet writes a complete preCheck without
  // needing JS round-trips. `motionMagnitudeSv` carries the same
  // signal as a normalized 0..1 magnitude (variance / threshold,
  // clamped) so the worklet can blend it into `gripSteadiness` rather
  // than getting a binary shaking/not-shaking verdict.
  const motionDetectedSv = useSharedValue<boolean>(false);
  const { shaking: motionDetected, motionMagnitudeSv } = useMotionVerdict();
  useEffect(() => {
    motionDetectedSv.value = motionDetected;
  }, [motionDetected, motionDetectedSv]);

  const bumpCapturedCheckpoints = useCallback(() => {
    const s = trackerStateSv.value;
    trackerStateSv.value = {
      ...s,
      capturedCheckpoints: s.capturedCheckpoints + 1,
    };
  }, [trackerStateSv]);

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      'worklet';

      const idx = frameIdxSv.value + 1;
      frameIdxSv.value = idx;
      if (idx % frameStrideSv.value !== 0) return;

      const t0 = performance.now();

      const buffer = resize(frame, {
        scale: { width: RESIZE_W, height: RESIZE_H },
        pixelFormat: 'rgb',
        dataType: 'uint8',
      });

      const w = RESIZE_W;
      const h = RESIZE_H;

      // Pick which buffer is "current"; the other holds the previous
      // frame's luma. Flipping the slot is equivalent to ping-ponging
      // a pair of allocations without ever allocating per-frame.
      const slot = lumaSlotSv.value;
      const currLuma = slot === 0 ? lumaA.value : lumaB.value;
      const prevLuma = slot === 0 ? lumaB.value : lumaA.value;

      const x0 = Math.floor(w * ROI_FRAC_X0);
      const x1 = Math.floor(w * ROI_FRAC_X1);
      const y0 = Math.floor(h * ROI_FRAC_Y0);
      const y1 = Math.floor(h * ROI_FRAC_Y1);
      const roiPixels = (x1 - x0) * (y1 - y0);

      // Pass 1: RGB → luma + ROI counters for glare & coverage.
      let glareCount = 0;
      let coverageCount = 0;
      for (let y = 0; y < h; y++) {
        const rowStart = y * w;
        for (let x = 0; x < w; x++) {
          const i = (rowStart + x) * 3;
          const r = buffer[i];
          const g = buffer[i + 1];
          const b = buffer[i + 2];
          const y8 = (r * 77 + g * 150 + b * 29) >> 8;
          currLuma[rowStart + x] = y8;

          if (x >= x0 && x < x1 && y >= y0 && y < y1) {
            if (y8 > GLARE_LUMA_CUTOFF) glareCount++;
            else if (y8 > COVERAGE_DARK_CUTOFF && y8 < COVERAGE_BRIGHT_CUTOFF) {
              coverageCount++;
            }
          }
        }
      }

      // Pass 2: Laplacian variance over the ROI for blur. Sharing the
      // same luma buffer with the silhouette and flow stages keeps
      // total memory traffic low.
      const lx0 = Math.max(1, x0);
      const lx1 = Math.min(w - 1, x1);
      const ly0 = Math.max(1, y0);
      const ly1 = Math.min(h - 1, y1);
      let lapSum = 0;
      let lapSumSq = 0;
      let lapN = 0;
      for (let y = ly0; y < ly1; y++) {
        const row = y * w;
        for (let x = lx0; x < lx1; x++) {
          const c = currLuma[row + x];
          const up = currLuma[row - w + x];
          const dn = currLuma[row + w + x];
          const lf = currLuma[row + x - 1];
          const rt = currLuma[row + x + 1];
          const lap = 4 * c - up - dn - lf - rt;
          lapSum += lap;
          lapSumSq += lap * lap;
          lapN++;
        }
      }
      const blurMean = lapN > 0 ? lapSum / lapN : 0;
      const blurVar = lapN > 0 ? lapSumSq / lapN - blurMean * blurMean : 0;
      const glareRatio = roiPixels > 0 ? glareCount / roiPixels : 0;
      const coverageRatio = roiPixels > 0 ? coverageCount / roiPixels : 0;

      // Bottle silhouette (no extra full-frame pass — the detector
      // walks rows in the central band and is bounded ~h band rows).
      const rawSilhouette = detectBottle(currLuma, w, h);
      const prevSteady = steadinessEmaSv.value;
      const nextSteady = rawSilhouette.detected
        ? prevSteady * (1 - STEADINESS_ALPHA) +
          rawSilhouette.steadinessScore * STEADINESS_ALPHA
        : prevSteady * (1 - STEADINESS_ALPHA);
      steadinessEmaSv.value = nextSteady;
      const silhouette: BottleSilhouette = rawSilhouette.detected
        ? { ...rawSilhouette, steadinessScore: nextSteady }
        : { ...EMPTY_SILHOUETTE, steadinessScore: nextSteady };

      // Flow needs a valid previous luma; on the very first accepted
      // frame we skip it.
      const flow: FlowMeasurement = hasPrevSv.value
        ? measureFlow(prevLuma, currLuma, w, h, silhouette)
        : EMPTY_FLOW;

      // Angular progress + velocity update.
      const nowMs = performance.now();
      const lastMs = lastFrameAtSv.value;
      const dtSec = lastMs > 0 ? (nowMs - lastMs) / 1000 : 0;

      const prior = trackerStateSv.value;
      const angle = computeAngularProgress(
        {
          coverage: prior.coverage,
          rotationDirection: prior.rotationDirection,
          angularVelocity: prior.angularVelocity,
          flowQuality: prior.flowQuality,
          dtSec,
        },
        flow,
        silhouette,
      );

      // Embodiment signals (Phase 1). gripSteadiness blends the
      // silhouette's per-frame edge tightness with the accelerometer
      // motion magnitude — a steady hand on a steady silhouette reads
      // 1.0; either source dragging the score down pulls it toward 0.
      // Drives the silhouette stroke-width tightening cue in the
      // overlay. Phase 2 will populate `handBox` from the palm
      // detector; for now it stays null.
      const motionMag = motionMagnitudeSv.value;
      const motionDamp = motionMag < 0 ? 0 : motionMag > 1 ? 1 : motionMag;
      const gripSteadiness = silhouette.steadinessScore * (1 - motionDamp);

      // Pre-check: priority order matches the existing camera screen
      // (motion → blur → glare → coverage → ready). Motion is the
      // accelerometer override.
      let preCheck: PreCheckVerdict;
      if (motionDetectedSv.value) {
        preCheck = { kind: 'warn', reason: 'motion' };
      } else if (blurVar < BLUR_THRESHOLD) {
        preCheck = { kind: 'warn', reason: 'blur' };
      } else if (glareRatio > GLARE_THRESHOLD) {
        preCheck = { kind: 'warn', reason: 'glare' };
      } else if (
        coverageRatio < COVERAGE_TOO_FAR ||
        coverageRatio > COVERAGE_TOO_CLOSE
      ) {
        preCheck = { kind: 'warn', reason: 'coverage' };
      } else {
        preCheck = { kind: 'ready' };
      }

      // Coverage-distance signal: only meaningful when the silhouette
      // is locked (we know the bottle's actual width). Without a lock
      // we leave it null so the state machine doesn't claim "too far"
      // when the camera is just looking at a wall.
      let coverageStatus: CoverageStatus = null;
      if (silhouette.detected) {
        const widthFrac = silhouette.widthPx / w;
        if (widthFrac < SILHOUETTE_WIDTH_TOO_FAR) {
          coverageStatus = 'too_far';
        } else if (widthFrac > SILHOUETTE_WIDTH_TOO_CLOSE) {
          coverageStatus = 'too_close';
        }
      }

      trackerStateSv.value = {
        silhouette,
        flow,
        coverage: angle.coverage,
        angularVelocity: angle.angularVelocity,
        rotationDirection: angle.rotationDirection,
        preCheck,
        coverageStatus,
        capturedCheckpoints: prior.capturedCheckpoints,
        frameTick: idx,
        handBox: prior.handBox,
        gripSteadiness,
        flowQuality: angle.flowQuality,
      };

      frameTickSv.value = idx;
      lastFrameAtSv.value = nowMs;
      lumaSlotSv.value = slot === 0 ? 1 : 0;
      hasPrevSv.value = true;

      // EMA + stride decision are folded into the worklet so the JS
      // thread doesn't wake on every accepted frame (~10–15 Hz). The
      // EMA is read by parent screens via `latencyEmaSv`; stride changes
      // are read by the worklet itself on the next tick.
      const ms = performance.now() - t0;
      const prevEma = latencyEmaSv.value;
      const nextEma =
        prevEma === 0
          ? ms
          : prevEma * (1 - LATENCY_EMA_ALPHA) + ms * LATENCY_EMA_ALPHA;
      latencyEmaSv.value = nextEma;
      const currentStride = frameStrideSv.value;
      if (currentStride === FRAME_STRIDE_FAST && nextEma > FRAME_BUDGET_MS) {
        frameStrideSv.value = FRAME_STRIDE_SLOW;
      } else if (
        currentStride === FRAME_STRIDE_SLOW &&
        nextEma < FRAME_BUDGET_MS * 0.7
      ) {
        frameStrideSv.value = FRAME_STRIDE_FAST;
      }
    },
    // SharedValue references are stable across renders (the wrapper
    // object never changes; only `.value` mutates). Including them
    // here is not just unnecessary — Vision Camera's internal
    // useFrameProcessor compares deps via Object.is, and that
    // comparison triggers `_value` access on the JS thread, which
    // Reanimated 3 throws on. Only stable JS-side closures belong here.
    [resize],
  );

  // JS-side staleness watchdog: if no frame has landed in
  // VERDICT_STALE_MS, fall the preCheck back to 'unknown' so the UI
  // doesn't display a stale verdict during thermal pauses.
  useEffect(() => {
    const interval = setInterval(() => {
      const last = lastFrameAtSv.value;
      if (last === 0) return;
      const stale = performance.now() - last > VERDICT_STALE_MS;
      if (!stale) return;
      const s = trackerStateSv.value;
      if (s.preCheck.kind === 'unknown') return;
      trackerStateSv.value = { ...s, preCheck: { kind: 'unknown' } };
    }, 250);
    return () => clearInterval(interval);
  }, [lastFrameAtSv, trackerStateSv]);

  return {
    frameProcessor,
    trackerStateSv,
    frameTickSv,
    latencyEmaSv,
    frameStrideSv,
    bumpCapturedCheckpoints,
  };
}

/**
 * Accelerometer-driven motion verdict. Variance of |acceleration|
 * over a rolling window high-passes gravity, so a stationary phone
 * reads as steady regardless of orientation.
 *
 * Returns both the boolean shaking verdict (for the pre-check pause
 * pathway) and a shared value carrying the variance normalized to
 * [0,1] against MOTION_VARIANCE_THRESHOLD — the worklet uses this
 * continuous signal to blend into `gripSteadiness`.
 */
export interface MotionVerdictResult {
  shaking: boolean;
  motionMagnitudeSv: SharedValue<number>;
}

export function useMotionVerdict(): MotionVerdictResult {
  const [shaking, setShaking] = useState(false);
  const motionMagnitudeSv = useSharedValue<number>(0);
  const windowRef = useRef<number[]>([]);

  useEffect(() => {
    Accelerometer.setUpdateInterval(ACCEL_INTERVAL_MS);

    const window = windowRef.current;
    const subscription = Accelerometer.addListener(
      (m: AccelerometerMeasurement) => {
        const magnitude = Math.sqrt(m.x * m.x + m.y * m.y + m.z * m.z);
        window.push(magnitude);
        if (window.length > ACCEL_WINDOW) window.shift();
        if (window.length < ACCEL_WINDOW) return;

        let mean = 0;
        for (const v of window) mean += v;
        mean /= window.length;
        let acc = 0;
        for (const v of window) acc += (v - mean) * (v - mean);
        const variance = acc / window.length;

        const normalized = variance / MOTION_VARIANCE_THRESHOLD;
        motionMagnitudeSv.value =
          normalized < 0 ? 0 : normalized > 1 ? 1 : normalized;

        setShaking((prev) => {
          const next = variance > MOTION_VARIANCE_THRESHOLD;
          return prev === next ? prev : next;
        });
      },
    );

    return () => {
      subscription.remove();
      windowRef.current = [];
      // Reset the React state immediately so a remount (e.g. thermal
      // pause + resume) doesn't inherit a stale `shaking=true` until
      // the listener refills its window (~1s). Without this, the
      // worklet would emit `paused/motion` for up to ACCEL_INTERVAL_MS
      // * ACCEL_WINDOW after every remount.
      setShaking(false);
      motionMagnitudeSv.value = 0;
    };
  }, [motionMagnitudeSv]);

  return { shaking, motionMagnitudeSv };
}
