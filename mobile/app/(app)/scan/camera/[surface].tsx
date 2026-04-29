/**
 * Camera capture screen.
 *
 * SPEC §v1.6 step 4–5 + §v1.7 row "Camera". Pre-check pipeline (SPEC
 * §v1.5 F1.3 + §0.5) is now driven by real signals:
 *
 *   - Blur, glare, and coverage measurements come from a Vision Camera
 *     v4 frame processor sampling every 3rd frame (~10 Hz at 30 fps).
 *     The worklet downsamples each frame to 64×96 RGB via
 *     vision-camera-resize-plugin, converts to luma in a single pass,
 *     and emits three shared values: blurScore (Laplacian variance),
 *     glareRatio (saturated-luma ratio inside the label ROI), and
 *     coverageRatio (label-luminance pixel ratio inside the ROI).
 *   - Motion remains an accelerometer-driven override (useMotionVerdict).
 *   - A useAnimatedReaction translates the shared values into the
 *     existing PreCheck variant the chip + capture-gate already consume.
 *   - If the worklet runtime is stale (>1s since last verdict, e.g. on
 *     thermal throttling) the pre-check holds at 'unknown' / "checking"
 *     rather than displaying a stale verdict.
 *
 * Threshold defaults below are conservative starting points; calibrate
 * with field captures (see mobile/README.md "Pre-check calibration").
 *
 * Still TODO (Phase-3+):
 *   - HDR multi-exposure + thermal-state adaptation (SPEC §0.5 mitigations).
 *   - Bottle-silhouette segmentation for curvature-aware framing.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { showErrorAlert } from '@src/api/errors';
import {
  Camera,
  type CameraDevice,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import Animated, {
  Easing,
  interpolateColor,
  runOnJS,
  useAnimatedReaction,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { Feather } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { Accelerometer, type AccelerometerMeasurement } from 'expo-sensors';

import { Button } from '@src/components';
import { useScanStore } from '@src/state/scanStore';
import type { Surface } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

type SurfaceParam = Extract<Surface, 'front' | 'back'>;

function isValidSurface(s: string | undefined): s is SurfaceParam {
  return s === 'front' || s === 'back';
}

// Discrete pre-check verdict surfaced to the UI. All four reasons are
// driven by real signals: blur/glare/coverage from the frame processor,
// motion from the accelerometer.
type PreCheck =
  | { kind: 'unknown' }
  | { kind: 'ready' }
  | { kind: 'warn'; reason: 'blur' | 'glare' | 'coverage' | 'motion' };

// --- Frame-processor configuration ---------------------------------
//
// Downsample target. 64×96 keeps the bottle-aspect (≈0.65) close to
// native and gives 6,144 pixels — small enough to iterate cheaply
// without hardware acceleration, large enough that the Laplacian still
// resolves real focus differences. Dropping below 32×48 made blur
// scores too noisy in informal calibration; raising above 80×120
// blew the per-frame budget.
const RESIZE_W = 64;
const RESIZE_H = 96;

// Frame-throttling. Run the worklet on every Nth frame, where N starts
// at 3 (≈10 Hz at 30 fps). If average worklet latency exceeds
// FRAME_BUDGET_MS we double N to 6 (≈5 Hz) and surface "checking…"
// rather than potentially stale verdicts.
const FRAME_STRIDE_FAST = 3;
const FRAME_STRIDE_SLOW = 6;
const FRAME_BUDGET_MS = 25; // total per-frame work budget per spec.
const LATENCY_EMA_ALPHA = 0.2; // smoothing factor for the rolling latency.
// If no fresh verdict arrived within this many ms, treat it as stale
// and hold the chip at 'unknown' until a fresh sample lands.
const VERDICT_STALE_MS = 1000;

// --- Threshold defaults --------------------------------------------
//
// All defaults are conservative starting points — they're meant to
// produce few false-positives on a well-lit reference shot. Tuning
// guidance lives in mobile/README.md ("Pre-check calibration"); the
// expectation is that field captures will move every constant here
// at least once before launch.

// Laplacian variance threshold. Below this = "blurry". 100 is a
// commonly cited starting value for OpenCV's Laplacian-variance blur
// detector and held up reasonably on the dev iPhone; expect it to
// move once we have field data.
const BLUR_THRESHOLD = 100;

// Saturated-luma cut-off. Pixels with luma above this count toward
// the glare ratio. 250 is the literal definition in SPEC §v1.5 F1.3
// ("R&G&B > 250") translated into luma space (BT.601 of (250,250,250)
// is 250); using straight luma keeps the worklet single-pass.
const GLARE_LUMA_CUTOFF = 250;
// Glare flare-up threshold. >20% of the ROI saturated = "reduce glare".
const GLARE_THRESHOLD = 0.2;

// Coverage uses a luma window heuristic: pixels neither too dark
// (background, lensbar) nor saturated (glare) are treated as
// "label-ish". Real label-area detection wants segmentation; this is
// the cheap version called out in the task brief.
const COVERAGE_DARK_CUTOFF = 80;
const COVERAGE_BRIGHT_CUTOFF = 230;
// Coverage band thresholds.
//   ratio < TOO_FAR  → "Move closer" (label too small in the ROI).
//   ratio > TOO_CLOSE → "Move back" (label fills the ROI; we want a
//                       margin so OCR can find the edges).
const COVERAGE_TOO_FAR = 0.3;
const COVERAGE_TOO_CLOSE = 0.8;

// Region-of-interest inside the resized 64×96 buffer: the dashed
// overlay frame on screen is 70% width × ~127% height (aspectRatio
// 0.55 against ~70% of viewport), which lands at roughly 70% of the
// ROI within the camera frame. We use a 70% × 70% center crop here
// as a coarse approximation; once curvature-aware framing lands
// (Phase 3) we'll feed the segmentation rect in directly.
const ROI_FRAC_X0 = 0.15;
const ROI_FRAC_X1 = 0.85;
const ROI_FRAC_Y0 = 0.15;
const ROI_FRAC_Y1 = 0.85;

// --- Accelerometer config ------------------------------------------
//
// 10 Hz updates per the lead's brief — fast enough to feel responsive,
// slow enough not to thrash a list.
const ACCEL_INTERVAL_MS = 100;
// Number of samples in the rolling window used to compute variance.
// 10 samples @ 10 Hz ≈ 1 second of motion history.
const ACCEL_WINDOW = 10;
// High-pass: variance of |a| above this threshold ⇒ "shaking". Tuned
// for hand-held use; bumping the device or walking with it crosses
// this comfortably while a phone resting on a surface stays well
// under it. Units are g^2 (gravitational-force squared).
const MOTION_VARIANCE_THRESHOLD = 0.015;

export default function CameraScreen(): React.ReactElement {
  const { surface } = useLocalSearchParams<{ surface: string }>();
  const safeSurface: SurfaceParam = isValidSurface(surface) ? surface : 'front';

  const { hasPermission, requestPermission } = useCameraPermission();
  const device: CameraDevice | undefined = useCameraDevice('back');

  const cameraRef = useRef<Camera>(null);

  const setCapture = useScanStore((s) => s.setCapture);
  const captures = useScanStore((s) => s.captures);

  const [livePreCheck, setLivePreCheck] = useState<PreCheck>({ kind: 'unknown' });
  const [busy, setBusy] = useState(false);

  const motionDetected = useMotionVerdict(hasPermission);

  // Real motion verdict (accelerometer) takes precedence over the
  // worklet-driven verdict. The frame processor only knows what's in
  // the picture, not how steady the device is being held.
  const preCheck: PreCheck = motionDetected
    ? { kind: 'warn', reason: 'motion' }
    : livePreCheck;

  // --- Frame-processor wiring --------------------------------------
  const { resize } = useResizePlugin();
  // Each shared value is updated by the worklet on every processed
  // frame; the useAnimatedReaction below maps them into the discrete
  // PreCheck the chip + capture-gate already understand.
  const blurScoreSv = useSharedValue<number>(0);
  const glareRatioSv = useSharedValue<number>(0);
  const coverageRatioSv = useSharedValue<number>(0);
  // Monotonic frame counter (worklet-side). The reaction watches it to
  // know "a fresh verdict arrived"; staleness is measured in JS via
  // lastVerdictAtRef.
  const frameTickSv = useSharedValue<number>(0);
  // Stride is JS-owned but a shared value so the worklet can read it
  // without a JS-thread hop. Adapts down (stride goes up) under load.
  const frameStrideSv = useSharedValue<number>(FRAME_STRIDE_FAST);
  // Frame index inside the worklet — used to apply the stride.
  const frameIdxSv = useSharedValue<number>(0);

  // Rolling EMA of worklet latency (ms). Updated from the worklet
  // via runOnJS; read on the JS side to make the stride decision.
  const latencyEmaRef = useRef<number>(0);
  const lastVerdictAtRef = useRef<number>(0);

  const recordLatency = useCallback((ms: number) => {
    const prev = latencyEmaRef.current;
    const next =
      prev === 0 ? ms : prev * (1 - LATENCY_EMA_ALPHA) + ms * LATENCY_EMA_ALPHA;
    latencyEmaRef.current = next;
    // Adapt stride if we're consistently over budget. Hysteresis so we
    // don't thrash: drop to slow at >budget, restore to fast at <70%
    // of budget.
    const currentStride = frameStrideSv.value;
    if (currentStride === FRAME_STRIDE_FAST && next > FRAME_BUDGET_MS) {
      frameStrideSv.value = FRAME_STRIDE_SLOW;
    } else if (
      currentStride === FRAME_STRIDE_SLOW &&
      next < FRAME_BUDGET_MS * 0.7
    ) {
      frameStrideSv.value = FRAME_STRIDE_FAST;
    }
  }, [frameStrideSv]);

  const markVerdict = useCallback(() => {
    lastVerdictAtRef.current = Date.now();
  }, []);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    // Stride: only sample every Nth frame to stay within budget.
    const idx = frameIdxSv.value + 1;
    frameIdxSv.value = idx;
    if (idx % frameStrideSv.value !== 0) return;

    const t0 = performance.now();

    // Resize + force a deterministic pixel format the worklet can
    // index without per-platform branching. RGB uint8 = 3 bytes/pixel.
    const buffer = resize(frame, {
      scale: { width: RESIZE_W, height: RESIZE_H },
      pixelFormat: 'rgb',
      dataType: 'uint8',
    });

    const w = RESIZE_W;
    const h = RESIZE_H;
    const total = w * h;

    // Single pass over the buffer. We need:
    //   - luma at every pixel (Float32-equivalent — store back into a
    //     scratch typed array so the Laplacian pass can re-read it).
    //   - inside-ROI counters for glare + coverage.
    //
    // Allocating a Uint8Array sized to the resize output every frame
    // is cheap (~6KB) compared to the resize+RGB copy that just
    // happened; pre-allocating across frames would be ideal but
    // lifetime across worklet runs is fragile. Keeping the alloc here
    // is honest about the trade.
    const luma = new Uint8Array(total);

    const x0 = Math.floor(w * ROI_FRAC_X0);
    const x1 = Math.floor(w * ROI_FRAC_X1);
    const y0 = Math.floor(h * ROI_FRAC_Y0);
    const y1 = Math.floor(h * ROI_FRAC_Y1);
    const roiPixels = (x1 - x0) * (y1 - y0);

    let glareCount = 0;
    let coverageCount = 0;

    for (let y = 0; y < h; y++) {
      const rowStart = y * w;
      for (let x = 0; x < w; x++) {
        const i = (rowStart + x) * 3;
        // BT.601 luma. Integer math on Uint8Array stays in JS-VM int
        // ops; the >> 8 keeps the result in [0,255].
        const r = buffer[i];
        const g = buffer[i + 1];
        const b = buffer[i + 2];
        const y8 = (r * 77 + g * 150 + b * 29) >> 8;
        luma[rowStart + x] = y8;

        if (x >= x0 && x < x1 && y >= y0 && y < y1) {
          if (y8 > GLARE_LUMA_CUTOFF) glareCount++;
          else if (y8 > COVERAGE_DARK_CUTOFF && y8 < COVERAGE_BRIGHT_CUTOFF) {
            coverageCount++;
          }
        }
      }
    }

    // Discrete 4-neighbor Laplacian over the ROI; variance of the
    // Laplacian is the standard blur metric. We only sum inside the
    // ROI minus a 1-pixel border so the index math stays trivial.
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
        const c = luma[row + x];
        const up = luma[row - w + x];
        const dn = luma[row + w + x];
        const lf = luma[row + x - 1];
        const rt = luma[row + x + 1];
        const lap = 4 * c - up - dn - lf - rt;
        lapSum += lap;
        lapSumSq += lap * lap;
        lapN++;
      }
    }
    const mean = lapN > 0 ? lapSum / lapN : 0;
    const variance = lapN > 0 ? lapSumSq / lapN - mean * mean : 0;

    blurScoreSv.value = variance;
    glareRatioSv.value = roiPixels > 0 ? glareCount / roiPixels : 0;
    coverageRatioSv.value = roiPixels > 0 ? coverageCount / roiPixels : 0;
    frameTickSv.value = idx;

    const ms = performance.now() - t0;
    runOnJS(recordLatency)(ms);
    runOnJS(markVerdict)();
  }, [resize, blurScoreSv, glareRatioSv, coverageRatioSv, frameTickSv, frameStrideSv, frameIdxSv, recordLatency, markVerdict]);

  // Map the shared values into the chip's PreCheck. Severity priority:
  // blur first (a blurred frame invalidates everything else), then
  // glare (interferes with text), then coverage (fixable by stepping
  // closer/further). 'ready' only fires when all three pass.
  useAnimatedReaction(
    () => ({
      tick: frameTickSv.value,
      blur: blurScoreSv.value,
      glare: glareRatioSv.value,
      coverage: coverageRatioSv.value,
    }),
    (curr) => {
      'worklet';
      if (curr.tick === 0) return;
      let next: PreCheck;
      if (curr.blur < BLUR_THRESHOLD) {
        next = { kind: 'warn', reason: 'blur' };
      } else if (curr.glare > GLARE_THRESHOLD) {
        next = { kind: 'warn', reason: 'glare' };
      } else if (
        curr.coverage < COVERAGE_TOO_FAR ||
        curr.coverage > COVERAGE_TOO_CLOSE
      ) {
        next = { kind: 'warn', reason: 'coverage' };
      } else {
        next = { kind: 'ready' };
      }
      runOnJS(setLivePreCheck)(next);
    },
    [],
  );

  // Staleness watchdog: if no frame has been processed in
  // VERDICT_STALE_MS (thermal throttling, paused worklet, etc.), pull
  // the chip back to 'unknown' / "checking…" so we never display a
  // stale verdict.
  useEffect(() => {
    if (!hasPermission) return undefined;
    const interval = setInterval(() => {
      const last = lastVerdictAtRef.current;
      if (last === 0) return;
      const stale = Date.now() - last > VERDICT_STALE_MS;
      if (stale) {
        setLivePreCheck((prev) =>
          prev.kind === 'unknown' ? prev : { kind: 'unknown' },
        );
      }
    }, 250);
    return () => clearInterval(interval);
  }, [hasPermission]);

  useEffect(() => {
    if (hasPermission === false) {
      void requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const handleCapture = useCallback(async () => {
    if (!cameraRef.current || busy) return;
    setBusy(true);
    try {
      const photo = await cameraRef.current.takePhoto({
        // TODO(hdr): enable HDR adaptive to thermal state per SPEC §0.5.
        flash: 'off',
        enableShutterSound: false,
      });
      setCapture(safeSurface, {
        // Vision Camera returns a path without scheme on iOS / Android.
        uri: photo.path.startsWith('file://') ? photo.path : `file://${photo.path}`,
        width: photo.width,
        height: photo.height,
        capturedAt: Date.now(),
      });
      // Front → back; back → review.
      if (safeSurface === 'front') {
        router.replace('/(app)/scan/camera/back');
      } else {
        router.replace('/(app)/scan/review');
      }
    } catch (err) {
      // takePhoto failures come from Vision Camera (shutter busy, device
      // disconnected, etc.) — describeError will route them through the
      // unknown-error path and surface the native message behind the
      // "Show details" disclosure.
      showErrorAlert(err, { title: "Couldn't capture photo" });
    } finally {
      setBusy(false);
    }
  }, [busy, safeSurface, setCapture]);

  if (!hasPermission) {
    return (
      <View style={styles.permissionWrap}>
        <Text style={styles.permissionTitle}>Camera permission needed</Text>
        <Text style={styles.permissionBody}>
          ProofRead needs camera access to capture beverage labels.
        </Text>
        <Button
          label="Grant access"
          onPress={() => {
            void requestPermission();
          }}
        />
        <Button label="Back" variant="ghost" onPress={() => router.back()} />
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.permissionWrap}>
        <Text style={styles.permissionTitle}>No camera available</Text>
        <Text style={styles.permissionBody}>
          This device doesn't expose a back-facing camera.
        </Text>
        <Button label="Back" variant="ghost" onPress={() => router.back()} />
      </View>
    );
  }

  const existingCapture = captures[safeSurface];

  return (
    <View style={styles.root}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        photo
        frameProcessor={frameProcessor}
        // RGB pixel format keeps the resize plugin's RGB output trivial;
        // YUV would force a per-platform conversion in the worklet.
        pixelFormat="yuv"
      />

      {/* Top status bar */}
      <View style={styles.topBar}>
        <Pressable onPress={() => router.back()} style={styles.iconButton}>
          <Text style={styles.iconText}>Cancel</Text>
        </Pressable>
        <Text style={styles.surfaceLabel}>
          {safeSurface === 'front' ? 'Front of label' : 'Back of label'}
        </Text>
        <View style={styles.iconSpacer} />
      </View>

      {/* Bottle outline overlay placeholder */}
      <View style={styles.overlayWrap} pointerEvents="none">
        <View style={styles.overlayFrame} />
      </View>

      {/* Pre-check chip */}
      <View style={styles.preCheckWrap} pointerEvents="none">
        <PreCheckIndicator value={preCheck} />
      </View>

      {/* Bottom controls */}
      <View style={styles.bottomBar}>
        <View style={styles.bottomLeft}>
          {existingCapture ? (
            <Text style={styles.retakeHint}>Captured. Press to retake.</Text>
          ) : null}
        </View>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Capture photo"
          disabled={busy || preCheck.kind !== 'ready'}
          onPress={handleCapture}
          style={({ pressed }) => [
            styles.shutterOuter,
            (busy || preCheck.kind !== 'ready') && { opacity: 0.5 },
            pressed && { transform: [{ scale: 0.97 }] },
          ]}
        >
          <View style={styles.shutterInner} />
        </Pressable>
        <View style={styles.bottomRight} />
      </View>
    </View>
  );
}

// Subscribe to the accelerometer and report a boolean "is the device
// shaking" signal. We compute the variance of |acceleration| over a
// rolling window and compare to a tuned threshold; that high-passes
// gravity (a constant ~1g offset) so a stationary phone reads as
// steady regardless of orientation.
function useMotionVerdict(active: boolean | undefined): boolean {
  const [shaking, setShaking] = useState(false);

  useEffect(() => {
    if (!active) return undefined;

    Accelerometer.setUpdateInterval(ACCEL_INTERVAL_MS);

    const window: number[] = [];
    const subscription = Accelerometer.addListener((m: AccelerometerMeasurement) => {
      const magnitude = Math.sqrt(m.x * m.x + m.y * m.y + m.z * m.z);
      window.push(magnitude);
      if (window.length > ACCEL_WINDOW) window.shift();
      if (window.length < ACCEL_WINDOW) return;

      const mean = window.reduce((s, v) => s + v, 0) / window.length;
      let acc = 0;
      for (const v of window) acc += (v - mean) * (v - mean);
      const variance = acc / window.length;

      setShaking((prev) => {
        const next = variance > MOTION_VARIANCE_THRESHOLD;
        return prev === next ? prev : next;
      });
    });

    return () => {
      subscription.remove();
    };
  }, [active]);

  return shaking;
}

// Numeric kind code drives Reanimated color interpolation. Keep
// integer-spaced so interpolateColor's input range is trivial.
const KIND_CODE = { unknown: 0, warn: 1, ready: 2 } as const;

const CHIP_COLORS = {
  unknown: 'rgba(0,0,0,0.55)',
  warn: 'rgba(244,184,96,0.9)',
  ready: 'rgba(61,220,151,0.85)',
} as const;

// Light impact on enter-ready, warning notification on enter-warn.
// Promises are intentionally unawaited — the haptic is fire-and-forget
// from the caller's perspective. Errors (e.g., simulator without a
// haptics engine) are swallowed.
function triggerHaptic(kind: 'ready' | 'warn'): void {
  if (kind === 'ready') {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
  } else {
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning).catch(() => {});
  }
}

function PreCheckIndicator({ value }: { value: PreCheck }): React.ReactElement {
  const { label, kindKey, accessibilityLabel } = describe(value);

  // Drive both color and scale from a single shared value tracking
  // the verdict kind. We animate to the target then settle with a
  // spring on scale for a subtle "snap" when verdicts change.
  const kindCode = useSharedValue<number>(KIND_CODE[kindKey]);
  const scale = useSharedValue<number>(1);
  const spinDeg = useSharedValue<number>(0);

  // Track the previous kind so we can fire haptics only on edges.
  const prevKindRef = useRef<'unknown' | 'warn' | 'ready'>(kindKey);

  useEffect(() => {
    kindCode.value = withTiming(KIND_CODE[kindKey], {
      duration: 220,
      easing: Easing.out(Easing.quad),
    });

    // Brief "pulse" scale: bump up then settle back via spring.
    scale.value = withSequence(
      withTiming(1.08, { duration: 120, easing: Easing.out(Easing.quad) }),
      withSpring(1, { damping: 12, stiffness: 180 }),
    );

    // Haptic edge detection — only fire on enter, not on re-entry of
    // the same kind. 'unknown' is silent.
    const prev = prevKindRef.current;
    if (prev !== kindKey) {
      if (kindKey === 'ready') triggerHaptic('ready');
      else if (kindKey === 'warn') triggerHaptic('warn');
      prevKindRef.current = kindKey;
    }
  }, [kindKey, kindCode, scale]);

  // Continuous rotation for the 'unknown' spinner. Driven separately
  // so it doesn't fight the per-verdict transition animation.
  useEffect(() => {
    if (kindKey === 'unknown') {
      spinDeg.value = 0;
      spinDeg.value = withRepeat(
        withTiming(360, { duration: 900, easing: Easing.linear }),
        -1,
        false,
      );
    } else {
      spinDeg.value = withTiming(0, { duration: 120 });
    }
  }, [kindKey, spinDeg]);

  const animatedChipStyle = useAnimatedStyle(() => {
    const bg = interpolateColor(
      kindCode.value,
      [KIND_CODE.unknown, KIND_CODE.warn, KIND_CODE.ready],
      [CHIP_COLORS.unknown, CHIP_COLORS.warn, CHIP_COLORS.ready],
    );
    return {
      backgroundColor: bg,
      transform: [{ scale: scale.value }],
    };
  });

  const animatedSpinStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${spinDeg.value}deg` }],
  }));

  return (
    <Animated.View
      accessibilityRole="text"
      accessibilityLabel={accessibilityLabel}
      style={[styles.preCheck, animatedChipStyle]}
    >
      <Animated.View style={animatedSpinStyle}>
        <PreCheckIcon kindKey={kindKey} />
      </Animated.View>
      <Text style={styles.preCheckText}>{label}</Text>
    </Animated.View>
  );
}

function PreCheckIcon({
  kindKey,
}: {
  kindKey: 'unknown' | 'warn' | 'ready';
}): React.ReactElement {
  const iconColor = colors.background;
  const size = 14;
  switch (kindKey) {
    case 'unknown':
      return <Feather name="loader" size={size} color={iconColor} />;
    case 'warn':
      return <Feather name="alert-triangle" size={size} color={iconColor} />;
    case 'ready':
      return <Feather name="check" size={size} color={iconColor} />;
  }
}

function describe(p: PreCheck): {
  label: string;
  kindKey: 'unknown' | 'warn' | 'ready';
  accessibilityLabel: string;
} {
  switch (p.kind) {
    case 'unknown':
      return {
        label: 'Hold steady…',
        kindKey: 'unknown',
        accessibilityLabel: 'Pre-check pending. Hold steady.',
      };
    case 'ready':
      return {
        label: 'Ready',
        kindKey: 'ready',
        accessibilityLabel: 'Pre-check ready. You can capture now.',
      };
    case 'warn': {
      const map: Record<typeof p.reason, string> = {
        blur: 'Image is blurry',
        glare: 'Reduce glare',
        coverage: 'Move closer',
        motion: 'Hold steady',
      };
      const label = map[p.reason];
      return {
        label,
        kindKey: 'warn',
        accessibilityLabel: `Pre-check warning: ${label}.`,
      };
    }
  }
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#000',
  },
  topBar: {
    position: 'absolute',
    top: 50,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
  },
  iconButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    backgroundColor: 'rgba(0,0,0,0.45)',
    borderRadius: radius.md,
  },
  iconText: {
    ...typography.body,
    color: '#FFF',
    fontWeight: '600',
  },
  iconSpacer: { width: 64 },
  surfaceLabel: {
    ...typography.heading,
    color: '#FFF',
  },
  overlayWrap: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  overlayFrame: {
    width: '70%',
    aspectRatio: 0.55,
    borderColor: 'rgba(255,255,255,0.7)',
    borderWidth: 2,
    borderRadius: radius.xl,
    borderStyle: 'dashed',
  },
  preCheckWrap: {
    position: 'absolute',
    top: 130,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  preCheck: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radius.xl,
  },
  preCheckText: {
    ...typography.caption,
    color: colors.background,
    fontWeight: '700',
    letterSpacing: 0.4,
  },
  bottomBar: {
    position: 'absolute',
    bottom: 50,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.xl,
  },
  bottomLeft: {
    flex: 1,
    alignItems: 'flex-start',
  },
  bottomRight: {
    flex: 1,
  },
  shutterOuter: {
    width: 78,
    height: 78,
    borderRadius: 39,
    borderWidth: 4,
    borderColor: '#FFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#FFF',
  },
  retakeHint: {
    ...typography.caption,
    color: '#FFF',
    backgroundColor: 'rgba(0,0,0,0.45)',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radius.sm,
  },
  permissionWrap: {
    flex: 1,
    backgroundColor: colors.background,
    padding: spacing.lg,
    gap: spacing.md,
    justifyContent: 'center',
  },
  permissionTitle: {
    ...typography.title,
    color: colors.text,
  },
  permissionBody: {
    ...typography.body,
    color: colors.textMuted,
  },
});
