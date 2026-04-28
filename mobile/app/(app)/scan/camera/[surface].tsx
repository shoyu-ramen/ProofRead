/**
 * Camera capture screen.
 *
 * SPEC §v1.6 step 4–5 + §v1.7 row "Camera". Substantive structural
 * stub — all the structural pieces are wired, but several pieces of
 * the live pipeline are explicit TODOs:
 *
 *   - Frame-processor pre-checks (focus / glare / coverage) per
 *     SPEC §v1.5 F1.3 are simulated by a deterministic state machine
 *     so the UI exercises every verdict realistically. The real
 *     implementation uses Vision Camera v4 frame processors +
 *     expo-ml-kit text recognition (SPEC §0 tech stack) feeding a
 *     worklet that surfaces a discrete pre-check verdict.
 *   - HDR + low-light + thermal adaptation per SPEC §0.5 mitigation
 *     table are TODOs; this screen captures with default settings.
 *
 * What this file _does_ get right today:
 *   - Camera permission gating
 *   - Front/back surface routing through the dynamic [surface] param
 *   - Capture button + retake / continue flow into scanStore
 *   - Live pre-check verdict pipeline: deterministic script for
 *     blur / glare / coverage / ready PLUS a real accelerometer-driven
 *     motion verdict that overrides the script when the device is
 *     shaking
 *   - Light haptic on transition into ready, warning haptic on
 *     transition into warn (expo-haptics)
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import {
  Camera,
  type CameraDevice,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import Animated, {
  Easing,
  interpolateColor,
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

// Discrete pre-check verdict surfaced to the UI. Motion is real
// (accelerometer-driven); blur / glare / coverage are still scripted
// until a Vision Camera frame processor lands.
type PreCheck =
  | { kind: 'unknown' }
  | { kind: 'ready' }
  | { kind: 'warn'; reason: 'blur' | 'glare' | 'coverage' | 'motion' };

// Scripted sequence the state machine cycles through. Each step has
// a duration (ms) the verdict is held before advancing. The sequence
// gives the user a chance to see every warning state before it
// settles into 'ready'. Motion is intentionally NOT scripted — it is
// driven live from the accelerometer (see useMotionVerdict below) and
// overrides the scripted verdict whenever the device is shaking.
//
// REAL IMPLEMENTATION for the still-scripted verdicts (SPEC §v1.5
// F1.3 — once Reanimated worklets are configured):
//
//   - focus      → Laplacian variance over the label ROI; below a
//                  threshold ⇒ {kind:'warn', reason:'blur'}.
//   - glare      → ratio of saturated pixels (R&G&B > 250) inside
//                  ROI; above threshold ⇒ {reason:'glare'}.
//   - coverage   → bbox area of detected text labels / frame area;
//                  below threshold ⇒ {reason:'coverage'}.
//
//   The frame processor would call a shared-value setter whose
//   useAnimatedReaction would dispatch into this same setPreCheck.
const VERDICT_SCRIPT: ReadonlyArray<{ verdict: PreCheck; durationMs: number }> = [
  { verdict: { kind: 'unknown' }, durationMs: 500 },
  { verdict: { kind: 'warn', reason: 'coverage' }, durationMs: 900 },
  { verdict: { kind: 'warn', reason: 'blur' }, durationMs: 750 },
  { verdict: { kind: 'warn', reason: 'glare' }, durationMs: 700 },
  { verdict: { kind: 'ready' }, durationMs: 5000 },
];

// Accelerometer config. 10 Hz updates per the lead's brief — fast
// enough to feel responsive, slow enough not to thrash a list.
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

  const [scriptedPreCheck, setScriptedPreCheck] = useState<PreCheck>({ kind: 'unknown' });
  const [busy, setBusy] = useState(false);

  const script = useMemo(() => VERDICT_SCRIPT, []);
  const motionDetected = useMotionVerdict(hasPermission);

  // Real motion verdict (accelerometer) takes precedence over the
  // scripted verdicts. Once a real frame processor lands the rest of
  // these will also become live signals.
  const preCheck: PreCheck = motionDetected
    ? { kind: 'warn', reason: 'motion' }
    : scriptedPreCheck;

  // Drive the verdict state machine while the camera is live. The
  // sequence loops indefinitely starting from the last 'ready' step,
  // mimicking ambient jitter. Real verdicts will replace this once a
  // frame processor lands (see comment on VERDICT_SCRIPT above for
  // the per-verdict signal recipe).
  useEffect(() => {
    if (!hasPermission) return undefined;

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    // Once we exhaust the scripted bring-up, loop back to the index
    // that holds 'ready' so subsequent cycles look like ambient
    // jitter rather than a re-bring-up.
    const loopAnchor = Math.max(
      0,
      script.findIndex((s) => s.verdict.kind === 'ready'),
    );

    const advance = (index: number): void => {
      if (cancelled) return;
      const step = script[index];
      if (!step) return;
      setScriptedPreCheck(step.verdict);
      timeoutId = setTimeout(() => {
        const next = index + 1 >= script.length ? loopAnchor : index + 1;
        advance(next);
      }, step.durationMs);
    };

    advance(0);

    return () => {
      cancelled = true;
      if (timeoutId !== undefined) clearTimeout(timeoutId);
    };
  }, [hasPermission, script]);

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
      const message = err instanceof Error ? err.message : 'Capture failed';
      Alert.alert('Capture failed', message);
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
        // TODO(reanimated): wire a frame-processor prop here once
        // Reanimated worklets are configured. Keeping this commented
        // ensures it doesn't fail on devices without the worklet
        // runtime during the scaffold phase.
        // frameProcessor={frameProcessor}
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
