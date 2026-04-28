/**
 * Camera capture screen.
 *
 * SPEC §v1.6 step 4–5 + §v1.7 row "Camera". Substantive structural
 * stub — all the structural pieces are wired, but several pieces of
 * the live pipeline are explicit TODOs:
 *
 *   - Frame-processor pre-checks (focus / glare / coverage / motion)
 *     per SPEC §v1.5 F1.3 are placeholders; the real implementation
 *     uses Vision Camera v4 frame processors + expo-ml-kit text
 *     recognition (SPEC §0 tech stack) feeding a worklet that
 *     surfaces a discrete pre-check verdict.
 *   - HDR + low-light + thermal adaptation per SPEC §0.5 mitigation
 *     table are TODOs; this screen captures with default settings.
 *
 * What this file _does_ get right today:
 *   - Camera permission gating
 *   - Front/back surface routing through the dynamic [surface] param
 *   - Capture button + retake / continue flow into scanStore
 *   - Visible pre-check indicator placeholder so the layout is real
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import {
  Camera,
  type CameraDevice,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';

import { Button } from '@src/components';
import { useScanStore } from '@src/state/scanStore';
import type { Surface } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

type SurfaceParam = Extract<Surface, 'front' | 'back'>;

function isValidSurface(s: string | undefined): s is SurfaceParam {
  return s === 'front' || s === 'back';
}

// Discrete pre-check verdict surfaced to the UI. Real values come
// from a frame processor; in the scaffold we hardcode "ready".
type PreCheck =
  | { kind: 'unknown' }
  | { kind: 'ready' }
  | { kind: 'warn'; reason: 'blur' | 'glare' | 'coverage' | 'motion' };

export default function CameraScreen(): React.ReactElement {
  const { surface } = useLocalSearchParams<{ surface: string }>();
  const safeSurface: SurfaceParam = isValidSurface(surface) ? surface : 'front';

  const { hasPermission, requestPermission } = useCameraPermission();
  const device: CameraDevice | undefined = useCameraDevice('back');

  const cameraRef = useRef<Camera>(null);

  const setCapture = useScanStore((s) => s.setCapture);
  const captures = useScanStore((s) => s.captures);

  const [preCheck, setPreCheck] = useState<PreCheck>({ kind: 'unknown' });
  const [busy, setBusy] = useState(false);

  // TODO(prechecks): wire a Vision Camera v4 frame processor that
  // computes Laplacian variance (focus), saturated-pixel ratio (glare),
  // detected-label area (coverage) and reports a verdict back to RN.
  // For now we just flip to 'ready' once the camera mounts so the UI
  // exercises both states.
  useEffect(() => {
    if (!hasPermission) return;
    const t = setTimeout(() => setPreCheck({ kind: 'ready' }), 600);
    return () => clearTimeout(t);
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

function PreCheckIndicator({ value }: { value: PreCheck }): React.ReactElement {
  const { bg, label } = labelFor(value);
  return (
    <View style={[styles.preCheck, { backgroundColor: bg }]}>
      <Text style={styles.preCheckText}>{label}</Text>
    </View>
  );
}

function labelFor(p: PreCheck): { bg: string; label: string } {
  switch (p.kind) {
    case 'unknown':
      return { bg: 'rgba(0,0,0,0.55)', label: 'Hold steady…' };
    case 'ready':
      return { bg: 'rgba(61,220,151,0.85)', label: 'Ready' };
    case 'warn': {
      const map: Record<typeof p.reason, string> = {
        blur: 'Image is blurry',
        glare: 'Reduce glare',
        coverage: 'Move closer',
        motion: 'Hold steady',
      };
      return { bg: 'rgba(244,184,96,0.9)', label: map[p.reason] };
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
