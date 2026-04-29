/**
 * BottleSilhouetteOverlay — stylized rounded-rect outline that traces
 * the detected bottle's left/right edges and its inferred top/bottom
 * caps (SCAN_DESIGN §3.3 + §4.1).
 *
 * Color, opacity, and glow respond to the active ScanState; positions
 * are driven entirely by `silhouetteSv`. The shape is intentionally
 * NOT a literal contour — it's an "I see a bottle here" hint, not a
 * curvature trace. We render with react-native-svg because the strokes,
 * caps, and drop shadow compose cleanly on the same layer.
 */

import React, { useEffect } from 'react';
import { StyleSheet, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedProps,
  useDerivedValue,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';
import Svg, { Rect } from 'react-native-svg';

import { colors, scanGeometry, scanMotion } from '@src/theme';
import type { ScanStateKind, PauseReason } from './ScanInstructions';

const AnimatedRect = Animated.createAnimatedComponent(Rect);

export interface SilhouetteFrame {
  /** Center X in screen-px (camera viewport space). */
  centerX: number;
  /** Center Y in screen-px. */
  centerY: number;
  /** Width in screen-px (silhouette diameter). */
  widthPx: number;
  /** Height in screen-px (silhouette vertical extent). */
  heightPx: number;
}

export interface BottleSilhouetteOverlayProps {
  /** Live geometry from the tracker, in screen-px. */
  silhouetteSv: SharedValue<SilhouetteFrame>;
  /** Whether the tracker currently has a confident silhouette lock. */
  detectedSv: SharedValue<boolean>;
  /** 0..1 from the tracker — the EMA steadiness over recent frames. */
  steadinessSv: SharedValue<number>;
  /** Active scan state — drives color + glow choices. */
  state: ScanStateKind;
  /** Pause reason when state is `paused`; `lost_bottle` triggers ghost. */
  pauseReason?: PauseReason;
  /** Camera viewport size — used to size the SVG canvas. */
  viewportWidth: number;
  viewportHeight: number;
}

interface VisualSpec {
  /** Stroke color reference token. */
  strokeColor: string;
  /** Glow color reference token (rgba). */
  glowColor: string;
  /** Target opacity in 0..1. */
  opacity: number;
  /** Target glow radius in px (used as shadowRadius). */
  glowRadius: number;
  /** Whether the live silhouette geometry should be tracked. */
  followLive: boolean;
  /** Whether the glow should breathe / pulse. */
  pulse: 'none' | 'ready' | 'scanning';
}

function specForState(
  state: ScanStateKind,
  pauseReason: PauseReason | undefined,
  steadiness: number,
  detected: boolean,
): VisualSpec {
  if (state === 'aligning') {
    if (!detected) {
      return {
        strokeColor: colors.scanIdle,
        glowColor: colors.scanIdleSoft,
        opacity: 0,
        glowRadius: 0,
        followLive: true,
        pulse: 'none',
      };
    }
    return {
      strokeColor: colors.scanIdle,
      glowColor: colors.scanIdleSoft,
      opacity: steadiness < 1 ? 0.6 : 1.0,
      glowRadius: 0,
      followLive: true,
      pulse: 'none',
    };
  }
  if (state === 'ready') {
    return {
      strokeColor: colors.scanReady,
      glowColor: colors.scanReadySoft,
      opacity: 1,
      glowRadius: 10,
      followLive: true,
      pulse: 'ready',
    };
  }
  if (state === 'scanning') {
    return {
      strokeColor: colors.scanIdle,
      glowColor: colors.scanIdleGlow,
      opacity: 1,
      glowRadius: 12,
      followLive: true,
      pulse: 'scanning',
    };
  }
  if (state === 'paused') {
    if (pauseReason === 'lost_bottle') {
      // §4.1: ghost the last-known silhouette at 0.3 opacity in fail
      // color. The 800ms-wait + 240ms-fade is the parent's job once it
      // drops us back to `aligning`.
      return {
        strokeColor: colors.scanFail,
        glowColor: colors.scanFailSoft,
        opacity: 0.3,
        glowRadius: 0,
        followLive: false,
        pulse: 'none',
      };
    }
    return {
      strokeColor: colors.scanWarn,
      glowColor: colors.scanWarnSoft,
      opacity: 0.9,
      glowRadius: 8,
      followLive: false,
      pulse: 'none',
    };
  }
  // complete + failed → invisible (parent owns reveal/transitions).
  return {
    strokeColor: colors.scanIdle,
    glowColor: colors.scanIdleSoft,
    opacity: 0,
    glowRadius: 0,
    followLive: false,
    pulse: 'none',
  };
}

export function BottleSilhouetteOverlay({
  silhouetteSv,
  detectedSv,
  steadinessSv,
  state,
  pauseReason,
  viewportWidth,
  viewportHeight,
}: BottleSilhouetteOverlayProps): React.ReactElement {
  // Sample steadiness/detected once on mount + on state changes; we
  // don't need them to drive per-frame style, only to derive the visual
  // spec at a state edge.
  const [steadinessSnapshot, setSteadinessSnapshot] = React.useState(0);
  const [detectedSnapshot, setDetectedSnapshot] = React.useState(false);

  useEffect(() => {
    // Lazy snapshot: shared values are read once at state change.
    setSteadinessSnapshot(steadinessSv.value);
    setDetectedSnapshot(detectedSv.value);
  }, [state, pauseReason, steadinessSv, detectedSv]);

  const spec = specForState(
    state,
    pauseReason,
    steadinessSnapshot,
    detectedSnapshot,
  );

  // Animated targets — opacity, stroke-width, glow radius. We tween
  // toward whatever the current spec dictates.
  const opacity = useSharedValue<number>(0);
  const strokeWidth = useSharedValue<number>(0);
  const glowOpacity = useSharedValue<number>(0);

  useEffect(() => {
    // Detection-lands curve (§4.1): opacity 0 → 0.6 → 1.0 in 220ms.
    if (state === 'aligning' && detectedSnapshot && spec.opacity >= 0.6) {
      opacity.value = withSequence(
        withTiming(0.6, {
          duration: 140,
          easing: Easing.out(Easing.cubic),
        }),
        withTiming(spec.opacity, {
          duration: 80,
          easing: Easing.out(Easing.cubic),
        }),
      );
      strokeWidth.value = withTiming(scanGeometry.silhouetteStrokeWidth, {
        duration: 220,
        easing: Easing.out(Easing.cubic),
      });
    } else if (spec.opacity === 0) {
      // Detection lost or hidden state — drop fast to a ghost.
      opacity.value = withTiming(spec.opacity, {
        duration: 80,
        easing: Easing.out(Easing.quad),
      });
    } else {
      opacity.value = withTiming(spec.opacity, scanMotion.midEase);
      strokeWidth.value = withTiming(scanGeometry.silhouetteStrokeWidth, {
        duration: 220,
        easing: Easing.out(Easing.cubic),
      });
    }

    // Glow pulse.
    glowOpacity.value = 0;
    if (spec.pulse === 'ready') {
      glowOpacity.value = withRepeat(
        withSequence(
          withTiming(1.0, {
            duration: 700,
            easing: Easing.inOut(Easing.sin),
          }),
          withTiming(0.4, {
            duration: 700,
            easing: Easing.inOut(Easing.sin),
          }),
        ),
        -1,
        false,
      );
    } else if (spec.pulse === 'scanning') {
      glowOpacity.value = withRepeat(
        withSequence(
          withTiming(0.85, {
            duration: 450,
            easing: Easing.inOut(Easing.sin),
          }),
          withTiming(0.5, {
            duration: 450,
            easing: Easing.inOut(Easing.sin),
          }),
        ),
        -1,
        false,
      );
    } else {
      glowOpacity.value = withTiming(spec.glowRadius > 0 ? 1 : 0, {
        duration: 220,
      });
    }
  }, [
    state,
    pauseReason,
    detectedSnapshot,
    spec.opacity,
    spec.pulse,
    spec.glowRadius,
    opacity,
    strokeWidth,
    glowOpacity,
  ]);

  // Stash last-live frame so the freeze freezes at the current value
  // and not at a stale one. Declared before `renderFrame` so the
  // worklet closure references a defined binding.
  const frozenFrame = useSharedValue<SilhouetteFrame>({
    centerX: viewportWidth / 2,
    centerY: viewportHeight / 2,
    widthPx: 200,
    heightPx: 360,
  });
  useDerivedValue(() => {
    if (spec.followLive) {
      frozenFrame.value = silhouetteSv.value;
    }
    return 0;
  }, [spec.followLive, silhouetteSv]);

  // Frozen-at-last-known geometry vs live-tracking geometry. We
  // implement this by deriving a "render frame" shared value: when
  // followLive is true it copies silhouetteSv, otherwise it falls
  // back to `frozenFrame` (last seen position).
  const renderFrame = useDerivedValue<SilhouetteFrame>(() => {
    if (spec.followLive) return silhouetteSv.value;
    return frozenFrame.value;
  }, [spec.followLive, silhouetteSv]);

  const animatedProps = useAnimatedProps(() => {
    const f = renderFrame.value;
    const x = f.centerX - f.widthPx / 2;
    const y = f.centerY - f.heightPx / 2;
    return {
      x,
      y,
      width: f.widthPx,
      height: f.heightPx,
      opacity: opacity.value,
      strokeWidth: strokeWidth.value,
    } as Partial<{
      x: number;
      y: number;
      width: number;
      height: number;
      opacity: number;
      strokeWidth: number;
    }>;
  });

  const glowAnimatedProps = useAnimatedProps(() => {
    const f = renderFrame.value;
    const x = f.centerX - f.widthPx / 2;
    const y = f.centerY - f.heightPx / 2;
    return {
      x,
      y,
      width: f.widthPx,
      height: f.heightPx,
      opacity: glowOpacity.value * (spec.glowRadius > 0 ? 1 : 0),
      strokeWidth: scanGeometry.silhouetteStrokeWidth + spec.glowRadius * 0.6,
    } as Partial<{
      x: number;
      y: number;
      width: number;
      height: number;
      opacity: number;
      strokeWidth: number;
    }>;
  });

  return (
    <View
      pointerEvents="none"
      style={[
        StyleSheet.absoluteFill,
        styles.shadowWrap,
      ]}
      accessibilityRole="image"
      accessibilityLabel="Bottle outline indicator"
    >
      <Svg width={viewportWidth} height={viewportHeight}>
        {/* Outer glow rectangle, drawn beneath the main stroke. */}
        <AnimatedRect
          rx={scanGeometry.silhouetteCornerRadius}
          ry={scanGeometry.silhouetteCornerRadius}
          fill="none"
          stroke={spec.glowColor}
          strokeLinecap="round"
          strokeLinejoin="round"
          animatedProps={glowAnimatedProps}
        />
        {/* Primary stroke. */}
        <AnimatedRect
          rx={scanGeometry.silhouetteCornerRadius}
          ry={scanGeometry.silhouetteCornerRadius}
          fill="none"
          stroke={spec.strokeColor}
          strokeLinecap="round"
          strokeLinejoin="round"
          animatedProps={animatedProps}
        />
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  shadowWrap: {
    // RN drop-shadow on iOS; on Android we'd need elevation. The
    // silhouette is mostly readable even without it.
    shadowColor: '#000',
    shadowOpacity: 0.6,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: scanGeometry.silhouetteShadowBlur,
  },
});
