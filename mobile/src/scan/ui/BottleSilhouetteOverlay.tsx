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
  useAnimatedReaction,
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
  /** Signed angular velocity in rev/s. Drives the scanning-glow pulse rate. */
  angularVelocitySv: SharedValue<number>;
  /** 0..1 grip-steadiness EMA. Tightens the silhouette stroke as grip settles. */
  gripSteadinessSv: SharedValue<number>;
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
    // Imperative target is the §4.1 ceiling (1.0). The live steadiness
    // blend in `liveOpacity` below caps the rendered value at
    // 0.6 + 0.4 * steadiness, so the 0.6 → 1.0 phase tracks the signal
    // instead of resolving from a stale snapshot.
    return {
      strokeColor: colors.scanIdle,
      glowColor: colors.scanIdleSoft,
      opacity: 1.0,
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
  angularVelocitySv,
  gripSteadinessSv,
  state,
  pauseReason,
  viewportWidth,
  viewportHeight,
}: BottleSilhouetteOverlayProps): React.ReactElement {
  // `detected` is sampled at state edges — the spec only branches on
  // detected-vs-not, not on its sub-frame jitter, so a snapshot is fine.
  // Live `steadiness` is read per-frame inside `liveOpacity` below.
  const [detectedSnapshot, setDetectedSnapshot] = React.useState(false);

  useEffect(() => {
    setDetectedSnapshot(detectedSv.value);
  }, [state, pauseReason, detectedSv]);

  const spec = specForState(state, pauseReason, detectedSnapshot);

  // Animated targets — opacity, stroke-width, glow radius. We tween
  // toward whatever the current spec dictates.
  const opacity = useSharedValue<number>(0);
  const strokeWidth = useSharedValue<number>(0);
  const glowOpacity = useSharedValue<number>(0);

  useEffect(() => {
    // Detection-lands curve (§4.1): opacity 0 → 0.6 → 1.0 in 220ms. The
    // `liveOpacity` derived value (below) caps this at the live
    // steadiness ceiling so the 0.6 → 1.0 phase tracks the signal
    // instead of holding at a snapshot.
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

    // Glow pulse. The 'scanning' branch only kicks off an initial
    // 900ms baseline cycle — the velocity-driven reaction below
    // re-launches the withRepeat with a duration that tracks the
    // user's rotation speed, so the pulse feels "tied" to the bottle
    // rather than running on its own clock.
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

  // Mirror the discrete state into a SharedValue so the
  // velocity-driven reaction below can gate itself on `scanning` from
  // the worklet thread without round-tripping through React props.
  const stateKindSv = useSharedValue<ScanStateKind>(state);
  useEffect(() => {
    stateKindSv.value = state;
  }, [state, stateKindSv]);

  // Velocity-driven pulse rate (SCAN_DESIGN §4.1): re-launch the
  // scanning glow's withRepeat whenever the rotation speed crosses a
  // bucket boundary so the user feels the silhouette breathing in
  // sync with their wrist. The 60ms hysteresis avoids re-launching on
  // every micro-jitter.
  useAnimatedReaction(
    () => {
      'worklet';
      if (stateKindSv.value !== 'scanning') return null;
      const v = Math.abs(angularVelocitySv.value);
      const denom = v < 0.1 ? 0.1 : v;
      const period = 900 / denom / 10;
      return period < 700 ? 700 : period > 1400 ? 1400 : period;
    },
    (duration, prev) => {
      'worklet';
      if (duration === null) return;
      if (prev !== null && Math.abs(duration - prev) < 60) return;
      const half = duration / 2;
      glowOpacity.value = withRepeat(
        withSequence(
          withTiming(0.85, {
            duration: half,
            easing: Easing.inOut(Easing.sin),
          }),
          withTiming(0.5, {
            duration: half,
            easing: Easing.inOut(Easing.sin),
          }),
        ),
        -1,
        false,
      );
    },
  );

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

  // §4.1 detection-lands phase: rendered opacity climbs 0.6 → 1.0 with
  // live steadiness. The imperative animation tweens `opacity` toward
  // the ceiling (1.0); this derived value caps the rendered value at
  // 0.6 + 0.4 * steadiness while we're still in `aligning + detected`,
  // so the user sees the silhouette firm up as their hand settles
  // instead of resolving from a state-edge snapshot.
  const liveOpacity = useDerivedValue<number>(() => {
    if (state === 'aligning' && detectedSnapshot) {
      const ceiling = 0.6 + 0.4 * steadinessSv.value;
      return Math.min(opacity.value, ceiling);
    }
    return opacity.value;
  }, [state, detectedSnapshot, opacity, steadinessSv]);

  const animatedProps = useAnimatedProps(() => {
    const f = renderFrame.value;
    const x = f.centerX - f.widthPx / 2;
    const y = f.centerY - f.heightPx / 2;
    // Phase 1 embodiment: the outline cinches as the user's grip
    // settles. The base strokeWidth tween still owns transition
    // shaping; this multiplier applies live grip damping on top so
    // the visible cinch tracks the SharedValue per frame.
    const tightness = 0.7 + 0.3 * gripSteadinessSv.value;
    return {
      x,
      y,
      width: f.widthPx,
      height: f.heightPx,
      opacity: liveOpacity.value,
      strokeWidth: strokeWidth.value * tightness,
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
    const tightness = 0.7 + 0.3 * gripSteadinessSv.value;
    return {
      x,
      y,
      width: f.widthPx,
      height: f.heightPx,
      opacity: glowOpacity.value * (spec.glowRadius > 0 ? 1 : 0),
      strokeWidth:
        scanGeometry.silhouetteStrokeWidth * tightness +
        spec.glowRadius * 0.6,
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
