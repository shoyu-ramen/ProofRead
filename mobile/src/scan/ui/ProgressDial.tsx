/**
 * ProgressDial — bottom-right radial dial showing the live 0–360°
 * coverage as a percentage (SCAN_DESIGN §3.5 + §4 milestone pulses).
 *
 * 64px outer diameter, gradient-filled arc, two-line center readout
 * ("47%" + "OF 360°"). Pulses on every quarter-turn milestone with a
 * scale 1.0→1.10→1.0 over 320ms plus a 24px feathered glow halo. The
 * dial keeps the numerical state legible against arbitrary camera
 * backdrops by sitting on a circular `scanOverlayDim` plate.
 */

import React, { useEffect, useRef } from 'react';
import { AccessibilityInfo, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Animated, {
  Easing,
  useAnimatedProps,
  useAnimatedStyle,
  useDerivedValue,
  useSharedValue,
  withRepeat,
  withSequence,
  withSpring,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';
import Svg, {
  Circle,
  Defs,
  LinearGradient,
  Stop,
} from 'react-native-svg';

import { colors, scanGeometry, scanMotion } from '@src/theme';
import type { ScanStateKind } from './ScanInstructions';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);
const AnimatedText = Animated.createAnimatedComponent(Text);

const TWO_PI = Math.PI * 2;
const MILESTONES = [0.25, 0.5, 0.75, 1.0] as const;

export interface ProgressDialProps {
  /** Live 0..1 coverage from the tracker. */
  coverageSv: SharedValue<number>;
  /** Active scan state — drives the "complete" check-mark swap. */
  state: ScanStateKind;
}

export function ProgressDial({
  coverageSv,
  state,
}: ProgressDialProps): React.ReactElement {
  const insets = useSafeAreaInsets();

  // Spring-smoothed visible coverage (re-uses §4.3 smoothing semantics).
  const arcSv = useSharedValue<number>(0);
  useDerivedValue(() => {
    arcSv.value = withSpring(coverageSv.value, scanMotion.spring);
    return 0;
  }, [coverageSv]);

  // Scale + glow shared values for the milestone pulse.
  const scale = useSharedValue<number>(1);
  const glowOpacity = useSharedValue<number>(0);

  const lastCovRef = useRef<number>(0);
  useEffect(() => {
    const id = setInterval(() => {
      const c = coverageSv.value;
      const prev = lastCovRef.current;
      lastCovRef.current = c;
      const crossed = MILESTONES.some((m) => prev < m && c >= m);
      if (!crossed) return;
      scale.value = withSequence(
        withTiming(1.1, {
          duration: 160,
          easing: Easing.out(Easing.cubic),
        }),
        withTiming(1, {
          duration: 160,
          easing: Easing.in(Easing.cubic),
        }),
      );
      glowOpacity.value = withSequence(
        withTiming(0.85, {
          duration: 160,
          easing: Easing.out(Easing.cubic),
        }),
        withTiming(0, {
          duration: 240,
          easing: Easing.in(Easing.cubic),
        }),
      );
    }, 50);
    return () => clearInterval(id);
  }, [coverageSv, scale, glowOpacity]);

  const containerStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const glowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
  }));

  // Geometry of the outer ring inside the 64×64 box.
  const D = scanGeometry.dialDiameter;
  const STROKE = scanGeometry.dialStrokeWidth;
  const r = (D - STROKE) / 2;
  const cx = D / 2;
  const cy = D / 2;
  const circ = TWO_PI * r;

  const arcAnimatedProps = useAnimatedProps(() => {
    const filled = circ * arcSv.value;
    return {
      strokeDasharray: [filled, Math.max(0.0001, circ - filled)],
      strokeDashoffset: 0,
    } as Partial<{ strokeDasharray: number[]; strokeDashoffset: number }>;
  });

  // Big-number text — re-derive the displayed integer from arcSv. We
  // can't use useAnimatedProps on a Text's children directly in
  // react-native-reanimated 3.10 without runtime support; cleanest
  // approach is to drive a JS state via a derived listener firing on
  // value changes (throttled).
  const [pct, setPct] = React.useState<number>(0);
  useEffect(() => {
    const id = setInterval(() => {
      // Read the visible (smoothed) value, not the raw coverage. We
      // floor + clamp to 100 so the readout reads cleanly during the
      // 100% lock.
      const vis = Math.max(0, Math.min(1, arcSv.value));
      const next = Math.floor(vis * 100);
      setPct((prev) => (prev === next ? prev : next));
    }, 80);
    return () => clearInterval(id);
  }, [arcSv]);

  const showCheck = state === 'complete';

  // Idle "spin slowly" affordance: when the dial reads 0% (no
  // measurable rotation yet), swap the bare percentage for a curved
  // unicode arrow + caption that telegraphs the action. Once the user
  // starts rotating the percentage takes over.
  const idleAffordance = !showCheck && pct === 0;
  const arrowSpin = useSharedValue<number>(0);
  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled().then((reduced) => {
      if (reduced) {
        arrowSpin.value = 0;
        return;
      }
      if (idleAffordance) {
        arrowSpin.value = 0;
        arrowSpin.value = withRepeat(
          withTiming(1, { duration: 2400, easing: Easing.linear }),
          -1,
          false,
        );
      } else {
        arrowSpin.value = 0;
      }
    }).catch(() => {});
  }, [idleAffordance, arrowSpin]);

  const arrowStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${arrowSpin.value * 360}deg` }],
  }));

  return (
    <View
      pointerEvents="none"
      style={[styles.wrap, { bottom: insets.bottom + scanGeometry.dialCornerInset }]}
      accessibilityRole="progressbar"
      accessibilityLabel={`Scan progress, ${pct} percent`}
    >
      <Animated.View style={[styles.glow, glowStyle]} />
      <Animated.View style={[styles.container, containerStyle]}>
        <View style={styles.plate} />
        <Svg width={D} height={D}>
          <Defs>
            <LinearGradient id="dial-fill" x1="0%" y1="0%" x2="100%" y2="0%">
              <Stop offset="0%" stopColor={colors.coverageFillStart} />
              <Stop offset="100%" stopColor={colors.coverageFillEnd} />
            </LinearGradient>
          </Defs>
          {/* Track */}
          <Circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={colors.coverageTrack}
            strokeWidth={STROKE}
            strokeLinecap="round"
          />
          {/* Filled arc — start at 12 o'clock, grow clockwise. */}
          <AnimatedCircle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke="url(#dial-fill)"
            strokeWidth={STROKE}
            strokeLinecap="round"
            transform={`rotate(-90 ${cx} ${cy})`}
            animatedProps={arcAnimatedProps}
          />
        </Svg>
        <View style={styles.center}>
          {showCheck ? (
            <Text
              style={[styles.bigNumber, { color: colors.scanReady }]}
              accessibilityElementsHidden
            >
              ✓
            </Text>
          ) : idleAffordance ? (
            <>
              <Animated.Text
                style={[styles.bigNumber, styles.arrow, arrowStyle]}
                accessibilityElementsHidden
              >
                ↻
              </Animated.Text>
              <Text style={styles.caption}>SPIN SLOWLY</Text>
            </>
          ) : (
            <>
              <AnimatedText style={styles.bigNumber}>{`${pct}%`}</AnimatedText>
              <Text style={styles.caption}>OF 360°</Text>
            </>
          )}
        </View>
      </Animated.View>
    </View>
  );
}

const D = scanGeometry.dialDiameter;

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    right: scanGeometry.dialCornerInset,
    width: D,
    height: D,
    alignItems: 'center',
    justifyContent: 'center',
  },
  container: {
    width: D,
    height: D,
    alignItems: 'center',
    justifyContent: 'center',
  },
  plate: {
    position: 'absolute',
    width: D,
    height: D,
    borderRadius: D / 2,
    backgroundColor: colors.scanOverlayDim,
  },
  glow: {
    position: 'absolute',
    width: D + 24,
    height: D + 24,
    borderRadius: (D + 24) / 2,
    backgroundColor: colors.scanIdleGlow,
  },
  center: {
    position: 'absolute',
    alignItems: 'center',
    justifyContent: 'center',
  },
  bigNumber: {
    fontSize: 20,
    fontWeight: '700',
    letterSpacing: -0.4,
    lineHeight: 22,
    color: colors.scanInk,
  },
  arrow: {
    fontSize: 22,
    lineHeight: 24,
    fontWeight: '600',
  },
  caption: {
    fontSize: 9,
    fontWeight: '500',
    letterSpacing: 0.6,
    lineHeight: 11,
    color: colors.scanInkFaint,
    marginTop: 1,
  },
});
