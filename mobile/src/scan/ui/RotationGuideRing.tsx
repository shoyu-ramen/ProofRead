/**
 * RotationGuideRing — circular guide overlay around the detected
 * bottle silhouette (SCAN_DESIGN §3.4 + §4.3).
 *
 * Renders the unfilled track, the gradient-filled coverage arc, the
 * leading-edge dot+chevron, milestone ticks at 25/50/75%, and an
 * idle-state shimmer that sweeps the empty ring while waiting for
 * the user to start rotating. Geometry tracks `silhouetteSv`; arc
 * length tracks `coverageSv` via `withSpring` for sub-perceptual
 * smoothing of the optical-flow estimator's jitter.
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
  withSpring,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';
import Svg, {
  Circle,
  Line,
  Defs,
  LinearGradient,
  Stop,
  G,
} from 'react-native-svg';

import { colors, scanGeometry, scanMotion } from '@src/theme';
import * as Haptics from 'expo-haptics';
import { AccessibilityInfo } from 'react-native';
import type { ScanStateKind } from './ScanInstructions';
import type { SilhouetteFrame } from './BottleSilhouetteOverlay';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);
const AnimatedG = Animated.createAnimatedComponent(G);
const AnimatedLine = Animated.createAnimatedComponent(Line);

const TWO_PI = Math.PI * 2;
const MILESTONES = [0.25, 0.5, 0.75] as const;

export interface RotationGuideRingProps {
  /** 0..1 coverage from the tracker. */
  coverageSv: SharedValue<number>;
  /** Live silhouette frame in screen-px. */
  silhouetteSv: SharedValue<SilhouetteFrame>;
  /** Active scan state — drives color + shimmer activation. */
  state: ScanStateKind;
  /** Camera viewport size — used to size the SVG canvas. */
  viewportWidth: number;
  viewportHeight: number;
  /**
   * Direction of rotation chosen by the tracker. Controls which way
   * the leading-edge chevron points and the shimmer's travel
   * direction. `null` reads as cw for visual consistency.
   */
  rotationDirection?: 'cw' | 'ccw' | null;
}

/**
 * Convert (cx, cy, rx, ry, theta) to a point on an axis-aligned
 * ellipse where theta is measured from "top" (12 o'clock) going
 * clockwise — matches how a user thinks about rotation around a
 * vertical-axis bottle.
 */
function ellipsePoint(
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  theta: number,
): { x: number; y: number } {
  // theta=0 → top of the ellipse, going cw means x grows, y grows
  // (toward bottom-right). Standard math: x = sin(theta), y = -cos.
  const x = cx + rx * Math.sin(theta);
  const y = cy - ry * Math.cos(theta);
  return { x, y };
}

export function RotationGuideRing({
  coverageSv,
  silhouetteSv,
  state,
  viewportWidth,
  viewportHeight,
  rotationDirection = 'cw',
}: RotationGuideRingProps): React.ReactElement {
  // Smooth the raw coverage shared value so the visible arc doesn't
  // jitter with optical-flow noise. `withSpring` tracks the target
  // continuously; we re-arm it whenever `coverageSv` changes.
  const arcFillSv = useSharedValue<number>(0);

  // Mirror coverageSv into arcFillSv with sub-perceptual spring
  // smoothing per §4.3.
  useDerivedValue(() => {
    arcFillSv.value = withSpring(coverageSv.value, scanMotion.spring);
    return 0;
  }, [coverageSv]);

  // Idle shimmer: a 30° arc sweeping the track at 1 rev / 2.4s.
  // Active only during `aligning`/`ready` (i.e. coverage 0).
  const shimmerSv = useSharedValue<number>(0);
  const reduceMotionRef = React.useRef<boolean>(false);

  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled().then((r) => {
      reduceMotionRef.current = r;
      // If reduce-motion is on, suppress the looping shimmer entirely.
      if (r) {
        shimmerSv.value = 0;
        return;
      }
      if (state === 'aligning' || state === 'ready') {
        shimmerSv.value = 0;
        shimmerSv.value = withRepeat(
          withTiming(1, { duration: 2400, easing: Easing.linear }),
          -1,
          false,
        );
      } else {
        shimmerSv.value = 0;
      }
    }).catch(() => {});
  }, [state, shimmerSv]);

  // Milestone-flash shared values, one per tick. We start them all at
  // 0 (track color) and pulse to 1 (green) when the leading edge
  // crosses, settling back to 0 (which we re-interpret as
  // coverageFillEnd in the render path so passed ticks read as
  // blue-on-blue). Hooks are declared one-per-milestone instead of
  // mapped because react-hooks lint requires fixed call order.
  const tickFlash25 = useSharedValue<number>(0);
  const tickFlash50 = useSharedValue<number>(0);
  const tickFlash75 = useSharedValue<number>(0);
  const tickPassed25 = useSharedValue<number>(0);
  const tickPassed50 = useSharedValue<number>(0);
  const tickPassed75 = useSharedValue<number>(0);
  const tickFlash = [tickFlash25, tickFlash50, tickFlash75];
  const tickPassed = [tickPassed25, tickPassed50, tickPassed75];
  const lastCoverageRef = React.useRef<number>(0);
  useEffect(() => {
    // Subscribe via a derived value to detect milestone crossings.
    const id = setInterval(() => {
      const c = coverageSv.value;
      const prev = lastCoverageRef.current;
      lastCoverageRef.current = c;
      MILESTONES.forEach((m, i) => {
        if (prev < m && c >= m) {
          // Flash + haptic.
          const fSv = tickFlash[i];
          const pSv = tickPassed[i];
          if (fSv) {
            fSv.value = withSequence(
              withTiming(1, { duration: 240, easing: Easing.out(Easing.cubic) }),
              withTiming(0, { duration: 240, easing: Easing.in(Easing.cubic) }),
            );
          }
          if (pSv) {
            pSv.value = withTiming(1, { duration: 480 });
          }
          void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(
            () => {},
          );
        }
      });
    }, 50);
    return () => clearInterval(id);
  }, [coverageSv, tickFlash, tickPassed]);

  // Geometry derived per-frame from the silhouette + ring offsets.
  const geom = useDerivedValue(() => {
    const s = silhouetteSv.value;
    const rx =
      s.widthPx / 2 +
      scanGeometry.ringGapFromSilhouette +
      scanGeometry.ringStrokeWidth / 2;
    const ry =
      s.heightPx / 2 +
      scanGeometry.ringGapFromSilhouette +
      scanGeometry.ringStrokeWidth / 2;
    return {
      cx: s.centerX,
      cy: s.centerY,
      rx,
      ry,
    };
  }, [silhouetteSv]);

  // Visual circumference for stroke-dasharray.  Since rx/ry vary, we
  // approximate with Ramanujan's formula and re-derive on each animation
  // tick. We render the arc with a circle of radius `rRender` (a single
  // average so the dasharray is well-defined); the asymmetry between
  // the silhouette's width and height is small enough that an averaged
  // circle reads as the intended ellipse.
  const rAvgSv = useDerivedValue(() => {
    const g = geom.value;
    return (g.rx + g.ry) / 2;
  });

  // Stroke-dasharray + stroke-dashoffset for the filled arc.
  const arcAnimatedProps = useAnimatedProps(() => {
    const r = rAvgSv.value;
    const circ = TWO_PI * r;
    const filled = circ * arcFillSv.value;
    return {
      strokeDasharray: [filled, Math.max(0.0001, circ - filled)],
      strokeDashoffset: 0,
      cx: geom.value.cx,
      cy: geom.value.cy,
      r,
    } as Partial<{
      strokeDasharray: number[];
      strokeDashoffset: number;
      cx: number;
      cy: number;
      r: number;
    }>;
  });

  // Track (the unfilled portion).
  const trackAnimatedProps = useAnimatedProps(() => {
    return {
      cx: geom.value.cx,
      cy: geom.value.cy,
      r: rAvgSv.value,
    } as Partial<{ cx: number; cy: number; r: number }>;
  });

  // Leading-edge dot + halo. We animate `transform: rotate(theta)` on a
  // group around the circle's center, with the dot pre-positioned at
  // 12 o'clock relative to the group origin.
  const leadingTransform = useAnimatedProps(() => {
    const theta = arcFillSv.value * 360;
    return {
      // SVG rotate(angle, cx, cy) – rotate around the silhouette center.
      transform: `rotate(${theta} ${geom.value.cx} ${geom.value.cy})`,
    } as { transform: string };
  });

  // Where to draw the dot (in the un-rotated coordinate frame). It sits
  // at the top of the ring (theta=0). The G's rotate animation moves it.
  const dotPositionProps = useAnimatedProps(() => {
    return {
      cx: geom.value.cx,
      cy: geom.value.cy - rAvgSv.value,
    } as Partial<{ cx: number; cy: number }>;
  });

  // Idle shimmer arc — a 30° wide segment sweeping the track. We
  // animate the dasharray's offset to slide a fixed-width dash around
  // the circumference.
  const shimmerProps = useAnimatedProps(() => {
    const r = rAvgSv.value;
    const circ = TWO_PI * r;
    const dashLen = circ * (30 / 360);
    const dir = rotationDirection === 'ccw' ? -1 : 1;
    return {
      cx: geom.value.cx,
      cy: geom.value.cy,
      r,
      strokeDasharray: [dashLen, Math.max(0.0001, circ - dashLen)],
      strokeDashoffset: dir * shimmerSv.value * circ,
      opacity:
        state === 'aligning' || state === 'ready' ? 0.3 : 0,
    } as Partial<{
      cx: number;
      cy: number;
      r: number;
      strokeDasharray: number[];
      strokeDashoffset: number;
      opacity: number;
    }>;
  });

  // Milestone tick props — three identical hook calls in fixed order
  // satisfy react-hooks rules and let each tick own its own
  // useAnimatedProps closure.
  const tick25Props = useMilestoneTickProps(0, geom, rAvgSv, tickFlash25, tickPassed25);
  const tick50Props = useMilestoneTickProps(1, geom, rAvgSv, tickFlash50, tickPassed50);
  const tick75Props = useMilestoneTickProps(2, geom, rAvgSv, tickFlash75, tickPassed75);

  return (
    <View
      pointerEvents="none"
      style={StyleSheet.absoluteFill}
      accessibilityRole="progressbar"
      accessibilityLabel="Rotation coverage"
    >
      <Svg width={viewportWidth} height={viewportHeight}>
        <Defs>
          <LinearGradient id="ring-fill" x1="0%" y1="0%" x2="100%" y2="0%">
            <Stop offset="0%" stopColor={colors.coverageFillStart} />
            <Stop offset="100%" stopColor={colors.coverageFillEnd} />
          </LinearGradient>
        </Defs>

        {/* Track */}
        <AnimatedCircle
          fill="none"
          stroke={colors.coverageTrack}
          strokeWidth={scanGeometry.ringStrokeWidth}
          strokeLinecap="round"
          animatedProps={trackAnimatedProps}
        />

        {/* Idle shimmer */}
        <AnimatedCircle
          fill="none"
          stroke={colors.coverageFillStart}
          strokeWidth={scanGeometry.ringStrokeWidth}
          strokeLinecap="round"
          animatedProps={shimmerProps}
        />

        {/* Filled arc — the static rotation orients dasharray's natural
            "3 o'clock start" to 12 o'clock so coverage grows from top. */}
        <AnimatedG
          rotation={rotationDirection === 'ccw' ? 90 : -90}
          originX={0}
          originY={0}
        >
          <AnimatedCircle
            fill="none"
            stroke={colors.coverageFillEnd}
            strokeWidth={scanGeometry.ringStrokeWidth}
            strokeLinecap="round"
            animatedProps={arcAnimatedProps}
          />
        </AnimatedG>

        {/* Milestone ticks */}
        <AnimatedLine
          stroke={colors.coverageTrack}
          strokeWidth={2}
          strokeLinecap="round"
          animatedProps={tick25Props}
        />
        <AnimatedLine
          stroke={colors.coverageTrack}
          strokeWidth={2}
          strokeLinecap="round"
          animatedProps={tick50Props}
        />
        <AnimatedLine
          stroke={colors.coverageTrack}
          strokeWidth={2}
          strokeLinecap="round"
          animatedProps={tick75Props}
        />

        {/* Leading-edge halo + dot */}
        <AnimatedG
          animatedProps={leadingTransform}
        >
          <AnimatedCircle
            fill={colors.coverageLeadingHalo}
            r={scanGeometry.ringLeadingHaloRadius}
            animatedProps={dotPositionProps}
          />
          <AnimatedCircle
            fill={colors.coverageLeadingDot}
            r={scanGeometry.ringLeadingDotRadius}
            animatedProps={dotPositionProps}
          />
        </AnimatedG>
      </Svg>
    </View>
  );
}

/**
 * Milestone tick at coverage `MILESTONES[i]`. The tick line goes from
 * the inner edge of the ring stroke `length` px inward toward the
 * silhouette center. Color flashes green during `flashSv`'s 1.0 peak,
 * then settles to `coverageFillEnd` once `passedSv` saturates.
 */
function useMilestoneTickProps(
  i: number,
  geom: SharedValue<{ cx: number; cy: number; rx: number; ry: number }>,
  rAvgSv: SharedValue<number>,
  flashSv: SharedValue<number>,
  passedSv: SharedValue<number>,
) {
  return useAnimatedProps(() => {
    const milestone = MILESTONES[i] ?? 0;
    const theta = TWO_PI * milestone;
    const r = rAvgSv.value;
    const inner = r - scanGeometry.ringStrokeWidth / 2;
    const tipR = inner - scanGeometry.ringMilestoneTickLength;
    const g = geom.value;
    const p1 = ellipsePoint(g.cx, g.cy, inner, inner, theta);
    const p2 = ellipsePoint(g.cx, g.cy, tipR, tipR, theta);
    // Color: flash > passed > track (snap, not interpolated; see
    // index.ts deviation #3).
    return {
      x1: p1.x,
      y1: p1.y,
      x2: p2.x,
      y2: p2.y,
      stroke:
        flashSv.value > 0.5
          ? colors.coverageMilestone
          : passedSv.value > 0.5
            ? colors.coverageFillEnd
            : colors.coverageTrack,
      strokeWidth: 2 + flashSv.value * 1.4,
    } as Partial<{
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      stroke: string;
      strokeWidth: number;
    }>;
  });
}
