/**
 * CompletionReveal — the magic-moment overlay that runs when coverage
 * reaches 1.0.
 *
 * Composes the §8 storyboard stages (pre-reveal lock, sparkle pass,
 * scrim ramp, hero copy, hand-off) on a compressed master clock so the
 * reveal feels celebratory without being a paywall: navigation fires at
 * 700ms regardless of whether longer animations are still resolving.
 * Tap-anywhere short-circuits the timer entirely. The strip lift itself
 * is owned by the panorama agent's PanoramaCanvas.
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  AccessibilityInfo,
  Dimensions,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withTiming,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';

import { colors } from '@src/theme';

const SPARKLE_TAIL_COUNT = 8;
// Master clock — navigation fires at this point even if some animations
// (e.g. the sparkle tail settle) are still resolving in the background.
const REVEAL_HANDOFF_MS = 700;

export interface CompletionRevealProps {
  /** True while the parent state is `complete`. Triggers the reveal. */
  active: boolean;
  /**
   * Where the panorama strip is currently rendered, in screen-px.
   * Used to drive the sparkle's left-to-right path. The strip lift
   * itself is the panorama agent's responsibility.
   */
  panoramaFrame: { x: number; y: number; width: number; height: number };
  /**
   * Fired at `t = REVEAL_HANDOFF_MS` (or immediately on tap) to signal
   * the parent screen it should navigate to /(app)/scan/review.
   */
  onComplete: () => void;
}

export function CompletionReveal({
  active,
  panoramaFrame,
  onComplete,
}: CompletionRevealProps): React.ReactElement | null {
  const screen = Dimensions.get('window');
  const [reduceMotion, setReduceMotion] = useState<boolean>(false);

  // Sparkle head X, in 0..1 of the panorama width.
  const sparkleProgress = useSharedValue<number>(0);
  // Sparkle head visibility.
  const sparkleOpacity = useSharedValue<number>(0);
  // Hero copy + caption opacity / translateY.
  const heroOpacity = useSharedValue<number>(0);
  const heroTranslateY = useSharedValue<number>(12);
  const captionOpacity = useSharedValue<number>(0);
  // Camera dim scrim opacity (ramps up so the strip stands out).
  const scrimOpacity = useSharedValue<number>(0);

  const completeFiredRef = useRef<boolean>(false);

  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled()
      .then(setReduceMotion)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!active) {
      completeFiredRef.current = false;
      sparkleOpacity.value = 0;
      sparkleProgress.value = 0;
      heroOpacity.value = 0;
      heroTranslateY.value = 12;
      captionOpacity.value = 0;
      scrimOpacity.value = 0;
      return;
    }

    // Fire success haptic at t = 0 (§7).
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(
      () => {},
    );

    if (reduceMotion) {
      // Reduced-motion path: skip sparkle + lift, crossfade in <120ms.
      heroOpacity.value = withTiming(1, { duration: 120 });
      captionOpacity.value = withDelay(120, withTiming(1, { duration: 120 }));
      const t = setTimeout(() => {
        if (!completeFiredRef.current) {
          completeFiredRef.current = true;
          onComplete();
        }
      }, 600);
      return () => clearTimeout(t);
    }

    // Stage 2 — Sparkle pass: t = 200..760ms (animation), 560ms head
    // sweep + 200ms tail settle.
    sparkleOpacity.value = withDelay(200, withTiming(1, { duration: 80 }));
    sparkleProgress.value = withDelay(
      200,
      withTiming(1, {
        duration: 560,
        easing: Easing.inOut(Easing.cubic),
      }),
    );
    // Fade the sparkle out as the tail settles.
    sparkleOpacity.value = withDelay(
      760,
      withTiming(0, { duration: 140, easing: Easing.in(Easing.cubic) }),
    );

    // Stage 3 — strip-lift companion: dim scrim ramps up t=320..900ms.
    scrimOpacity.value = withDelay(
      320,
      withTiming(0.8, { duration: 580, easing: Easing.inOut(Easing.cubic) }),
    );

    // Stage 4 — Hero copy: opacity in t=580..860ms, caption t=800..1040ms.
    heroOpacity.value = withDelay(
      580,
      withTiming(1, { duration: 280, easing: Easing.out(Easing.cubic) }),
    );
    heroTranslateY.value = withDelay(
      580,
      withTiming(0, { duration: 280, easing: Easing.out(Easing.cubic) }),
    );
    captionOpacity.value = withDelay(
      800,
      withTiming(1, { duration: 240, easing: Easing.out(Easing.cubic) }),
    );

    // Stage 5 — Hand-off at t = REVEAL_HANDOFF_MS. Some of the
    // animations above run longer than this; that's fine, they keep
    // resolving as the next screen mounts.
    const tHandoff = setTimeout(() => {
      if (!completeFiredRef.current) {
        completeFiredRef.current = true;
        onComplete();
      }
    }, REVEAL_HANDOFF_MS);
    return () => clearTimeout(tHandoff);
  }, [
    active,
    reduceMotion,
    onComplete,
    sparkleProgress,
    sparkleOpacity,
    heroOpacity,
    heroTranslateY,
    captionOpacity,
    scrimOpacity,
  ]);

  // Sparkle head + halo position, derived from sparkleProgress and the
  // panorama frame.
  const sparkleHeadStyle = useAnimatedStyle(() => {
    const t = sparkleProgress.value;
    const x = panoramaFrame.x + 12 + t * (panoramaFrame.width - 24);
    const y = panoramaFrame.y + panoramaFrame.height / 2;
    return {
      transform: [{ translateX: x - 14 }, { translateY: y - 14 }],
      opacity: sparkleOpacity.value,
    };
  });

  // Build the tail particles. Each tail particle lags the head by an
  // index-based delay; we render them as siblings driven by the same
  // sparkleProgress shared value.
  const tailParticles = Array.from({ length: SPARKLE_TAIL_COUNT }, (_, i) => i);

  const heroStyle = useAnimatedStyle(() => ({
    opacity: heroOpacity.value,
    transform: [{ translateY: heroTranslateY.value }],
  }));
  const captionStyle = useAnimatedStyle(() => ({
    opacity: captionOpacity.value,
  }));
  const scrimStyle = useAnimatedStyle(() => ({
    opacity: scrimOpacity.value,
  }));

  const handleSkip = () => {
    if (completeFiredRef.current) return;
    completeFiredRef.current = true;
    onComplete();
  };

  if (!active) return null;

  return (
    <Pressable
      onPress={handleSkip}
      accessibilityRole="alert"
      accessibilityLabel="Scan complete. Tap to continue."
      accessibilityHint="Skips the reveal animation"
      style={StyleSheet.absoluteFill}
    >
      {/* Stage 3 dim scrim — ramps the camera underneath toward black. */}
      <Animated.View
        style={[
          StyleSheet.absoluteFill,
          { backgroundColor: colors.scanOverlayDim },
          scrimStyle,
        ]}
      />

      {/* Stage 2 sparkle layer */}
      <View
        pointerEvents="none"
        style={StyleSheet.absoluteFill}
      >
        {/* Tail particles, drawn under the head. */}
        {tailParticles.map((i) => (
          <SparkleTailParticle
            key={i}
            index={i}
            sparkleProgress={sparkleProgress}
            sparkleOpacity={sparkleOpacity}
            panoramaFrame={panoramaFrame}
          />
        ))}
        {/* Sparkle head + halo. */}
        <Animated.View style={[styles.sparkleHead, sparkleHeadStyle]}>
          <View style={styles.sparkleHalo} />
          <View style={styles.sparkleCore} />
        </Animated.View>
      </View>

      {/* Stage 4 hero copy + caption — centered below the panorama. */}
      <View
        pointerEvents="none"
        style={[
          styles.heroWrap,
          {
            top:
              panoramaFrame.y +
              Math.max(panoramaFrame.height, screen.height * 0.32) +
              28,
          },
        ]}
      >
        <Animated.Text style={[styles.hero, heroStyle]}>
          Label captured{' '}
          <Animated.Text style={{ color: colors.scanReady, fontWeight: '700' }}>
            ✓
          </Animated.Text>
        </Animated.Text>
        <Animated.Text style={[styles.caption, captionStyle]}>
          Checking compliance…
        </Animated.Text>
      </View>
    </Pressable>
  );
}

interface SparkleTailParticleProps {
  index: number;
  sparkleProgress: Animated.SharedValue<number>;
  sparkleOpacity: Animated.SharedValue<number>;
  panoramaFrame: { x: number; y: number; width: number; height: number };
}

function SparkleTailParticle({
  index,
  sparkleProgress,
  sparkleOpacity,
  panoramaFrame,
}: SparkleTailParticleProps): React.ReactElement {
  // Each tail particle is offset behind the head by a fraction of the
  // total path. 4px of jitter on Y deterministic per-index so we don't
  // cause a re-render flicker on remount.
  const lag = (index + 1) / (SPARKLE_TAIL_COUNT + 2);
  const yJitter = ((index * 37) % 9) - 4;
  const size = 2 + ((index * 5) % 3);

  const style = useAnimatedStyle(() => {
    const t = Math.max(0, sparkleProgress.value - lag * 0.18);
    const x = panoramaFrame.x + 12 + t * (panoramaFrame.width - 24);
    const y = panoramaFrame.y + panoramaFrame.height / 2 + yJitter;
    return {
      transform: [{ translateX: x - size / 2 }, { translateY: y - size / 2 }],
      // Tail fades out behind the head.
      opacity:
        sparkleOpacity.value *
        Math.max(0, 1 - lag * 1.4) *
        (sparkleProgress.value > lag * 0.18 ? 1 : 0),
      width: size,
      height: size,
      borderRadius: size / 2,
    };
  });

  return <Animated.View style={[styles.tailParticle, style]} />;
}

const styles = StyleSheet.create({
  sparkleHead: {
    position: 'absolute',
    width: 28,
    height: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sparkleHalo: {
    position: 'absolute',
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.sparkleAccent,
    opacity: 0.6,
  },
  sparkleCore: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.sparkleCore,
  },
  tailParticle: {
    position: 'absolute',
    backgroundColor: colors.sparkleAccent,
  },
  heroWrap: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  hero: {
    fontSize: 22,
    fontWeight: '700',
    letterSpacing: -0.5,
    lineHeight: 26,
    color: colors.scanInk,
  },
  caption: {
    marginTop: 6,
    fontSize: 13,
    fontWeight: '500',
    letterSpacing: 0.2,
    lineHeight: 18,
    color: colors.scanInkDim,
  },
});
