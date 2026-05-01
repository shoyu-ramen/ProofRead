/**
 * Skeleton — shimmer placeholder for loading content.
 *
 * Replaces the previous "ActivityIndicator + 'Loading…'" pattern that
 * the home / history / report screens used. A skeleton conveys both
 * "data is on its way" AND the rough shape of what's coming, which
 * makes the wait feel shorter and prevents a layout pop when the real
 * content lands.
 *
 * Implementation: a single Reanimated `useSharedValue` driving an
 * opacity loop (~1.4s, ease in-out) on a translucent surface tile. We
 * intentionally keep this lightweight — no gradient sweep with masks,
 * just opacity pulsing — so dozens of skeleton rows can render at once
 * without thrashing the UI thread.
 *
 * Usage:
 *   <Skeleton width="100%" height={56} radius={radius.md} />
 *   <Skeleton width={48} height={48} radius={24} />     // circle
 */
import React, { useEffect } from 'react';
import { StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';

import { colors } from '@src/theme';

export interface SkeletonProps {
  /** Required width — number for fixed px, string for percentage. */
  width: number | `${number}%`;
  /** Required height — number for fixed px, string for percentage. */
  height: number | `${number}%`;
  /** Optional border-radius. Defaults to 6 for a soft tile look. */
  radius?: number;
  /** Optional override style — composed on top of the base block style. */
  style?: ViewStyle;
  /** Override testID for unit tests. */
  testID?: string;
}

/**
 * Loop duration in ms for the pulse cycle. Tuned to ~1.4s — slow
 * enough to read as "ambient motion" rather than a busy spinner, but
 * fast enough that the user gets clear "still loading" feedback.
 */
const PULSE_MS = 1400;

export function Skeleton({
  width,
  height,
  radius = 6,
  style,
  testID,
}: SkeletonProps): React.ReactElement {
  // Drive opacity 0.45 → 1 → 0.45 in a smooth loop. The base block is
  // already a low-contrast translucent fill (`surfaceAlt` over the
  // app's dark background) so the multiplied opacity reads as a soft
  // pulse rather than a strobe.
  const pulse = useSharedValue<number>(0.6);
  useEffect(() => {
    pulse.value = withRepeat(
      withTiming(1, {
        duration: PULSE_MS / 2,
        easing: Easing.inOut(Easing.ease),
      }),
      -1,
      true,
    );
  }, [pulse]);

  const animStyle = useAnimatedStyle(() => ({
    opacity: pulse.value,
  }));

  return (
    <Animated.View
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
      testID={testID}
      style={[
        styles.block,
        { width, height, borderRadius: radius },
        style,
        animStyle,
      ]}
    />
  );
}

const styles = StyleSheet.create({
  block: {
    backgroundColor: colors.surfaceAlt,
  },
});
