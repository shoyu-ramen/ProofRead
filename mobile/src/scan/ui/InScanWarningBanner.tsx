/**
 * InScanWarningBanner — non-blocking soft warning surfaced during the
 * cylindrical scan when capture coverage suggests the back label
 * (where the §16.21 Government Warning lives) might not have been
 * rotated through yet.
 *
 * Heuristic-only — this component reads existing scan-state signals and
 * never mutates them. Specifically: when the user has covered between
 * 240° and 300° of the bottle (i.e. coverage ∈ [0.667, 0.833] in the
 * tracker's 0..1-revolutions scale), we display a one-line nudge that
 * fades out the moment they cross 300°. The user is never blocked; if
 * they choose to stop here we still ship them to review (the rule
 * engine will surface the missing-warning verdict downstream).
 *
 * Visual language matches the existing pause / quality banner pattern
 * (capsule pill, Reanimated slide-in, scrim background) — `QualityChip`
 * is the closest sibling. The banner is positioned just below the
 * panorama strip rather than at the top-right corner so it lives in
 * the user's natural reading band without colliding with the chip.
 */

import React, { useEffect, useRef } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';

import { colors, scanGeometry, scanMotion } from '@src/theme';

/**
 * Coverage band where the warning is active. The scan tracker measures
 * coverage in revolutions (0..1 ≡ 0..360°), so the spec's "≥240°" and
 * "<300°" map to these constants.
 */
export const IN_SCAN_WARNING_LOWER = 240 / 360; // 0.6667
export const IN_SCAN_WARNING_UPPER = 300 / 360; // 0.8333

export interface InScanWarningBannerProps {
  /**
   * Current scan state kind. The banner is only ever visible during
   * `scanning` — pause / complete / failed states surface their own
   * chrome and we don't want to fight them.
   */
  state: 'aligning' | 'ready' | 'scanning' | 'paused' | 'complete' | 'failed';
  /** 0..1 coverage from the tracker. */
  coverage: number;
  /**
   * Top offset for the banner. Callers should pass
   * `panoramaFrame.y + panoramaFrame.height + 12` so the banner lands
   * just below the live strip without overlapping it.
   */
  topPx: number;
  /**
   * Override visibility — primarily for testing. When true, ignores
   * the coverage heuristic and shows the banner; when false, ignores
   * the heuristic and hides it. `undefined` (default) → use heuristic.
   */
  forceVisible?: boolean;
}

/**
 * Pure predicate exported for tests and the parent unwrap screen.
 * Returns true when the banner should be visible according to the
 * heuristic alone (state is `scanning` and coverage is in the late-
 * stretch band). Test cases drive this directly.
 */
export function shouldShowInScanWarning(
  state: InScanWarningBannerProps['state'],
  coverage: number,
): boolean {
  if (state !== 'scanning') return false;
  return coverage >= IN_SCAN_WARNING_LOWER && coverage < IN_SCAN_WARNING_UPPER;
}

export function InScanWarningBanner({
  state,
  coverage,
  topPx,
  forceVisible,
}: InScanWarningBannerProps): React.ReactElement {
  const visible =
    typeof forceVisible === 'boolean'
      ? forceVisible
      : shouldShowInScanWarning(state, coverage);

  // Reanimated state — slide in from -8px above the resting Y plus a
  // fade. Mirrors the QualityChip motion language but vertical instead
  // of horizontal because we sit center-screen rather than at a corner.
  const translateY = useSharedValue<number>(visible ? 0 : -8);
  const opacity = useSharedValue<number>(visible ? 1 : 0);
  const visibleRef = useRef<boolean>(visible);

  useEffect(() => {
    if (visible && !visibleRef.current) {
      translateY.value = withTiming(0, {
        duration: scanMotion.midEase.duration,
        easing: Easing.out(Easing.cubic),
      });
      opacity.value = withTiming(1, {
        duration: scanMotion.midEase.duration,
        easing: Easing.out(Easing.cubic),
      });
    } else if (!visible && visibleRef.current) {
      translateY.value = withTiming(-8, {
        duration: scanMotion.fastEase.duration,
        easing: Easing.in(Easing.cubic),
      });
      opacity.value = withTiming(0, {
        duration: scanMotion.fastEase.duration,
        easing: Easing.in(Easing.cubic),
      });
    }
    visibleRef.current = visible;
  }, [visible, translateY, opacity]);

  const animStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
    opacity: opacity.value,
  }));

  return (
    <Animated.View
      pointerEvents="none"
      accessibilityRole="text"
      accessibilityLabel="Rotate further — the back label has the health warning."
      accessibilityLiveRegion="polite"
      style={[styles.wrap, { top: topPx }, animStyle]}
    >
      <View style={styles.pill}>
        <View
          style={styles.iconWrap}
          accessibilityElementsHidden
          importantForAccessibility="no"
        >
          <Text style={styles.iconGlyph}>!</Text>
        </View>
        <Text style={styles.text} numberOfLines={2}>
          Rotate further — the back label has the health warning.
        </Text>
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    left: scanGeometry.panoramaPaddingHorizontal,
    right: scanGeometry.panoramaPaddingHorizontal,
    alignItems: 'center',
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 999,
    backgroundColor: colors.scanOverlayScrim,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.panoramaFrameStroke,
    // Narrow accent stripe on the leading edge calls out the warning
    // class without changing the chip's overall identity.
    borderLeftWidth: 3,
    borderLeftColor: colors.scanWarn,
    maxWidth: '100%',
  },
  iconWrap: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.scanWarn,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.scanWarnSoft,
  },
  iconGlyph: {
    fontSize: 12,
    fontWeight: '700',
    lineHeight: 14,
    color: colors.scanWarn,
  },
  text: {
    flex: 1,
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.2,
    lineHeight: 18,
    color: colors.scanInk,
  },
});
