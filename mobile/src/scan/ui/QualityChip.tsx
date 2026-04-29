/**
 * QualityChip — corner-mounted pre-check indicator (SCAN_DESIGN §3.7).
 *
 * Lifts the visual language of the original `PreCheckIndicator` from
 * scan/camera/[surface].tsx (state-mapped color interpolation, scale
 * pulse on verdict change, spin-while-unknown) and refits it for the
 * cylindrical scan corner. New behavior versus the original: the chip
 * is hidden during clean active scanning and slides in from the right
 * edge only when there's an actionable warning.
 */

import React, { useEffect, useRef } from 'react';
import { StyleSheet, Text } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
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

import { colors } from '@src/theme';
import type { PreCheckVerdict } from '@src/scan/tracker/types';
import type { ScanStateKind, PauseReason } from './ScanInstructions';

const KIND_CODE = { unknown: 0, warn: 1, ready: 2 } as const;

const CHIP_COLORS = {
  unknown: 'rgba(0,0,0,0.55)',
  warn: colors.scanWarn,
  ready: colors.scanReady,
} as const;

export interface QualityChipProps {
  /** Active scan state — drives visibility (hidden during scanning). */
  state: ScanStateKind;
  /** Live pre-check verdict from the tracker. */
  preCheck: PreCheckVerdict;
  /** Pause reason if state is `paused`; supersedes preCheck for label. */
  pauseReason?: PauseReason;
}

interface ChipDescriptor {
  label: string;
  kindKey: 'unknown' | 'warn' | 'ready';
  accessibilityLabel: string;
}

function describePauseReason(reason: PauseReason): string {
  switch (reason) {
    case 'too_fast':
      return 'Too fast';
    case 'too_slow':
      return 'Too slow';
    case 'lost_bottle':
      return 'Bottle lost';
    case 'blur':
      return 'Hold steady';
    case 'glare':
      return 'Reduce glare';
    case 'motion':
      return 'Phone shake';
    default:
      return 'Paused';
  }
}

function describe(props: QualityChipProps): ChipDescriptor {
  const { state, preCheck, pauseReason } = props;
  if (state === 'paused') {
    const label =
      pauseReason !== undefined
        ? describePauseReason(pauseReason)
        : 'Paused';
    const isFail = pauseReason === 'lost_bottle';
    return {
      label,
      kindKey: 'warn',
      accessibilityLabel: isFail
        ? `Scan paused: ${label}`
        : `Scan paused: ${label}`,
    };
  }
  if (state === 'aligning' || state === 'ready' || state === 'scanning') {
    if (preCheck.kind === 'ready') {
      return {
        label: 'Ready',
        kindKey: 'ready',
        accessibilityLabel: 'Pre-check ready.',
      };
    }
    if (preCheck.kind === 'warn') {
      const map: Record<typeof preCheck.reason, string> = {
        blur: 'Hold steady',
        glare: 'Reduce glare',
        coverage: 'Move closer',
        motion: 'Phone shake',
      };
      const label = map[preCheck.reason];
      return {
        label,
        kindKey: 'warn',
        accessibilityLabel: `Pre-check warning: ${label}.`,
      };
    }
    return {
      label: 'Hold steady…',
      kindKey: 'unknown',
      accessibilityLabel: 'Pre-check pending.',
    };
  }
  return {
    label: '',
    kindKey: 'unknown',
    accessibilityLabel: '',
  };
}

/**
 * Visibility — see §3.7.
 *  - aligning / ready / paused: visible
 *  - scanning: visible only when there's a non-`ready` preCheck
 *  - complete / failed: hidden (parent screen takes over)
 */
function shouldBeVisible(props: QualityChipProps): boolean {
  const { state, preCheck } = props;
  if (state === 'aligning' || state === 'ready' || state === 'paused') return true;
  if (state === 'scanning') return preCheck.kind === 'warn';
  return false;
}

export function QualityChip(props: QualityChipProps): React.ReactElement {
  const insets = useSafeAreaInsets();
  const { kindKey, label, accessibilityLabel } = describe(props);
  const visible = shouldBeVisible(props);

  // Reanimated state — translateX (slide-in/out from the right edge)
  // and opacity for the visibility transition; kindCode + scale + spin
  // for the verdict change pulse, lifted from PreCheckIndicator.
  const translateX = useSharedValue<number>(120);
  const opacity = useSharedValue<number>(0);
  const kindCode = useSharedValue<number>(KIND_CODE[kindKey]);
  const scale = useSharedValue<number>(1);
  const spinDeg = useSharedValue<number>(0);

  const prevKindRef = useRef<'unknown' | 'warn' | 'ready'>(kindKey);
  const visibleRef = useRef<boolean>(visible);

  useEffect(() => {
    if (visible && !visibleRef.current) {
      // Slide in over 220ms.
      translateX.value = withTiming(0, {
        duration: 220,
        easing: Easing.out(Easing.cubic),
      });
      opacity.value = withTiming(1, {
        duration: 220,
        easing: Easing.out(Easing.cubic),
      });
    } else if (!visible && visibleRef.current) {
      // Slide out over 180ms.
      translateX.value = withTiming(120, {
        duration: 180,
        easing: Easing.in(Easing.cubic),
      });
      opacity.value = withTiming(0, {
        duration: 180,
        easing: Easing.in(Easing.cubic),
      });
    }
    visibleRef.current = visible;
  }, [visible, translateX, opacity]);

  useEffect(() => {
    kindCode.value = withTiming(KIND_CODE[kindKey], {
      duration: 220,
      easing: Easing.out(Easing.quad),
    });
    scale.value = withSequence(
      withTiming(1.08, { duration: 120, easing: Easing.out(Easing.quad) }),
      withSpring(1, { damping: 12, stiffness: 180 }),
    );
    prevKindRef.current = kindKey;
  }, [kindKey, kindCode, scale]);

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

  const wrapStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: translateX.value }],
    opacity: opacity.value,
  }));

  const chipStyle = useAnimatedStyle(() => {
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

  const spinStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${spinDeg.value}deg` }],
  }));

  return (
    <Animated.View
      pointerEvents="none"
      accessibilityRole="text"
      accessibilityLabel={accessibilityLabel}
      style={[styles.wrap, { top: insets.top + 14 }, wrapStyle]}
    >
      <Animated.View style={[styles.chip, chipStyle]}>
        <Animated.View style={spinStyle}>
          <ChipIcon kindKey={kindKey} />
        </Animated.View>
        <Text style={styles.text}>{label}</Text>
      </Animated.View>
    </Animated.View>
  );
}

function ChipIcon({
  kindKey,
}: {
  kindKey: 'unknown' | 'warn' | 'ready';
}): React.ReactElement {
  const iconColor = colors.background;
  const size = 12;
  switch (kindKey) {
    case 'unknown':
      return <Feather name="loader" size={size} color={iconColor} />;
    case 'warn':
      return <Feather name="alert-triangle" size={size} color={iconColor} />;
    case 'ready':
      return <Feather name="check" size={size} color={iconColor} />;
  }
}

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    right: 16,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
  },
  text: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.4,
    lineHeight: 14,
    color: colors.background,
  },
});
