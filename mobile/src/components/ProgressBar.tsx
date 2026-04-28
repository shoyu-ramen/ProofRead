import React from 'react';
import { StyleSheet, View, ViewStyle } from 'react-native';
import { colors, radius } from '@src/theme';

export interface ProgressBarProps {
  // 0..1; values outside the range are clamped.
  value: number;
  // null/undefined renders an indeterminate-styled bar (currently a
  // partially-filled track). Real animated indeterminate state is a
  // TODO once Reanimated is wired up.
  indeterminate?: boolean;
  height?: number;
  style?: ViewStyle;
  testID?: string;
}

export function ProgressBar({
  value,
  indeterminate = false,
  height = 8,
  style,
  testID,
}: ProgressBarProps): React.ReactElement {
  const clamped = Math.max(0, Math.min(1, value));
  // TODO(progress): drive an indeterminate animation via Reanimated
  // once we wire reanimated v3 entry/exit transitions.
  const fillPct = indeterminate ? 0.35 : clamped;

  return (
    <View
      style={[styles.track, { height }, style]}
      accessibilityRole="progressbar"
      accessibilityValue={
        indeterminate
          ? undefined
          : { min: 0, max: 100, now: Math.round(clamped * 100) }
      }
      testID={testID}
    >
      <View
        style={[
          styles.fill,
          { width: `${Math.round(fillPct * 100)}%`, height },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  track: {
    backgroundColor: colors.surfaceAlt,
    borderRadius: radius.sm,
    overflow: 'hidden',
  },
  fill: {
    backgroundColor: colors.primary,
    borderRadius: radius.sm,
  },
});
