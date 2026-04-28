import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { OverallStatus, RuleStatus } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

export interface StatusBadgeProps {
  status: OverallStatus | RuleStatus;
  label?: string;
  size?: 'sm' | 'md';
}

export function StatusBadge({ status, label, size = 'md' }: StatusBadgeProps) {
  const palette = paletteFor(status);
  return (
    <View
      style={[
        styles.badge,
        size === 'sm' && styles.badgeSm,
        { backgroundColor: palette.bg, borderColor: palette.border },
      ]}
      accessibilityLabel={`Status: ${status}`}
    >
      <Text style={[styles.text, size === 'sm' && styles.textSm, { color: palette.fg }]}>
        {(label ?? status).toUpperCase()}
      </Text>
    </View>
  );
}

function paletteFor(status: OverallStatus | RuleStatus): {
  fg: string;
  bg: string;
  border: string;
} {
  switch (status) {
    case 'pass':
      return { fg: colors.pass, bg: 'rgba(61,220,151,0.15)', border: colors.pass };
    case 'fail':
      return { fg: colors.fail, bg: 'rgba(255,107,107,0.15)', border: colors.fail };
    case 'advisory':
      return { fg: colors.advisory, bg: 'rgba(244,184,96,0.15)', border: colors.advisory };
  }
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radius.sm,
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
  badgeSm: {
    paddingHorizontal: spacing.xs,
    paddingVertical: 2,
  },
  text: {
    ...typography.caption,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  textSm: {
    fontSize: 11,
  },
});
