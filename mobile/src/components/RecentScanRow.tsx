import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import type { HistoryItem } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';
import { StatusBadge } from './StatusBadge';

export interface RecentScanRowProps {
  scan: HistoryItem;
  onPress?: () => void;
}

// Row used by the history list (and, in time, by home's recent-scans
// rail — the inline version in home.tsx mirrors this shape).
export function RecentScanRow({ scan, onPress }: RecentScanRowProps): React.ReactElement {
  const handlePress =
    onPress ?? (() => router.push(`/(app)/scan/report/${scan.scan_id}`));
  return (
    <Pressable
      style={({ pressed }) => [styles.rowCard, pressed && styles.pressed]}
      onPress={handlePress}
      accessibilityRole="button"
      accessibilityLabel={`Open scan ${scan.label}, status ${scan.overall}`}
    >
      <View style={styles.titleCol}>
        <Text style={styles.rowTitle} numberOfLines={1}>
          {scan.label}
        </Text>
        <Text style={styles.rowMeta} numberOfLines={1}>
          {scan.scanned_at}
        </Text>
      </View>
      <StatusBadge status={scan.overall} size="sm" />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  rowCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    gap: spacing.sm,
  },
  pressed: {
    opacity: 0.85,
  },
  titleCol: {
    flex: 1,
    gap: 2,
  },
  rowTitle: {
    ...typography.heading,
    color: colors.text,
  },
  rowMeta: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
