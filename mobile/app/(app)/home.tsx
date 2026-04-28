/**
 * Home screen — Big "Scan new label" CTA + recent 3 scans.
 *
 * Per SPEC §v1.7. The recent-scans rail is wired to a tanstack-query
 * fetch against the backend's history endpoint when one exists; today
 * the v1 API surface lists GET /v1/scans (history) but the current
 * scaffold backend only ships single-scan endpoints, so this renders
 * an empty state until the history endpoint lands.
 *
 * TODO(history-endpoint): replace the placeholder array with a query
 * once GET /v1/scans paginated history is implemented backend-side.
 */

import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { Button, Screen, SectionHeader, StatusBadge } from '@src/components';
import { useScanStore } from '@src/state/scanStore';
import type { OverallStatus } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

interface RecentScanSummary {
  scan_id: string;
  label: string;
  overall: OverallStatus;
  scanned_at: string;
}

// TODO(history-endpoint): swap for live data.
const RECENT_PLACEHOLDER: RecentScanSummary[] = [];

export default function Home(): React.ReactElement {
  const reset = useScanStore((s) => s.reset);

  const handleStartScan = () => {
    reset();
    router.push('/(app)/scan/beverage-type');
  };

  return (
    <Screen>
      <View style={styles.heroBlock}>
        <Text style={styles.headline}>Verify a label.</Text>
        <Text style={styles.subhead}>
          Capture front and back, get a TTB-compliance report in under
          30 seconds.
        </Text>
      </View>

      <Button
        label="Scan new label"
        size="lg"
        fullWidth
        onPress={handleStartScan}
      />

      <View style={styles.row}>
        <Button
          label="History"
          variant="secondary"
          fullWidth
          onPress={() => router.push('/(app)/history')}
          style={{ flex: 1 }}
        />
        <Button
          label="Settings"
          variant="secondary"
          fullWidth
          onPress={() => router.push('/(app)/settings')}
          style={{ flex: 1 }}
        />
      </View>

      <SectionHeader title="Recent scans" subtitle="Last three" />

      {RECENT_PLACEHOLDER.length === 0 ? (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyTitle}>No scans yet</Text>
          <Text style={styles.emptyBody}>
            Your three most recent scans show up here.
          </Text>
        </View>
      ) : (
        <View style={styles.list}>
          {RECENT_PLACEHOLDER.slice(0, 3).map((s) => (
            <RecentScanRow key={s.scan_id} scan={s} />
          ))}
        </View>
      )}
    </Screen>
  );
}

function RecentScanRow({ scan }: { scan: RecentScanSummary }): React.ReactElement {
  return (
    <Pressable
      style={({ pressed }) => [styles.rowCard, pressed && { opacity: 0.85 }]}
      onPress={() => router.push(`/(app)/scan/report/${scan.scan_id}`)}
    >
      <View style={{ flex: 1, gap: 2 }}>
        <Text style={styles.rowTitle}>{scan.label}</Text>
        <Text style={styles.rowMeta}>{scan.scanned_at}</Text>
      </View>
      <StatusBadge status={scan.overall} size="sm" />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  heroBlock: {
    gap: spacing.sm,
    paddingTop: spacing.md,
  },
  headline: {
    ...typography.display,
    color: colors.text,
  },
  subhead: {
    ...typography.body,
    color: colors.textMuted,
  },
  row: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  list: {
    gap: spacing.sm,
  },
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
  rowTitle: {
    ...typography.heading,
    color: colors.text,
  },
  rowMeta: {
    ...typography.caption,
    color: colors.textMuted,
  },
  emptyCard: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
    gap: spacing.xs,
  },
  emptyTitle: {
    ...typography.heading,
    color: colors.text,
  },
  emptyBody: {
    ...typography.body,
    color: colors.textMuted,
  },
});
