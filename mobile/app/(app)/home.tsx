/**
 * Home screen — Big "Scan new label" CTA + recent 3 scans.
 *
 * Per SPEC §v1.7. The recent-scans rail is wired to a tanstack-query
 * fetch against the backend's history endpoint when one exists; today
 * the v1 API surface lists GET /v1/scans (history) but the current
 * scaffold backend only ships single-scan endpoints, so the query will
 * 404 and the rail silently degrades to the empty state. (We do not
 * surface errors on the home rail — it's a secondary surface; the full
 * history screen is where retry lives.)
 */

import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { useQuery } from '@tanstack/react-query';

import { Button, RecentScanRow, Screen, SectionHeader } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { useScanStore } from '@src/state/scanStore';
import { colors, radius, spacing, typography } from '@src/theme';

const HOME_RECENT_LIMIT = 3;

export default function Home(): React.ReactElement {
  const reset = useScanStore((s) => s.reset);

  // TODO(history-endpoint): backend GET /v1/scans isn't implemented
  // yet — until it lands the query errors and we render the empty
  // state. Sharing the queryKeys.history() cache so this rail and
  // the full history screen stay in sync once the route exists.
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.history(),
    queryFn: () => apiClient.getHistory(),
  });

  const items = data?.items ?? [];
  const showSkeleton = isLoading;
  // Error → empty (silent degradation). Real history screen handles retry.
  const showEmpty = !showSkeleton && (error != null || items.length === 0);

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

      {showSkeleton ? (
        <RecentSkeletons count={HOME_RECENT_LIMIT} />
      ) : showEmpty ? (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyTitle}>No scans yet</Text>
          <Text style={styles.emptyBody}>
            Your three most recent scans show up here.
          </Text>
        </View>
      ) : (
        <View style={styles.list}>
          {items.slice(0, HOME_RECENT_LIMIT).map((s) => (
            <RecentScanRow key={s.scan_id} scan={s} />
          ))}
        </View>
      )}
    </Screen>
  );
}

function RecentSkeletons({ count }: { count: number }): React.ReactElement {
  return (
    <View
      style={styles.list}
      accessibilityRole="progressbar"
      accessibilityLabel="Loading recent scans"
    >
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonRow key={i} delayMs={i * 120} />
      ))}
    </View>
  );
}

function SkeletonRow({ delayMs }: { delayMs: number }): React.ReactElement {
  const opacity = useRef(new Animated.Value(0.4)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, {
          toValue: 0.8,
          duration: 700,
          delay: delayMs,
          useNativeDriver: true,
        }),
        Animated.timing(opacity, {
          toValue: 0.4,
          duration: 700,
          useNativeDriver: true,
        }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [opacity, delayMs]);

  return <Animated.View style={[styles.skeletonRow, { opacity }]} />;
}

const SKELETON_HEIGHT = 64;

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
  skeletonRow: {
    height: SKELETON_HEIGHT,
    borderRadius: radius.md,
    backgroundColor: colors.surfaceAlt,
    borderWidth: 1,
    borderColor: colors.border,
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
