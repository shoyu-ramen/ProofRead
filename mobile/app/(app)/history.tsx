/**
 * Scan history list. SPEC §v1.5 F1.11 + §v1.7 row "History".
 *
 * Backend exposes `GET /v1/scans` per SPEC §v1.9 but the current
 * scaffold backend (scans.py) only implements per-scan endpoints —
 * the request will 404 today. The screen is wired to tanstack-query
 * so it'll light up the moment the route lands; until then users see
 * the error state with a retry button.
 *
 * Once FileSystem persistence is in (SPEC §0.5 offline matrix), we'll
 * cache the last 50 reports on device and merge them into the list.
 */

import React, { useCallback, useEffect, useRef } from 'react';
import {
  Animated,
  FlatList,
  RefreshControl,
  StyleSheet,
  Text,
  View,
  type ListRenderItemInfo,
} from 'react-native';
import { router } from 'expo-router';
import { useQuery } from '@tanstack/react-query';

import { Button, RecentScanRow, Screen, SectionHeader } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import type { HistoryItem } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

const SKELETON_COUNT = 5;

export default function HistoryScreen(): React.ReactElement {
  // TODO(history-endpoint): apiClient.getHistory() targets GET /v1/scans
  // which isn't implemented backend-side yet. Until then the query
  // resolves to the error state, which is a deliberately useful demo
  // of the retry path.
  const { data, error, isLoading, isRefetching, refetch } = useQuery({
    queryKey: queryKeys.history(),
    queryFn: () => apiClient.getHistory(),
    // History rows are cheap to refetch and we want fresh state on
    // re-entry (e.g. after a new scan completes).
    staleTime: 0,
  });

  return (
    <Screen scroll={false}>
      <SectionHeader title="Your scans" subtitle="Most recent first." />

      <View style={styles.bodyWrap}>
        {isLoading ? (
          <SkeletonList />
        ) : error ? (
          <ErrorCard onRetry={() => void refetch()} />
        ) : data && data.items.length > 0 ? (
          <HistoryList
            items={data.items}
            refreshing={isRefetching}
            onRefresh={() => void refetch()}
          />
        ) : (
          <EmptyCard />
        )}
      </View>

      <Button
        label="Scan a new label"
        size="lg"
        fullWidth
        onPress={() => router.push('/(app)/scan/beverage-type')}
      />
    </Screen>
  );
}

// ---------- States ----------

function HistoryList({
  items,
  refreshing,
  onRefresh,
}: {
  items: HistoryItem[];
  refreshing: boolean;
  onRefresh: () => void;
}): React.ReactElement {
  const renderItem = useCallback(
    ({ item }: ListRenderItemInfo<HistoryItem>) => <RecentScanRow scan={item} />,
    [],
  );
  return (
    <FlatList
      data={items}
      keyExtractor={(item) => item.scan_id}
      renderItem={renderItem}
      ItemSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
      contentContainerStyle={styles.listContent}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={colors.text}
        />
      }
    />
  );
}

function SkeletonList(): React.ReactElement {
  return (
    <View
      style={styles.skeletonList}
      accessibilityRole="progressbar"
      accessibilityLabel="Loading scan history"
    >
      {Array.from({ length: SKELETON_COUNT }).map((_, i) => (
        <SkeletonRow key={i} delayMs={i * 120} />
      ))}
    </View>
  );
}

// Pulsing-opacity skeleton. Reanimated would give us a sweep shimmer
// but Animated keeps this dependency-free and is honest about the
// fact that we're just signalling "loading" rather than rendering a
// faux content preview.
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

function ErrorCard({ onRetry }: { onRetry: () => void }): React.ReactElement {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Couldn't load history</Text>
      <Text style={styles.cardBody}>
        We couldn't reach the server. Check your connection and try again.
      </Text>
      <View style={styles.cardActions}>
        <Button label="Retry" onPress={onRetry} />
      </View>
    </View>
  );
}

function EmptyCard(): React.ReactElement {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>No scans yet</Text>
      <Text style={styles.cardBody}>
        Completed scans will appear here. Pull down to refresh once you've
        finished one.
      </Text>
    </View>
  );
}

// ---------- Styles ----------

const SKELETON_HEIGHT = 64;

const styles = StyleSheet.create({
  bodyWrap: {
    flex: 1,
  },
  listContent: {
    paddingBottom: spacing.md,
  },
  skeletonList: {
    gap: spacing.sm,
  },
  skeletonRow: {
    height: SKELETON_HEIGHT,
    borderRadius: radius.md,
    backgroundColor: colors.surfaceAlt,
    borderWidth: 1,
    borderColor: colors.border,
  },
  card: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.lg,
    gap: spacing.sm,
  },
  cardTitle: {
    ...typography.heading,
    color: colors.text,
  },
  cardBody: {
    ...typography.body,
    color: colors.textMuted,
  },
  cardActions: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginTop: spacing.xs,
  },
});
