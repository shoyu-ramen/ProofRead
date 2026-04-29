/**
 * Scan history — full list of the user's recent scans.
 *
 * Reads `GET /v1/scans` (HistoryResponse) via tanstack-query, renders
 * a reverse-chronological list (the backend already sorts), and routes
 * each row to the report screen on tap. Empty / loading / error states
 * are inline so the screen never appears blank.
 *
 * Pull-to-refresh re-runs the query — useful after a brand-new scan
 * lands while the user was on this screen.
 */

import React, { useCallback } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { router } from 'expo-router';
import { useQuery } from '@tanstack/react-query';

import { Button, StatusBadge } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { colors, radius, spacing, typography } from '@src/theme';
import type { HistoryItem } from '@src/api/types';
import { SafeAreaView } from 'react-native-safe-area-context';

/**
 * Convert an ISO timestamp into a human-readable relative phrase.
 * Mirrors the SPEC §v1.7 "2 hours ago / Yesterday / absolute date"
 * pattern. Falls back to a locale date string when the input is
 * unparseable so the row never renders an empty meta.
 */
export function formatRelativeTime(iso: string, now: Date = new Date()): string {
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) {
    // Server should always return a parseable ISO — but if not, surface
    // the raw value so we can debug rather than silently dropping it.
    return iso;
  }
  const then = new Date(ts);
  const diffMs = now.getTime() - then.getTime();
  const diffSec = Math.round(diffMs / 1000);
  const diffMin = Math.round(diffSec / 60);
  const diffHr = Math.round(diffMin / 60);
  const diffDay = Math.round(diffHr / 24);

  if (diffSec < 45) return 'Just now';
  if (diffMin < 60) return `${diffMin} minute${diffMin === 1 ? '' : 's'} ago`;
  if (diffHr < 24) return `${diffHr} hour${diffHr === 1 ? '' : 's'} ago`;
  if (diffDay === 1) return 'Yesterday';
  if (diffDay < 7) return `${diffDay} days ago`;
  // Beyond a week: absolute date in the user's locale (no time).
  return then.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

export default function HistoryScreen(): React.ReactElement {
  const { data, isLoading, isRefetching, refetch, error } = useQuery({
    queryKey: queryKeys.history(),
    queryFn: () => apiClient.listScans(),
  });

  const onRefresh = useCallback(() => {
    void refetch();
  }, [refetch]);

  if (isLoading) {
    return (
      <SafeAreaView style={styles.center} edges={['top', 'bottom']}>
        <ActivityIndicator color={colors.primary} />
        <Text style={styles.muted}>Loading scans…</Text>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.center} edges={['top', 'bottom']}>
        <Text style={styles.title}>Couldn't load history</Text>
        <Text style={styles.muted}>
          Check your connection and try again.
        </Text>
        <Button label="Retry" onPress={onRefresh} />
      </SafeAreaView>
    );
  }

  const items = data?.items ?? [];

  if (items.length === 0) {
    return (
      <SafeAreaView style={styles.center} edges={['top', 'bottom']}>
        <Text style={styles.title}>No scans yet</Text>
        <Text style={styles.muted}>
          Scan your first label to see it here.
        </Text>
        <Button
          label="Scan your first label"
          onPress={() => router.replace('/(app)/scan/setup')}
        />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root} edges={['bottom']}>
      <FlatList
        data={items}
        keyExtractor={(item) => item.scan_id}
        renderItem={({ item }) => <HistoryRow item={item} />}
        contentContainerStyle={styles.listContent}
        ItemSeparatorComponent={() => <View style={styles.separator} />}
        refreshControl={
          <RefreshControl
            refreshing={isRefetching}
            onRefresh={onRefresh}
            tintColor={colors.text}
          />
        }
      />
    </SafeAreaView>
  );
}

/**
 * One row in the history list. Local component (not the shared
 * RecentScanRow) because the spec calls for relative-time formatting on
 * `scanned_at`, plus a fallback when `label` is missing — neither of
 * which RecentScanRow does today, and we don't want to widen its API
 * just for the history screen.
 */
function HistoryRow({ item }: { item: HistoryItem }): React.ReactElement {
  const title = item.label && item.label.trim().length > 0
    ? item.label
    : 'Untitled scan';
  const when = formatRelativeTime(item.scanned_at);

  return (
    <Pressable
      onPress={() => router.push(`/(app)/scan/report/${item.scan_id}`)}
      accessibilityRole="button"
      accessibilityLabel={`Open scan ${title}, ${item.overall}, ${when}`}
      style={({ pressed }) => [
        styles.row,
        pressed && styles.rowPressed,
      ]}
    >
      <View style={styles.rowText}>
        <Text style={styles.rowTitle} numberOfLines={1}>
          {title}
        </Text>
        <Text style={styles.rowMeta} numberOfLines={1}>
          {when}
        </Text>
      </View>
      <StatusBadge status={item.overall} size="sm" />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  center: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.lg,
    gap: spacing.md,
  },
  title: {
    ...typography.title,
    color: colors.text,
    textAlign: 'center',
  },
  muted: {
    ...typography.body,
    color: colors.textMuted,
    textAlign: 'center',
  },
  listContent: {
    padding: spacing.lg,
  },
  separator: {
    height: spacing.sm,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.md,
  },
  rowPressed: {
    opacity: 0.85,
  },
  rowText: {
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
