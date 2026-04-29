/**
 * Home screen — primary "Scan new label" CTA + recent-scans rail.
 *
 * Reads `GET /v1/scans` (HistoryResponse) via tanstack-query and
 * conditionally renders the rail:
 *   - 0 scans → hero + scan CTA + settings only.
 *   - 1+ scans → scan CTA on top, "Recent scans" with up to 3 rows,
 *     and a "View all" link to /history when more exist.
 *
 * The rail uses the locally-cached panorama (scanStore.recentPanoramas)
 * for thumbnails when available — keeps the home screen feeling fast
 * after a scan completes, no second download required.
 */

import React from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { router } from 'expo-router';
import { useQuery } from '@tanstack/react-query';

import { Button, Screen, StatusBadge } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { useScanStore } from '@src/state/scanStore';
import { colors, radius, spacing, typography } from '@src/theme';
import type { HistoryItem } from '@src/api/types';
import { formatRelativeTime } from './history';

const RAIL_LIMIT = 3;

export default function Home(): React.ReactElement {
  const reset = useScanStore((s) => s.reset);
  const recentPanoramas = useScanStore((s) => s.recentPanoramas);

  // History query — failure here shouldn't block the scan CTA, so the
  // error path quietly hides the rail rather than rendering an error
  // card on the home screen.
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.history(),
    queryFn: () => apiClient.listScans(),
  });

  const items = data?.items ?? [];
  const hasItems = items.length > 0;
  const railItems = items.slice(0, RAIL_LIMIT);
  const showViewAll = items.length > RAIL_LIMIT;

  const handleStartScan = () => {
    reset();
    router.push('/(app)/scan/setup');
  };

  return (
    <Screen>
      <View style={styles.heroBlock}>
        <Text style={styles.headline}>Verify a label.</Text>
        <Text style={styles.subhead}>
          Rotate the bottle once. We read every side.
        </Text>
      </View>

      <Button
        label="Scan new label"
        size="lg"
        fullWidth
        onPress={handleStartScan}
      />

      {/* Loading: show a small inline spinner so the rail doesn't pop
          in late and shift the layout abruptly. Errors silently hide
          the rail (the scan CTA is still useful on its own). */}
      {isLoading && !hasItems ? (
        <View style={styles.railLoading}>
          <ActivityIndicator color={colors.textMuted} size="small" />
        </View>
      ) : null}

      {hasItems && !error ? (
        <View style={styles.rail}>
          <View style={styles.railHeader}>
            <Text style={styles.railTitle}>Recent scans</Text>
            {showViewAll ? (
              <Pressable
                onPress={() => router.push('/(app)/history')}
                hitSlop={8}
                accessibilityRole="link"
                accessibilityLabel="View all scans"
              >
                {({ pressed }) => (
                  <Text
                    style={[
                      styles.viewAllLink,
                      pressed && { opacity: 0.7 },
                    ]}
                  >
                    View all
                  </Text>
                )}
              </Pressable>
            ) : null}
          </View>

          {railItems.map((item) => (
            <RecentRow
              key={item.scan_id}
              item={item}
              thumbnailUri={recentPanoramas[item.scan_id]?.uri ?? null}
            />
          ))}
        </View>
      ) : null}

      {/* Bottom row — Settings stays where it was, History link added
          beside it so users can reach the full list even when the rail
          is hidden (zero scans / error). */}
      <View style={styles.bottomActions}>
        <Button
          label="Settings"
          variant="secondary"
          fullWidth
          onPress={() => router.push('/(app)/settings')}
        />
        <Pressable
          onPress={() => router.push('/(app)/history')}
          hitSlop={8}
          accessibilityRole="link"
          accessibilityLabel="Open scan history"
          style={styles.historyLink}
        >
          {({ pressed }) => (
            <Text
              style={[styles.historyLinkText, pressed && { opacity: 0.7 }]}
            >
              {hasItems ? 'View full history' : 'History'}
            </Text>
          )}
        </Pressable>
      </View>
    </Screen>
  );
}

/**
 * Compact recent-scan row used by the home rail. Shows a tiny panorama
 * thumbnail when we have one cached locally — otherwise just the title
 * + relative-time + status badge. Falls back to "Untitled scan" when
 * the backend's `label` is empty.
 */
function RecentRow({
  item,
  thumbnailUri,
}: {
  item: HistoryItem;
  thumbnailUri: string | null;
}): React.ReactElement {
  const title = item.label && item.label.trim().length > 0
    ? item.label
    : 'Untitled scan';
  const when = formatRelativeTime(item.scanned_at);

  return (
    <Pressable
      onPress={() => router.push(`/(app)/scan/report/${item.scan_id}`)}
      accessibilityRole="button"
      accessibilityLabel={`Open scan ${title}, ${item.overall}, ${when}`}
      style={({ pressed }) => [styles.row, pressed && styles.rowPressed]}
    >
      <View style={styles.thumbWrap}>
        {thumbnailUri ? (
          <Image
            source={{ uri: thumbnailUri }}
            style={styles.thumb}
            resizeMode="cover"
          />
        ) : (
          <View style={styles.thumbPlaceholder} />
        )}
      </View>
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
  rail: {
    gap: spacing.sm,
    marginTop: spacing.sm,
  },
  railLoading: {
    paddingVertical: spacing.md,
    alignItems: 'flex-start',
  },
  railHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  railTitle: {
    ...typography.heading,
    color: colors.text,
  },
  viewAllLink: {
    ...typography.caption,
    color: colors.primary,
    fontWeight: '600',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.sm,
  },
  rowPressed: {
    opacity: 0.85,
  },
  thumbWrap: {
    width: 56,
    height: 40,
    borderRadius: radius.sm,
    overflow: 'hidden',
    backgroundColor: colors.surfaceAlt,
  },
  thumb: {
    width: '100%',
    height: '100%',
  },
  thumbPlaceholder: {
    flex: 1,
    backgroundColor: colors.surfaceAlt,
  },
  rowText: {
    flex: 1,
    gap: 2,
  },
  rowTitle: {
    ...typography.body,
    color: colors.text,
    fontWeight: '600',
  },
  rowMeta: {
    ...typography.caption,
    color: colors.textMuted,
  },
  bottomActions: {
    marginTop: spacing.md,
    gap: spacing.sm,
    alignItems: 'stretch',
  },
  historyLink: {
    alignSelf: 'center',
    paddingVertical: spacing.xs,
  },
  historyLinkText: {
    ...typography.caption,
    color: colors.primary,
    fontWeight: '600',
  },
});
