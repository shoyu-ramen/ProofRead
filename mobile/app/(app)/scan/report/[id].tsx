/**
 * Report screen — header (pass/fail icon), per-rule list, image overlay
 * drawer.
 *
 * SPEC §v1.6 step 8 + §v1.7 row "Report".
 *
 * The image-overlay drawer is rendered as a thumbnail strip that opens
 * the rule-detail screen on tap, where the full bbox-on-image view
 * lives. (Inline modal drawer would also satisfy the spec; we picked
 * navigation-based for simpler v1 layout.)
 */

import React from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { useQuery } from '@tanstack/react-query';

import {
  Button,
  RuleResultCard,
  SectionHeader,
  StatusBadge,
} from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { useScanStore } from '@src/state/scanStore';
import { colors, radius, spacing, typography } from '@src/theme';
import type { Surface } from '@src/api/types';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function ReportScreen(): React.ReactElement {
  const { id } = useLocalSearchParams<{ id: string }>();
  const scanId = typeof id === 'string' ? id : '';
  const captures = useScanStore((s) => s.captures);

  const { data, isLoading, isRefetching, refetch, error } = useQuery({
    queryKey: queryKeys.report(scanId),
    enabled: scanId.length > 0,
    queryFn: () => apiClient.getReport(scanId),
  });

  if (isLoading) {
    return (
      <SafeAreaView style={styles.loadingWrap}>
        <ActivityIndicator color={colors.primary} />
        <Text style={styles.loadingText}>Loading report…</Text>
      </SafeAreaView>
    );
  }

  if (error || !data) {
    return (
      <SafeAreaView style={styles.loadingWrap}>
        <Text style={styles.loadingTitle}>Report unavailable</Text>
        <Text style={styles.loadingText}>
          We couldn't load this report. Check your connection and try again.
        </Text>
        <Button label="Retry" onPress={() => void refetch()} />
        <Button
          label="Back to home"
          variant="ghost"
          onPress={() => router.replace('/(app)/home')}
        />
      </SafeAreaView>
    );
  }

  const grouped = groupByStatus(data.rule_results);

  return (
    <SafeAreaView style={styles.root} edges={['bottom']}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl
            refreshing={isRefetching}
            onRefresh={() => void refetch()}
            tintColor={colors.text}
          />
        }
      >
        <View style={styles.headerCard}>
          <View style={styles.headerRow}>
            <Text style={styles.headerTitle}>Overall</Text>
            <StatusBadge status={data.overall} />
          </View>
          <Text style={styles.headerCaption}>
            {data.rule_results.length} rule
            {data.rule_results.length === 1 ? '' : 's'} evaluated
          </Text>
        </View>

        {/* Image strip — taps open rule detail with the relevant surface */}
        <SectionHeader title="Captures" />
        <View style={styles.imageStrip}>
          {(['front', 'back'] as Surface[]).map((surface) => {
            const cap = captures[surface];
            return (
              <View key={surface} style={styles.imageThumb}>
                <Text style={styles.imageLabel}>{surface}</Text>
                <View style={styles.imageBox}>
                  {cap ? (
                    <Image
                      source={{ uri: cap.uri }}
                      style={styles.imageInner}
                      resizeMode="cover"
                    />
                  ) : (
                    <Text style={styles.imageMissing}>not local</Text>
                  )}
                </View>
              </View>
            );
          })}
        </View>

        {grouped.fail.length > 0 && (
          <Section title="Failed" results={grouped.fail} scanId={scanId} />
        )}
        {grouped.advisory.length > 0 && (
          <Section title="Advisory" results={grouped.advisory} scanId={scanId} />
        )}
        {grouped.pass.length > 0 && (
          <Section title="Passing" results={grouped.pass} scanId={scanId} />
        )}

        <View style={styles.actions}>
          <Button
            label="Rescan"
            variant="secondary"
            fullWidth
            onPress={() => router.replace('/(app)/scan/beverage-type')}
          />
          <Button
            label="Done"
            fullWidth
            onPress={() => router.replace('/(app)/home')}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function Section({
  title,
  results,
  scanId,
}: {
  title: string;
  results: ReadonlyArray<import('@src/api/types').RuleResultDTO>;
  scanId: string;
}): React.ReactElement {
  return (
    <View style={{ gap: spacing.sm }}>
      <SectionHeader title={title} />
      {results.map((r) => (
        <Pressable
          key={r.rule_id}
          onPress={() =>
            router.push({
              pathname: '/(app)/scan/rule/[ruleId]',
              params: { ruleId: r.rule_id, scanId },
            })
          }
        >
          <RuleResultCard result={r} />
        </Pressable>
      ))}
    </View>
  );
}

function groupByStatus(
  results: ReadonlyArray<import('@src/api/types').RuleResultDTO>
): {
  pass: import('@src/api/types').RuleResultDTO[];
  fail: import('@src/api/types').RuleResultDTO[];
  advisory: import('@src/api/types').RuleResultDTO[];
} {
  const out = {
    pass: [] as import('@src/api/types').RuleResultDTO[],
    fail: [] as import('@src/api/types').RuleResultDTO[],
    advisory: [] as import('@src/api/types').RuleResultDTO[],
  };
  for (const r of results) out[r.status].push(r);
  return out;
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scrollContent: {
    padding: spacing.lg,
    gap: spacing.md,
  },
  headerCard: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
    gap: spacing.sm,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerTitle: {
    ...typography.title,
    color: colors.text,
  },
  headerCaption: {
    ...typography.caption,
    color: colors.textMuted,
  },
  imageStrip: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  imageThumb: {
    flex: 1,
    gap: spacing.xs,
  },
  imageLabel: {
    ...typography.caption,
    color: colors.textMuted,
    textTransform: 'capitalize',
  },
  imageBox: {
    aspectRatio: 0.65,
    backgroundColor: colors.surfaceAlt,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageInner: {
    width: '100%',
    height: '100%',
  },
  imageMissing: {
    ...typography.caption,
    color: colors.textMuted,
  },
  actions: {
    gap: spacing.sm,
    marginTop: spacing.md,
  },
  loadingWrap: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.lg,
    gap: spacing.md,
  },
  loadingTitle: {
    ...typography.title,
    color: colors.text,
    textAlign: 'center',
  },
  loadingText: {
    ...typography.body,
    color: colors.textMuted,
    textAlign: 'center',
  },
});
