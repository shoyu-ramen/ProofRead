/**
 * Per-surface captures preview — full-bleed captured image with every
 * rule's bbox overlaid, color-coded by status.
 *
 * Reached from the report screen's captures strip. The bbox geometry
 * lives in the same image-pixel space the rule-detail overlay uses;
 * here we just render every rule's bbox at once so the user can see
 * the whole label state in context.
 */

import React, { useMemo } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery } from '@tanstack/react-query';

import { BBoxOverlay, Button, StatusBadge } from '@src/components';
import type { BBoxOverlayItem } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { surfaceForPanel, useScanStore } from '@src/state/scanStore';
import type { RuleResultDTO, Surface } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

export default function CapturesPreviewScreen(): React.ReactElement {
  const params = useLocalSearchParams<{ surface: string; scanId?: string }>();
  const surface = normalizeSurface(params.surface);
  const scanId =
    typeof params.scanId === 'string' && params.scanId.length > 0
      ? params.scanId
      : (useScanStore.getState().scanId ?? '');

  const capture = useScanStore((s) => (surface ? s.captures[surface] : undefined));

  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.report(scanId),
    enabled: scanId.length > 0,
    queryFn: () => apiClient.getReport(scanId),
  });

  // Every rule with a bbox AND that maps to this capture becomes an
  // overlay item. Backend rule_results carry a `surface` panel id
  // (e.g. "panel_0"); surfaceForPanel maps it to a local capture slot
  // and we keep only the matches. When `surface` is missing on the
  // wire (scans.py is mid-rollout — verify.py emits the field today)
  // the rule is treated as un-attributed and drawn on every capture
  // the user opens, which preserves today's behavior. Once both
  // endpoints ship the field, the un-attributed branch can go.
  const items = useMemo<BBoxOverlayItem[]>(() => {
    if (!data) return [];
    return data.rule_results
      .filter((r): r is RuleResultDTO & { bbox: NonNullable<RuleResultDTO['bbox']> } =>
        r.bbox !== null
      )
      .filter((r) => {
        const mapped = surfaceForPanel(r.surface);
        return mapped === null || mapped === surface;
      })
      .map((r) => ({ id: r.rule_id, bbox: r.bbox, status: r.status }));
  }, [data, surface]);

  if (!surface) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.muted}>Unknown surface "{params.surface}".</Text>
        <Button label="Back" variant="ghost" onPress={() => router.back()} />
      </SafeAreaView>
    );
  }

  if (isLoading) {
    return (
      <SafeAreaView style={styles.center}>
        <ActivityIndicator color={colors.primary} />
        <Text style={styles.muted}>Loading captures…</Text>
      </SafeAreaView>
    );
  }

  if (error || !data) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.muted}>
          We couldn't load this report. Try again from the report.
        </Text>
        <Button label="Back" variant="ghost" onPress={() => router.back()} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root} edges={['bottom']}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.headerRow}>
          <Text style={styles.title}>{capitalize(surface)} capture</Text>
          <StatusBadge status={data.overall} />
        </View>
        <Text style={styles.caption}>
          {items.length} region{items.length === 1 ? '' : 's'} highlighted
        </Text>

        {capture ? (
          <BBoxOverlay image={capture} items={items} />
        ) : (
          <View style={styles.missingBox}>
            <Text style={styles.muted}>
              Original {surface} image isn't on this device — overlay
              drawing skipped. Reload after a fresh scan to see the
              captured frame.
            </Text>
          </View>
        )}

        {items.length > 0 ? (
          <View style={styles.legend}>
            <LegendDot color={colors.fail} label="fail" />
            <LegendDot color={colors.advisory} label="advisory" />
            <LegendDot color={colors.pass} label="pass" />
          </View>
        ) : null}

        {items.length > 0 ? (
          <View style={styles.ruleList}>
            <Text style={styles.sectionTitle}>Regions</Text>
            {items.map((item) => {
              const rule = data.rule_results.find((r) => r.rule_id === item.id);
              if (!rule) return null;
              return (
                <Pressable
                  key={item.id}
                  accessibilityRole="button"
                  onPress={() =>
                    router.push({
                      pathname: '/(app)/scan/rule/[ruleId]',
                      params: { ruleId: rule.rule_id, scanId },
                    })
                  }
                  style={({ pressed }) => [
                    styles.ruleRow,
                    pressed && styles.ruleRowPressed,
                  ]}
                >
                  <View style={styles.ruleRowText}>
                    <Text style={styles.ruleRowTitle} numberOfLines={1}>
                      {humanizeRuleId(rule.rule_id)}
                    </Text>
                    <Text style={styles.ruleRowCitation}>{rule.citation}</Text>
                  </View>
                  <StatusBadge status={rule.status} size="sm" />
                </Pressable>
              );
            })}
          </View>
        ) : null}

        <Button label="Back to report" variant="ghost" onPress={() => router.back()} />
      </ScrollView>
    </SafeAreaView>
  );
}

function normalizeSurface(raw: unknown): Surface | null {
  if (raw !== 'front' && raw !== 'back' && raw !== 'side' && raw !== 'neck') return null;
  return raw;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function humanizeRuleId(ruleId: string): string {
  // beer.health_warning.exact_text → "Health warning · exact text"
  const parts = ruleId.split('.');
  if (parts.length < 2) return ruleId;
  const [, ...rest] = parts;
  return rest
    .map((p) => p.replace(/_/g, ' '))
    .map(capitalize)
    .join(' · ');
}

function LegendDot({ color, label }: { color: string; label: string }): React.ReactElement {
  return (
    <View style={styles.legendItem}>
      <View style={[styles.legendDot, { borderColor: color }]} />
      <Text style={styles.legendText}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.lg,
    gap: spacing.md,
  },
  center: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.md,
    padding: spacing.lg,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.sm,
  },
  title: {
    ...typography.title,
    color: colors.text,
    flex: 1,
  },
  caption: {
    ...typography.caption,
    color: colors.textMuted,
  },
  missingBox: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
  },
  legend: {
    flexDirection: 'row',
    gap: spacing.md,
    flexWrap: 'wrap',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  legendDot: {
    width: 14,
    height: 14,
    borderWidth: 2,
    borderRadius: 2,
  },
  legendText: {
    ...typography.caption,
    color: colors.textMuted,
    textTransform: 'capitalize',
  },
  ruleList: {
    gap: spacing.sm,
  },
  sectionTitle: {
    ...typography.heading,
    color: colors.text,
  },
  ruleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    gap: spacing.sm,
  },
  ruleRowPressed: {
    opacity: 0.85,
  },
  ruleRowText: {
    flex: 1,
    gap: 2,
  },
  ruleRowTitle: {
    ...typography.heading,
    color: colors.text,
  },
  ruleRowCitation: {
    ...typography.caption,
    color: colors.textMuted,
  },
  muted: {
    ...typography.body,
    color: colors.textMuted,
  },
});
