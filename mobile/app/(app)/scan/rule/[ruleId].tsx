/**
 * Rule detail — bounding box + expected vs found + citation + fix.
 *
 * SPEC §v1.7 row "Rule detail".
 *
 * Routed with both ruleId and scanId in the params object so we can
 * fetch the full report and pick out this rule's result. Backend has
 * no per-rule endpoint today; we filter client-side from the cached
 * report query.
 */

import React, { useMemo } from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery } from '@tanstack/react-query';

import { Button, StatusBadge } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { useScanStore } from '@src/state/scanStore';
import type { BBox, RuleResultDTO } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

export default function RuleDetailScreen(): React.ReactElement {
  const params = useLocalSearchParams<{ ruleId: string; scanId?: string }>();
  const ruleId = typeof params.ruleId === 'string' ? params.ruleId : '';
  const scanId =
    typeof params.scanId === 'string' && params.scanId.length > 0
      ? params.scanId
      : (useScanStore.getState().scanId ?? '');

  const captures = useScanStore((s) => s.captures);

  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.report(scanId),
    enabled: scanId.length > 0,
    queryFn: () => apiClient.getReport(scanId),
  });

  const rule = useMemo<RuleResultDTO | null>(() => {
    if (!data) return null;
    return data.rule_results.find((r) => r.rule_id === ruleId) ?? null;
  }, [data, ruleId]);

  if (isLoading) {
    return (
      <SafeAreaView style={styles.center}>
        <ActivityIndicator color={colors.primary} />
        <Text style={styles.muted}>Loading rule…</Text>
      </SafeAreaView>
    );
  }
  if (error || !rule) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.muted}>
          {error
            ? "We couldn't load this rule. Try again from the report."
            : `Rule ${ruleId} isn't in this report.`}
        </Text>
        <Button label="Back" variant="ghost" onPress={() => router.back()} />
      </SafeAreaView>
    );
  }

  // Heuristic: in v1, beer rules are mostly back-of-label (health
  // warning, name+address). Default to back if available, else front.
  // The bbox is in the coordinate space of one captured image; the
  // backend doesn't tell us which one today, so this is a TODO.
  // TODO(bbox-image-source): backend should return image_id alongside
  // bbox so we can pick the correct surface deterministically.
  const surface = captures.back ? 'back' : 'front';
  const cap = captures[surface];

  return (
    <SafeAreaView style={styles.root} edges={['bottom']}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.headerRow}>
          <Text style={styles.title}>{humanizeRuleId(rule.rule_id)}</Text>
          <StatusBadge status={rule.status} />
        </View>
        <Text style={styles.citation}>{rule.citation}</Text>

        <Section title="Expected">
          <Text style={styles.body}>{rule.expected ?? '—'}</Text>
        </Section>

        <Section title="Found">
          <Text style={styles.body}>{rule.finding ?? '—'}</Text>
        </Section>

        <Section title="Bounding box">
          <BoundingBoxView bbox={rule.bbox} surface={surface} hasImage={Boolean(cap)} />
        </Section>

        {rule.fix_suggestion ? (
          <Section title="How to fix">
            <Text style={styles.body}>{rule.fix_suggestion}</Text>
          </Section>
        ) : null}

        <Button
          label="Flag this result"
          variant="secondary"
          fullWidth
          onPress={() => {
            // TODO(flag): wire POST /v1/scans/:id/rule-results/:rid/flag
            // once a comment input UI is added.
          }}
        />
      </ScrollView>
    </SafeAreaView>
  );
}

function humanizeRuleId(ruleId: string): string {
  // beer.health_warning.exact_text → "Health warning · exact text"
  const parts = ruleId.split('.');
  if (parts.length < 2) return ruleId;
  const [, ...rest] = parts;
  return rest
    .map((p) => p.replace(/_/g, ' '))
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(' · ');
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

function BoundingBoxView({
  bbox,
  surface,
  hasImage,
}: {
  bbox: BBox | null;
  surface: 'front' | 'back';
  hasImage: boolean;
}): React.ReactElement {
  if (!bbox) {
    return <Text style={styles.muted}>No region recorded for this rule.</Text>;
  }
  const [x, y, w, h] = bbox;
  return (
    <View style={styles.bboxCard}>
      <Text style={styles.muted}>
        Surface: <Text style={styles.body}>{surface}</Text>
      </Text>
      <Text style={styles.muted}>
        x={x} y={y} w={w} h={h}
      </Text>
      {!hasImage ? (
        <Text style={styles.muted}>
          Original image not on device — overlay drawing skipped. The
          server-side report still has the region; reload after a
          fresh scan to see the overlay.
        </Text>
      ) : (
        // TODO(bbox-overlay): render the captured image with a scaled
        // bbox overlay (Reanimated / SVG). The scaffold draws a
        // schematic instead so the layout is honest.
        <View style={styles.bboxSchematic}>
          <View
            style={[
              styles.bboxSchematicFrame,
              {
                left: `${(x / Math.max(1, x + w + 1)) * 100}%`,
                top: `${(y / Math.max(1, y + h + 1)) * 100}%`,
                width: `${Math.max(10, w / 4)}%`,
                height: `${Math.max(10, h / 4)}%`,
              },
            ]}
          />
        </View>
      )}
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
    ...typography.heading,
    color: colors.text,
    flex: 1,
  },
  citation: {
    ...typography.caption,
    color: colors.textMuted,
  },
  section: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    gap: spacing.xs,
  },
  sectionTitle: {
    ...typography.heading,
    color: colors.text,
  },
  body: {
    ...typography.body,
    color: colors.text,
  },
  muted: {
    ...typography.body,
    color: colors.textMuted,
  },
  bboxCard: {
    gap: spacing.xs,
  },
  bboxSchematic: {
    aspectRatio: 0.65,
    backgroundColor: colors.surfaceAlt,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.sm,
    marginTop: spacing.sm,
  },
  bboxSchematicFrame: {
    position: 'absolute',
    borderColor: colors.fail,
    borderWidth: 2,
    borderRadius: 2,
  },
});
