/**
 * Rule detail — bounding box + expected vs found + citation + fix.
 *
 * SPEC §v1.7 row "Rule detail".
 *
 * Routed with both ruleId and scanId in the params object so we can
 * fetch the full report and pick out this rule's result. Backend has
 * no per-rule endpoint today; we filter client-side from the cached
 * report query.
 *
 * Per ARCH §6/§7, the bbox is drawn on the single panorama image —
 * there is no per-surface fan-out anymore. BBoxOverlay accepts
 * `image: { uri, width, height }` and computes letterbox math from
 * the intrinsic dimensions, so the panorama (a wide aspect ratio)
 * just produces side-letterboxing instead of the previous
 * portrait-letterboxing.
 */

import React, { useMemo } from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery } from '@tanstack/react-query';

import { BBoxOverlay, Button, StatusBadge } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { useScanStore } from '@src/state/scanStore';
import type { BBox, RuleResultDTO, RuleStatus } from '@src/api/types';
import type { UnrolledPanorama } from '@src/state/scanStore';
import { colors, radius, spacing, typography } from '@src/theme';

export default function RuleDetailScreen(): React.ReactElement {
  const params = useLocalSearchParams<{ ruleId: string; scanId?: string }>();
  const ruleId = typeof params.ruleId === 'string' ? params.ruleId : '';
  const scanId =
    typeof params.scanId === 'string' && params.scanId.length > 0
      ? params.scanId
      : (useScanStore.getState().scanId ?? '');

  const panorama = useScanStore((s) => s.panorama);

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

  return (
    <SafeAreaView style={styles.root} edges={['bottom']}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.headerRow}>
          <Text style={styles.title}>{humanizeRuleId(rule.rule_id)}</Text>
          <StatusBadge status={rule.status} />
        </View>
        <Text style={styles.citation}>{rule.citation}</Text>

        {rule.status === 'advisory' ? <AdvisoryBanner /> : null}

        <Section title="Expected">
          <Text style={styles.body}>{rule.expected ?? '—'}</Text>
        </Section>

        <Section title="Found">
          <Text style={styles.body}>{rule.finding ?? '—'}</Text>
        </Section>

        <Section title="Bounding box">
          <BoundingBoxView
            bbox={rule.bbox}
            panorama={panorama}
            status={rule.status}
          />
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

function AdvisoryBanner(): React.ReactElement {
  return (
    <View style={styles.advisoryBanner} accessible accessibilityRole="text">
      <Text style={styles.advisoryIcon} accessibilityElementsHidden>
        ⚠
      </Text>
      <View style={styles.advisoryTextCol}>
        <Text style={styles.advisoryTitle}>
          Couldn't verify with confidence — rescan recommended
        </Text>
        <Text style={styles.advisorySubtitle}>
          The label was readable but extraction confidence was low. The
          finding below may be incomplete or off; capturing a sharper,
          straighter shot usually resolves it.
        </Text>
      </View>
    </View>
  );
}

function BoundingBoxView({
  bbox,
  panorama,
  status,
}: {
  bbox: BBox | null;
  panorama: UnrolledPanorama | null;
  status: RuleStatus;
}): React.ReactElement {
  if (!bbox) {
    return <Text style={styles.muted}>No region recorded for this rule.</Text>;
  }
  const [x, y, w, h] = bbox;
  // Override BBoxOverlay's portrait default with the panorama's wide
  // aspect ratio so the rendered image isn't squished to 0.65:1.
  const overlayStyle = panorama
    ? { aspectRatio: panorama.width / panorama.height }
    : undefined;
  return (
    <View style={styles.bboxCard}>
      <Text style={styles.muted}>
        x={x} y={y} w={w} h={h}
      </Text>
      {!panorama ? (
        <Text style={styles.muted}>
          Original panorama not on device — overlay drawing skipped. The
          server-side report still has the region; reload after a fresh
          scan to see the overlay.
        </Text>
      ) : (
        <BBoxOverlay
          image={panorama}
          items={[{ bbox, status }]}
          style={[styles.bboxOverlay, overlayStyle]}
        />
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
  bboxOverlay: {
    marginTop: spacing.sm,
  },
  advisoryBanner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    padding: spacing.md,
    backgroundColor: 'rgba(244,184,96,0.12)',
    borderRadius: radius.md,
    borderLeftWidth: 3,
    borderLeftColor: colors.advisory,
    borderWidth: 1,
    borderColor: colors.advisory,
  },
  advisoryIcon: {
    ...typography.heading,
    color: colors.advisory,
    fontWeight: '700',
    lineHeight: 22,
  },
  advisoryTextCol: {
    flex: 1,
    gap: spacing.xs,
  },
  advisoryTitle: {
    ...typography.heading,
    color: colors.advisory,
  },
  advisorySubtitle: {
    ...typography.body,
    color: colors.text,
  },
});
