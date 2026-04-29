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

import { BBoxOverlay, Button, StatusBadge } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { surfaceForPanel, useScanStore } from '@src/state/scanStore';
import type { BBox, RuleResultDTO, RuleStatus } from '@src/api/types';
import type { CapturedImage } from '@src/state/scanStore';
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

  // Pick the capture this rule's bbox lives in. Backend rule_results
  // carry a `surface` panel id (e.g. "panel_0"); surfaceForPanel maps
  // it to the local capture slot. v1 only captures front + back — if
  // the backend ever points at `side`/`neck` we fall back to the
  // legacy heuristic ("back if we have one, else front"). Same fallback
  // covers the still-rolling /v1/scans/:id/report endpoint which
  // hasn't shipped `surface` yet (verify.py emits it; scans.py is
  // following). Once scans.py ships and v1 still only captures two
  // surfaces, the fallback can shrink to just the side/neck guard.
  const mapped = surfaceForPanel(rule.surface);
  const surface: 'front' | 'back' =
    mapped === 'front' || mapped === 'back'
      ? mapped
      : captures.back
      ? 'back'
      : 'front';
  const cap = captures[surface];

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
            surface={surface}
            capture={cap}
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
  surface,
  capture,
  status,
}: {
  bbox: BBox | null;
  surface: 'front' | 'back';
  capture: CapturedImage | undefined;
  status: RuleStatus;
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
      {!capture ? (
        <Text style={styles.muted}>
          Original image not on device — overlay drawing skipped. The
          server-side report still has the region; reload after a
          fresh scan to see the overlay.
        </Text>
      ) : (
        <BBoxOverlay
          image={capture}
          items={[{ bbox, status }]}
          style={styles.bboxOverlay}
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
