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
 *
 * Track C UX surface: per the audit, the backend already returns
 * fix_suggestion, image_quality (+ notes), per-field confidence, and
 * citation strings. We render all of that here so the user gets the
 * full review without leaving the screen, and add Share + a tappable
 * eCFR deep link on each rule.
 */

import React, { useEffect, useMemo, useRef } from 'react';
import {
  Image,
  Linking,
  Pressable,
  RefreshControl,
  ScrollView,
  Share,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import * as Haptics from 'expo-haptics';
import { router, useLocalSearchParams } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withSequence,
  withTiming,
} from 'react-native-reanimated';

import {
  Button,
  CaptureQualityPill,
  ConfidenceBar,
  ErrorState,
  ExternalMatchCard,
  HealthWarningCard,
  isHealthWarningRule,
  RuleResultCard,
  ruleIsSurfacedByHero,
  SectionHeader,
  Skeleton,
  StatusBadge,
} from '@src/components';
import { apiClient } from '@src/api/client';
import { useToast } from '@src/hooks/useToast';
import { queryKeys } from '@src/state/queryClient';
import { useScanStore } from '@src/state/scanStore';
import { colors, radius, scanMotion, spacing, typography } from '@src/theme';
import type {
  FieldSummary,
  ImageQuality,
  OverallStatus,
  ReportResponse,
  RuleResultDTO,
} from '@src/api/types';
import { SafeAreaView } from 'react-native-safe-area-context';

// Field-id → rule-id mapping. The DTO returns extracted fields keyed by
// `field_id` (e.g. "brand_name") and rule_results keyed by `rule_id`
// (e.g. "beer.brand_name.presence"). The rule yaml encodes which field
// each rule reads from; we mirror that here so we can attach the
// per-field confidence bar to the right rule card. Conservative on
// purpose — if a rule_id isn't mapped, we just skip the bar rather than
// guess wrong.
const FIELD_FOR_RULE: Record<string, string> = {
  'beer.brand_name.presence': 'brand_name',
  'beer.class_type.presence': 'class_type',
  'beer.alcohol_content.format': 'alcohol_content',
  'beer.net_contents.presence': 'net_contents',
  'beer.name_address.presence': 'name_address',
  'beer.health_warning.exact_text': 'health_warning',
  'beer.health_warning.presence': 'health_warning',
  'beer.country_of_origin.presence': 'country_of_origin',
};

export default function ReportScreen(): React.ReactElement {
  const { id } = useLocalSearchParams<{ id: string }>();
  const scanId = typeof id === 'string' ? id : '';
  const panorama = useScanStore((s) => s.panorama);
  const { show: showToast } = useToast();

  const { data, isLoading, isRefetching, refetch, error } = useQuery({
    queryKey: queryKeys.report(scanId),
    enabled: scanId.length > 0,
    queryFn: () => apiClient.getReport(scanId),
  });

  // Surface a top-of-screen toast on any failure path that reaches the
  // report screen — covers the /v1/scans/:id/report fetch, which is the
  // mobile client's compliance verifier endpoint (the legacy /v1/verify
  // is web-only). One toast per error transition; we ref-guard so
  // re-render churn from refetch doesn't double-fire.
  const lastErrorIdRef = useRef<unknown>(null);
  useEffect(() => {
    if (error && error !== lastErrorIdRef.current) {
      lastErrorIdRef.current = error;
      showToast({
        variant: 'error',
        message: "Couldn't reach the verifier. Tap to retry.",
      });
    }
    if (!error) {
      lastErrorIdRef.current = null;
    }
  }, [error, showToast]);

  // If the backend response carries a `cached=true` field (durable L3
  // cache short-circuit), surface a quick info toast so the user knows
  // they're looking at history-derived data rather than a fresh
  // verification. Indexed by scanId so we only fire once per report
  // load. The field is currently optional on the DTO (older API rows
  // may not include it); the truthy check covers both shapes.
  const cachedToastFiredRef = useRef<string | null>(null);
  useEffect(() => {
    if (!data) return;
    const isCached = (data as ReportResponse & { cached?: boolean }).cached;
    if (isCached && cachedToastFiredRef.current !== scanId) {
      cachedToastFiredRef.current = scanId;
      showToast({
        variant: 'info',
        message: 'Loaded from history.',
      });
    }
  }, [data, scanId, showToast]);

  // Outcome haptic: success notification on overall=pass, warning on
  // fail/advisory. Fires once per scanId — refetches don't re-trigger.
  // expo-haptics is a JS-thread call; the native side handles the
  // platform-correct pattern (taptic engine on iOS, vibration on
  // Android) so we don't have to fan out by Platform.OS here.
  const hapticFiredRef = useRef<string | null>(null);
  useEffect(() => {
    if (!data || !scanId) return;
    if (hapticFiredRef.current === scanId) return;
    hapticFiredRef.current = scanId;
    const fire = async () => {
      try {
        if (data.overall === 'pass') {
          await Haptics.notificationAsync(
            Haptics.NotificationFeedbackType.Success,
          );
        } else {
          // Advisory and fail share the same warning haptic — both are
          // "something needs your attention" rather than a clean win.
          await Haptics.notificationAsync(
            Haptics.NotificationFeedbackType.Warning,
          );
        }
      } catch {
        // Older Android devices can throw on notificationAsync if the
        // pattern isn't supported. Swallow — haptics is enhancement, not
        // load-bearing.
      }
    };
    void fire();
  }, [data, scanId]);

  if (isLoading) {
    return (
      <SafeAreaView style={styles.root} edges={['bottom']}>
        <ScrollView contentContainerStyle={styles.scrollContent}>
          {/* Skeleton stack mirrors the real report shape: header card,
              hero card, capture thumbnail, two rule sections. Lines up
              row heights with the live screen so the layout doesn't pop
              when data lands. */}
          <Skeleton width="100%" height={120} radius={radius.md} />
          <Skeleton width="100%" height={180} radius={radius.lg} />
          <Skeleton width={120} height={22} radius={4} />
          <Skeleton width="100%" height={140} radius={radius.md} />
          <Skeleton width={120} height={22} radius={4} />
          <Skeleton width="100%" height={72} radius={radius.md} />
          <Skeleton width="100%" height={72} radius={radius.md} />
        </ScrollView>
      </SafeAreaView>
    );
  }

  if (error || !data) {
    return (
      <SafeAreaView style={styles.errorWrap} edges={['bottom']}>
        <ErrorState
          title="Report unavailable"
          description="We couldn't load this report. Check your connection and try again."
          retry={() => void refetch()}
          secondaryAction={{
            label: 'Back to home',
            onPress: () => router.replace('/(app)/home'),
          }}
        />
      </SafeAreaView>
    );
  }

  const grouped = groupByStatus(data.rule_results);

  // Find the health-warning exact-text rule so the hero card can render
  // the verbatim comparison. The rule may be absent (older reports, or
  // backend versions where the rule wasn't evaluated); the card handles
  // that as the `unverified` state.
  const healthWarningRule = data.rule_results.find(isHealthWarningRule) ?? null;

  const onShare = async () => {
    try {
      await Share.share({ message: composeShareText(data) });
    } catch {
      // Share sheet dismissed / failed — non-fatal.
    }
  };

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
        {/* TTB COLA reverse-image-lookup match — only renders when the
            backend supplies one. Sits above the verdict header so the
            user sees the regulatory anchor before the rule breakdown. */}
        {data.external_match ? <ExternalMatchCard match={data.external_match} /> : null}
        <HeaderCard data={data} grouped={grouped} />

        {/* Government Warning hero card — surfaced ABOVE the rule list
            because §16.21 is the highest-stakes label rule. The
            corresponding `health_warning.exact_text` rule still appears
            in the rule list below; ruleIsSurfacedByHero() lets the rule
            list flag duplication if the design ever wants to dim the
            row. */}
        <HealthWarningCard rule={healthWarningRule} />

        {/* Single panorama thumbnail — captures the entire label in one
            unrolled image. The deleted /(app)/scan/captures/[surface]
            route is gone; rule bboxes are viewed via the rule detail
            screen, which draws them on the panorama directly. */}
        <SectionHeader title="Capture" />
        <View style={styles.panoramaThumb}>
          {panorama ? (
            <Image
              source={{ uri: panorama.uri }}
              style={styles.panoramaImage}
              resizeMode="cover"
            />
          ) : (
            <View style={styles.panoramaMissing}>
              <Text style={styles.imageMissing}>not local</Text>
            </View>
          )}
        </View>
        <CaptureQualityPill
          quality={normalizeQuality(data.image_quality)}
          notes={data.image_quality_notes}
        />

        {grouped.fail.length > 0 && (
          <Section
            title="Failed"
            results={grouped.fail}
            scanId={scanId}
            fields={data.fields_summary}
          />
        )}
        {grouped.advisory.length > 0 && (
          <Section
            title="Advisory"
            results={grouped.advisory}
            scanId={scanId}
            fields={data.fields_summary}
          />
        )}
        {grouped.pass.length > 0 && (
          <Section
            title="Passing"
            results={grouped.pass}
            scanId={scanId}
            fields={data.fields_summary}
          />
        )}

        <View style={styles.actions}>
          <Button label="Share" variant="secondary" fullWidth onPress={onShare} />
          <Button
            label="Rescan"
            variant="secondary"
            fullWidth
            onPress={() => router.replace('/(app)/scan/setup')}
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

/**
 * Animated header. `pass` gets a brief scale-in on entrance (0.96 → 1.0
 * over ~320ms) and a one-shot green pulse on the ✓ glyph; non-pass
 * states render statically so the celebration doesn't fire when the
 * label has issues.
 */
function HeaderCard({
  data,
  grouped,
}: {
  data: ReportResponse;
  grouped: ReturnType<typeof groupByStatus>;
}): React.ReactElement {
  const isPass = data.overall === 'pass';
  const cardScale = useSharedValue<number>(isPass ? 0.96 : 1);
  const glowOpacity = useSharedValue<number>(0);

  useEffect(() => {
    if (!isPass) return;
    cardScale.value = withTiming(1, { duration: scanMotion.midEase.duration });
    glowOpacity.value = withDelay(
      120,
      withSequence(
        withTiming(1, { duration: 220 }),
        withTiming(0, { duration: 480 })
      )
    );
  }, [isPass, cardScale, glowOpacity]);

  const cardAnim = useAnimatedStyle(() => ({
    transform: [{ scale: cardScale.value }],
  }));
  const glowAnim = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
  }));

  return (
    <Animated.View style={[styles.headerCard, cardAnim]}>
      <View style={styles.headerRow}>
        <View style={styles.headerTitleRow}>
          <View style={styles.overallIconWrap}>
            <OverallIcon status={data.overall} />
            {isPass ? <Animated.View style={[styles.passGlow, glowAnim]} /> : null}
          </View>
          <Text style={styles.headerTitle}>Overall</Text>
        </View>
        <StatusBadge status={data.overall} />
      </View>
      <Text style={styles.headerCaption}>
        {data.rule_results.length} rule
        {data.rule_results.length === 1 ? '' : 's'} evaluated
      </Text>
      <View style={styles.headerCounts}>
        <CountChip count={grouped.fail.length} label="failed" color={colors.fail} />
        <Text style={styles.countSep}>·</Text>
        <CountChip
          count={grouped.advisory.length}
          label="advisory"
          color={colors.advisory}
        />
        <Text style={styles.countSep}>·</Text>
        <CountChip count={grouped.pass.length} label="passing" color={colors.pass} />
      </View>
    </Animated.View>
  );
}

function OverallIcon({ status }: { status: OverallStatus }): React.ReactElement {
  const palette = overallPalette(status);
  return (
    <View
      accessibilityLabel={`Overall ${status}`}
      style={[
        styles.overallIcon,
        { backgroundColor: palette.bg, borderColor: palette.border },
      ]}
    >
      <Text style={[styles.overallGlyph, { color: palette.fg }]}>{palette.glyph}</Text>
    </View>
  );
}

function overallPalette(status: OverallStatus): {
  fg: string;
  bg: string;
  border: string;
  glyph: string;
} {
  switch (status) {
    case 'pass':
      return {
        fg: colors.pass,
        bg: 'rgba(61,220,151,0.15)',
        border: colors.pass,
        glyph: '✓',
      };
    case 'fail':
      return {
        fg: colors.fail,
        bg: 'rgba(255,107,107,0.15)',
        border: colors.fail,
        glyph: '✕',
      };
    case 'advisory':
      return {
        fg: colors.advisory,
        bg: 'rgba(244,184,96,0.15)',
        border: colors.advisory,
        glyph: '!',
      };
  }
}

function CountChip({
  count,
  label,
  color,
}: {
  count: number;
  label: string;
  color: string;
}): React.ReactElement {
  return (
    <Text style={styles.countText}>
      <Text style={[styles.countNumber, { color }]}>{count}</Text>
      <Text style={styles.countLabel}> {label}</Text>
    </Text>
  );
}

function Section({
  title,
  results,
  scanId,
  fields,
}: {
  title: string;
  results: ReadonlyArray<RuleResultDTO>;
  scanId: string;
  fields: ReportResponse['fields_summary'];
}): React.ReactElement {
  return (
    <View style={{ gap: spacing.sm }}>
      <SectionHeader title={title} />
      {results.map((r) => {
        const fieldId = FIELD_FOR_RULE[r.rule_id];
        const summary =
          fieldId && fields ? (fields[fieldId] as FieldSummary | undefined) : undefined;
        const confidence =
          summary && typeof summary.confidence === 'number' ? summary.confidence : null;
        const ecfrUrl = ecfrUrlForCitation(r.citation);
        return (
          <Pressable
            key={r.rule_id}
            onPress={() =>
              router.push({
                pathname: '/(app)/scan/rule/[ruleId]',
                params: { ruleId: r.rule_id, scanId },
              })
            }
          >
            <View style={{ gap: spacing.xs }}>
              <RuleResultCard result={r} />
              {ruleIsSurfacedByHero(r) ? (
                <Text style={styles.heroAnnotation}>
                  Also shown above in the Government Warning card
                </Text>
              ) : null}
              {(confidence != null && confidence > 0) || ecfrUrl ? (
                <View style={styles.ruleMeta}>
                  {confidence != null && confidence > 0 ? (
                    <View style={styles.metaItem}>
                      <Text style={styles.metaLabel}>Extraction</Text>
                      <ConfidenceBar value={confidence} />
                    </View>
                  ) : null}
                  {ecfrUrl ? (
                    <Pressable
                      onPress={() => {
                        void Linking.openURL(ecfrUrl);
                      }}
                      hitSlop={8}
                      accessibilityRole="link"
                      accessibilityLabel={`Open ${r.citation} on eCFR`}
                      style={({ pressed }) => [
                        styles.citationLink,
                        pressed && { opacity: 0.7 },
                      ]}
                    >
                      <Text style={styles.citationLinkText}>{r.citation}</Text>
                      <Text style={styles.citationLinkIcon}>↗</Text>
                    </Pressable>
                  ) : null}
                </View>
              ) : null}
            </View>
          </Pressable>
        );
      })}
    </View>
  );
}

function groupByStatus(results: ReadonlyArray<RuleResultDTO>): {
  pass: RuleResultDTO[];
  fail: RuleResultDTO[];
  advisory: RuleResultDTO[];
} {
  const out = {
    pass: [] as RuleResultDTO[],
    fail: [] as RuleResultDTO[],
    advisory: [] as RuleResultDTO[],
  };
  for (const r of results) out[r.status].push(r);
  return out;
}

function normalizeQuality(q: string | null | undefined): ImageQuality {
  const v = (q ?? '').toLowerCase();
  if (v === 'good' || v === 'fair' || v === 'poor') return v;
  // Backend can also return statuses outside the simple trio (older rows
  // / different extractor profiles). Treat unknown as "fair" so the pill
  // still renders something rather than crashing the whole header.
  return 'fair';
}

/**
 * Map a CFR citation string (e.g. "27 CFR 7.62(a)") to an eCFR.gov URL.
 * Returns null when the citation is empty or doesn't parse — call sites
 * skip rendering the link in that case rather than fall back to a
 * search URL that might point somewhere unrelated.
 *
 * Format: https://www.ecfr.gov/current/title-{title}/section-{section}
 * Subsection refs ("(a)" etc.) are dropped for the URL — eCFR's
 * canonical section pages cover the whole section in-page.
 */
export function ecfrUrlForCitation(citation: string | null | undefined): string | null {
  if (!citation) return null;
  // Match e.g. "27 CFR 7.62" or "27 CFR §7.62" with optional "(a)" / ".(b)" trail.
  const m = citation.match(/(\d+)\s*CFR\s*§?\s*([\d.]+)/i);
  if (!m) return null;
  const title = m[1];
  // Strip trailing dots so "7.62." doesn't break the URL.
  const section = m[2].replace(/\.+$/, '');
  if (!title || !section) return null;
  return `https://www.ecfr.gov/current/title-${title}/section-${section}`;
}

/**
 * Compose plain-text Share.share() payload from the report. Truncates
 * to ~2000 chars by chopping the lowest-priority section last (advisory
 * before failed) and trimming citations if needed. Mirrors the
 * structure called out in the Track C brief.
 */
export function composeShareText(data: ReportResponse): string {
  const grouped = groupByStatus(data.rule_results);
  const date = new Date().toLocaleDateString();
  const brand =
    typeof (data.fields_summary?.brand_name as FieldSummary | undefined)?.value === 'string'
      ? ((data.fields_summary.brand_name as FieldSummary).value as string)
      : 'Beer label';
  const issueCount = grouped.fail.length + grouped.advisory.length;

  const ruleLine = (r: RuleResultDTO) => {
    const title = humanizeRuleTitle(r.rule_id);
    const fix = r.fix_suggestion ? ` — ${r.fix_suggestion}` : '';
    const citation = r.citation ? `\n  (${r.citation})` : '';
    return `• ${title}${fix}${citation}`;
  };

  const sections: string[] = [];
  sections.push(`ProofRead Compliance Review — ${brand} ${date}`);
  sections.push(
    `Overall: ${data.overall.toUpperCase()} (${issueCount} issue${issueCount === 1 ? '' : 's'})`
  );
  if (grouped.fail.length > 0) {
    sections.push(`\nFAILED:\n${grouped.fail.map(ruleLine).join('\n')}`);
  }
  if (grouped.advisory.length > 0) {
    sections.push(`\nADVISORY:\n${grouped.advisory.map(ruleLine).join('\n')}`);
  }
  sections.push('\nReviewed via ProofRead');

  let out = sections.join('\n');
  // Soft 2000-char budget: drop citations first, then advisory, then
  // truncate. Keeps the failure list — the most important payload —
  // intact for as long as possible.
  if (out.length > 2000) {
    out = out.replace(/\n  \([^)]+\)/g, '');
  }
  if (out.length > 2000) {
    out = out.split('\nADVISORY:')[0] + '\n\nReviewed via ProofRead';
  }
  if (out.length > 2000) {
    out = out.slice(0, 1990) + '…';
  }
  return out;
}

function humanizeRuleTitle(ruleId: string): string {
  const parts = ruleId.split('.');
  if (parts.length < 2) return ruleId;
  const [, ...rest] = parts;
  return rest
    .map((p) => p.replace(/_/g, ' '))
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(' · ');
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
  headerTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    flex: 1,
  },
  headerTitle: {
    ...typography.titleMd,
    color: colors.text,
  },
  headerCaption: {
    ...typography.caption,
    color: colors.textMuted,
  },
  headerCounts: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: spacing.xs,
  },
  overallIconWrap: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  overallIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  passGlow: {
    position: 'absolute',
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.pass,
    opacity: 0,
  },
  overallGlyph: {
    fontSize: 18,
    fontWeight: '700',
    lineHeight: 20,
  },
  countText: {
    ...typography.body,
  },
  countNumber: {
    fontWeight: '700',
  },
  countLabel: {
    color: colors.textMuted,
  },
  countSep: {
    ...typography.body,
    color: colors.textMuted,
  },
  panoramaThumb: {
    aspectRatio: 3,
    backgroundColor: colors.surfaceAlt,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  panoramaImage: {
    width: '100%',
    height: '100%',
  },
  panoramaMissing: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageMissing: {
    ...typography.caption,
    color: colors.textMuted,
  },
  ruleMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: spacing.md,
    paddingHorizontal: spacing.md,
    paddingTop: spacing.xs,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  metaLabel: {
    ...typography.caption,
    color: colors.textMuted,
  },
  citationLink: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  citationLinkText: {
    ...typography.caption,
    color: colors.primary,
    textDecorationLine: 'underline',
  },
  citationLinkIcon: {
    ...typography.caption,
    color: colors.primary,
  },
  heroAnnotation: {
    ...typography.caption,
    color: colors.textMuted,
    fontStyle: 'italic',
    paddingHorizontal: spacing.md,
  },
  actions: {
    gap: spacing.sm,
    marginTop: spacing.md,
  },
  errorWrap: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.lg,
  },
});
