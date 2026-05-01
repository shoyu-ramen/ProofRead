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
 *
 * Track C: the citation is rendered as a tappable eCFR.gov link, and
 * the "Flag this result" button now collects a comment via a modal
 * sheet and POSTs to /v1/scans/:id/rule-results/:rid/flag.
 */

import React, { useMemo, useState } from 'react';
import {
  KeyboardAvoidingView,
  Linking,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery } from '@tanstack/react-query';

import {
  BBoxOverlay,
  Button,
  ErrorState,
  Skeleton,
  StatusBadge,
} from '@src/components';
import { apiClient } from '@src/api/client';
import { useToast } from '@src/hooks/useToast';
import { queryKeys } from '@src/state/queryClient';
import { useScanStore } from '@src/state/scanStore';
import type { BBox, RuleResultDTO, RuleStatus } from '@src/api/types';
import type { UnrolledPanorama } from '@src/state/scanStore';
import { colors, radius, spacing, typography } from '@src/theme';
import { ecfrUrlForCitation } from '../report/[id]';

export default function RuleDetailScreen(): React.ReactElement {
  const params = useLocalSearchParams<{ ruleId: string; scanId?: string }>();
  const ruleId = typeof params.ruleId === 'string' ? params.ruleId : '';
  const scanId =
    typeof params.scanId === 'string' && params.scanId.length > 0
      ? params.scanId
      : (useScanStore.getState().scanId ?? '');

  const panorama = useScanStore((s) => s.panorama);
  const { show: showToast } = useToast();

  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.report(scanId),
    enabled: scanId.length > 0,
    queryFn: () => apiClient.getReport(scanId),
  });

  const rule = useMemo<RuleResultDTO | null>(() => {
    if (!data) return null;
    return data.rule_results.find((r) => r.rule_id === ruleId) ?? null;
  }, [data, ruleId]);

  // Flag-as-incorrect modal state. Kept local — submit posts directly
  // and surfaces success/error inline. We intentionally do NOT
  // invalidate the report query: a flagged result still appears in
  // the report list (the backend just records the flag for later
  // operator review per scans.py:532-554).
  const [flagOpen, setFlagOpen] = useState(false);
  const [flagComment, setFlagComment] = useState('');
  const [flagSubmitting, setFlagSubmitting] = useState(false);
  const [flagToast, setFlagToast] = useState<{ kind: 'ok' | 'err'; msg: string } | null>(
    null
  );

  if (isLoading) {
    // Skeleton mirrors the live rule layout: title row (heading + status
    // pill), citation chip, four labeled section cards, and the bbox
    // preview at the bottom. Sized so the layout doesn't pop on land.
    return (
      <SafeAreaView style={styles.root} edges={['bottom']}>
        <ScrollView contentContainerStyle={styles.content}>
          <View style={styles.skeletonHeaderRow}>
            <Skeleton width="60%" height={24} radius={4} />
            <Skeleton width={64} height={22} radius={radius.sm} />
          </View>
          <Skeleton width={140} height={16} radius={4} />
          <Skeleton width="100%" height={88} radius={radius.md} />
          <Skeleton width="100%" height={88} radius={radius.md} />
          <Skeleton width="100%" height={140} radius={radius.md} />
        </ScrollView>
      </SafeAreaView>
    );
  }
  if (error || !rule) {
    return (
      <SafeAreaView style={styles.center}>
        {error ? (
          <ErrorState
            title="Rule unavailable"
            description="We couldn't load this rule. Check your connection and try again."
            retry={() => router.back()}
            retryLabel="Back to report"
          />
        ) : (
          <ErrorState
            title="Rule not found"
            description={`Rule ${ruleId} isn't in this report.`}
            retry={() => router.back()}
            retryLabel="Back"
          />
        )}
      </SafeAreaView>
    );
  }

  const ecfrUrl = ecfrUrlForCitation(rule.citation);

  const submitFlag = async () => {
    if (flagSubmitting) return;
    setFlagSubmitting(true);
    setFlagToast(null);
    try {
      await apiClient.flagRuleResult(scanId, rule.rule_id, {
        comment: flagComment.trim(),
      });
      setFlagOpen(false);
      setFlagComment('');
      // Inline acknowledgement stays for the screen-reader live region;
      // the global toast surface gives sighted users a transient cue
      // that doesn't push the rule body around.
      setFlagToast({ kind: 'ok', msg: 'Thanks — flagged for review.' });
      showToast({
        variant: 'success',
        message: 'Thanks — flagged for review.',
      });
    } catch (e) {
      setFlagToast({
        kind: 'err',
        msg: "Couldn't submit flag. Check connection and try again.",
      });
      showToast({
        variant: 'error',
        message: "Couldn't reach the verifier. Tap to retry.",
      });
    } finally {
      setFlagSubmitting(false);
    }
  };

  return (
    <SafeAreaView style={styles.root} edges={['bottom']}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.headerRow}>
          <Text style={styles.title}>{humanizeRuleId(rule.rule_id)}</Text>
          <StatusBadge status={rule.status} />
        </View>
        {ecfrUrl ? (
          <Pressable
            onPress={() => {
              void Linking.openURL(ecfrUrl);
            }}
            hitSlop={8}
            accessibilityRole="link"
            accessibilityLabel={`Open ${rule.citation} on eCFR`}
            style={({ pressed }) => [
              styles.citationRow,
              pressed && { opacity: 0.7 },
            ]}
          >
            <Text style={styles.citationLink}>{rule.citation}</Text>
            <Text style={styles.citationIcon}>↗</Text>
          </Pressable>
        ) : (
          <Text style={styles.citation}>{rule.citation}</Text>
        )}

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

        {flagToast ? (
          <View
            style={[
              styles.toast,
              flagToast.kind === 'ok' ? styles.toastOk : styles.toastErr,
            ]}
            accessibilityLiveRegion="polite"
          >
            <Text
              style={[
                styles.toastText,
                {
                  color: flagToast.kind === 'ok' ? colors.pass : colors.fail,
                },
              ]}
            >
              {flagToast.msg}
            </Text>
          </View>
        ) : null}

        {/* Flag-as-incorrect — opens a modal sheet to collect a comment.
            The Button primitive picks up the new `interaction` tokens
            (pressed scale + opacity, disabled fade) so the touch feel
            matches the rest of the app. */}
        <Button
          label="Flag as incorrect"
          variant="secondary"
          fullWidth
          onPress={() => setFlagOpen(true)}
        />
      </ScrollView>

      <FlagModal
        visible={flagOpen}
        comment={flagComment}
        onChangeComment={setFlagComment}
        submitting={flagSubmitting}
        onCancel={() => {
          if (flagSubmitting) return;
          setFlagOpen(false);
        }}
        onSubmit={() => void submitFlag()}
      />
    </SafeAreaView>
  );
}

function FlagModal({
  visible,
  comment,
  onChangeComment,
  submitting,
  onCancel,
  onSubmit,
}: {
  visible: boolean;
  comment: string;
  onChangeComment: (s: string) => void;
  submitting: boolean;
  onCancel: () => void;
  onSubmit: () => void;
}): React.ReactElement {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onCancel}
    >
      <KeyboardAvoidingView
        style={styles.modalRoot}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <Pressable style={styles.modalBackdrop} onPress={onCancel} />
        <View style={styles.modalCard}>
          <Text style={styles.modalTitle}>Flag this result</Text>
          <Text style={styles.modalSubtitle}>
            Tell us what's wrong. Our reviewers use these notes to fix the
            extractor and rules.
          </Text>
          <TextInput
            style={styles.modalInput}
            placeholder="What's incorrect about this finding?"
            placeholderTextColor={colors.textMuted}
            value={comment}
            onChangeText={onChangeComment}
            multiline
            numberOfLines={4}
            editable={!submitting}
            textAlignVertical="top"
          />
          <View style={styles.modalActions}>
            <Button
              label="Cancel"
              variant="ghost"
              onPress={onCancel}
              disabled={submitting}
            />
            <Button
              label={submitting ? 'Sending…' : 'Submit'}
              onPress={onSubmit}
              disabled={submitting}
            />
          </View>
        </View>
      </KeyboardAvoidingView>
    </Modal>
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
  skeletonHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.sm,
  },
  title: {
    ...typography.headingLg,
    color: colors.text,
    flex: 1,
  },
  citation: {
    ...typography.caption,
    color: colors.textMuted,
  },
  citationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    alignSelf: 'flex-start',
  },
  citationLink: {
    ...typography.caption,
    color: colors.primary,
    textDecorationLine: 'underline',
  },
  citationIcon: {
    ...typography.caption,
    color: colors.primary,
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
    ...typography.headingSm,
    color: colors.text,
  },
  body: {
    ...typography.bodyMd,
    color: colors.text,
  },
  muted: {
    ...typography.bodyMd,
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
  toast: {
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
  },
  toastOk: {
    backgroundColor: 'rgba(61,220,151,0.10)',
    borderColor: colors.pass,
  },
  toastErr: {
    backgroundColor: 'rgba(255,107,107,0.10)',
    borderColor: colors.fail,
  },
  toastText: {
    ...typography.body,
  },
  modalRoot: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  modalCard: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: radius.xl,
    borderTopRightRadius: radius.xl,
    padding: spacing.lg,
    gap: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  modalTitle: {
    ...typography.title,
    color: colors.text,
  },
  modalSubtitle: {
    ...typography.body,
    color: colors.textMuted,
  },
  modalInput: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surfaceAlt,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    minHeight: 96,
  },
  modalActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: spacing.sm,
  },
});
