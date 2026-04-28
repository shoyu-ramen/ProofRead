/**
 * Processing screen — staged progress + cancel.
 *
 * SPEC §v1.6 step 7. Polls GET /v1/scans/:id with tanstack-query at
 * a short interval until status === 'complete' (or 'failed'), then
 * routes to the report.
 *
 * The backend's process_scan() runs synchronously in v1, so by the
 * time the finalize call returns the scan should already be complete.
 * We still poll to handle the v2 async case and to give the UI time
 * to render the processing animation.
 *
 * The backend `status` field is coarse — just uploading | processing |
 * complete | failed. To give users a more legible sense of what's
 * happening, we synthesize sub-stages (OCR → field extraction → rule
 * engine) over the long `processing` window on a wall-clock timer.
 * These transitions are estimates, not real backend signals; the UI
 * surfaces that honestly via an "estimating" caption.
 */

import React, { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { useQuery } from '@tanstack/react-query';

import { Button, Screen, SectionHeader } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import type { ScanStatus } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

// Stage identifiers for the staged checklist. Order matters — index
// is used to compare current-vs-target stage.
type StageId = 'uploading' | 'ocr' | 'extract' | 'rules' | 'done';

interface Stage {
  id: StageId;
  label: string;
  // Whether this stage maps 1:1 to a real backend status, or is
  // synthesized on the client. Used to show the "estimating" caption.
  synthesized: boolean;
}

const STAGES: ReadonlyArray<Stage> = [
  { id: 'uploading', label: 'Uploading images', synthesized: false },
  { id: 'ocr', label: 'OCR (Google Vision)', synthesized: true },
  { id: 'extract', label: 'Extracting fields', synthesized: true },
  { id: 'rules', label: 'Running TTB rule engine', synthesized: true },
  { id: 'done', label: 'Done', synthesized: false },
];

// Rough wall-clock spacing used to step through the synthesized
// sub-stages while the backend sits on `processing`. Real timing is
// not on the wire today (would need per-stage status from §v2.7).
const SUB_STAGE_INTERVAL_MS = 1_200;

export default function ProcessingScreen(): React.ReactElement {
  const { id } = useLocalSearchParams<{ id: string }>();
  const scanId = typeof id === 'string' ? id : '';

  const { data, error, isFetching } = useQuery({
    queryKey: queryKeys.scan(scanId),
    enabled: scanId.length > 0,
    queryFn: () => apiClient.getScan(scanId),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      if (status === 'complete' || status === 'failed') return false;
      // SPEC §v1.10: poll until complete. 1s is plenty for the v1
      // synchronous backend; v2 will move to websockets per §v2.7.
      return 1_000;
    },
  });

  // Tick a synthesized sub-stage forward while the backend is in the
  // long `processing` state. Reset whenever processing starts so each
  // visit to the screen plays the animation from the top.
  const [subStageIdx, setSubStageIdx] = useState(0);
  useEffect(() => {
    if (data?.status !== 'processing') {
      setSubStageIdx(0);
      return;
    }
    const t = setInterval(() => {
      setSubStageIdx((i) => Math.min(i + 1, 2)); // 0:ocr, 1:extract, 2:rules
    }, SUB_STAGE_INTERVAL_MS);
    return () => clearInterval(t);
  }, [data?.status]);

  useEffect(() => {
    if (data?.status === 'complete') {
      router.replace(`/(app)/scan/report/${scanId}`);
    }
    if (data?.status === 'failed') {
      Alert.alert('Scan failed', 'Processing failed. Please retry.', [
        { text: 'OK', onPress: () => router.replace('/(app)/home') },
      ]);
    }
  }, [data?.status, scanId]);

  const currentStageIdx = useMemo(
    () => stageIndexFor(data?.status, subStageIdx),
    [data?.status, subStageIdx],
  );

  const isFailed = data?.status === 'failed';
  const currentStage = STAGES[currentStageIdx];
  const synthesizing =
    !isFailed && data?.status === 'processing' && currentStage?.synthesized === true;

  return (
    <Screen>
      <SectionHeader
        title="Analyzing"
        subtitle="OCR, field extraction, and TTB rule checks."
      />

      <View style={styles.stageList}>
        {STAGES.map((stage, idx) => (
          <StageRow
            key={stage.id}
            label={stage.label}
            state={
              isFailed
                ? idx < currentStageIdx
                  ? 'done'
                  : idx === currentStageIdx
                  ? 'failed'
                  : 'pending'
                : idx < currentStageIdx
                ? 'done'
                : idx === currentStageIdx
                ? 'active'
                : 'pending'
            }
          />
        ))}
      </View>

      <View style={styles.captionBlock}>
        {error ? (
          <Text style={styles.errorCaption}>
            Couldn't reach the server. Retrying…
          </Text>
        ) : null}
        {synthesizing ? (
          <Text style={styles.estimateCaption}>
            Sub-stage timing is estimated — backend reports a single
            "processing" status today.
          </Text>
        ) : null}
        <Text style={styles.scanId}>Scan {scanId.slice(0, 8)}…</Text>
      </View>

      <View style={{ flex: 1 }} />
      <Button
        label="Cancel"
        variant="secondary"
        onPress={() => {
          // TODO(cancel): backend has no cancel endpoint in v1; this just
          // navigates away and abandons polling. The scan continues
          // server-side and surfaces in history.
          router.replace('/(app)/home');
        }}
        // While submitting we don't want the user to bail mid-upload —
        // but once we're polling, cancel is fine.
        disabled={isFetching && !data}
      />
    </Screen>
  );
}

type StageState = 'done' | 'active' | 'pending' | 'failed';

function StageRow({
  label,
  state,
}: {
  label: string;
  state: StageState;
}): React.ReactElement {
  return (
    <View style={styles.stageRow} accessibilityRole="text">
      <View style={[styles.indicator, indicatorStyleFor(state)]}>
        {state === 'active' ? (
          <ActivityIndicator size="small" color={colors.primary} />
        ) : state === 'done' ? (
          <Text style={styles.indicatorGlyph}>✓</Text>
        ) : state === 'failed' ? (
          <Text style={[styles.indicatorGlyph, { color: colors.fail }]}>!</Text>
        ) : null}
      </View>
      <Text
        style={[
          styles.stageLabel,
          state === 'pending' && styles.stageLabelPending,
          state === 'active' && styles.stageLabelActive,
          state === 'failed' && styles.stageLabelFailed,
        ]}
      >
        {label}
      </Text>
    </View>
  );
}

function indicatorStyleFor(state: StageState) {
  switch (state) {
    case 'done':
      return {
        backgroundColor: 'rgba(61,220,151,0.15)',
        borderColor: colors.pass,
      };
    case 'active':
      return {
        backgroundColor: 'rgba(110,168,254,0.12)',
        borderColor: colors.primary,
      };
    case 'failed':
      return {
        backgroundColor: 'rgba(255,107,107,0.12)',
        borderColor: colors.fail,
      };
    case 'pending':
      return {
        backgroundColor: colors.surfaceAlt,
        borderColor: colors.border,
      };
  }
}

function stageIndexFor(status: ScanStatus | undefined, subStageIdx: number): number {
  // Maps the backend's coarse status (and the synthesized sub-stage
  // counter while processing) to an index into STAGES.
  // STAGES = [uploading, ocr, extract, rules, done]
  switch (status) {
    case undefined:
      return 0;
    case 'uploading':
      return 0;
    case 'processing':
      // 0 → ocr (1), 1 → extract (2), 2 → rules (3)
      return 1 + Math.max(0, Math.min(2, subStageIdx));
    case 'complete':
      return 4;
    case 'failed':
      // Park at "rules" so the failure marker lands on the latest
      // pipeline step — we don't know which stage actually failed.
      return 3;
  }
}

const styles = StyleSheet.create({
  stageList: {
    marginTop: spacing.lg,
    gap: spacing.md,
  },
  stageRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  indicator: {
    width: 28,
    height: 28,
    borderRadius: radius.lg,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  indicatorGlyph: {
    ...typography.caption,
    color: colors.pass,
    fontWeight: '700',
  },
  stageLabel: {
    ...typography.body,
    color: colors.text,
    flex: 1,
  },
  stageLabelPending: {
    color: colors.textMuted,
  },
  stageLabelActive: {
    fontWeight: '600',
  },
  stageLabelFailed: {
    color: colors.fail,
  },
  captionBlock: {
    gap: spacing.xs,
    marginTop: spacing.md,
  },
  estimateCaption: {
    ...typography.caption,
    color: colors.textMuted,
    fontStyle: 'italic',
  },
  errorCaption: {
    ...typography.caption,
    color: colors.advisory,
  },
  scanId: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
