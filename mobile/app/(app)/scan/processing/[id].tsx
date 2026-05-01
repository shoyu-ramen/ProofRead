/**
 * Processing screen — indeterminate progress while the backend processes
 * the scan.
 *
 * SPEC §v1.6 step 7. Polls GET /v1/scans/:id with tanstack-query at a
 * short interval until status === 'complete' (or 'failed'), then routes
 * to the report.
 *
 * The backend reports a single coarse `processing` status today, but the
 * elapsed timer drives a synthesized 3-phase indicator
 * ("Reading text" → "Extracting fields" → "Checking compliance") so the
 * user has something to read while we wait. The phases are pure UI —
 * the backend has no concept of them.
 *
 * On `failed`, an inline failure card replaces the spinner with three
 * concrete actions ("Try again", "Rescan label", "Back to home"). The
 * panorama and scan_id are preserved in the Zustand store so "Try
 * again" can re-enter /scan/review and re-run the upload pipeline.
 */

import React, { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { useQuery } from '@tanstack/react-query';

import {
  Button,
  ErrorState,
  ProgressBar,
  Screen,
  SectionHeader,
} from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { colors, radius, spacing, typography } from '@src/theme';

interface ProcessingPhase {
  key: 'reading' | 'extracting' | 'compliance';
  label: string;
  // Inclusive lower bound (seconds) at which this phase becomes active.
  startSec: number;
  // Exclusive upper bound. `null` for the final phase (no upper bound).
  endSec: number | null;
}

const PHASES: ProcessingPhase[] = [
  { key: 'reading', label: 'Reading text', startSec: 0, endSec: 6 },
  { key: 'extracting', label: 'Extracting fields', startSec: 6, endSec: 14 },
  { key: 'compliance', label: 'Checking compliance', startSec: 14, endSec: null },
];

const FAILURE_DEFAULT_MESSAGE =
  'Analysis failed. The scan may not have been clear enough.';

function phaseFor(elapsedSec: number): { phase: ProcessingPhase; index: number } {
  for (let i = 0; i < PHASES.length; i += 1) {
    const p = PHASES[i]!;
    if (p.endSec === null || elapsedSec < p.endSec) {
      return { phase: p, index: i };
    }
  }
  // Defensive fallback — PHASES always ends with an open-ended phase.
  return { phase: PHASES[PHASES.length - 1]!, index: PHASES.length - 1 };
}

/**
 * Translate the elapsed seconds into a 0..1 progress value across the
 * full phase schedule. The final phase has no upper bound, so we stop
 * advancing the bar at the boundary into "Checking compliance" and let
 * the bar sit at 100% with the indeterminate spinner doing the talking
 * after that. Backend has no real progress signal, so this is a
 * synthesized hint, not a promise.
 */
function progressFor(elapsedSec: number): number {
  const last = PHASES[PHASES.length - 1]!;
  const total = last.startSec; // 14s in the current schedule
  if (total <= 0) return 0;
  return Math.max(0, Math.min(1, elapsedSec / total));
}

export default function ProcessingScreen(): React.ReactElement {
  const { id } = useLocalSearchParams<{ id: string }>();
  const scanId = typeof id === 'string' ? id : '';

  const { data, error } = useQuery({
    queryKey: queryKeys.scan(scanId),
    enabled: scanId.length > 0,
    queryFn: () => apiClient.getScan(scanId),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      if (status === 'complete' || status === 'failed') return false;
      return 1_000;
    },
  });

  // Tick a wall-clock counter for the phase indicator. Real time is
  // real information, even if the phase boundaries themselves are
  // synthesized.
  const [elapsedSec, setElapsedSec] = useState(0);
  useEffect(() => {
    if (data?.status === 'complete' || data?.status === 'failed') return;
    const t = setInterval(() => setElapsedSec((n) => n + 1), 1_000);
    return () => clearInterval(t);
  }, [data?.status]);

  useEffect(() => {
    if (data?.status === 'complete') {
      router.replace(`/(app)/scan/report/${scanId}`);
    }
    // No auto-navigation on `failed`; the inline failure card below
    // owns the recovery flow.
  }, [data?.status, scanId]);

  const { phase, index: phaseIndex } = useMemo(
    () => phaseFor(elapsedSec),
    [elapsedSec],
  );
  const progress = useMemo(() => progressFor(elapsedSec), [elapsedSec]);

  const isFailed = data?.status === 'failed';
  // ScanStatusResponse currently has no `reason` field, but be
  // forgiving in case the backend grows one before this code is
  // updated again.
  const failureMessage =
    (isFailed &&
      typeof (data as unknown as { reason?: unknown })?.reason === 'string' &&
      ((data as unknown as { reason: string }).reason.trim() || null)) ||
    FAILURE_DEFAULT_MESSAGE;

  return (
    <Screen>
      <SectionHeader
        title={isFailed ? 'Scan failed' : 'Analyzing'}
        subtitle={
          isFailed
            ? "We couldn't finish processing this label."
            : 'OCR, field extraction, and TTB rule checks.'
        }
      />

      {isFailed ? (
        <View>
          <ErrorState
            title="Analysis didn't complete"
            description={failureMessage}
            retry={() => {
              // The panorama + scan_id remain in the Zustand store;
              // the review screen will re-run createScan + upload +
              // finalize on mount.
              router.replace('/(app)/scan/review');
            }}
            retryLabel="Try again"
            secondaryAction={{
              label: 'Rescan label',
              onPress: () => router.replace('/(app)/scan/setup'),
            }}
          />
          <Button
            label="Back to home"
            variant="ghost"
            fullWidth
            onPress={() => router.replace('/(app)/home')}
          />
        </View>
      ) : (
        <View style={styles.progressCard}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.phaseLabel}>{phase.label}</Text>
          <ProgressBar value={progress} style={styles.progressBar} />
          <Text style={styles.elapsedText}>
            Step {phaseIndex + 1} of {PHASES.length} · {elapsedSec}s
          </Text>
        </View>
      )}

      {!isFailed ? (
        <View style={styles.captionBlock}>
          {error ? (
            <Text style={styles.errorCaption}>
              Couldn't reach the server. Retrying…
            </Text>
          ) : null}
          <Text style={styles.scanId}>Scan {scanId.slice(0, 8)}…</Text>
        </View>
      ) : (
        <View style={styles.captionBlock}>
          <Text style={styles.scanId}>Scan {scanId.slice(0, 8)}…</Text>
        </View>
      )}

      <View style={{ flex: 1 }} />
      {!isFailed ? (
        <Button
          label="Run in background"
          variant="secondary"
          onPress={() => {
            // Backend has no cancel endpoint in v1; leaving abandons
            // polling but the scan still lands in history.
            router.replace('/(app)/home');
          }}
        />
      ) : null}
    </Screen>
  );
}

const styles = StyleSheet.create({
  progressCard: {
    marginTop: spacing.lg,
    padding: spacing.xl,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.md,
  },
  phaseLabel: {
    ...typography.headingMd,
    color: colors.text,
    textAlign: 'center',
  },
  progressBar: {
    alignSelf: 'stretch',
  },
  elapsedText: {
    ...typography.caption,
    color: colors.textMuted,
  },
  captionBlock: {
    gap: spacing.xs,
    marginTop: spacing.md,
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
