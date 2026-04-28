/**
 * Processing screen — progress + cancel.
 *
 * SPEC §v1.6 step 7. Polls GET /v1/scans/:id with tanstack-query at
 * a short interval until status === 'complete' (or 'failed'), then
 * routes to the report.
 *
 * The backend's process_scan() runs synchronously in v1, so by the
 * time the finalize call returns the scan should already be complete.
 * We still poll to handle the v2 async case and to give the UI time
 * to render the processing animation.
 */

import React, { useEffect, useMemo } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { useQuery } from '@tanstack/react-query';

import { Button, ProgressBar, Screen, SectionHeader } from '@src/components';
import { apiClient } from '@src/api/client';
import { queryKeys } from '@src/state/queryClient';
import { colors, spacing, typography } from '@src/theme';

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

  const message = useMemo(() => {
    if (error) return "Couldn't reach the server. Retrying…";
    if (!data) return 'Connecting…';
    switch (data.status) {
      case 'uploading':
        return 'Uploading images…';
      case 'processing':
        return 'Running OCR + rule engine…';
      case 'complete':
        return 'Done. Opening report…';
      case 'failed':
        return 'Processing failed.';
    }
  }, [data, error]);

  return (
    <Screen>
      <SectionHeader
        title="Analyzing"
        subtitle="OCR, field extraction, and TTB rule checks."
      />

      <View style={styles.progressBlock}>
        <ProgressBar
          value={progressFor(data?.status)}
          indeterminate={data?.status !== 'complete' && data?.status !== 'failed'}
        />
        <View style={styles.progressLabelRow}>
          {data?.status !== 'complete' && data?.status !== 'failed' ? (
            <ActivityIndicator size="small" color={colors.primary} />
          ) : null}
          <Text style={styles.progressLabel}>{message}</Text>
        </View>
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

function progressFor(status: string | undefined): number {
  switch (status) {
    case 'uploading':
      return 0.25;
    case 'processing':
      return 0.65;
    case 'complete':
      return 1;
    case 'failed':
      return 0;
    default:
      return 0.1;
  }
}

const styles = StyleSheet.create({
  progressBlock: {
    gap: spacing.sm,
    marginTop: spacing.lg,
  },
  progressLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  progressLabel: {
    ...typography.body,
    color: colors.text,
  },
  scanId: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
