/**
 * KnownLabelSheet — recognition decision UI inside `confirming{detected}`.
 *
 * Rendered by `unwrap.tsx`'s ConfirmingOverlay when
 * `DetectContainerResponse.known_label` is non-null
 * (KNOWN_LABEL_DESIGN.md Decision 7). Three actions:
 *
 *   - **View results** (primary) → POST /v1/scans/from-cache with the
 *     payload from `known_label`, navigate to the report on success.
 *     On failure, render inline `ErrorState` (full-UI decision point —
 *     no toasts).
 *   - **Scan anyway** (ghost) → clear the recognition payload (callback
 *     sets `knownLabel=null`) and fall through to the normal capture
 *     pipeline (`confirmStart`).
 *   - **Reshoot** (ghost) → clear the recognition payload and re-arm
 *     (`confirmRetry`).
 *
 * Lives in its own file so unit tests can mount it without pulling in
 * unwrap.tsx's native deps (Camera / Skia / Reanimated frame processor).
 */

import React, { useCallback, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Button, ErrorState, StatusBadge } from '@src/components';
import { apiClient } from '@src/api/client';
import { coerceKnownLabelOverall } from '@src/api/types';
import type { KnownLabelPayload } from '@src/api/types';
import { colors, spacing, typography } from '@src/theme';

export interface KnownLabelSheetProps {
  knownLabel: KnownLabelPayload;
  /** Called when the user picks "Scan anyway" — clears `knownLabel` then `confirmStart()`. */
  onScanAnyway: () => void;
  /** Called when the user picks "Reshoot" — clears `knownLabel` then `confirmRetry()`. */
  onReshoot: () => void;
}

export function KnownLabelSheet({
  knownLabel,
  onScanAnyway,
  onReshoot,
}: KnownLabelSheetProps): React.ReactElement {
  const insets = useSafeAreaInsets();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleViewResults = useCallback(async () => {
    if (submitting) return;
    setError(null);
    setSubmitting(true);
    try {
      const created = await apiClient.createScanFromCache({
        entry_id: knownLabel.entry_id,
        beverage_type: knownLabel.beverage_type,
        container_size_ml: knownLabel.container_size_ml,
        is_imported: knownLabel.is_imported,
      });
      router.replace(`/(app)/scan/report/${created.scan_id}`);
    } catch (err) {
      console.warn('[unwrap] createScanFromCache failed', err);
      const message =
        err instanceof Error
          ? err.message
          : 'Could not load the saved verdict. Try again or scan anyway.';
      setError(message);
    } finally {
      setSubmitting(false);
    }
  }, [submitting, knownLabel]);

  const handleRetryFromCache = useCallback(() => {
    setError(null);
    void handleViewResults();
  }, [handleViewResults]);

  if (error) {
    // Full-UI decision point — no toast. The user needs to act on this:
    // retry the cache call, scan anyway, or reshoot.
    return (
      <View style={styles.errorWrap}>
        <ErrorState
          title="Couldn't load the saved verdict."
          description={error}
          retry={handleRetryFromCache}
          retryLabel="Try again"
          secondaryAction={{ label: 'Scan anyway', onPress: onScanAnyway }}
        />
      </View>
    );
  }

  const headline = knownLabel.brand_name ?? 'Recognized label';
  const fancifulName = knownLabel.fanciful_name;
  return (
    <View style={styles.scrim} pointerEvents="box-none">
      <View
        style={[
          styles.card,
          { marginBottom: insets.bottom + spacing.lg },
        ]}
      >
        <Text style={styles.subtitle}>Recognized from a previous scan.</Text>
        <Text style={styles.brand} numberOfLines={2}>
          {headline}
        </Text>
        {fancifulName ? (
          <Text style={styles.fanciful} numberOfLines={2}>
            {fancifulName}
          </Text>
        ) : null}
        <View style={styles.badgeWrap}>
          {/* The backend's `overall` is wider than OverallStatus
              (`warn`, `unreadable`). Coerce so StatusBadge's existing
              palette covers every case. */}
          <StatusBadge
            status={coerceKnownLabelOverall(
              knownLabel.verdict_summary.overall,
            )}
          />
        </View>
        <View style={styles.actions}>
          <Button
            label="View results"
            onPress={handleViewResults}
            loading={submitting}
            fullWidth
          />
          <Button
            label="Scan anyway"
            variant="ghost"
            onPress={onScanAnyway}
            disabled={submitting}
            fullWidth
          />
          <Button
            label="Reshoot"
            variant="ghost"
            onPress={onReshoot}
            disabled={submitting}
            fullWidth
          />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  scrim: {
    ...StyleSheet.absoluteFillObject,
    // Match the existing ConfirmingOverlay scrim so the recognition
    // sheet reads as the same family of UI.
    backgroundColor: colors.scanOverlayScrim,
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  errorWrap: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: colors.scanOverlayDim,
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    position: 'absolute',
    left: spacing.lg,
    right: spacing.lg,
    bottom: 0,
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    gap: spacing.sm,
  },
  subtitle: {
    ...typography.caption,
    color: colors.textMuted,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  brand: {
    ...typography.titleLg,
    color: colors.text,
  },
  fanciful: {
    ...typography.body,
    color: colors.textMuted,
  },
  badgeWrap: {
    marginTop: spacing.xs,
    flexDirection: 'row',
  },
  actions: {
    marginTop: spacing.md,
    gap: spacing.sm,
  },
});
