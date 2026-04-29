/**
 * Review screen — full-bleed panorama preview + retake + Analyze CTA.
 *
 * SPEC §v1.6 step 6, ARCH §6 (single-panorama scan store), and SCAN_DESIGN
 * §8 stage-5 hand-off (the panorama image is rendered at the same
 * top-of-screen position the unwrap screen left it in, so the cross-
 * navigation feels continuous).
 *
 * "Analyze" is the moment we hit the backend:
 *   1. POST /v1/scans
 *   2. PUT the single panorama upload to upload_urls[0]
 *   3. POST /v1/scans/:id/finalize
 *   4. Navigate to the processing screen which polls GET /v1/scans/:id
 */

import React, { useCallback, useState } from 'react';
import { Image, StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { Button, Screen, SectionHeader } from '@src/components';
import { apiClient } from '@src/api/client';
import { showErrorAlert } from '@src/api/errors';
import { useScanStore } from '@src/state/scanStore';
import { colors, radius, spacing, typography } from '@src/theme';

export default function ReviewScreen(): React.ReactElement {
  const beverageType = useScanStore((s) => s.beverageType);
  const containerSizeMl = useScanStore((s) => s.containerSizeMl);
  const isImported = useScanStore((s) => s.isImported);
  const panorama = useScanStore((s) => s.panorama);
  const setScanId = useScanStore((s) => s.setScanId);

  const [submitting, setSubmitting] = useState(false);

  const allReady =
    beverageType !== null && containerSizeMl !== null && panorama !== null;

  const handleAnalyze = useCallback(async () => {
    if (!allReady || !beverageType || containerSizeMl === null || !panorama) {
      return;
    }
    setSubmitting(true);
    try {
      // 1. Create scan — backend returns one upload URL with surface
      // "panorama" (per ARCH §7.2 / BACKEND_HANDOFF.md).
      const created = await apiClient.createScan({
        beverage_type: beverageType,
        container_size_ml: containerSizeMl,
        is_imported: isImported,
      });

      // 2. Upload the panorama JPEG to the single signed URL.
      const upload = created.upload_urls[0];
      if (!upload) {
        throw new Error('backend returned no upload URLs');
      }
      const bytes = await fetchUriAsBytes(panorama.uri);
      await apiClient.uploadImage(upload.signed_url, bytes, 'image/jpeg');

      // 3. Finalize → backend kicks off processing.
      await apiClient.finalizeScan(created.scan_id);

      setScanId(created.scan_id);
      router.replace(`/(app)/scan/processing/${created.scan_id}`);
    } catch (err) {
      showErrorAlert(err, { title: "Couldn't submit scan" });
    } finally {
      setSubmitting(false);
    }
  }, [allReady, beverageType, containerSizeMl, isImported, panorama, setScanId]);

  return (
    <Screen>
      <SectionHeader
        title="Review your label"
        subtitle="Retake to rotate again, or analyze."
      />

      <View style={styles.panoramaCard}>
        {panorama ? (
          <Image
            source={{ uri: panorama.uri }}
            style={styles.panoramaImage}
            resizeMode="cover"
          />
        ) : (
          <View style={styles.panoramaEmpty}>
            <Text style={styles.panoramaEmptyText}>No scan yet</Text>
          </View>
        )}
      </View>
      {panorama ? (
        <Text style={styles.panoramaCaption}>
          {panorama.frameCount} strip{panorama.frameCount === 1 ? '' : 's'} ·{' '}
          {(panorama.durationMs / 1000).toFixed(1)}s
        </Text>
      ) : null}

      <View style={styles.metaCard}>
        <Text style={styles.metaTitle}>Scan details</Text>
        <Text style={styles.metaRow}>
          Beverage: <Text style={styles.metaValue}>{beverageType ?? '—'}</Text>
        </Text>
        <Text style={styles.metaRow}>
          Container:{' '}
          <Text style={styles.metaValue}>
            {containerSizeMl !== null ? `${containerSizeMl} mL` : '—'}
          </Text>
        </Text>
        <Text style={styles.metaRow}>
          Imported:{' '}
          <Text style={styles.metaValue}>{isImported ? 'Yes' : 'No'}</Text>
        </Text>
      </View>

      <View style={{ flex: 1 }} />
      <View style={styles.actions}>
        <Button
          label={panorama ? 'Retake' : 'Capture'}
          variant="secondary"
          fullWidth
          onPress={() => router.replace('/(app)/scan/unwrap')}
        />
        <Button
          label="Analyze"
          size="lg"
          fullWidth
          loading={submitting}
          disabled={!allReady}
          onPress={handleAnalyze}
        />
      </View>
    </Screen>
  );
}

/**
 * Read a file:// URI into bytes for upload. Uses fetch() since RN
 * supports it for file:// — avoids a dependency on expo-file-system at
 * the upload layer (the panorama subsystem already depends on it for
 * persistence).
 */
async function fetchUriAsBytes(uri: string): Promise<ArrayBuffer> {
  const res = await fetch(uri);
  return res.arrayBuffer();
}

const styles = StyleSheet.create({
  panoramaCard: {
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
  panoramaEmpty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  panoramaEmptyText: {
    ...typography.caption,
    color: colors.textMuted,
  },
  panoramaCaption: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: -spacing.sm,
  },
  metaCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.md,
    gap: spacing.xs,
  },
  metaTitle: {
    ...typography.heading,
    color: colors.text,
    marginBottom: spacing.xs,
  },
  metaRow: {
    ...typography.body,
    color: colors.textMuted,
  },
  metaValue: {
    color: colors.text,
    fontWeight: '600',
  },
  actions: {
    gap: spacing.sm,
  },
});
