/**
 * Review screen — thumbnail strip + retake + Analyze CTA.
 *
 * SPEC §v1.6 step 6.
 *
 * "Analyze" is the moment we hit the backend:
 *   1. POST /v1/scans
 *   2. PUT each signed_url with the captured image bytes
 *   3. POST /v1/scans/:id/finalize
 *   4. Navigate to the processing screen which polls GET /v1/scans/:id
 */

import React, { useCallback, useState } from 'react';
import { Alert, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { Button, Screen, SectionHeader } from '@src/components';
import { apiClient } from '@src/api/client';
import {
  REQUIRED_CAPTURE_SURFACES,
  useScanStore,
} from '@src/state/scanStore';
import type { Surface, UploadURL } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

export default function ReviewScreen(): React.ReactElement {
  const beverageType = useScanStore((s) => s.beverageType);
  const containerSizeMl = useScanStore((s) => s.containerSizeMl);
  const isImported = useScanStore((s) => s.isImported);
  const captures = useScanStore((s) => s.captures);
  const setScanId = useScanStore((s) => s.setScanId);

  const [submitting, setSubmitting] = useState(false);

  const allReady =
    beverageType !== null &&
    containerSizeMl !== null &&
    REQUIRED_CAPTURE_SURFACES.every((s) => Boolean(captures[s]));

  const handleAnalyze = useCallback(async () => {
    if (!allReady || !beverageType || containerSizeMl === null) return;
    setSubmitting(true);
    try {
      // 1. Create scan
      const created = await apiClient.createScan({
        beverage_type: beverageType,
        container_size_ml: containerSizeMl,
        is_imported: isImported,
      });

      // 2. Upload each surface to its signed URL.
      await Promise.all(
        created.upload_urls.map(async (u: UploadURL) => {
          const cap = captures[u.surface as Surface];
          if (!cap) {
            throw new Error(`missing capture for surface ${u.surface}`);
          }
          const bytes = await fetchUriAsBytes(cap.uri);
          await apiClient.uploadImage(u.signed_url, bytes, 'image/jpeg');
        })
      );

      // 3. Finalize → backend kicks off processing.
      await apiClient.finalizeScan(created.scan_id);

      setScanId(created.scan_id);
      router.replace(`/(app)/scan/processing/${created.scan_id}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Submit failed';
      Alert.alert("Couldn't submit scan", message);
    } finally {
      setSubmitting(false);
    }
  }, [allReady, beverageType, captures, containerSizeMl, isImported, setScanId]);

  return (
    <Screen>
      <SectionHeader
        title="Review your captures"
        subtitle="Retake either side, then analyze."
      />

      <View style={styles.thumbStrip}>
        {REQUIRED_CAPTURE_SURFACES.map((surface) => {
          const cap = captures[surface];
          return (
            <View key={surface} style={styles.thumbCard}>
              <Text style={styles.surfaceLabel}>
                {surface === 'front' ? 'Front' : 'Back'}
              </Text>
              <View style={styles.thumb}>
                {cap ? (
                  <Image source={{ uri: cap.uri }} style={styles.thumbImage} />
                ) : (
                  <View style={styles.thumbEmpty}>
                    <Text style={styles.thumbEmptyText}>Not captured</Text>
                  </View>
                )}
              </View>
              <Pressable
                style={({ pressed }) => [
                  styles.retake,
                  pressed && { opacity: 0.85 },
                ]}
                onPress={() =>
                  router.push(`/(app)/scan/camera/${surface}` as `/(app)/scan/camera/${string}`)
                }
              >
                <Text style={styles.retakeText}>
                  {cap ? 'Retake' : 'Capture'}
                </Text>
              </Pressable>
            </View>
          );
        })}
      </View>

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
      <Button
        label="Analyze"
        size="lg"
        fullWidth
        loading={submitting}
        disabled={!allReady}
        onPress={handleAnalyze}
      />
    </Screen>
  );
}

/**
 * Read a file:// URI into bytes for upload.
 *
 * Uses fetch() since RN supports it for file:// — avoids a dependency
 * on expo-file-system at the scaffold stage. If perf becomes an issue,
 * swap for FileSystem.readAsStringAsync + base64 decode.
 */
async function fetchUriAsBytes(uri: string): Promise<ArrayBuffer> {
  const res = await fetch(uri);
  return res.arrayBuffer();
}

const styles = StyleSheet.create({
  thumbStrip: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  thumbCard: {
    flex: 1,
    gap: spacing.xs,
  },
  surfaceLabel: {
    ...typography.heading,
    color: colors.text,
  },
  thumb: {
    aspectRatio: 0.65,
    backgroundColor: colors.surfaceAlt,
    borderRadius: radius.md,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: colors.border,
  },
  thumbImage: {
    width: '100%',
    height: '100%',
  },
  thumbEmpty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  thumbEmptyText: {
    ...typography.caption,
    color: colors.textMuted,
  },
  retake: {
    paddingVertical: spacing.sm,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    alignItems: 'center',
  },
  retakeText: {
    ...typography.button,
    color: colors.text,
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
});
