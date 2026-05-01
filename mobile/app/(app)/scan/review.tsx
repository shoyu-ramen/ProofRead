/**
 * Review screen — full-bleed panorama preview + auto-submit.
 *
 * SPEC §v1.6 step 6, ARCH §6 (single-panorama scan store), and SCAN_DESIGN
 * §8 stage-5 hand-off (the panorama image is rendered at the same
 * top-of-screen position the unwrap screen left it in, so the cross-
 * navigation feels continuous).
 *
 * On mount we kick off the submit pipeline automatically:
 *   1. POST /v1/scans
 *   2. PUT the single panorama upload to upload_urls[0]
 *   3. POST /v1/scans/:id/finalize
 *   4. Navigate to the processing screen which polls GET /v1/scans/:id
 *
 * Retake stays prominent so the user can bail at any time; if they hit
 * Retake mid-flight, the AbortController cancels the in-flight requests.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Animated,
  Easing,
  Image,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { router } from 'expo-router';
import { Button, ErrorState, Screen, SectionHeader, Skeleton } from '@src/components';
import { apiClient } from '@src/api/client';
import { describeError } from '@src/api/errors';
import { useToast } from '@src/hooks/useToast';
import { useScanStore } from '@src/state/scanStore';
import { colors, radius, spacing, typography } from '@src/theme';

type Phase = 'submitting' | 'uploading' | 'analyzing' | 'error';

interface PhaseStep {
  key: Exclude<Phase, 'error'>;
  label: string;
}

const PHASE_STEPS: PhaseStep[] = [
  { key: 'submitting', label: 'Submitting' },
  { key: 'uploading', label: 'Uploading' },
  { key: 'analyzing', label: 'Analyzing' },
];

interface ErrorState {
  title: string;
  message: string;
}

const UPLOAD_TIMEOUT_MS = 30_000;
const FINALIZE_TIMEOUT_MS = 60_000;

export default function ReviewScreen(): React.ReactElement {
  const beverageType = useScanStore((s) => s.beverageType);
  const containerSizeMl = useScanStore((s) => s.containerSizeMl);
  const isImported = useScanStore((s) => s.isImported);
  const panorama = useScanStore((s) => s.panorama);
  const setScanId = useScanStore((s) => s.setScanId);
  // Threaded through to /finalize so enrichment can stamp the L3 cache
  // row's first_frame_signature_hex (KNOWN_LABEL_DESIGN.md Decision 6).
  // null when detect-container didn't return a hash (e.g. older backend
  // or the call failed) — finalize tolerates absence.
  const firstFrameSignatureHex = useScanStore(
    (s) => s.firstFrameSignatureHex,
  );

  const [phase, setPhase] = useState<Phase>('submitting');
  const [error, setError] = useState<ErrorState | null>(null);
  const [attempt, setAttempt] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const { show: showToast } = useToast();

  const allReady =
    beverageType !== null && containerSizeMl !== null && panorama !== null;

  const submit = useCallback(async () => {
    if (!allReady || !beverageType || containerSizeMl === null || !panorama) {
      return;
    }
    const controller = new AbortController();
    abortRef.current?.abort();
    abortRef.current = controller;

    setError(null);
    setPhase('submitting');

    try {
      const created = await runWithTimeout(
        apiClient.createScan({
          beverage_type: beverageType,
          container_size_ml: containerSizeMl,
          is_imported: isImported,
        }),
        UPLOAD_TIMEOUT_MS,
        controller.signal,
      );
      if (controller.signal.aborted) return;

      const upload = created.upload_urls[0];
      if (!upload) {
        throw new Error('backend returned no upload URLs');
      }

      setPhase('uploading');
      const bytes = await fetchUriAsBytes(panorama.uri);
      if (controller.signal.aborted) return;

      await runWithTimeout(
        apiClient.uploadImage(upload.signed_url, bytes, 'image/jpeg'),
        UPLOAD_TIMEOUT_MS,
        controller.signal,
      );
      if (controller.signal.aborted) return;

      setPhase('analyzing');
      await runWithTimeout(
        apiClient.finalizeScan(created.scan_id, {
          firstFrameSignatureHex,
        }),
        FINALIZE_TIMEOUT_MS,
        controller.signal,
      );
      if (controller.signal.aborted) return;

      setScanId(created.scan_id);
      router.replace(`/(app)/scan/processing/${created.scan_id}`);
    } catch (err) {
      if (controller.signal.aborted) return;
      const v = describeError(err);
      setError({
        title: v.title,
        message:
          v.kind === 'network'
            ? 'Network error — tap to retry.'
            : `${v.message} Tap to retry.`,
      });
      setPhase('error');
      // Mirror the inline error to a top-of-screen toast so the user
      // sees a consistent failure cue regardless of which screen the
      // verify pipeline died on.
      showToast({
        variant: 'error',
        message: "Couldn't reach the verifier. Tap to retry.",
      });
    }
  }, [
    allReady,
    beverageType,
    containerSizeMl,
    isImported,
    panorama,
    setScanId,
    showToast,
    firstFrameSignatureHex,
  ]);

  useEffect(() => {
    void submit();
    return () => {
      abortRef.current?.abort();
    };
  }, [submit, attempt]);

  // Drive the in-flight progress bar with a looping animation while we
  // wait on network. Stops the moment we hit the error state so the
  // user's eye can land on the error box without competing motion.
  const shimmer = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (phase === 'error') {
      shimmer.stopAnimation();
      shimmer.setValue(0);
      return;
    }
    shimmer.setValue(0);
    const loop = Animated.loop(
      Animated.timing(shimmer, {
        toValue: 1,
        duration: 1_400,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: false,
      }),
    );
    loop.start();
    return () => {
      loop.stop();
    };
  }, [phase, shimmer]);

  const handleRetake = () => {
    abortRef.current?.abort();
    router.replace('/(app)/scan/unwrap');
  };

  const handleRetry = () => {
    setAttempt((n) => n + 1);
  };

  const headerCopy = headerCopyFor(phase);
  const panoramaCaption = panorama ? formatPanoramaCaption(panorama.durationMs) : null;
  const activePhaseIndex = PHASE_STEPS.findIndex((s) => s.key === phase);

  // Map the looping 0..1 driver onto a sliding band that traverses the
  // track. Two outputs: the band's left offset and its width fraction,
  // both expressed as percentages so we don't need to measure the
  // track on layout.
  const shimmerLeft = shimmer.interpolate({
    inputRange: [0, 1],
    outputRange: ['-40%', '100%'],
  });

  return (
    <Screen>
      <SectionHeader title={headerCopy.title} subtitle={headerCopy.subtitle} />

      <View style={styles.panoramaCard}>
        {panorama ? (
          <Image
            source={{ uri: panorama.uri }}
            style={styles.panoramaImage}
            resizeMode="cover"
          />
        ) : (
          // Skeleton fills the panorama frame while the local capture is
          // wiring up — the user briefly sees a pulsing tile rather than
          // bare "No scan yet" copy on a flash of grey.
          <Skeleton width="100%" height="100%" radius={radius.md} />
        )}
      </View>
      {panoramaCaption ? (
        <Text style={styles.panoramaCaption}>{panoramaCaption}</Text>
      ) : null}

      {phase !== 'error' ? (
        <View style={styles.statusBlock}>
          <View style={styles.progressWrap}>
            <View style={styles.stepRow}>
              {PHASE_STEPS.map((step, idx) => {
                const isActive = idx === activePhaseIndex;
                const isDone = idx < activePhaseIndex;
                return (
                  <View key={step.key} style={styles.stepItem}>
                    <View
                      style={[
                        styles.stepDot,
                        isActive && styles.stepDotActive,
                        isDone && styles.stepDotDone,
                      ]}
                    />
                    <Text
                      style={[
                        styles.stepLabel,
                        (isActive || isDone) && styles.stepLabelOn,
                      ]}
                      numberOfLines={1}
                    >
                      {step.label}
                    </Text>
                  </View>
                );
              })}
            </View>
            <View style={styles.progressTrack}>
              <Animated.View
                style={[styles.progressBand, { left: shimmerLeft }]}
              />
            </View>
          </View>
        </View>
      ) : null}

      {phase === 'error' && error ? (
        <ErrorState
          title={error.title}
          description={error.message}
          retry={handleRetry}
          retryLabel="Retry submission"
        />
      ) : null}

      <View style={{ flex: 1 }} />
      <Button
        label={panorama ? 'Retake' : 'Capture'}
        variant="secondary"
        size="lg"
        fullWidth
        onPress={handleRetake}
      />
    </Screen>
  );
}

function headerCopyFor(phase: Phase): { title: string; subtitle: string } {
  switch (phase) {
    case 'submitting':
      return {
        title: 'Submitting scan',
        subtitle: 'Sending to compliance check.',
      };
    case 'uploading':
      return {
        title: 'Submitting scan',
        subtitle: 'Uploading panorama to the server.',
      };
    case 'analyzing':
      return {
        title: 'Submitting scan',
        subtitle: 'Handing off for analysis.',
      };
    case 'error':
      return {
        title: 'Review',
        subtitle: 'Something went wrong submitting this scan.',
      };
  }
}

function formatPanoramaCaption(durationMs: number): string | null {
  if (durationMs < 1_000) return null;
  const seconds = Math.round(durationMs / 1000);
  return `Full label captured in ${seconds} second${seconds === 1 ? '' : 's'}`;
}

/**
 * Read a file:// URI into bytes for upload. Uses fetch() since RN
 * supports it for file:// — avoids a dependency on expo-file-system at
 * the upload layer.
 */
async function fetchUriAsBytes(uri: string): Promise<ArrayBuffer> {
  const res = await fetch(uri);
  return res.arrayBuffer();
}

/**
 * Race a promise against a timeout + an abort signal. The underlying
 * apiClient calls don't accept signals today, so abort/timeouts here
 * unblock the UI; in-flight fetches will resolve in the background and
 * the result is dropped.
 */
function runWithTimeout<T>(
  p: Promise<T>,
  ms: number,
  signal: AbortSignal,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => {
      reject(new Error(`timeout after ${Math.round(ms / 1000)}s`));
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(new DOMException('aborted', 'AbortError'));
    };
    if (signal.aborted) {
      onAbort();
      return;
    }
    signal.addEventListener('abort', onAbort);
    p.then(
      (v) => {
        clearTimeout(t);
        signal.removeEventListener('abort', onAbort);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        signal.removeEventListener('abort', onAbort);
        reject(e);
      },
    );
  });
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
  panoramaCaption: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: -spacing.sm,
  },
  statusBlock: {
    gap: spacing.sm,
    minHeight: 64,
  },
  progressWrap: {
    gap: spacing.sm,
  },
  stepRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: spacing.sm,
  },
  stepItem: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  stepDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.border,
  },
  stepDotActive: {
    backgroundColor: colors.primary,
  },
  stepDotDone: {
    backgroundColor: colors.pass,
  },
  stepLabel: {
    ...typography.caption,
    color: colors.textMuted,
    flexShrink: 1,
  },
  stepLabelOn: {
    color: colors.text,
  },
  progressTrack: {
    height: 4,
    backgroundColor: colors.surfaceAlt,
    borderRadius: radius.sm,
    overflow: 'hidden',
  },
  progressBand: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: '40%',
    backgroundColor: colors.primary,
    borderRadius: radius.sm,
  },
});
