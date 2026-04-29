/**
 * useRotationCapture — checkpoint-driven photo capture (ARCH §4.5).
 *
 * Subscribes to the tracker's coverage; when coverage advances past
 * the next angular checkpoint (every 1/N revolution, default N=36 for
 * 10°), schedules a `Camera.takePhoto()`, runs `extractStrip()` on the
 * resulting JPEG, and appends the strip to the panorama state.
 *
 * Performance contract (ARCH §4.5 / brief #4):
 *
 *   - takePhoto is async and slow (~80–150ms); never awaited from the
 *     worklet.
 *   - At most 2 captures may be in flight or queued. New triggers that
 *     arrive while we're at capacity are *dropped* (not buffered) — a
 *     missed checkpoint is a one-tick gap, while a bloated queue would
 *     sluggishly trail the rotation.
 *   - The frame processor keeps streaming tracker state regardless of
 *     queue depth. The hook returns a cancel() that drains pending
 *     captures and prevents any later strips from being committed.
 *
 * The returned `panoramaState` is a JS-side accumulator passed to the
 * `<PanoramaCanvas>` component for live drawing.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  runOnJS,
  useAnimatedReaction,
  type SharedValue,
} from 'react-native-reanimated';
import { Camera } from 'react-native-vision-camera';

import {
  N_CHECKPOINTS,
  createEmptyPanoramaState,
  extractStrip,
  type PanoramaState,
  type StripCheckpoint,
} from '@src/scan/panorama';
import type { TrackerState } from '@src/scan/tracker';
import type { ScanFrame } from '@src/state/scanStore';

/** Hard cap on simultaneously in-flight + pending captures (brief #4). */
const QUEUE_CAPACITY = 2;

/**
 * Resize buffer dims used by the tracker (must match the constants in
 * `mobile/src/scan/tracker/frameProcessor.ts`). The strip extractor
 * uses these to scale the silhouette into photo-pixel space.
 */
const TRACKER_FRAME_W = 160;
const TRACKER_FRAME_H = 240;

export interface UseRotationCaptureOptions {
  /** Camera ref — `cameraRef.current?.takePhoto()` triggers capture. */
  cameraRef: React.RefObject<Camera>;
  /** Tracker shared value — read coverage + silhouette per checkpoint. */
  trackerStateSv: SharedValue<TrackerState>;
  /** Caller-driven setter so the tracker bumps its own checkpoint count. */
  bumpCapturedCheckpoints: () => void;
  /** Fired once per appended frame so the parent can store ScanFrames. */
  onFrameCaptured?: (frame: ScanFrame) => void;
  /** True iff we should accept new triggers (false during pause/complete). */
  enabled: boolean;
  /** Number of checkpoints around the bottle. Defaults to N_CHECKPOINTS=36. */
  numCheckpoints?: number;
}

export interface UseRotationCaptureResult {
  /** Live accumulator the PanoramaCanvas reads. */
  panoramaState: PanoramaState;
  /** Index of the highest checkpoint we've completed (for diagnostics). */
  lastCheckpoint: number;
  /** Drains the queue + ignores any pending strip results. */
  cancel: () => void;
}

export function useRotationCapture(
  opts: UseRotationCaptureOptions,
): UseRotationCaptureResult {
  const {
    cameraRef,
    trackerStateSv,
    bumpCapturedCheckpoints,
    onFrameCaptured,
    enabled,
    numCheckpoints = N_CHECKPOINTS,
  } = opts;

  const [panoramaState, setPanoramaState] = useState<PanoramaState>(() =>
    createEmptyPanoramaState(),
  );
  const [lastCheckpoint, setLastCheckpoint] = useState(-1);

  // Queue depth across in-flight + pending captures. Tracked as a ref
  // so the worklet-side trigger reads the latest value without a
  // re-render cycle.
  const queueDepthRef = useRef(0);
  // Last checkpoint index we've decided to capture. Worklet-readable
  // via a ref.
  const lastTriggeredRef = useRef<number>(-1);
  // Cancel flag — set on unmount or explicit cancel; pending strips
  // landing after this won't mutate state.
  const cancelledRef = useRef(false);
  // Single-deferred checkpoint slot. When the queue is at capacity we
  // drop the inbound trigger but remember the most recent one; once
  // capacity opens up we replay it against current coverage so the
  // panorama doesn't keep a permanent gap. Capped at one — a deeper
  // queue would bloat behind the rotation per ARCH §4.5.
  const pendingCkptRef = useRef<number | null>(null);
  // Self-reference for the deferred-checkpoint retry inside
  // `triggerCapture`'s finally block. Using a ref instead of a direct
  // recursive reference avoids stale-closure issues across renders.
  const triggerCaptureRef = useRef<
    ((ckpt: number, coverage: number) => Promise<void>) | null
  >(null);

  /** JS-side trigger invoked from the worklet via runOnJS. */
  const triggerCapture = useCallback(
    async (checkpointIdx: number, coverageAtCapture: number) => {
      if (cancelledRef.current) return;
      if (queueDepthRef.current >= QUEUE_CAPACITY) {
        // At capacity — defer this checkpoint into the single-slot
        // recovery ring so we can retry it once the queue drains.
        // Overwrites any older deferral: only the most recent missed
        // checkpoint is worth recovering (newer is closer to where the
        // user actually is).
        if (checkpointIdx > lastTriggeredRef.current) {
          pendingCkptRef.current = checkpointIdx;
        }
        return;
      }
      const camera = cameraRef.current;
      if (!camera) return;

      queueDepthRef.current += 1;
      try {
        // Snapshot the silhouette at trigger time as a *fallback* —
        // the photo may resolve in a frame where the silhouette is
        // briefly mis-detected, so we keep this as a backstop for the
        // strip-extractor crop.
        const triggerTs = trackerStateSv.value;
        const triggerSilhouette = triggerTs.silhouette;

        const photo = await camera.takePhoto({
          flash: 'off',
          enableShutterSound: false,
        });
        if (cancelledRef.current) return;

        // Re-read coverage immediately after the shutter resolves.
        // Audit finding: takePhoto() takes 80–150ms — at ~0.15 rev/s
        // that's ~5° of drift, accumulated over 36 strips into a
        // visible mid-panorama shear. The trigger-time snapshot is
        // still useful for *which* checkpoint to bind to, but the
        // *placement* angle should reflect where the camera actually
        // was when the shutter fired.
        const shutterTs = trackerStateSv.value;
        const placementCoverage = shutterTs.coverage;
        // Prefer the post-shutter silhouette if still detected; fall
        // back to the trigger-time snapshot otherwise.
        const silhouette = shutterTs.silhouette.detected
          ? shutterTs.silhouette
          : triggerSilhouette;

        const photoUri = photo.path.startsWith('file://')
          ? photo.path
          : `file://${photo.path}`;

        // Strip extraction is heavy; run it after takePhoto resolves
        // so the next checkpoint can fire while this is still working.
        const strip: StripCheckpoint = await extractStrip(
          photoUri,
          silhouette.detected ? silhouette : null,
          {
            coverage: placementCoverage,
            silhouetteSourceWidth: TRACKER_FRAME_W,
            silhouetteSourceHeight: TRACKER_FRAME_H,
          },
        );
        if (cancelledRef.current) return;

        // Commit: append to the panorama state, bump the tracker's
        // checkpoint counter (so the state machine knows we made
        // progress), and notify the parent of the raw frame URI.
        setPanoramaState((prev) => ({
          ...prev,
          strips: [...prev.strips, strip],
        }));
        setLastCheckpoint(checkpointIdx);
        bumpCapturedCheckpoints();
        onFrameCaptured?.({
          uri: photoUri,
          coverage: placementCoverage,
          capturedAt: Date.now(),
        });
      } catch (err) {
        // A modal alert mid-rotation would derail the magic-moment UX
        // (the user's hand is on the bottle and a system Alert blocks
        // them). The dot-pattern fill already shows the gap visually,
        // and the next checkpoint will try again — that's the recovery.
        // The console.warn keeps a trail for `npx react-native log-ios`.
        if (!cancelledRef.current) {
          // eslint-disable-next-line no-console
          console.warn('[useRotationCapture] checkpoint capture failed', err);
        }
      } finally {
        queueDepthRef.current = Math.max(0, queueDepthRef.current - 1);

        // Deferred-checkpoint retry: drain the single-slot recovery
        // ring once we have headroom. Coverage may have advanced past
        // the pending index in the meantime — if it has by more than
        // one checkpoint, the gap is too wide to recover meaningfully
        // and we drop the deferred slot rather than capture stale
        // angles. The retry uses the *current* coverage so the strip
        // angle reflects what the camera is actually seeing now.
        const pending = pendingCkptRef.current;
        if (
          !cancelledRef.current &&
          pending !== null &&
          queueDepthRef.current < QUEUE_CAPACITY
        ) {
          pendingCkptRef.current = null;
          const liveCoverage = trackerStateSv.value.coverage;
          const liveCkpt = Math.floor(liveCoverage * (numCheckpoints));
          if (
            pending > lastTriggeredRef.current &&
            liveCkpt - pending <= 1
          ) {
            lastTriggeredRef.current = pending;
            void triggerCaptureRef.current?.(pending, liveCoverage);
          }
        }
      }
    },
    [
      cameraRef,
      trackerStateSv,
      bumpCapturedCheckpoints,
      onFrameCaptured,
      numCheckpoints,
    ],
  );

  // Keep the ref pointing at the current callback so the deferred-
  // checkpoint retry sees the latest closure across renders.
  useEffect(() => {
    triggerCaptureRef.current = triggerCapture;
  }, [triggerCapture]);

  // Worklet reaction: every coverage tick, decide whether to trigger
  // the next checkpoint. The check is cheap (a single comparison), so
  // we run it on every frame; the queue cap upstream throttles
  // duplicate triggers.
  useAnimatedReaction(
    () => trackerStateSv.value.coverage,
    (coverage) => {
      'worklet';
      if (!enabled) return;
      if (coverage <= 0) return;
      // Compute the next checkpoint index based on the coverage
      // boundary. We want strips at coverage = (i + 0.5) / N for i in
      // [0, N-1], so the strip lands centred in its angular slice.
      const ckpt = Math.floor(coverage * numCheckpoints);
      if (ckpt > numCheckpoints - 1) return;
      // Cross the threshold *exactly once* per checkpoint. We compare
      // to lastTriggeredRef on the JS side via runOnJS.
      runOnJS(maybeTrigger)(ckpt, coverage);
    },
    // Stable-identity deps only — SharedValue refs trigger
    // Reanimated's "_value access from JS" guard during dep
    // comparison (see frameProcessor.ts deps comment).
    [enabled, numCheckpoints],
  );

  const maybeTrigger = useCallback(
    (ckpt: number, coverage: number) => {
      if (cancelledRef.current) return;
      if (ckpt <= lastTriggeredRef.current) return;
      // Coverage has advanced past the deferred checkpoint by more
      // than one slot — recovering it would capture a stale angle, so
      // drop the deferral.
      const pending = pendingCkptRef.current;
      if (pending !== null && ckpt - pending > 1) {
        pendingCkptRef.current = null;
      }
      lastTriggeredRef.current = ckpt;
      void triggerCapture(ckpt, coverage);
    },
    [triggerCapture],
  );

  /**
   * Cancel: ignore in-flight strip results, reset the queue. Does NOT
   * reset the panorama state — the parent decides whether to keep or
   * scrap the partial scan.
   */
  const cancel = useCallback(() => {
    cancelledRef.current = true;
    queueDepthRef.current = 0;
    pendingCkptRef.current = null;
  }, []);

  // Auto-cancel on unmount so a remount starts clean.
  useEffect(() => {
    return () => {
      cancelledRef.current = true;
      queueDepthRef.current = 0;
      pendingCkptRef.current = null;
    };
  }, []);

  return useMemo(
    () => ({ panoramaState, lastCheckpoint, cancel }),
    [panoramaState, lastCheckpoint, cancel],
  );
}
