/**
 * unwrap.tsx — the live cylindrical scan screen.
 *
 * The user holds a bottle in front of the camera and rotates it; the
 * screen tracks rotation visually, captures a photo at every angular
 * checkpoint, extracts a vertical strip from each, and paints them
 * into a live unrolled-panorama Skia canvas at the top of the screen
 * (ARCH §3 + §6). When coverage hits 1.0 the panorama is stitched into
 * a single JPEG, stored in the scan store, and the user is navigated
 * to /(app)/scan/review.
 *
 * State machine (ARCH §3): aligning → ready → scanning ↔ paused →
 * complete | failed. Owned by `useScanStateMachine` — driven by a
 * Reanimated `useAnimatedReaction` that watches the tracker's frame
 * tick + tracker fields.
 *
 * Checkpoint capture (ARCH §4.5): `useRotationCapture` queues at most
 * 2 takePhoto() calls; new triggers above capacity are dropped (a
 * single missed strip is recoverable, a stale queue would lag the
 * rotation visibly). The frame processor never blocks on capture —
 * the worklet keeps streaming tracker state regardless of queue depth.
 *
 * Comments cite section numbers; the prose lives in
 * .claude/CYLINDRICAL_SCAN_ARCHITECTURE.md and SCAN_DESIGN.md.
 */

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { router } from 'expo-router';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  Camera,
  type CameraDevice,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { useDerivedValue, useSharedValue } from 'react-native-reanimated';
import type { SkImage } from '@shopify/react-native-skia';

import { Button } from '@src/components';
import {
  PanoramaCanvas,
  stitchPanorama,
} from '@src/scan/panorama';
import {
  useTrackerFrameProcessor,
  type PreCheckVerdict,
} from '@src/scan/tracker';
import {
  BottleSilhouetteOverlay,
  CancelButton,
  CompletionReveal,
  InScanWarningBanner,
  ProgressDial,
  QualityChip,
  RotationGuideRing,
  ScanInstructions,
  type SilhouetteFrame,
} from '@src/scan/ui';
import { coverageOf } from '@src/scan/state/scanMachine';
import { useRotationCapture } from '@src/scan/hooks/useRotationCapture';
import { useScanHaptics } from '@src/scan/hooks/useScanHaptics';
import { useScanStateMachine } from '@src/scan/hooks/useScanStateMachine';
import { useScanStore } from '@src/state/scanStore';
import { colors, scanGeometry, spacing, typography } from '@src/theme';

// Tracker frame dims — must match `RESIZE_W`/`RESIZE_H` in
// `mobile/src/scan/tracker/frameProcessor.ts`. Used to translate
// silhouette coords from tracker-frame px into screen-px.
const TRACKER_FRAME_W = 160;
const TRACKER_FRAME_H = 240;

// Quiet window after coverage hits 1.0 before we encode the JPEG.
// `useRotationCapture` runs takePhoto+extractStrip asynchronously
// (queue cap = 2, ~150ms each) and triggers via runOnJS, so the last
// 1-2 strips can still be in flight when the state machine sees
// coverage = 1.0. Re-arming this timer on every panoramaState change
// gives the queue a chance to drain — without it, trailing strips end
// up missing from the stitched panorama.
const STITCH_QUIET_MS = 320;

interface TrackerSnapshot {
  preCheck: PreCheckVerdict;
  rotationDirection: 'cw' | 'ccw' | null;
  coverage: number;
  steadiness: number;
  detected: boolean;
}

function preCheckChanged(a: PreCheckVerdict, b: PreCheckVerdict): boolean {
  if (a.kind !== b.kind) return true;
  if (a.kind === 'warn' && b.kind === 'warn') return a.reason !== b.reason;
  return false;
}

export default function CylindricalScanScreen(): React.ReactElement {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device: CameraDevice | undefined = useCameraDevice('back');

  // Permission flow — same pattern as the deleted camera/[surface].tsx.
  useEffect(() => {
    if (hasPermission === false) {
      void requestPermission();
    }
  }, [hasPermission, requestPermission]);

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.permissionWrap}>
        <Text style={styles.permissionTitle}>Camera permission needed</Text>
        <Text style={styles.permissionBody}>
          ProofRead needs camera access to capture beverage labels.
        </Text>
        <Button
          label="Grant access"
          onPress={() => void requestPermission()}
        />
        <Button label="Back" variant="ghost" onPress={() => router.back()} />
      </SafeAreaView>
    );
  }

  if (!device) {
    return (
      <SafeAreaView style={styles.permissionWrap}>
        <Text style={styles.permissionTitle}>No camera available</Text>
        <Text style={styles.permissionBody}>
          This device doesn't expose a back-facing camera.
        </Text>
        <Button label="Back" variant="ghost" onPress={() => router.back()} />
      </SafeAreaView>
    );
  }

  return <CylindricalScan device={device} />;
}

interface CylindricalScanProps {
  device: CameraDevice;
}

function CylindricalScan({ device }: CylindricalScanProps): React.ReactElement {
  const insets = useSafeAreaInsets();
  const screen = useWindowDimensions();
  const cameraRef = useRef<Camera>(null);

  const setPanorama = useScanStore((s) => s.setPanorama);
  const appendFrame = useScanStore((s) => s.appendFrame);
  const clearScanCaptures = useScanStore((s) => s.clearScanCaptures);

  // Live panorama snapshot — owned here so `stitchPanorama` can encode
  // the already-painted bytes from PanoramaCanvas's off-screen surface
  // without allocating a second 12 MB Skia surface or repainting every
  // strip. PanoramaCanvas writes a fresh `makeImageSnapshot()` here on
  // every strip arrival; we read once at completion.
  const snapshotSv = useSharedValue<SkImage | null>(null);

  // Tracker — frame processor + shared values.
  const tracker = useTrackerFrameProcessor();

  // Scan state machine. Reads tracker state via worklet reaction;
  // dispatches discrete transitions on the JS side.
  const {
    state,
    fail,
    complete,
    cancel,
    reset,
    autoCaptureCandidate,
    manualOverrideAvailable,
    manualStart,
  } = useScanStateMachine(
    tracker.trackerStateSv,
    tracker.frameTickSv,
  );

  // Wall-clock when scanning began — passed to `stitchPanorama` to
  // compute durationMs. Initialised lazily on first scanning tick.
  const startedAtRef = useRef<number | null>(null);
  useEffect(() => {
    if (
      state.kind === 'scanning' &&
      startedAtRef.current === null
    ) {
      startedAtRef.current = Date.now();
    }
  }, [state.kind]);

  // Reset the scan store on mount so a re-entry from a previous abort
  // starts clean. (The rotation-capture hook handles its own teardown
  // on unmount; nothing for us to clean up here.)
  useEffect(() => {
    clearScanCaptures();
  }, [clearScanCaptures]);

  // Live silhouette frame in screen-px — translates the tracker's
  // 160×240 buffer coords to the camera viewport. When the detector
  // surfaces a real vertical extent we use it; when it can't (cap
  // narrowing escapes the column-tolerance window), we fall back to
  // the bottle-aspect heuristic so the overlay still has a stable
  // shape.
  const silhouetteSv = useDerivedValue<SilhouetteFrame>(() => {
    const s = tracker.trackerStateSv.value.silhouette;
    const sx = screen.width / TRACKER_FRAME_W;
    const sy = screen.height / TRACKER_FRAME_H;
    if (!s.detected) {
      // Default-centered guide while we wait for the detector. The
      // overlay's opacity is 0 in this state so these numbers never
      // render — but keeping the frame stable means detection lands
      // smoothly without a jump. centerY sits at 60% of screen height
      // so the ring's leading-edge dot at 12 o'clock falls below the
      // panorama strip footprint if its visibility ever leaks.
      return {
        centerX: screen.width / 2,
        centerY: screen.height * 0.6,
        widthPx: screen.width * 0.55,
        heightPx: screen.height * 0.55,
      };
    }
    const widthScreen = s.widthPx * sx;
    const hasMeasuredHeight = s.heightPx > 0;
    return {
      centerX: s.centerX * sx,
      centerY: hasMeasuredHeight
        ? ((s.edgeTopY + s.edgeBottomY) * 0.5) * sy
        : screen.height / 2,
      widthPx: widthScreen,
      heightPx: hasMeasuredHeight ? s.heightPx * sy : widthScreen / 0.55,
    };
  }, [tracker.trackerStateSv]);

  const detectedSv = useDerivedValue<boolean>(
    () => tracker.trackerStateSv.value.silhouette.detected,
    [tracker.trackerStateSv],
  );
  const steadinessSv = useDerivedValue<number>(
    () => tracker.trackerStateSv.value.silhouette.steadinessScore,
    [tracker.trackerStateSv],
  );
  const coverageSv = useDerivedValue<number>(
    () => tracker.trackerStateSv.value.coverage,
    [tracker.trackerStateSv],
  );
  // Phase 1 embodiment derivations — feed the silhouette overlay so
  // its scanning-glow pulse rate tracks rotation speed and its stroke
  // width tightens with the user's grip steadiness.
  const angularVelocitySv = useDerivedValue<number>(
    () => tracker.trackerStateSv.value.angularVelocity,
    [tracker.trackerStateSv],
  );
  const gripSteadinessSv = useDerivedValue<number>(
    () => tracker.trackerStateSv.value.gripSteadiness,
    [tracker.trackerStateSv],
  );

  // Audit finding: the biggest source of strip-tone variation across a
  // scan is the camera's auto-exposure re-converging between
  // checkpoints — each takePhoto() can land with a different brightness
  // baseline, banding the unrolled panorama. On entering `ready`
  // (bottle detected and steady), call Camera.focus() at the
  // silhouette's center to lock AF — on iOS this also briefly locks AE
  // for the focus convergence window, which is enough to flatten the
  // tone variance across the 8–15s scan.
  //
  // VisionCamera v4 does NOT expose a "read current exposure" API; the
  // only `exposure` surface is a bias prop premultiplied onto the
  // device's auto-exposure (range = device.minExposure..maxExposure,
  // 0 = neutral). Since we can't snapshot the current AE point and pin
  // it, we accept some residual AE drift and rely on focus() for the
  // AF lock. AE/AF re-engages naturally on the next scan when this
  // screen re-mounts (failed/cancel paths route through router.back or
  // router.replace), so no explicit "release" is needed.
  useEffect(() => {
    if (state.kind !== 'ready') return;
    const silhouette = silhouetteSv.value;
    const x = silhouette.centerX;
    const y = silhouette.centerY;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    void (async () => {
      try {
        await cameraRef.current?.focus({ x, y });
      } catch (err) {
        // Camera.focus() throws if the device is busy mid-capture or
        // not yet initialized — both transient and recoverable. Log
        // and move on; a missed AF lock degrades to "scan with mild
        // tone variance", not failure.
        console.warn('Camera.focus() failed', err);
      }
    })();
  }, [state.kind, silhouetteSv]);

  // JS-side snapshots of tracker fields the React chrome reads. The
  // shared value updates at frame rate; the UI bands (instruction copy
  // thresholds, chip reasons, ring direction) change at human cadence,
  // so a single 5 Hz interval with per-field deduplication is plenty
  // and avoids running four parallel timers off the same source.
  const [snap, setSnap] = useState<TrackerSnapshot>(() => {
    const ts = tracker.trackerStateSv.value;
    return {
      preCheck: ts.preCheck,
      rotationDirection: ts.rotationDirection,
      coverage: ts.coverage,
      steadiness: ts.silhouette.steadinessScore,
      detected: ts.silhouette.detected,
    };
  });
  useEffect(() => {
    const id = setInterval(() => {
      const ts = tracker.trackerStateSv.value;
      setSnap((prev) => {
        const nextPreCheck = preCheckChanged(prev.preCheck, ts.preCheck)
          ? ts.preCheck
          : prev.preCheck;
        const nextDir =
          ts.rotationDirection === prev.rotationDirection
            ? prev.rotationDirection
            : ts.rotationDirection;
        const nextCov =
          Math.abs(ts.coverage - prev.coverage) > 0.005
            ? ts.coverage
            : prev.coverage;
        const nextSteady =
          Math.abs(ts.silhouette.steadinessScore - prev.steadiness) > 0.05
            ? ts.silhouette.steadinessScore
            : prev.steadiness;
        const nextDetected =
          ts.silhouette.detected === prev.detected
            ? prev.detected
            : ts.silhouette.detected;

        if (
          nextPreCheck === prev.preCheck &&
          nextDir === prev.rotationDirection &&
          nextCov === prev.coverage &&
          nextSteady === prev.steadiness &&
          nextDetected === prev.detected
        ) {
          return prev;
        }
        return {
          preCheck: nextPreCheck,
          rotationDirection: nextDir,
          coverage: nextCov,
          steadiness: nextSteady,
          detected: nextDetected,
        };
      });
    }, 200);
    return () => clearInterval(id);
  }, [tracker.trackerStateSv]);

  // Capture pipeline — fires takePhoto at each angular checkpoint and
  // appends the resulting strip to the panorama state. Disabled when
  // we're not actively scanning so paused / complete states don't
  // accumulate spurious strips.
  const captureEnabled = state.kind === 'scanning';
  const { panoramaState, cancel: cancelCaptures } = useRotationCapture({
    cameraRef,
    trackerStateSv: tracker.trackerStateSv,
    bumpCapturedCheckpoints: tracker.bumpCapturedCheckpoints,
    onFrameCaptured: appendFrame,
    enabled: captureEnabled,
  });

  // Stitch + commit once coverage hits 1.0 *and* the strip queue has
  // been quiet for STITCH_QUIET_MS. Each new strip arrives via
  // panoramaState; this effect reschedules the timer on every change,
  // so we naturally wait for the trailing 1-2 in-flight captures to
  // land before encoding. `stitchedRef` flips inside the timer so the
  // effect can keep re-running until the queue settles. `cancelledRef`
  // is checked after the async work so a cancel mid-stitch resolves
  // into a no-op rather than pushing a panorama into a torn-down scan.
  const stitchedRef = useRef(false);
  const cancelledRef = useRef(false);
  useEffect(() => {
    if (stitchedRef.current) return;
    if (state.kind !== 'scanning') return;
    if (state.coverage < 1.0) return;

    const timer = setTimeout(() => {
      if (stitchedRef.current || cancelledRef.current) return;
      stitchedRef.current = true;
      void (async () => {
        try {
          const liveSnapshot = snapshotSv.value;
          if (!liveSnapshot) {
            // Surface allocation must have failed earlier — without it
            // there's nothing to encode. Treat as capture error rather
            // than shipping an empty JPEG.
            console.warn('stitchPanorama: snapshotSv null at completion');
            fail('capture_error');
            return;
          }
          const result = await stitchPanorama(
            liveSnapshot,
            panoramaState,
            { startedAt: startedAtRef.current ?? Date.now() },
          );
          if (cancelledRef.current) return;
          setPanorama(result);
          complete(result.uri);
        } catch (err) {
          if (cancelledRef.current) return;
          console.warn('stitchPanorama failed', err);
          fail('capture_error');
        }
      })();
    }, STITCH_QUIET_MS);

    return () => clearTimeout(timer);
  }, [state, panoramaState, setPanorama, complete, fail, snapshotSv]);

  // Haptics — fire-and-forget edge events.
  useScanHaptics(state);

  // Cancel handler — drains in-flight captures, resets the store,
  // dismisses the modal back to the previous screen. Sets
  // `cancelledRef` first so an in-flight stitch resolves into a no-op.
  const handleCancel = useCallback(() => {
    cancelledRef.current = true;
    cancelCaptures();
    cancel();
    clearScanCaptures();
    reset();
    router.back();
  }, [cancelCaptures, cancel, clearScanCaptures, reset]);

  // Completion reveal hand-off to the review screen. Fires only once
  // the §8 storyboard timer signals it's safe to navigate.
  const handleRevealComplete = useCallback(() => {
    router.replace('/(app)/scan/review');
  }, []);

  // Failure recovery: re-mount this screen so all integrators (state
  // machine, capture queue, panorama) start clean.
  const handleFailureRetry = useCallback(() => {
    router.replace('/(app)/scan/unwrap');
  }, []);

  // Layout for the panorama strip — top 35% of screen. Used both to
  // size the canvas and as the `panoramaFrame` prop for the
  // CompletionReveal sparkle path.
  const panoramaFrame = useMemo(
    () => ({
      x: scanGeometry.panoramaPaddingHorizontal,
      y: insets.top + scanGeometry.panoramaPaddingTop,
      width: screen.width - scanGeometry.panoramaPaddingHorizontal * 2,
      height:
        screen.height * scanGeometry.panoramaTopHeightFraction -
        (insets.top + scanGeometry.panoramaPaddingTop) -
        12,
    }),
    [insets.top, screen.width, screen.height],
  );

  // Discriminated payloads — destructure the unions outside JSX so the
  // sub-component prop types narrow correctly.
  const pauseReason = state.kind === 'paused' ? state.reason : undefined;
  const failReason = state.kind === 'failed' ? state.reason : undefined;
  const stateKind = state.kind;

  // Camera is only "live" while the user is actively capturing. During
  // the 1340ms completion reveal and the failure card, freezing the
  // camera releases the worklet, the resize plugin, and (most
  // importantly) frees GPU/encoder bandwidth for the stitcher's JPEG
  // encode + the reveal animation. Cancelled also takes the camera
  // down so a back-press doesn't keep frames flowing during route
  // transition.
  const cameraActive =
    stateKind === 'aligning' ||
    stateKind === 'ready' ||
    stateKind === 'scanning' ||
    stateKind === 'paused';

  return (
    <View style={styles.root}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={cameraActive}
        photo
        frameProcessor={tracker.frameProcessor}
        // YUV pixel format keeps the resize plugin's RGB output trivial
        // (matches the deleted camera/[surface].tsx).
        pixelFormat="yuv"
      />

      {/* Top-of-screen panorama strip — the live unrolled label. The
          PanoramaCanvas owns its own Skia surface and only repaints
          new strips, so its per-frame cost is flat regardless of how
          many strips have landed. */}
      <View
        pointerEvents="none"
        style={[
          styles.panoramaWrap,
          {
            top: panoramaFrame.y,
            left: panoramaFrame.x,
            width: panoramaFrame.width,
            height: panoramaFrame.height,
          },
        ]}
      >
        <PanoramaCanvas
          state={panoramaState}
          snapshotSv={snapshotSv}
          style={styles.panoramaCanvas}
        />
      </View>

      {/* Bottle silhouette overlay — drawn in screen space over the
          camera feed, follows `silhouetteSv`. */}
      <BottleSilhouetteOverlay
        silhouetteSv={silhouetteSv}
        detectedSv={detectedSv}
        steadinessSv={steadinessSv}
        angularVelocitySv={angularVelocitySv}
        gripSteadinessSv={gripSteadinessSv}
        state={stateKind}
        pauseReason={pauseReason}
        viewportWidth={screen.width}
        viewportHeight={screen.height}
      />

      {/* Rotation guide ring — concentric with the silhouette;
          arc fills with coverage. Hidden during aligning so its
          leading-edge dot doesn't read as a stray UI artifact before
          the bottle is detected. */}
      {stateKind !== 'aligning' && (
        <RotationGuideRing
          coverageSv={coverageSv}
          silhouetteSv={silhouetteSv}
          state={stateKind}
          viewportWidth={screen.width}
          viewportHeight={screen.height}
          rotationDirection={snap.rotationDirection}
        />
      )}

      {/* Cancel button — top-left, with confirm-dialog when coverage
          > 0.05 (CancelButton owns the confirm). Hidden once the scan
          is complete or failed: the user already gets the reveal /
          failure card to dismiss, and a confirm dialog over a finished
          scan would feel broken. */}
      {stateKind !== 'complete' && stateKind !== 'failed' && (
        <CancelButton
          coverage={coverageOf(state)}
          onCancel={handleCancel}
        />
      )}

      {/* Quality chip — top-right; visibility rules per SCAN_DESIGN §3.7. */}
      <QualityChip
        state={stateKind}
        preCheck={snap.preCheck}
        pauseReason={pauseReason}
      />

      {/* In-scan warning banner — soft nudge during the late-stretch
          coverage band (240°..300°) that the back label, where the
          §16.21 Government Warning typically lives, may not have been
          captured yet. Heuristic-only; no streaming verify. The banner
          self-hides once coverage clears 300° (the user has rotated
          enough to confidently catch the back label). */}
      <InScanWarningBanner
        state={stateKind}
        coverage={snap.coverage}
        topPx={panoramaFrame.y + panoramaFrame.height + 12}
      />

      {/* Adaptive instruction copy — center-bottom pill. */}
      <ScanInstructions
        state={stateKind}
        coverage={snap.coverage}
        steadiness={snap.steadiness}
        pauseReason={pauseReason}
        failReason={failReason}
        bottleDetected={snap.detected}
      />

      {/* Auto-capture countdown pill. Visible only while we're in
          `ready` and the live predicate is true but the 1.5s hold
          hasn't latched yet — once it latches, the state machine
          flips to `scanning` and this disappears. The pill lives just
          below the silhouette so the cue reads near the bottle the
          user is holding, not down by the dial. */}
      {stateKind === 'ready' && autoCaptureCandidate ? (
        <View
          pointerEvents="none"
          style={[styles.autoCapturePill, { top: insets.top + scanGeometry.panoramaPaddingTop + 12 }]}
        >
          <Text style={styles.autoCapturePillText}>
            Hold steady — starting in 1.5s
          </Text>
        </View>
      ) : null}

      {/* Manual-override button. Surfaces only after 8 seconds in
          `ready` without the auto-capture predicate ever sustaining
          (dim labels, harsh contrast, anything the heuristic can't
          confidently classify). Sits above the instruction pill so
          the user can tap their way through without hunting for it. */}
      {stateKind === 'ready' && manualOverrideAvailable ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Tap to start scanning manually"
          onPress={manualStart}
          style={({ pressed }) => [
            styles.manualStartButton,
            { bottom: insets.bottom + 96 + 56 },
            pressed && styles.manualStartButtonPressed,
          ]}
        >
          <Text style={styles.manualStartButtonText}>
            Tap to start manually
          </Text>
        </Pressable>
      ) : null}

      {/* Progress dial — bottom-right, big number + caption. */}
      <ProgressDial coverageSv={coverageSv} state={stateKind} />

      {/* Completion reveal — overlays everything when state.kind ===
          'complete', then signals the review hand-off. */}
      <CompletionReveal
        active={stateKind === 'complete'}
        panoramaFrame={panoramaFrame}
        onComplete={handleRevealComplete}
      />

      {/* Failure recovery card — sits above the camera once we've
          entered a terminal failure state. */}
      {stateKind === 'failed' && failReason !== 'cancelled' ? (
        <FailureCard
          reason={failReason}
          onRetry={handleFailureRetry}
          onDismiss={handleCancel}
        />
      ) : null}
    </View>
  );
}

function FailureCard({
  reason,
  onRetry,
  onDismiss,
}: {
  reason: 'permission_denied' | 'no_camera' | 'capture_error' | undefined;
  onRetry: () => void;
  onDismiss: () => void;
}): React.ReactElement {
  const text =
    reason === 'permission_denied'
      ? 'Camera access needed'
      : reason === 'no_camera'
        ? 'No camera available'
        : 'Something went wrong — try again';
  return (
    <View style={styles.failureWrap} pointerEvents="auto">
      <Text style={styles.failureTitle}>{text}</Text>
      <View style={styles.failureActions}>
        <Button label="Try again" onPress={onRetry} />
        <Button label="Cancel" variant="ghost" onPress={onDismiss} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#000',
  },
  panoramaWrap: {
    position: 'absolute',
    borderRadius: scanGeometry.panoramaCornerRadius,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.panoramaFrameStroke,
    overflow: 'hidden',
  },
  panoramaCanvas: {
    flex: 1,
    backgroundColor: colors.panoramaBg,
  },
  permissionWrap: {
    flex: 1,
    backgroundColor: colors.background,
    padding: spacing.lg,
    gap: spacing.md,
    justifyContent: 'center',
  },
  permissionTitle: {
    ...typography.title,
    color: colors.text,
  },
  permissionBody: {
    ...typography.body,
    color: colors.textMuted,
  },
  failureWrap: {
    position: 'absolute',
    left: spacing.lg,
    right: spacing.lg,
    top: '40%',
    backgroundColor: colors.surface,
    borderRadius: 14,
    padding: spacing.lg,
    gap: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  failureTitle: {
    ...typography.heading,
    color: colors.text,
    textAlign: 'center',
  },
  failureActions: {
    flexDirection: 'row',
    gap: spacing.md,
    width: '100%',
    justifyContent: 'center',
  },
  // Auto-capture countdown pill. Reuses the same scrim + pill shape as
  // ScanInstructions for visual continuity (small, peripheral pill on
  // the camera feed) but sits in the upper-third near the silhouette
  // outline rather than at the bottom — the bottle is what the user is
  // looking at and the cue should land where their attention is. No
  // new colors; reuses scanReadySoft border so the pill reads as the
  // "almost-ready" cue without competing with the warn / fail palette.
  autoCapturePill: {
    position: 'absolute',
    alignSelf: 'center',
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  autoCapturePillText: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: colors.scanOverlayScrim,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.scanReadySoft,
    color: colors.scanInk,
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.2,
    overflow: 'hidden',
  },
  // Manual-override button. Subtle, low-emphasis affordance — the
  // spec calls for "subtle" because in the happy path the user never
  // sees it; only the long-tail lighting cases ever surface it.
  manualStartButton: {
    position: 'absolute',
    alignSelf: 'center',
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 999,
    backgroundColor: colors.scanOverlayDim,
    borderWidth: 1,
    borderColor: colors.scanIdleSoft,
  },
  manualStartButtonPressed: {
    opacity: 0.7,
  },
  manualStartButtonText: {
    color: colors.scanInk,
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.2,
    textAlign: 'center',
  },
});
