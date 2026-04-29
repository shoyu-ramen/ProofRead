/**
 * ScanInstructions — the single-line adaptive copy band centered above
 * the progress dial (SCAN_DESIGN §3.6 + §4.5 + §6).
 *
 * Reads the active ScanState (and current coverage for the scanning
 * substates), derives the canonical string from the §6 copy ladder,
 * and crossfades between strings via a paired translateY on the
 * outgoing/incoming line. `accessibilityLiveRegion="polite"` makes the
 * announcement land on screen readers without interrupting the user.
 */

import React, { useEffect, useRef, useState } from 'react';
import { AccessibilityInfo, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withTiming,
} from 'react-native-reanimated';

import { colors } from '@src/theme';

export type ScanStateKind =
  | 'aligning'
  | 'ready'
  | 'scanning'
  | 'paused'
  | 'complete'
  | 'failed';

export type PauseReason =
  | 'too_fast'
  | 'too_slow'
  | 'lost_bottle'
  | 'blur'
  | 'glare'
  | 'motion'
  | 'too_far'
  | 'too_close'
  | 'untrackable_surface';

export type FailReason =
  | 'permission_denied'
  | 'no_camera'
  | 'capture_error'
  | 'cancelled';

export interface ScanInstructionsProps {
  /** Top-level discriminator from the scan machine. */
  state: ScanStateKind;
  /**
   * Current 0..1 coverage. Selects the substate copy when state is
   * `scanning` via `selectScanningBand` (thresholds at
   * 0.20 / 0.45 / 0.60 / 0.75 / 0.85, with a 0.005 hysteresis dead-band).
   */
  coverage?: number;
  /**
   * Steadiness score of the bottle silhouette, 0..1. >0.9 picks the
   * "Almost there — hold steady" intermediate when state is `aligning`.
   */
  steadiness?: number;
  /** Pause reason when state is `paused`. */
  pauseReason?: PauseReason;
  /** Fail reason when state is `failed`. */
  failReason?: FailReason;
  /**
   * Whether the tracker currently has a confident silhouette lock.
   * Used during `aligning` to swap copy from "Hold the bottle in the
   * frame" to "Center, then rotate slowly" once detection lands, so
   * the user pre-loads the rotation action.
   */
  bottleDetected?: boolean;
}

interface CopyDescriptor {
  /** Literal text to display, matches §6 ladder. */
  text: string;
  /**
   * If set, a trailing accent token of the displayed string is colored
   * via `instructionAccent` (the ✓ on "Got it ✓").
   */
  accentColor?: string;
  /** Number of trailing characters to color with accentColor. */
  accentLength?: number;
}

/**
 * Scanning-state copy bands. Upper bounds are the *enter* thresholds;
 * a 0.005 hysteresis dead-band on exit (see `selectScanningBand`)
 * prevents flicker when coverage oscillates around a boundary.
 */
type ScanningBand = 0 | 1 | 2 | 3 | 4 | 5;
const SCANNING_BAND_UPPER: readonly number[] = [0.2, 0.45, 0.6, 0.75, 0.85];
const SCANNING_BAND_TEXT: readonly string[] = [
  'Keep rotating',
  "You're getting it",
  'About halfway',
  'Two-thirds there',
  'Almost done',
  'Almost done',
];
const BAND_HYSTERESIS = 0.005;

function selectScanningBand(coverage: number, prev: ScanningBand): ScanningBand {
  // Walk the thresholds; only switch bands once coverage clears the
  // boundary by `BAND_HYSTERESIS` in the appropriate direction.
  let next: ScanningBand = 0;
  for (let i = 0; i < SCANNING_BAND_UPPER.length; i += 1) {
    if (coverage >= SCANNING_BAND_UPPER[i]) next = (i + 1) as ScanningBand;
  }
  if (next === prev) return prev;
  if (next > prev) {
    // Rising: require coverage to clear the entry threshold by the dead-band.
    const enterThreshold = SCANNING_BAND_UPPER[prev];
    return coverage >= enterThreshold + BAND_HYSTERESIS ? next : prev;
  }
  // Falling: require coverage to drop below the exit threshold by the dead-band.
  const exitThreshold = SCANNING_BAND_UPPER[next];
  return coverage <= exitThreshold - BAND_HYSTERESIS ? next : prev;
}

function describe(
  props: ScanInstructionsProps,
  scanningBand: ScanningBand,
): CopyDescriptor | null {
  const {
    state,
    steadiness = 0,
    pauseReason,
    failReason,
    bottleDetected = false,
  } = props;
  switch (state) {
    case 'aligning':
      if (steadiness >= 0.9) return { text: 'Almost there — hold steady' };
      if (bottleDetected) return { text: 'Center, then rotate slowly' };
      return { text: 'Hold the bottle in the frame' };
    case 'ready':
      return { text: 'Ready. Rotate slowly →' };
    case 'scanning':
      return { text: SCANNING_BAND_TEXT[scanningBand] };
    case 'paused':
      switch (pauseReason) {
        case 'too_fast':
          return { text: 'Slow down a little' };
        case 'too_slow':
          return { text: 'Keep rotating' };
        case 'lost_bottle':
          return { text: 'Bring the bottle back' };
        case 'blur':
          return { text: 'Hold steady' };
        case 'glare':
          return { text: 'Reduce glare' };
        case 'motion':
          return { text: 'Hold the phone still' };
        case 'too_far':
          return { text: 'Move closer to the bottle' };
        case 'too_close':
          return { text: 'Move the bottle back a bit' };
        case 'untrackable_surface':
          return { text: 'Try a labeled bottle or can' };
        default:
          return { text: 'Hold steady' };
      }
    case 'complete':
      return {
        text: 'Got it ✓',
        accentColor: colors.scanReady,
        accentLength: 1,
      };
    case 'failed':
      switch (failReason) {
        case 'permission_denied':
          return { text: 'Camera access needed' };
        case 'no_camera':
          return { text: 'No camera available' };
        case 'capture_error':
          return { text: 'Something went wrong — try again' };
        case 'cancelled':
        default:
          return null;
      }
    default:
      return null;
  }
}

export function ScanInstructions(
  props: ScanInstructionsProps,
): React.ReactElement | null {
  const insets = useSafeAreaInsets();
  // Tracks the last-displayed scanning copy band. Persisting across
  // renders + applying a 0.005 hysteresis dead-band in selectScanningBand
  // keeps the line steady when coverage jitters around a threshold.
  const scanningBandRef = useRef<ScanningBand>(0);
  if (props.state === 'scanning') {
    scanningBandRef.current = selectScanningBand(
      props.coverage ?? 0,
      scanningBandRef.current,
    );
  } else {
    scanningBandRef.current = 0;
  }
  const desc = describe(props, scanningBandRef.current);

  // Two layered Animated.Text — outgoing and incoming. We toggle which
  // slot owns the next copy each transition to avoid identity churn.
  const [activeText, setActiveText] = useState<CopyDescriptor | null>(desc);
  const [outgoingText, setOutgoingText] =
    useState<CopyDescriptor | null>(null);
  const lastTextRef = useRef<string | null>(desc?.text ?? null);

  const inOpacity = useSharedValue<number>(desc ? 1 : 0);
  const inTranslateY = useSharedValue<number>(0);
  const outOpacity = useSharedValue<number>(0);
  const outTranslateY = useSharedValue<number>(0);

  useEffect(() => {
    const nextText = desc?.text ?? null;
    if (nextText === lastTextRef.current) return;
    lastTextRef.current = nextText;

    // Transition: outgoing slot animates out, incoming animates in with
    // an 80ms overlap delay (§4.5). The state rotation is committed
    // before kicking off the timings so the JS-side render owns the
    // text we're animating.
    setOutgoingText(activeText);
    setActiveText(desc ?? null);

    outOpacity.value = 1;
    outTranslateY.value = 0;
    outOpacity.value = withTiming(0, {
      duration: 160,
      easing: Easing.in(Easing.cubic),
    });
    outTranslateY.value = withTiming(-6, {
      duration: 160,
      easing: Easing.in(Easing.cubic),
    });

    inOpacity.value = 0;
    inTranslateY.value = 6;
    inOpacity.value = withDelay(
      80,
      withTiming(desc ? 1 : 0, {
        duration: 220,
        easing: Easing.out(Easing.cubic),
      }),
    );
    inTranslateY.value = withDelay(
      80,
      withTiming(0, { duration: 220, easing: Easing.out(Easing.cubic) }),
    );
  }, [
    desc,
    activeText,
    inOpacity,
    inTranslateY,
    outOpacity,
    outTranslateY,
  ]);

  const inStyle = useAnimatedStyle(() => ({
    opacity: inOpacity.value,
    transform: [{ translateY: inTranslateY.value }],
  }));
  const outStyle = useAnimatedStyle(() => ({
    opacity: outOpacity.value,
    transform: [{ translateY: outTranslateY.value }],
  }));

  // Reduce-motion: skip the slide and just drop opacity to 0/1. The
  // shared values above still drive opacity; the translateY is harmless
  // at 0 so we leave it.
  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled().then((reduced) => {
      if (!reduced) return;
      inTranslateY.value = 0;
      outTranslateY.value = 0;
    }).catch(() => {});
  }, [inTranslateY, outTranslateY]);

  if (!activeText && !outgoingText) return null;

  return (
    <View
      pointerEvents="none"
      accessibilityRole="text"
      accessibilityLiveRegion="polite"
      accessibilityLabel={activeText?.text ?? ''}
      style={[styles.wrap, { bottom: insets.bottom + 96 }]}
    >
      <View style={styles.pill}>
        {/* Outgoing layer */}
        {outgoingText ? (
          <Animated.Text
            numberOfLines={1}
            style={[styles.text, styles.absoluteFill, outStyle]}
          >
            {renderWithAccent(outgoingText)}
          </Animated.Text>
        ) : null}
        {/* Incoming layer */}
        {activeText ? (
          <Animated.Text numberOfLines={1} style={[styles.text, inStyle]}>
            {renderWithAccent(activeText)}
          </Animated.Text>
        ) : null}
      </View>
    </View>
  );
}

function renderWithAccent(
  desc: CopyDescriptor,
): React.ReactElement | string {
  if (!desc.accentLength || !desc.accentColor) return desc.text;
  const head = desc.text.slice(0, desc.text.length - desc.accentLength);
  const tail = desc.text.slice(desc.text.length - desc.accentLength);
  return (
    <>
      {head}
      <Animated.Text style={{ color: desc.accentColor, fontWeight: '700' }}>
        {tail}
      </Animated.Text>
    </>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  pill: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 999,
    backgroundColor: colors.scanOverlayScrim,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.panoramaFrameStroke,
    minHeight: 22 + 20,
    minWidth: 120,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    fontSize: 17,
    fontWeight: '600',
    letterSpacing: -0.1,
    lineHeight: 22,
    color: colors.scanInk,
  },
  absoluteFill: {
    position: 'absolute',
    left: 16,
    right: 16,
    top: 10,
    textAlign: 'center',
  },
});
