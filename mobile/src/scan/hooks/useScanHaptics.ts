/**
 * useScanHaptics — fires the haptic taps the scan flow specifies in
 * SCAN_DESIGN.md §7. Side-effect only; returns nothing.
 *
 * Edge-detection rule: every haptic fires on the *transition*, never
 * on a steady state. Coverage-milestone haptics are owned by
 * `RotationGuideRing` per §7, so this hook only watches `ScanState`.
 *
 * Errors from `expo-haptics` are swallowed — simulators and devices
 * without haptics engines silently no-op.
 */

import { useEffect, useRef } from 'react';
import * as Haptics from 'expo-haptics';

import type { ScanState } from '@src/scan/state/scanMachine';

export function useScanHaptics(state: ScanState): void {
  const prevKindRef = useRef<ScanState['kind'] | null>(null);
  const prevPauseReasonRef = useRef<string | null>(null);

  useEffect(() => {
    const prevKind = prevKindRef.current;
    const kind = state.kind;
    if (kind === prevKind) {
      // Pause-reason edge while still paused: surface a single
      // warning if the reason actually changes (debounced upstream).
      if (kind === 'paused') {
        const r = state.reason;
        if (prevPauseReasonRef.current !== r) {
          void Haptics.notificationAsync(
            Haptics.NotificationFeedbackType.Warning,
          ).catch(() => {});
          prevPauseReasonRef.current = r;
        }
      }
      return;
    }
    prevKindRef.current = kind;
    if (kind !== 'paused') prevPauseReasonRef.current = null;

    if (kind === 'ready') {
      // aligning → ready. Light impact.
      void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(
        () => {},
      );
    } else if (kind === 'scanning' && prevKind === 'ready') {
      // ready → scanning. Medium impact (first measurable rotation).
      void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(
        () => {},
      );
    } else if (kind === 'scanning' && prevKind === 'paused') {
      // paused → scanning recovery. Light impact reward.
      void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(
        () => {},
      );
    } else if (kind === 'paused') {
      void Haptics.notificationAsync(
        Haptics.NotificationFeedbackType.Warning,
      ).catch(() => {});
      prevPauseReasonRef.current = state.reason;
    } else if (kind === 'complete') {
      // Coverage hits 1.0 → success. The CompletionReveal component
      // also fires Success at t=0; we let that be the single source
      // of truth for the completion notification (see
      // CompletionReveal.tsx). No-op here.
    } else if (kind === 'failed') {
      void Haptics.notificationAsync(
        Haptics.NotificationFeedbackType.Error,
      ).catch(() => {});
    }
  }, [state]);
}
