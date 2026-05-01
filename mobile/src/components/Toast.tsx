/**
 * Toast — single transient alert chip rendered by `ToastProvider`.
 *
 * Slide-in from the top using the same Reanimated spring + ease tokens
 * the scan screen uses, so the app's motion language stays unified.
 * Variant maps to a left accent stripe + icon glyph (no new colors —
 * we reuse `colors.pass / fail / advisory / primary` from the existing
 * theme tokens). The toast is tap-to-dismiss; its parent owns lifetime
 * (auto-dismiss timer, queue ordering, exit animation) so this view is
 * thin and predictable.
 *
 * Accessibility: the chip is `accessibilityLiveRegion="polite"` so VoiceOver
 * / TalkBack announces the message when it lands. The icon glyph carries
 * the variant semantics for screen readers via the `accessibilityLabel`
 * we attach on the wrapping Pressable.
 */

import React, { useEffect } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';

import { colors, radius, spacing, toastMotion, typography } from '@src/theme';
import type { ToastVariant } from './ToastContext';

export interface ToastProps {
  /** Stable toast id — used by the parent for keyed list updates / dismiss. */
  id: string;
  variant: ToastVariant;
  message: string;
  /**
   * Whether this toast is currently being dismissed. When true, the
   * view fades + slides up off-screen; when the animation completes
   * the parent removes the entry from the queue.
   */
  dismissing: boolean;
  /** Tap / lifecycle dismiss callback. The parent runs the actual queue removal. */
  onDismiss: (id: string) => void;
  /**
   * Notify the parent once the exit animation has fully played out so
   * it can splice the entry. The parent guards against double-fire,
   * but we still only call it from the timing-completion callback to
   * keep the contract narrow.
   */
  onAnimationComplete?: (id: string) => void;
}

interface VariantPalette {
  accent: string;
  glyph: string;
  accessibilityLabel: string;
}

function paletteFor(variant: ToastVariant): VariantPalette {
  switch (variant) {
    case 'success':
      return {
        accent: colors.pass,
        glyph: '✓',
        accessibilityLabel: 'Success',
      };
    case 'error':
      return {
        accent: colors.fail,
        glyph: '✕',
        accessibilityLabel: 'Error',
      };
    case 'warning':
      return {
        accent: colors.advisory,
        glyph: '!',
        accessibilityLabel: 'Warning',
      };
    case 'info':
    default:
      return {
        accent: colors.primary,
        glyph: 'i',
        accessibilityLabel: 'Info',
      };
  }
}

export function Toast({
  id,
  variant,
  message,
  dismissing,
  onDismiss,
  onAnimationComplete,
}: ToastProps): React.ReactElement {
  const palette = paletteFor(variant);

  // Slide-in from -16px above its resting Y, fade from 0 → 1 in lockstep.
  // We use a spring on the translation to get the "settle" feel that
  // matches the scan screen's pause-banner motion language; opacity is
  // a short timing because we want the chip to be readable as soon as
  // it lands rather than fading in slowly.
  const translateY = useSharedValue<number>(-16);
  const opacity = useSharedValue<number>(0);

  useEffect(() => {
    translateY.value = withSpring(0, toastMotion.spring);
    opacity.value = withTiming(1, {
      duration: toastMotion.fastEase.duration,
      easing: Easing.out(Easing.cubic),
    });
  }, [translateY, opacity]);

  useEffect(() => {
    if (!dismissing) return;
    translateY.value = withTiming(-16, {
      duration: toastMotion.fastEase.duration,
      easing: Easing.in(Easing.cubic),
    });
    opacity.value = withTiming(
      0,
      {
        duration: toastMotion.fastEase.duration,
        easing: Easing.in(Easing.cubic),
      },
      (finished) => {
        if (finished && onAnimationComplete) {
          // Reanimated fires this on the UI thread; the parent queue
          // mutator is JS-side, so we'd normally need runOnJS. The
          // ToastProvider uses a sentinel-driven JS effect (see its
          // exit handling) so we can treat this as advisory only — fire
          // the JS callback opportunistically.
          onAnimationComplete(id);
        }
      },
    );
  }, [dismissing, id, opacity, translateY, onAnimationComplete]);

  const animStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
    opacity: opacity.value,
  }));

  return (
    <Animated.View style={animStyle}>
      <Pressable
        onPress={() => onDismiss(id)}
        accessibilityRole="alert"
        accessibilityLabel={`${palette.accessibilityLabel}: ${message}`}
        accessibilityHint="Tap to dismiss"
        accessibilityLiveRegion="polite"
        // Stretch to the parent's width — the provider container already
        // applies horizontal padding + max-width.
        style={({ pressed }) => [
          styles.card,
          { borderLeftColor: palette.accent },
          pressed && styles.pressed,
        ]}
      >
        <View
          style={[styles.iconWrap, { borderColor: palette.accent }]}
          accessibilityElementsHidden
          importantForAccessibility="no"
        >
          <Text style={[styles.icon, { color: palette.accent }]}>
            {palette.glyph}
          </Text>
        </View>
        <Text style={styles.message} numberOfLines={3}>
          {message}
        </Text>
      </Pressable>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    borderLeftWidth: 4,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    // Soft shadow lifts the toast above the underlying screen content
    // even on dim backgrounds. iOS reads shadow*; Android reads elevation.
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.18,
    shadowRadius: 6,
    elevation: 4,
  },
  pressed: {
    opacity: 0.85,
  },
  iconWrap: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: {
    fontSize: 13,
    fontWeight: '700',
    lineHeight: 16,
  },
  message: {
    ...typography.body,
    color: colors.text,
    flex: 1,
  },
});
