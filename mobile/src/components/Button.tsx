import React from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from 'react-native';
import { colors, interaction, radius, spacing, typography } from '@src/theme';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'md' | 'lg';

export interface ButtonProps {
  label: string;
  onPress?: () => void;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  /**
   * When true, the label is replaced with a centered spinner. The
   * outer button width is preserved (the label is rendered with
   * `opacity: 0` rather than removed) so the layout doesn't shift on
   * state change.
   */
  loading?: boolean;
  fullWidth?: boolean;
  style?: ViewStyle;
  testID?: string;
}

/**
 * Button — primary/secondary/ghost/danger CTA used throughout the app.
 *
 * Interaction states (UI/UX pass):
 *   - **Pressed** — `interaction.pressed.opacity` (0.78) + a subtle
 *     scale-down (0.98). Both pulled from the shared theme tokens so
 *     all interactive surfaces feel uniform.
 *   - **Disabled** — `interaction.disabled.opacity` (0.4). No pointer
 *     events; `accessibilityState.disabled` set so VoiceOver / TalkBack
 *     announce it correctly.
 *   - **Loading** — visually identical to pressed-but-not-tappable. The
 *     label is overlaid by a spinner that takes up the same vertical
 *     space, preventing a layout-jump when the loading flips on/off.
 */
export function Button({
  label,
  onPress,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  fullWidth = false,
  style,
  testID,
}: ButtonProps): React.ReactElement {
  const isInactive = disabled || loading;
  return (
    <Pressable
      onPress={isInactive ? undefined : onPress}
      disabled={isInactive}
      accessibilityRole="button"
      accessibilityState={{ disabled: isInactive, busy: loading }}
      testID={testID}
      style={({ pressed }) => [
        styles.base,
        styles[`size_${size}`],
        styles[`variant_${variant}`],
        fullWidth && styles.fullWidth,
        pressed && !isInactive && styles.pressed,
        disabled && styles.disabled,
        loading && styles.loadingState,
        style,
      ]}
    >
      {/*
        Stack-mounted label + spinner so the button width is identical
        in both states. We keep the label rendered (visibility-hidden)
        when loading so the Pressable sizes itself to the same
        intrinsic width — important for `fullWidth={false}` callers
        whose label measurement defines the box.
      */}
      <View style={styles.row}>
        <Text
          style={[
            styles.text,
            { color: textColorFor(variant) },
            loading && styles.labelHidden,
          ]}
        >
          {label}
        </Text>
        {loading ? (
          <View style={styles.spinnerOverlay} pointerEvents="none">
            <ActivityIndicator size="small" color={textColorFor(variant)} />
          </View>
        ) : null}
      </View>
    </Pressable>
  );
}

function textColorFor(variant: ButtonVariant): string {
  switch (variant) {
    case 'primary':
      return colors.onPrimary;
    case 'danger':
      return colors.onDanger;
    case 'secondary':
      return colors.text;
    case 'ghost':
      return colors.text;
  }
}

const styles = StyleSheet.create({
  base: {
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  fullWidth: { alignSelf: 'stretch' },
  pressed: {
    opacity: interaction.pressed.opacity,
    transform: [{ scale: interaction.pressed.scale }],
  },
  disabled: { opacity: interaction.disabled.opacity },
  // Loading uses a slightly higher opacity than disabled so the spinner
  // is fully readable; users still get a clear "busy" cue without
  // dimming the chrome below threshold.
  loadingState: { opacity: 0.85 },
  size_md: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  size_lg: { paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  variant_primary: { backgroundColor: colors.primary },
  variant_secondary: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
  },
  variant_ghost: { backgroundColor: 'transparent' },
  variant_danger: { backgroundColor: colors.danger },
  text: {
    ...typography.button,
  },
  labelHidden: {
    opacity: 0,
  },
  spinnerOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
