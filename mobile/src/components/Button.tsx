import React from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from 'react-native';
import { colors, radius, spacing, typography } from '@src/theme';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'md' | 'lg';

export interface ButtonProps {
  label: string;
  onPress?: () => void;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
  style?: ViewStyle;
  testID?: string;
}

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
        isInactive && styles.inactive,
        style,
      ]}
    >
      <View style={styles.row}>
        {loading ? (
          <ActivityIndicator size="small" color={textColorFor(variant)} />
        ) : (
          <Text style={[styles.text, { color: textColorFor(variant) }]}>{label}</Text>
        )}
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
    gap: spacing.sm,
  },
  fullWidth: { alignSelf: 'stretch' },
  pressed: { opacity: 0.85 },
  inactive: { opacity: 0.5 },
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
});
