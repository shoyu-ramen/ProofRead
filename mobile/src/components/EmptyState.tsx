/**
 * EmptyState — centered message + optional CTA for empty lists / no-
 * results screens.
 *
 * Replaces the previous "small heading + muted body + buried button"
 * pattern that home / history used. The icon glyph carries the
 * semantic load (an empty inbox, a clock for "no recent scans") and
 * the optional `action` is rendered as a primary Button so the user
 * always has a clear next step.
 *
 * Accessibility: the wrapping View carries an `accessibilityRole` of
 * `summary` so VoiceOver groups the title + description as one unit
 * rather than reading each chunk separately.
 */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { colors, spacing, typography } from '@src/theme';
import { Button } from './Button';
import { Icon, type IconName } from './Icon';

export interface EmptyStateAction {
  label: string;
  onPress: () => void;
}

export interface EmptyStateProps {
  /** Optional Icon name — uses Feather/SFSymbol via the Icon wrapper. */
  icon?: IconName;
  title: string;
  description?: string;
  action?: EmptyStateAction;
  /** Optional override for testing. */
  testID?: string;
}

/**
 * Default icon when none is supplied — generic "nothing here yet".
 * Picked `inbox` because it reads as universal for "empty container".
 */
const DEFAULT_ICON: IconName = 'inbox';

export function EmptyState({
  icon = DEFAULT_ICON,
  title,
  description,
  action,
  testID,
}: EmptyStateProps): React.ReactElement {
  return (
    <View
      accessible
      accessibilityRole="summary"
      accessibilityLabel={
        description ? `${title}. ${description}` : title
      }
      testID={testID}
      style={styles.wrap}
    >
      <View style={styles.iconCircle}>
        <Icon name={icon} size={28} color={colors.textMuted} />
      </View>
      <Text style={styles.title}>{title}</Text>
      {description ? (
        <Text style={styles.description}>{description}</Text>
      ) : null}
      {action ? (
        <View style={styles.actionWrap}>
          <Button label={action.label} onPress={action.onPress} />
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    padding: spacing.xl,
  },
  iconCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.surfaceAlt,
    borderColor: colors.border,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.xs,
  },
  title: {
    ...typography.titleMd,
    color: colors.text,
    textAlign: 'center',
  },
  description: {
    ...typography.bodyMd,
    color: colors.textMuted,
    textAlign: 'center',
    maxWidth: 280,
  },
  actionWrap: {
    marginTop: spacing.sm,
  },
});
