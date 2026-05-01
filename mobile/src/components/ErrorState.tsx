/**
 * ErrorState — centered failure card with retry affordance.
 *
 * Used when an async load fails (review submission, report fetch,
 * history fetch). Visual weight is intentionally heavier than
 * `EmptyState`: the icon ring inherits `colors.fail`, the title is
 * a `titleLg`, and the retry button is the primary CTA so the user
 * doesn't need to hunt for it.
 *
 * Errors are surfaced as a *blocking* surface — by design we don't
 * show a small inline pill for these, because the screen has nothing
 * else to render until the load succeeds.
 */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { colors, spacing, typography } from '@src/theme';
import { Button } from './Button';
import { Icon } from './Icon';

export interface ErrorStateProps {
  title: string;
  description?: string;
  /** Required: errors should always offer a retry. */
  retry: () => void;
  /** Override the default "Retry" button label. */
  retryLabel?: string;
  /** Optional secondary action (e.g. "Back to home"). */
  secondaryAction?: { label: string; onPress: () => void };
  testID?: string;
}

export function ErrorState({
  title,
  description,
  retry,
  retryLabel = 'Retry',
  secondaryAction,
  testID,
}: ErrorStateProps): React.ReactElement {
  return (
    <View
      accessible
      accessibilityRole="alert"
      accessibilityLabel={
        description ? `${title}. ${description}` : title
      }
      testID={testID}
      style={styles.wrap}
    >
      <View style={styles.iconCircle}>
        <Icon name="alert-circle" size={28} color={colors.fail} />
      </View>
      <Text style={styles.title}>{title}</Text>
      {description ? (
        <Text style={styles.description}>{description}</Text>
      ) : null}
      <View style={styles.actionWrap}>
        <Button label={retryLabel} onPress={retry} />
        {secondaryAction ? (
          <Button
            label={secondaryAction.label}
            variant="ghost"
            onPress={secondaryAction.onPress}
          />
        ) : null}
      </View>
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
    // Subtle red wash so the icon ring announces the error severity
    // without overwhelming the surrounding chrome.
    backgroundColor: 'rgba(255,107,107,0.10)',
    borderColor: colors.fail,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.xs,
  },
  title: {
    ...typography.titleLg,
    color: colors.text,
    textAlign: 'center',
  },
  description: {
    ...typography.bodyMd,
    color: colors.textMuted,
    textAlign: 'center',
    maxWidth: 320,
  },
  actionWrap: {
    marginTop: spacing.sm,
    gap: spacing.sm,
    alignItems: 'center',
  },
});
