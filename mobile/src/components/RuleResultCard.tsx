import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import type { RuleResultDTO } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';
import { StatusBadge } from './StatusBadge';

export interface RuleResultCardProps {
  result: RuleResultDTO;
  onPress?: () => void;
}

export function RuleResultCard({ result, onPress }: RuleResultCardProps) {
  const inner = (
    <>
      <View style={styles.headerRow}>
        <View style={styles.titleCol}>
          <Text style={styles.ruleId} numberOfLines={1}>
            {humanizeRuleId(result.rule_id)}
          </Text>
          <Text style={styles.citation}>{result.citation}</Text>
        </View>
        <StatusBadge status={result.status} size="sm" />
      </View>
      {result.finding ? (
        <Text style={styles.finding} numberOfLines={2}>
          {result.finding}
        </Text>
      ) : null}
    </>
  );

  if (onPress) {
    return (
      <Pressable
        onPress={onPress}
        accessibilityRole="button"
        style={({ pressed }) => [styles.card, pressed && styles.pressed]}
      >
        {inner}
      </Pressable>
    );
  }
  return <View style={styles.card}>{inner}</View>;
}

function humanizeRuleId(ruleId: string): string {
  // beer.health_warning.exact_text → "Health warning · exact text"
  const parts = ruleId.split('.');
  if (parts.length < 2) return ruleId;
  const [, ...rest] = parts;
  return rest
    .map((p) => p.replace(/_/g, ' '))
    .map(capitalize)
    .join(' · ');
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    gap: spacing.sm,
  },
  pressed: {
    opacity: 0.85,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.sm,
  },
  titleCol: {
    flex: 1,
    gap: 2,
  },
  ruleId: {
    ...typography.heading,
    color: colors.text,
  },
  citation: {
    ...typography.caption,
    color: colors.textMuted,
  },
  finding: {
    ...typography.body,
    color: colors.textMuted,
  },
});
