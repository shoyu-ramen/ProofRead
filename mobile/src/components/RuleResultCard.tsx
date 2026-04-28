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
  const isAdvisory = result.status === 'advisory';

  const inner = (
    <>
      {isAdvisory ? <AdvisoryBanner /> : null}
      <View style={styles.body}>
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
      </View>
    </>
  );

  const cardStyle = [styles.card, isAdvisory && styles.cardAdvisory];

  if (onPress) {
    return (
      <Pressable
        onPress={onPress}
        accessibilityRole="button"
        accessibilityHint={
          isAdvisory ? "Couldn't verify with confidence. Rescan recommended." : undefined
        }
        style={({ pressed }) => [...cardStyle, pressed && styles.pressed]}
      >
        {inner}
      </Pressable>
    );
  }
  return <View style={cardStyle}>{inner}</View>;
}

function AdvisoryBanner(): React.ReactElement {
  return (
    <View style={styles.advisoryBanner} accessible accessibilityRole="text">
      <Text style={styles.advisoryIcon} accessibilityElementsHidden>
        ⚠
      </Text>
      <View style={styles.advisoryTextCol}>
        <Text style={styles.advisoryTitle}>Couldn't verify with confidence</Text>
        <Text style={styles.advisorySubtitle}>Rescan recommended</Text>
      </View>
    </View>
  );
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
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  // Advisory cards get a left accent stripe so the "couldn't verify"
  // state reads as a distinct visual class — not a softer fail.
  cardAdvisory: {
    borderLeftWidth: 3,
    borderLeftColor: colors.advisory,
  },
  pressed: {
    opacity: 0.85,
  },
  body: {
    padding: spacing.md,
    gap: spacing.sm,
  },
  advisoryBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: 'rgba(244,184,96,0.12)',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.advisory,
  },
  advisoryIcon: {
    ...typography.body,
    color: colors.advisory,
    fontWeight: '700',
  },
  advisoryTextCol: {
    flex: 1,
  },
  advisoryTitle: {
    ...typography.caption,
    color: colors.advisory,
    fontWeight: '700',
  },
  advisorySubtitle: {
    ...typography.caption,
    color: colors.textMuted,
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
