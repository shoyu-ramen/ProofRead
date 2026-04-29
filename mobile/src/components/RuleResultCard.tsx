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
  // Show fix_suggestion on the card itself for fail/advisory so the user
  // doesn't have to drill into rule detail to know what to do. Pass
  // results never need a fix, so we suppress to keep the report scannable.
  const showFix =
    (result.status === 'fail' || result.status === 'advisory') &&
    typeof result.fix_suggestion === 'string' &&
    result.fix_suggestion.trim().length > 0;

  // AI-generated explanation: only surface for fail/advisory rules and
  // only when the backend produced one. Pass rules don't need extra
  // context, and the field is optional/nullable in the DTO.
  const showExplanation =
    (result.status === 'fail' || result.status === 'advisory') &&
    typeof result.explanation === 'string' &&
    result.explanation.trim().length > 0;

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
        {showFix ? (
          <Text style={styles.fix} numberOfLines={3}>
            <Text style={styles.fixLabel}>Fix: </Text>
            {result.fix_suggestion}
          </Text>
        ) : null}
        {result.finding ? (
          <Text style={styles.finding} numberOfLines={2}>
            {result.finding}
          </Text>
        ) : null}
        {showExplanation ? (
          <Text style={styles.explanation} numberOfLines={4}>
            <Text style={styles.explanationLabel}>WHY · </Text>
            {result.explanation}
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
  fix: {
    ...typography.body,
    color: colors.text,
    fontStyle: 'italic',
  },
  fixLabel: {
    fontWeight: '700',
    fontStyle: 'normal',
    color: colors.text,
  },
  // AI-generated "Why · …" supplementary context. Indented with a left
  // border accent so it reads as commentary rather than a primary
  // finding, italicised to mark the soft, narrative tone.
  explanation: {
    ...typography.caption,
    color: colors.textMuted,
    fontStyle: 'italic',
    marginLeft: spacing.sm,
    paddingLeft: spacing.sm,
    borderLeftWidth: 2,
    borderLeftColor: colors.border,
  },
  explanationLabel: {
    ...typography.caption,
    color: colors.textMuted,
    fontStyle: 'normal',
    fontWeight: '700',
    letterSpacing: 0.5,
  },
});
