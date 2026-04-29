/**
 * ExternalMatchCard — surfaces a TTB COLA reverse-image-lookup hit at
 * the top of the report screen.
 *
 * The card frames the match as a positive signal: the user's label
 * aligns with an existing TTB approval. Tapping opens the source URL
 * (TTB COLA Online detail page) externally when present.
 *
 * The component itself is null-safe to render — when no match exists,
 * the parent screen simply skips it. This component assumes a match
 * object is provided.
 */

import React from 'react';
import { Linking, Pressable, StyleSheet, Text, View } from 'react-native';
import type { ExternalMatchDTO } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

export interface ExternalMatchCardProps {
  match: ExternalMatchDTO;
  // Override for the default Linking.openURL behaviour. Useful for
  // tests or for hosts that want to capture analytics around taps.
  onPress?: () => void;
}

export function ExternalMatchCard({
  match,
  onPress,
}: ExternalMatchCardProps): React.ReactElement {
  const sourceLabel = sourceLabelFor(match.source);
  const subtitleLine = composeSubtitle(match);
  const approvalLine = match.approval_date
    ? `Approved ${formatApprovalDate(match.approval_date)}`
    : null;
  const hasUrl = typeof match.source_url === 'string' && match.source_url.length > 0;

  const handlePress = () => {
    if (onPress) {
      onPress();
      return;
    }
    if (!hasUrl || !match.source_url) return;
    try {
      void Linking.openURL(match.source_url);
    } catch {
      // Malformed URL or no handler installed for the scheme — swallow
      // so a bad backend payload can't crash the report screen.
    }
  };

  const inner = (
    <View style={styles.body}>
      <View style={styles.titleRow}>
        <View
          style={styles.checkBadge}
          accessibilityElementsHidden
          importantForAccessibility="no"
        >
          <Text style={styles.checkGlyph}>✓</Text>
        </View>
        <View style={styles.titleCol}>
          <Text style={styles.title}>Approved by TTB</Text>
          <Text style={styles.sourceLabel}>{sourceLabel}</Text>
        </View>
      </View>
      {subtitleLine ? (
        <Text style={styles.subtitle} numberOfLines={2}>
          {subtitleLine}
        </Text>
      ) : null}
      <View style={styles.metaRow}>
        <Text style={styles.metaLabel}>COLA ID</Text>
        <Text style={styles.metaValue}>{match.source_id}</Text>
      </View>
      {approvalLine ? <Text style={styles.approvalLine}>{approvalLine}</Text> : null}
      {hasUrl ? <Text style={styles.tapHint}>Tap to view on TTB ↗</Text> : null}
    </View>
  );

  if (hasUrl || onPress) {
    return (
      <Pressable
        onPress={handlePress}
        accessibilityRole="link"
        accessibilityLabel={`Approved by TTB COLA ${match.source_id}`}
        accessibilityHint={hasUrl ? 'Opens the TTB COLA approval page' : undefined}
        style={({ pressed }) => [styles.card, pressed && styles.pressed]}
      >
        {inner}
      </Pressable>
    );
  }

  return <View style={styles.card}>{inner}</View>;
}

function sourceLabelFor(source: string): string {
  switch (source) {
    case 'ttb_cola':
      return 'Source: TTB COLA Online';
    default:
      // Surface the raw identifier for forward-compat — beats hiding the
      // attribution outright if a new source ships before the UI catches up.
      return `Source: ${source}`;
  }
}

function composeSubtitle(match: ExternalMatchDTO): string | null {
  const parts: string[] = [];
  if (match.brand) parts.push(match.brand);
  if (match.fanciful_name && match.fanciful_name !== match.brand) {
    parts.push(match.fanciful_name);
  }
  if (match.class_type) parts.push(match.class_type);
  if (parts.length === 0) return null;
  return parts.join(' · ');
}

function formatApprovalDate(iso: string): string {
  // The DTO ships ISO-8601 dates; render in the user's locale. If
  // parsing fails (legacy rows, malformed values), fall back to the raw
  // string so we never blank the line.
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString();
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    borderLeftWidth: 3,
    borderLeftColor: colors.pass,
    overflow: 'hidden',
  },
  pressed: {
    opacity: 0.85,
  },
  body: {
    padding: spacing.md,
    gap: spacing.sm,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  checkBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(61,220,151,0.15)',
    borderWidth: 1,
    borderColor: colors.pass,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkGlyph: {
    ...typography.body,
    color: colors.pass,
    fontWeight: '700',
    lineHeight: 18,
  },
  titleCol: {
    flex: 1,
    gap: 2,
  },
  title: {
    ...typography.heading,
    color: colors.text,
  },
  sourceLabel: {
    ...typography.caption,
    color: colors.textMuted,
  },
  subtitle: {
    ...typography.body,
    color: colors.text,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  metaLabel: {
    ...typography.caption,
    color: colors.textMuted,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  metaValue: {
    ...typography.caption,
    color: colors.text,
  },
  approvalLine: {
    ...typography.caption,
    color: colors.textMuted,
  },
  tapHint: {
    ...typography.caption,
    color: colors.primary,
  },
});
