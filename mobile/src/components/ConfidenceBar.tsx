/**
 * Compact horizontal confidence bar — mirrors the web demo's `.conf-bar`
 * (backend/app/static/index.html lines ~455–466 and ~1798–1803). Renders
 * a small filled track plus a "92%" label so per-field extraction
 * confidence is visible inline with the value.
 *
 * Value is 0..1; out-of-range inputs are clamped. Returns null for
 * non-finite or zero values so callers can splat <ConfidenceBar
 * value={summary.confidence} /> without guarding upstream.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { colors, radius, spacing, typography } from '@src/theme';

export interface ConfidenceBarProps {
  /** 0..1; values outside the range are clamped. */
  value: number | null | undefined;
  /** Hides the trailing "%" label. Default false. */
  hideLabel?: boolean;
}

export function ConfidenceBar({
  value,
  hideLabel = false,
}: ConfidenceBarProps): React.ReactElement | null {
  if (value == null || !Number.isFinite(value) || value <= 0) return null;
  const clamped = Math.max(0, Math.min(1, value));
  const pct = Math.round(clamped * 100);
  const fillColor =
    clamped >= 0.8 ? colors.pass : clamped >= 0.5 ? colors.advisory : colors.fail;
  return (
    <View
      style={styles.row}
      accessibilityRole="progressbar"
      accessibilityLabel={`Extraction confidence ${pct}%`}
      accessibilityValue={{ min: 0, max: 100, now: pct }}
    >
      <View style={styles.track}>
        <View style={[styles.fill, { width: `${pct}%`, backgroundColor: fillColor }]} />
      </View>
      {hideLabel ? null : <Text style={styles.label}>{pct}%</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  track: {
    width: 40,
    height: 4,
    backgroundColor: colors.border,
    borderRadius: radius.sm,
    overflow: 'hidden',
  },
  fill: {
    height: '100%',
    borderRadius: radius.sm,
  },
  label: {
    ...typography.caption,
    color: colors.textMuted,
    fontVariant: ['tabular-nums'],
  },
});
