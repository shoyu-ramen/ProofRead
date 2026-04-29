/**
 * Capture-quality verdict pill for the report header.
 *
 * Sourced from `ReportResponse.image_quality` (good|fair|poor) plus the
 * optional `image_quality_notes` string. The pill carries the verdict
 * color; the notes (when present) render below as muted body text so
 * the user understands *why* the capture got flagged. Mirrors the web
 * demo's per-field/per-capture confidence treatment.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { ImageQuality } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

export interface CaptureQualityPillProps {
  quality: ImageQuality;
  notes?: string | null;
}

export function CaptureQualityPill({
  quality,
  notes,
}: CaptureQualityPillProps): React.ReactElement {
  const palette = paletteFor(quality);
  return (
    <View style={styles.wrap}>
      <View
        style={[
          styles.pill,
          { backgroundColor: palette.bg, borderColor: palette.border },
        ]}
        accessibilityLabel={`Capture quality: ${palette.label}`}
      >
        <View style={[styles.dot, { backgroundColor: palette.fg }]} />
        <Text style={styles.pillLabel}>Capture quality</Text>
        <Text style={[styles.pillValue, { color: palette.fg }]}>{palette.label}</Text>
      </View>
      {notes ? <Text style={styles.notes}>{notes}</Text> : null}
    </View>
  );
}

function paletteFor(quality: ImageQuality): {
  fg: string;
  bg: string;
  border: string;
  label: string;
} {
  switch (quality) {
    case 'good':
      return {
        fg: colors.pass,
        bg: 'rgba(61,220,151,0.12)',
        border: colors.pass,
        label: 'Good',
      };
    case 'fair':
      return {
        fg: colors.advisory,
        bg: 'rgba(244,184,96,0.12)',
        border: colors.advisory,
        label: 'Fair',
      };
    case 'poor':
      return {
        fg: colors.fail,
        bg: 'rgba(255,107,107,0.12)',
        border: colors.fail,
        label: 'Poor',
      };
  }
}

const styles = StyleSheet.create({
  wrap: {
    gap: spacing.xs,
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: spacing.xs,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radius.xl,
    borderWidth: 1,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  pillLabel: {
    ...typography.caption,
    color: colors.textMuted,
  },
  pillValue: {
    ...typography.caption,
    fontWeight: '700',
  },
  notes: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
