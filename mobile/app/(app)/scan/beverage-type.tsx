/**
 * Beverage type picker. Per SPEC §v1.6 step 2 — only Beer enabled in v1.
 */

import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { Button, Screen, SectionHeader } from '@src/components';
import { useScanStore } from '@src/state/scanStore';
import type { BeverageType } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

interface BeverageOption {
  type: BeverageType;
  label: string;
  enabled: boolean;
  helper?: string;
}

const OPTIONS: BeverageOption[] = [
  { type: 'beer', label: 'Beer', enabled: true },
  { type: 'wine', label: 'Wine', enabled: false, helper: 'v2' },
  { type: 'spirits', label: 'Spirits', enabled: false, helper: 'v2' },
];

export default function BeverageTypePicker(): React.ReactElement {
  const beverageType = useScanStore((s) => s.beverageType);
  const setBeverageType = useScanStore((s) => s.setBeverageType);

  const handleNext = () => {
    if (!beverageType) return;
    router.push('/(app)/scan/container-size');
  };

  return (
    <Screen>
      <SectionHeader
        title="What's on the label?"
        subtitle="v1 supports beer. Wine and spirits land in v2."
      />

      <View style={styles.options}>
        {OPTIONS.map((opt) => {
          const selected = beverageType === opt.type;
          return (
            <Pressable
              key={opt.type}
              accessibilityRole="radio"
              accessibilityState={{ selected, disabled: !opt.enabled }}
              disabled={!opt.enabled}
              onPress={() => setBeverageType(opt.type)}
              style={({ pressed }) => [
                styles.option,
                selected && styles.optionSelected,
                !opt.enabled && styles.optionDisabled,
                pressed && opt.enabled && { opacity: 0.85 },
              ]}
            >
              <View style={{ flex: 1 }}>
                <Text style={styles.optionLabel}>{opt.label}</Text>
                {opt.helper ? (
                  <Text style={styles.optionHelper}>{opt.helper}</Text>
                ) : null}
              </View>
              <View style={[styles.radio, selected && styles.radioOn]}>
                {selected ? <View style={styles.radioDot} /> : null}
              </View>
            </Pressable>
          );
        })}
      </View>

      <View style={{ flex: 1 }} />
      <Button
        label="Continue"
        size="lg"
        fullWidth
        disabled={!beverageType}
        onPress={handleNext}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  options: {
    gap: spacing.sm,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.md,
    gap: spacing.md,
  },
  optionSelected: {
    borderColor: colors.primary,
  },
  optionDisabled: {
    opacity: 0.5,
  },
  optionLabel: {
    ...typography.heading,
    color: colors.text,
  },
  optionHelper: {
    ...typography.caption,
    color: colors.textMuted,
  },
  radio: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  radioOn: {
    borderColor: colors.primary,
  },
  radioDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.primary,
  },
});
