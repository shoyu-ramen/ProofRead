/**
 * Scan setup — combined beverage-type + container-size picker.
 *
 * v1 only enables Beer; sensible defaults (355 mL, not imported) are
 * pre-populated so the user can tap "Continue to capture" without
 * touching anything else.
 */

import React, { useEffect, useMemo, useState } from 'react';
import {
  Keyboard,
  Pressable,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import { router } from 'expo-router';
import { Button, Screen, SectionHeader } from '@src/components';
import {
  DEFAULT_CONTAINER_SIZES,
  useScanStore,
} from '@src/state/scanStore';
import type { BeverageType } from '@src/api/types';
import { colors, radius, spacing, typography } from '@src/theme';

interface BeverageOption {
  type: BeverageType;
  label: string;
  enabled: boolean;
  helper?: string;
}

const BEVERAGES: BeverageOption[] = [
  { type: 'beer', label: 'Beer', enabled: true },
  { type: 'wine', label: 'Wine', enabled: false, helper: 'Coming soon' },
  { type: 'spirits', label: 'Spirits', enabled: false, helper: 'Coming soon' },
];

const DEFAULT_CONTAINER_ML = 355;

export default function ScanSetup(): React.ReactElement {
  const beverageType = useScanStore((s) => s.beverageType);
  const setBeverageType = useScanStore((s) => s.setBeverageType);
  const containerSizeMl = useScanStore((s) => s.containerSizeMl);
  const setContainerSize = useScanStore((s) => s.setContainerSize);
  const isImported = useScanStore((s) => s.isImported);
  const setIsImported = useScanStore((s) => s.setIsImported);

  // Pre-select Beer + 355 mL on first mount if nothing is set yet.
  useEffect(() => {
    if (beverageType === null) setBeverageType('beer');
    if (containerSizeMl === null) setContainerSize(DEFAULT_CONTAINER_ML);
  }, [beverageType, containerSizeMl, setBeverageType, setContainerSize]);

  const presetSizes = useMemo(
    () => DEFAULT_CONTAINER_SIZES.map((s) => s.ml),
    []
  );
  const isCustomActive =
    containerSizeMl !== null && !presetSizes.includes(containerSizeMl);

  const [customDraft, setCustomDraft] = useState<string>(
    isCustomActive && containerSizeMl !== null ? String(containerSizeMl) : ''
  );

  const applyCustom = () => {
    const n = Number.parseInt(customDraft, 10);
    if (!Number.isFinite(n) || n <= 0 || n > 10_000) {
      return;
    }
    setContainerSize(n);
    Keyboard.dismiss();
  };

  const canContinue =
    beverageType !== null &&
    containerSizeMl !== null &&
    containerSizeMl > 0;

  const handleNext = () => {
    if (!canContinue) return;
    router.push('/(app)/scan/unwrap');
  };

  return (
    <Screen>
      <SectionHeader
        title="Scan setup"
        subtitle="Defaults are good for a 12 oz can. Adjust if needed."
      />

      <Text style={styles.sectionLabel}>Beverage</Text>
      <View style={styles.options}>
        {BEVERAGES.map((opt) => {
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

      <Text style={styles.sectionLabel}>Container size</Text>
      <View style={styles.options}>
        {DEFAULT_CONTAINER_SIZES.map((preset) => {
          const selected = containerSizeMl === preset.ml;
          return (
            <Pressable
              key={preset.ml}
              accessibilityRole="radio"
              accessibilityState={{ selected }}
              onPress={() => setContainerSize(preset.ml)}
              style={({ pressed }) => [
                styles.option,
                selected && styles.optionSelected,
                pressed && { opacity: 0.85 },
              ]}
            >
              <View style={{ flex: 1 }}>
                <Text style={styles.optionLabel}>{preset.label}</Text>
                <Text style={styles.optionHelper}>{preset.ml} mL</Text>
              </View>
              <View style={[styles.radio, selected && styles.radioOn]}>
                {selected ? <View style={styles.radioDot} /> : null}
              </View>
            </Pressable>
          );
        })}
      </View>

      <View style={styles.customRow}>
        <Text style={styles.customLabel}>Custom (mL)</Text>
        <TextInput
          value={customDraft}
          onChangeText={setCustomDraft}
          onBlur={applyCustom}
          onSubmitEditing={applyCustom}
          keyboardType="number-pad"
          placeholder="e.g. 750"
          placeholderTextColor={colors.textMuted}
          style={[styles.input, isCustomActive && styles.inputActive]}
          maxLength={5}
        />
      </View>

      <View style={styles.toggleRow}>
        <View style={{ flex: 1 }}>
          <Text style={styles.toggleLabel}>Imported product</Text>
          <Text style={styles.toggleHelper}>
            Required for the country-of-origin rule.
          </Text>
        </View>
        <Switch
          value={isImported}
          onValueChange={setIsImported}
          trackColor={{ false: colors.border, true: colors.primary }}
        />
      </View>

      <View style={{ flex: 1 }} />
      <Button
        label="Continue to capture"
        size="lg"
        fullWidth
        disabled={!canContinue}
        onPress={handleNext}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  sectionLabel: {
    ...typography.heading,
    color: colors.text,
    marginTop: spacing.sm,
  },
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
  customRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  customLabel: {
    ...typography.body,
    color: colors.text,
    flex: 1,
  },
  input: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    minWidth: 110,
    textAlign: 'right',
  },
  inputActive: {
    borderColor: colors.primary,
  },
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: spacing.sm,
  },
  toggleLabel: {
    ...typography.body,
    color: colors.text,
  },
  toggleHelper: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
