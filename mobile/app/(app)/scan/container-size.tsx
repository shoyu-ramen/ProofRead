/**
 * Container size picker. Per SPEC §v1.5 F1.4: defaults 355, 473, 500,
 * 650 mL plus a custom mL input.
 */

import React, { useMemo, useState } from 'react';
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
import { colors, radius, spacing, typography } from '@src/theme';

export default function ContainerSizePicker(): React.ReactElement {
  const containerSizeMl = useScanStore((s) => s.containerSizeMl);
  const setContainerSize = useScanStore((s) => s.setContainerSize);
  const isImported = useScanStore((s) => s.isImported);
  const setIsImported = useScanStore((s) => s.setIsImported);

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
      // Backend caps container_size_ml at 10,000.
      return;
    }
    setContainerSize(n);
    Keyboard.dismiss();
  };

  const handleNext = () => {
    if (containerSizeMl === null || containerSizeMl <= 0) return;
    router.push('/(app)/scan/camera/front');
  };

  return (
    <Screen>
      <SectionHeader
        title="Container size"
        subtitle="Tap a preset, or type a custom volume in mL."
      />

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
          style={[
            styles.input,
            isCustomActive && styles.inputActive,
          ]}
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
        disabled={containerSizeMl === null || containerSizeMl <= 0}
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
    marginTop: spacing.sm,
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
    marginTop: spacing.sm,
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
