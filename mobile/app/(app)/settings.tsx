/**
 * Settings — account, sign out, image retention.
 *
 * SPEC §v1.7 row "Settings". Image retention default is 90 days,
 * configurable per company up to 7 years per SPEC §0 security
 * baseline. The retention picker is a stub today; the backend
 * doesn't yet expose a per-company retention setting endpoint.
 */

import React, { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { Button, Screen, SectionHeader } from '@src/components';
import { useAuthStore } from '@src/state/auth';
import { useScanStore } from '@src/state/scanStore';
import { colors, interaction, radius, spacing, typography } from '@src/theme';

interface RetentionOption {
  label: string;
  days: number;
}

const RETENTION_OPTIONS: RetentionOption[] = [
  { label: '30 days', days: 30 },
  { label: '90 days (default)', days: 90 },
  { label: '1 year', days: 365 },
  { label: '7 years (max)', days: 365 * 7 },
];

export default function SettingsScreen(): React.ReactElement {
  const user = useAuthStore((s) => s.user);
  const signOut = useAuthStore((s) => s.signOut);
  const resetScan = useScanStore((s) => s.reset);

  const [retentionDays, setRetentionDays] = useState(90);

  const handleSignOut = () => {
    resetScan();
    signOut();
    router.replace('/signin');
  };

  return (
    <Screen>
      <SectionHeader title="Account" />
      <View style={styles.card}>
        <Text style={styles.cardLabel}>Email</Text>
        <Text style={styles.cardValue}>{user?.email ?? '—'}</Text>
      </View>
      <View style={styles.card}>
        <Text style={styles.cardLabel}>Role</Text>
        <Text style={styles.cardValue}>{user?.role ?? '—'}</Text>
      </View>

      <SectionHeader
        title="Image retention"
        subtitle="How long captures are kept on the server."
      />
      <View style={styles.retentionList}>
        {RETENTION_OPTIONS.map((opt) => {
          const selected = retentionDays === opt.days;
          return (
            <Pressable
              key={opt.days}
              accessibilityRole="radio"
              accessibilityState={{ selected }}
              onPress={() => setRetentionDays(opt.days)}
              style={({ pressed }) => [
                styles.retentionRow,
                selected && styles.retentionRowSelected,
                pressed && {
                  opacity: interaction.pressed.opacity,
                  transform: [{ scale: interaction.pressed.scale }],
                },
              ]}
            >
              <Text style={styles.retentionLabel}>{opt.label}</Text>
              <View style={[styles.radio, selected && styles.radioOn]}>
                {selected ? <View style={styles.radioDot} /> : null}
              </View>
            </Pressable>
          );
        })}
      </View>
      <Text style={styles.footnote}>
        Retention changes apply to future scans. The backend endpoint
        for persisting this preference lands with the company-settings
        API in v2.
      </Text>

      <View style={{ flex: 1 }} />
      <Button label="Sign out" variant="danger" fullWidth onPress={handleSignOut} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.md,
  },
  cardLabel: {
    ...typography.caption,
    color: colors.textMuted,
  },
  cardValue: {
    ...typography.headingMd,
    color: colors.text,
  },
  retentionList: {
    gap: spacing.sm,
  },
  retentionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.md,
  },
  retentionRowSelected: {
    borderColor: colors.primary,
  },
  retentionLabel: {
    flex: 1,
    ...typography.bodyMd,
    color: colors.text,
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
  footnote: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
