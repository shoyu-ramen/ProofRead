/**
 * Scan history list. SPEC §v1.5 F1.11 + §v1.7 row "History".
 *
 * Backend exposes `GET /v1/scans` per SPEC §v1.9 but the current
 * scaffold backend (scans.py) only implements per-scan endpoints.
 * Until that lands we render an explicit empty state with a
 * documented TODO.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { Button, Screen, SectionHeader } from '@src/components';
import { colors, radius, spacing, typography } from '@src/theme';

export default function HistoryScreen(): React.ReactElement {
  // TODO(history-endpoint): replace the empty state below with a
  // tanstack-query fetch against GET /v1/scans once the backend
  // implements paginated history. Cache last 50 reports on device
  // (SPEC §0.5 offline matrix) once we have FileSystem persistence.
  const items: Array<{ id: string }> = [];

  return (
    <Screen>
      <SectionHeader
        title="Your scans"
        subtitle="Most recent first."
      />

      {items.length === 0 ? (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyTitle}>Nothing here yet</Text>
          <Text style={styles.emptyBody}>
            Completed scans will appear here. Pull-to-refresh once the
            history endpoint ships.
          </Text>
        </View>
      ) : null}

      <View style={{ flex: 1 }} />
      <Button
        label="Scan a new label"
        size="lg"
        fullWidth
        onPress={() => router.push('/(app)/scan/beverage-type')}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  emptyCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.lg,
    gap: spacing.xs,
  },
  emptyTitle: {
    ...typography.heading,
    color: colors.text,
  },
  emptyBody: {
    ...typography.body,
    color: colors.textMuted,
  },
});
