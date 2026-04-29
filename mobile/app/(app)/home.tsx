/**
 * Home screen — Big "Scan new label" CTA.
 *
 * Per SPEC §v1.7, home was specced with a "recent 3 scans" rail. The
 * backend `GET /v1/scans` (history) endpoint isn't implemented yet, so
 * the rail and the History nav entry are hidden until it lands; until
 * then the home surface is just the hero + scan CTA + settings entry.
 *
 * Re-add when the backend ships:
 *   - import RecentScanRow + queryKeys.history + apiClient.getHistory
 *   - render the rail conditionally on a successful query
 *   - restore the History button beside Settings
 *   - restore the Stack.Screen entry in (app)/_layout.tsx and the
 *     history.tsx route file
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';

import { Button, Screen } from '@src/components';
import { useScanStore } from '@src/state/scanStore';
import { colors, spacing, typography } from '@src/theme';

export default function Home(): React.ReactElement {
  const reset = useScanStore((s) => s.reset);

  const handleStartScan = () => {
    reset();
    router.push('/(app)/scan/beverage-type');
  };

  return (
    <Screen>
      <View style={styles.heroBlock}>
        <Text style={styles.headline}>Verify a label.</Text>
        <Text style={styles.subhead}>
          Rotate the bottle once. We read every side.
        </Text>
      </View>

      <Button
        label="Scan new label"
        size="lg"
        fullWidth
        onPress={handleStartScan}
      />

      <Button
        label="Settings"
        variant="secondary"
        fullWidth
        onPress={() => router.push('/(app)/settings')}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  heroBlock: {
    gap: spacing.sm,
    paddingTop: spacing.md,
  },
  headline: {
    ...typography.display,
    color: colors.text,
  },
  subhead: {
    ...typography.body,
    color: colors.textMuted,
  },
});
