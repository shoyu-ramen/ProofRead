/**
 * Authenticated route group. Redirects to /signin when no user.
 */

import React from 'react';
import { Redirect, Stack } from 'expo-router';
import { useAuthStore } from '@src/state/auth';
import { colors } from '@src/theme';

export default function AuthenticatedLayout(): React.ReactElement {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);

  if (!isAuthenticated) {
    return <Redirect href="/signin" />;
  }

  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: colors.background },
        headerTintColor: colors.text,
        headerTitleStyle: { color: colors.text },
        headerShadowVisible: false,
        contentStyle: { backgroundColor: colors.background },
      }}
    >
      <Stack.Screen name="home" options={{ title: 'ProofRead' }} />
      <Stack.Screen name="history" options={{ title: 'Scan history' }} />
      <Stack.Screen name="settings" options={{ title: 'Settings' }} />

      <Stack.Screen
        name="scan/setup"
        options={{ title: 'Scan setup' }}
      />
      <Stack.Screen
        name="scan/unwrap"
        options={{ headerShown: false, presentation: 'fullScreenModal' }}
      />
      <Stack.Screen name="scan/review" options={{ title: 'Review' }} />
      <Stack.Screen
        name="scan/processing/[id]"
        options={{ title: 'Processing', headerBackVisible: false }}
      />
      <Stack.Screen name="scan/report/[id]" options={{ title: 'Report' }} />
      <Stack.Screen
        name="scan/rule/[ruleId]"
        options={{ title: 'Rule detail' }}
      />
    </Stack>
  );
}
