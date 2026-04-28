/**
 * Sign-in screen — stub.
 *
 * Real impl uses Auth0 hosted login (SPEC §v1.5 F1.1). For the scaffold,
 * a single button calls signIn() to populate a fake user and routes to
 * the authenticated layout.
 *
 * TODO(auth0): replace this with an Auth0 hosted-login flow + token
 * exchange against POST /v1/auth/exchange.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { Button, Screen } from '@src/components';
import { useAuthStore } from '@src/state/auth';
import { colors, spacing, typography } from '@src/theme';

export default function SignIn(): React.ReactElement {
  const signIn = useAuthStore((s) => s.signIn);

  const handleStubSignIn = () => {
    signIn();
    router.replace('/(app)/home');
  };

  return (
    <Screen contentStyle={styles.content}>
      <View style={styles.brandBlock}>
        <Text style={styles.title}>ProofRead</Text>
        <Text style={styles.subtitle}>
          TTB-compliance review for beer labels.
        </Text>
      </View>

      <View style={styles.formBlock}>
        <Button label="Sign in (stub)" size="lg" fullWidth onPress={handleStubSignIn} />
        <Button
          label="Continue with Google"
          variant="secondary"
          size="lg"
          fullWidth
          disabled
        />
        <Text style={styles.footnote}>
          Auth0 hosted login lands once backend exchange endpoint is
          wired. Tap "Sign in (stub)" to enter the app.
        </Text>
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: {
    justifyContent: 'space-between',
    paddingVertical: spacing.xxl,
  },
  brandBlock: {
    gap: spacing.sm,
    marginTop: spacing.xxl,
  },
  title: {
    ...typography.display,
    color: colors.text,
  },
  subtitle: {
    ...typography.body,
    color: colors.textMuted,
  },
  formBlock: {
    gap: spacing.md,
    marginBottom: spacing.xl,
  },
  footnote: {
    ...typography.caption,
    color: colors.textMuted,
    textAlign: 'center',
  },
});
