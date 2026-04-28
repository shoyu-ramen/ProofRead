/**
 * Splash / entry redirect.
 *
 * SPEC §v1.7: "Splash — Logo, 1s timeout". For the scaffold we redirect
 * synchronously based on auth state; the 1-second logo animation can
 * be added later by gating on a small useEffect timer.
 */

import React from 'react';
import { Redirect } from 'expo-router';
import { useAuthStore } from '@src/state/auth';

export default function Index(): React.ReactElement {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  if (!isAuthenticated) return <Redirect href="/signin" />;
  return <Redirect href="/(app)/home" />;
}
