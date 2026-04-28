/**
 * Auth store — fake user for v1 scaffold.
 *
 * Real impl will use Auth0 (SPEC §0 tech stack) and persist tokens via
 * expo-secure-store. For now this is a zustand store with a stub user
 * and a no-op token, so screens can gate on `isAuthenticated`.
 *
 * TODO(auth0): swap setSignedIn() for an Auth0 hosted-login flow,
 * exchange the Auth0 token at POST /v1/auth/exchange, and persist the
 * resulting session token via expo-secure-store.
 */

import { create } from 'zustand';

export interface FakeUser {
  id: string;
  email: string;
  role: 'producer' | 'consultant' | 'admin';
}

interface AuthState {
  user: FakeUser | null;
  // Token is null today — backend's get_current_user() returns a fixed
  // user regardless. Wired so the API client picks it up the moment
  // auth lands.
  token: string | null;
  isAuthenticated: boolean;
  signIn: (user?: Partial<FakeUser>) => void;
  signOut: () => void;
}

const DEFAULT_USER: FakeUser = {
  id: '00000000-0000-0000-0000-000000000001',
  email: 'test@proofread.local',
  role: 'producer',
};

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isAuthenticated: false,
  signIn: (user) => {
    set({
      user: { ...DEFAULT_USER, ...(user ?? {}) },
      token: 'stub-token',
      isAuthenticated: true,
    });
  },
  signOut: () => {
    set({ user: null, token: null, isAuthenticated: false });
  },
}));

// Selector helpers — used by the API client and route guards so they
// don't need to subscribe to the whole store.
export function getAuthToken(): string | null {
  return useAuthStore.getState().token;
}
