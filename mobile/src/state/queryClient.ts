/**
 * tanstack-query client. Tuned for the v1 polling pattern from SPEC
 * §v1.10 (mobile polls /scans/:id until status == complete).
 */

import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Keep results fresh briefly so screens that re-mount don't refetch
      // the moment a user backs into them.
      staleTime: 5_000,
      gcTime: 5 * 60_000,
      retry: 2,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 0,
    },
  },
});

// Query-key registry — colocated so screens stay in sync.
export const queryKeys = {
  scan: (id: string) => ['scan', id] as const,
  report: (id: string) => ['scan', id, 'report'] as const,
  history: () => ['scans', 'history'] as const,
} as const;
