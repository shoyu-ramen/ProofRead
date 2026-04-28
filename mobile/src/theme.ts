/**
 * Minimal design tokens. Intentionally tiny and string-typed; the v1
 * mobile app is not aiming for a polished design system, just enough
 * to make screens render readably during development.
 */

export const colors = {
  background: '#0E1116',
  surface: '#171B22',
  surfaceAlt: '#1F242D',
  border: '#2A2F3A',
  text: '#F4F6FA',
  textMuted: '#9AA0AB',
  primary: '#6EA8FE',
  onPrimary: '#0E1116',
  pass: '#3DDC97',
  fail: '#FF6B6B',
  advisory: '#F4B860',
  danger: '#E85D75',
  onDanger: '#FFFFFF',
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 20,
  xl: 28,
  xxl: 40,
} as const;

export const radius = {
  sm: 4,
  md: 8,
  lg: 12,
  xl: 20,
} as const;

export const typography = {
  display: {
    fontSize: 28,
    fontWeight: '700' as const,
    letterSpacing: -0.5,
  },
  title: {
    fontSize: 22,
    fontWeight: '700' as const,
  },
  heading: {
    fontSize: 18,
    fontWeight: '600' as const,
  },
  body: {
    fontSize: 15,
    fontWeight: '400' as const,
  },
  caption: {
    fontSize: 13,
    fontWeight: '400' as const,
  },
  button: {
    fontSize: 16,
    fontWeight: '600' as const,
  },
  mono: {
    fontSize: 13,
    fontFamily: 'Courier',
  },
} as const;
