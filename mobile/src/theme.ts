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

  // --- Scan-specific overlay tokens (SCAN_DESIGN.md §2) ---
  scanOverlayDim: 'rgba(8, 11, 16, 0.58)',
  scanOverlayScrim: 'rgba(8, 11, 16, 0.32)',
  scanInk: '#F4F6FA',
  scanInkDim: 'rgba(244, 246, 250, 0.72)',
  scanInkFaint: 'rgba(244, 246, 250, 0.44)',

  // --- State accents (mapped to scanMachine ScanState) ---
  scanIdle: '#6EA8FE',
  scanIdleSoft: 'rgba(110, 168, 254, 0.22)',
  scanIdleGlow: 'rgba(110, 168, 254, 0.55)',
  scanReady: '#3DDC97',
  scanReadySoft: 'rgba(61, 220, 151, 0.20)',
  scanWarn: '#F4B860',
  scanWarnSoft: 'rgba(244, 184, 96, 0.22)',
  scanFail: '#FF6B6B',
  scanFailSoft: 'rgba(255, 107, 107, 0.22)',

  // --- Panorama strip canvas tokens ---
  panoramaBg: '#0B0F15',
  panoramaEmptyDot: 'rgba(110, 168, 254, 0.18)',
  panoramaWritingEdge: '#F4F6FA',
  panoramaWritingEdgeGlow: 'rgba(110, 168, 254, 0.85)',
  panoramaFilledShadow: 'rgba(0, 0, 0, 0.45)',
  panoramaFrameStroke: 'rgba(244, 246, 250, 0.10)',
  panoramaFadeMask: '#0B0F15',

  // --- Coverage track tokens ---
  coverageTrack: 'rgba(244, 246, 250, 0.14)',
  coverageFillStart: '#6EA8FE',
  coverageFillEnd: '#9BC1FF',
  coverageMilestone: '#3DDC97',
  coverageLeadingDot: '#FFFFFF',
  coverageLeadingHalo: 'rgba(110, 168, 254, 0.65)',

  // --- Completion reveal tokens ---
  sparkleCore: '#FFFFFF',
  sparkleAccent: '#9BC1FF',
  sparklePass: '#3DDC97',
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

// --- Scan geometry (SCAN_DESIGN.md §2 + §3) ---
export const scanGeometry = {
  panoramaAspect: 16 / 5,
  panoramaPaddingTop: 56,
  panoramaPaddingHorizontal: 12,
  panoramaCornerRadius: 14,
  panoramaTopHeightFraction: 0.35,
  panoramaEmptyDotSize: 3,
  panoramaEmptyDotPitch: 14,
  panoramaWritingEdgeWidth: 1.5,

  silhouetteStrokeWidth: 2.5,
  silhouetteCornerRadius: 18,
  silhouetteShadowBlur: 10,

  ringStrokeWidth: 6,
  ringGapFromSilhouette: 18,
  ringMilestoneTickLength: 10,
  ringLeadingDotRadius: 4,
  ringLeadingHaloRadius: 14,

  dialDiameter: 64,
  dialStrokeWidth: 4,
  dialCornerInset: 16,
} as const;

// --- Scan motion timing constants (SCAN_DESIGN.md §2 + §4) ---
export const scanMotion = {
  spring: { damping: 18, stiffness: 220 },
  springSnappy: { damping: 14, stiffness: 320 },
  fastEase: { duration: 180 },
  midEase: { duration: 320 },
  slowEase: { duration: 520 },
  hero: { duration: 900 },
} as const;

/**
 * Toast subsystem tokens (in-app transient alerts surfaced via the
 * ToastProvider hook). The toast slide-in motion deliberately reuses
 * the scan screen's spring + mid-ease values so the app's motion
 * language stays consistent — a toast feels like the same family of
 * UI as the scan-screen banners. The default 4s lifetime sits between
 * "long enough to read a 6-word message" and "short enough that a
 * neglected toast doesn't crowd the next one".
 */
export const toastMotion = {
  spring: scanMotion.spring,
  midEase: scanMotion.midEase,
  fastEase: scanMotion.fastEase,
  defaultDurationMs: 4000,
  // Vertical gap between stacked toasts.
  stackGapPx: 8,
  // Maximum number of toasts visible at once.
  maxVisible: 3,
} as const;
