/**
 * Design tokens.
 *
 * The non-scan UI/UX pass added a more complete typography scale,
 * interaction state tokens, and motion durations/easings while
 * preserving the original flat token names as aliases — older screens
 * that import `typography.body / heading / title / display / caption`
 * keep rendering identically until each is migrated to the new scale.
 */
import { Easing } from 'react-native-reanimated';

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

  /**
   * High-contrast focus ring color used by the `interaction.focusRing`
   * token. Sits a few notches lighter than `primary` so the ring is
   * legible against both `surface` and `surfaceAlt` backgrounds.
   */
  focus: '#7BB7FF',

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

/**
 * Typography scale.
 *
 * Two-tier system to keep legacy screens from breaking while letting
 * new code reach for a richer, more semantic token set:
 *
 * - **Named scale** (`displayLg, titleMd, headingSm, bodyLg, ...`) —
 *   each style provides `fontSize`, `fontWeight`, `lineHeight`, and a
 *   `letterSpacing` where the optical correction matters. Use these
 *   in new screens.
 *
 * - **Legacy aliases** (`display, title, heading, body, caption,
 *   button, mono`) — kept identical in semantics to their pre-pass
 *   shapes so a screen that does `...typography.body` keeps rendering
 *   the same. Aliases point at the closest new style.
 */
const named = {
  /** Hero numerals + welcome screens. Big, tightly tracked. */
  displayLg: {
    fontSize: 34,
    fontWeight: '700' as const,
    lineHeight: 40,
    letterSpacing: -0.6,
  },
  /** Standard display — used by home headline today. */
  displayMd: {
    fontSize: 28,
    fontWeight: '700' as const,
    lineHeight: 34,
    letterSpacing: -0.5,
  },
  /** Modal / sheet titles. */
  titleLg: {
    fontSize: 24,
    fontWeight: '700' as const,
    lineHeight: 30,
    letterSpacing: -0.3,
  },
  /** Card titles, section headers. */
  titleMd: {
    fontSize: 22,
    fontWeight: '700' as const,
    lineHeight: 28,
  },
  /** Page subheaders inside a section. */
  headingLg: {
    fontSize: 20,
    fontWeight: '600' as const,
    lineHeight: 26,
  },
  /** Default heading inside cards. */
  headingMd: {
    fontSize: 18,
    fontWeight: '600' as const,
    lineHeight: 24,
  },
  /** Compact heading for list rows / chips. */
  headingSm: {
    fontSize: 16,
    fontWeight: '600' as const,
    lineHeight: 22,
  },
  /** Generous body for hero copy / longform. */
  bodyLg: {
    fontSize: 17,
    fontWeight: '400' as const,
    lineHeight: 24,
  },
  /** Default body text. */
  bodyMd: {
    fontSize: 15,
    fontWeight: '400' as const,
    lineHeight: 22,
  },
  /** Compact body for dense lists. */
  bodySm: {
    fontSize: 14,
    fontWeight: '400' as const,
    lineHeight: 20,
  },
  /** Meta text — timestamps, helper hints. */
  caption: {
    fontSize: 13,
    fontWeight: '400' as const,
    lineHeight: 18,
  },
  /** Caption with stronger weight; used for tags / status pills. */
  captionStrong: {
    fontSize: 13,
    fontWeight: '700' as const,
    lineHeight: 18,
    letterSpacing: 0.4,
  },
  /** Form labels / small all-caps badges. */
  label: {
    fontSize: 12,
    fontWeight: '600' as const,
    lineHeight: 16,
    letterSpacing: 0.5,
  },
  /** Button label — same shape regardless of size. */
  button: {
    fontSize: 16,
    fontWeight: '600' as const,
    lineHeight: 22,
  },
  /** Monospace for pixel coordinates / debug overlays. */
  mono: {
    fontSize: 13,
    fontFamily: 'Courier',
    lineHeight: 18,
  },
} as const;

export const typography = {
  ...named,

  // --- Legacy aliases (identical semantics to pre-pass values) ---
  /** Alias → `displayMd`. Identical to the original `display` token. */
  display: named.displayMd,
  /** Alias → `titleMd`. */
  title: named.titleMd,
  /** Alias → `headingMd`. */
  heading: named.headingMd,
  /** Alias → `bodyMd`. */
  body: named.bodyMd,
} as const;

/**
 * Interaction state tokens — used by Pressable surfaces to keep
 * hover/pressed/disabled feedback consistent across the app.
 *
 * `pressed.scale` pairs with the existing scan motion language: the
 * scan instructions chip uses 0.96–0.98 on tap; the rest of the UI
 * inherits the same convention so the touch-down feel is uniform.
 */
export const interaction = {
  /** Web/tablet hover affordance. iOS doesn't fire hover so this is
      effectively no-op there, but kept for parity. */
  hover: {
    opacity: 0.92,
  },
  /** Active touch — opacity dim + tiny scale-down. */
  pressed: {
    opacity: 0.78,
    scale: 0.98,
  },
  /** Inactive controls (loading, disabled). */
  disabled: {
    opacity: 0.4,
  },
  /** Focus ring (keyboard / external-pointer focus indicators). */
  focusRing: {
    borderColor: colors.focus,
    borderWidth: 2,
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

/**
 * App-wide motion tokens for non-scan surfaces.
 *
 * `durations` is referenced in milliseconds by `withTiming` callsites;
 * `easings` reuses `react-native-reanimated`'s `Easing.bezier` so we
 * can hand the same curve to either Animated or Reanimated targets.
 *
 * Where possible the values mirror the existing scan/toast tokens so
 * the whole app feels like one motion system:
 *   - `fast` (120) is a hair quicker than `scanMotion.fastEase` (180);
 *     used for pressed-state feedback that wants to feel instant.
 *   - `base` (200) matches typical RN-Animated default; used for tab
 *     transitions and skeleton fades.
 *   - `slow` (320) === `scanMotion.midEase`; used for inline state
 *     swaps (loading → loaded).
 *   - `page` (460) is between `slow` and `slowEase` (520); used by
 *     screen-level enters that need a clearer hand-off than `slow`.
 */
export const motion = {
  durations: {
    fast: 120,
    base: 200,
    slow: 320,
    page: 460,
  },
  /**
   * Material-3 inspired curves expressed as Reanimated `Easing.bezier`.
   * - `standard` is the default for most UI motion.
   * - `emphasized` is reserved for hero / page-level transitions.
   */
  easings: {
    standard: Easing.bezier(0.2, 0, 0, 1),
    emphasized: Easing.bezier(0.3, 0, 0, 1),
  },
} as const;
