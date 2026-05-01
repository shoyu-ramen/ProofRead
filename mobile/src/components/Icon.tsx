/**
 * Icon — cross-platform icon wrapper.
 *
 * On iOS, renders an `expo-symbols` SFSymbol; on Android (and any
 * other host where SymbolView isn't available), falls back to the
 * Feather glyph that best matches the SF Symbol name. The mapping is
 * a small, hand-curated table — we only map names actually used in
 * the app, so callers get a clear TypeScript error if they reach for
 * a name we haven't vetted.
 *
 * Adding a new icon: extend `ICON_MAP` with both the SFSymbol and its
 * Feather equivalent; both must exist in their respective glyph sets
 * (we picked Feather as the Android fallback because the project
 * already ships it for the scan UI).
 */
import React from 'react';
import { Platform, View, ViewStyle } from 'react-native';
import { Feather } from '@expo/vector-icons';

import { colors } from '@src/theme';

/**
 * The SF Symbol name + its Feather fallback, keyed by the **name our
 * code uses**. The shared name is intentionally human-friendly (and
 * generally matches Feather since that's the larger established
 * vocabulary in the app); we map it to the right SF Symbol underneath.
 */
const ICON_MAP = {
  /** Camera — used on the home scan CTA on iOS. */
  camera: { sf: 'camera.viewfinder', feather: 'camera' },
  /** Right-facing chevron for list disclosure / navigation hints. */
  'chevron-right': { sf: 'chevron.right', feather: 'chevron-right' },
  /** Solid alert circle — error/empty states. */
  'alert-circle': {
    sf: 'exclamationmark.circle.fill',
    feather: 'alert-circle',
  },
  /** Solid check circle — success states. */
  'check-circle': {
    sf: 'checkmark.circle.fill',
    feather: 'check-circle',
  },
  /** Refresh / retry. */
  'refresh-cw': { sf: 'arrow.clockwise', feather: 'refresh-cw' },
  /** Empty inbox — empty list state. */
  inbox: { sf: 'tray', feather: 'inbox' },
  /** Generic image icon — used by panorama placeholders. */
  image: { sf: 'photo', feather: 'image' },
  /** Clock — relative-time / history. */
  clock: { sf: 'clock', feather: 'clock' },
  /** Info circle. */
  info: { sf: 'info.circle', feather: 'info' },
  /** Close / cancel. */
  x: { sf: 'xmark', feather: 'x' },
} as const;

export type IconName = keyof typeof ICON_MAP;

export interface IconProps {
  name: IconName;
  /** Size in px — applied uniformly to width/height. Default 20. */
  size?: number;
  /** Tint color. Defaults to `colors.text`. */
  color?: string;
  /** Optional wrapping style (e.g. for centering inside a circle). */
  style?: ViewStyle;
  /** Optional accessibility label. Without this the icon is silent. */
  accessibilityLabel?: string;
}

/**
 * Whether to use SF Symbols. Lazy import is intentional — `expo-symbols`
 * pulls a native module that doesn't exist on web/Android, so we only
 * touch it inside the iOS branch.
 */
const USE_SF_SYMBOLS = Platform.OS === 'ios';

let SymbolViewCache: React.ComponentType<{
  name: string;
  size?: number;
  tintColor?: string | null;
  weight?: 'regular' | 'medium' | 'semibold' | 'bold';
  resizeMode?: 'scaleAspectFit' | 'scaleAspectFill';
  style?: unknown;
}> | null = null;

function getSymbolView(): typeof SymbolViewCache {
  if (SymbolViewCache !== null) return SymbolViewCache;
  try {
    // Require so the bundler doesn't try to follow this on Android.
    // Wrapped in try/catch so the app still launches if the native
    // module is missing for any reason.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require('expo-symbols');
    SymbolViewCache = mod.SymbolView;
    return SymbolViewCache;
  } catch {
    return null;
  }
}

export function Icon({
  name,
  size = 20,
  color = colors.text,
  style,
  accessibilityLabel,
}: IconProps): React.ReactElement {
  const entry = ICON_MAP[name];
  const a11yProps = accessibilityLabel
    ? { accessible: true, accessibilityLabel, accessibilityRole: 'image' as const }
    : { accessibilityElementsHidden: true, importantForAccessibility: 'no' as const };

  if (USE_SF_SYMBOLS) {
    const SymbolView = getSymbolView();
    if (SymbolView) {
      return (
        <View style={[{ width: size, height: size }, style]} {...a11yProps}>
          <SymbolView
            name={entry.sf}
            size={size}
            tintColor={color}
            weight="medium"
            resizeMode="scaleAspectFit"
            style={{ width: size, height: size }}
          />
        </View>
      );
    }
  }

  // Android / web / SF unavailable → Feather fallback.
  return (
    <View style={[{ width: size, height: size }, style]} {...a11yProps}>
      <Feather name={entry.feather} size={size} color={color} />
    </View>
  );
}

/**
 * For tests + storybook hosts: lets a caller assert which fallback path
 * is expected without rendering the full component tree.
 */
export function _iconResolveFor(
  name: IconName,
  platform: typeof Platform.OS,
): { engine: 'sf' | 'feather'; symbol: string } {
  if (platform === 'ios') {
    return { engine: 'sf', symbol: ICON_MAP[name].sf };
  }
  return { engine: 'feather', symbol: ICON_MAP[name].feather };
}
