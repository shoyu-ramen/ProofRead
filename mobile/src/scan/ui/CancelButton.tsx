/**
 * CancelButton — top-left back affordance for the cylindrical scan
 * screen (SCAN_DESIGN §3.7 + §4.6).
 *
 * 36×36 circular target sitting on `scanOverlayDim`, hit-slop 12 on all
 * sides. Press feedback is the spec'd opacity dip only — no scale. If
 * the parent indicates active progress (coverage > 0.05), the button
 * surfaces a confirm dialog before invoking `onCancel`; otherwise it
 * goes straight back.
 */

import React, { useCallback } from 'react';
import { Alert, Pressable, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Feather } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

import { colors } from '@src/theme';

export interface CancelButtonProps {
  /**
   * Current scan coverage in 0..1. Used purely to decide whether to
   * surface a confirm dialog — over 0.05 means the user has already
   * invested rotation effort and shouldn't lose it on a stray tap.
   */
  coverage: number;
  /** Invoked when the user confirms cancel (or no confirm was needed). */
  onCancel: () => void;
}

export function CancelButton({
  coverage,
  onCancel,
}: CancelButtonProps): React.ReactElement {
  const insets = useSafeAreaInsets();

  const handlePress = useCallback(() => {
    void Haptics.selectionAsync().catch(() => {});
    if (coverage > 0.05) {
      Alert.alert(
        'Cancel scan?',
        "You'll lose your rotation progress.",
        [
          { text: 'Keep scanning', style: 'cancel' },
          { text: 'Cancel scan', style: 'destructive', onPress: onCancel },
        ],
        { cancelable: true },
      );
      return;
    }
    onCancel();
  }, [coverage, onCancel]);

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel="Cancel scan"
      hitSlop={12}
      onPress={handlePress}
      style={({ pressed }) => [
        styles.button,
        { top: insets.top + 14 },
        { opacity: pressed ? 0.65 : 0.9 },
      ]}
    >
      <Feather name="x" size={18} color={colors.scanInk} />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    position: 'absolute',
    left: 16,
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.scanOverlayDim,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
