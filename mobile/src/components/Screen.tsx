import React from 'react';
import { ScrollView, StyleSheet, View, ViewStyle } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { colors, spacing } from '@src/theme';

export interface ScreenProps {
  children: React.ReactNode;
  scroll?: boolean;
  contentStyle?: ViewStyle;
  edges?: Array<'top' | 'bottom' | 'left' | 'right'>;
  // Pass-through for the SafeAreaView background; useful for the
  // camera screen to go edge-to-edge.
  background?: string;
}

export function Screen({
  children,
  scroll = true,
  contentStyle,
  edges = ['top', 'bottom', 'left', 'right'],
  background,
}: ScreenProps): React.ReactElement {
  return (
    <SafeAreaView
      edges={edges}
      style={[styles.safe, background ? { backgroundColor: background } : null]}
    >
      {scroll ? (
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={[styles.content, contentStyle]}
          keyboardShouldPersistTaps="handled"
        >
          {children}
        </ScrollView>
      ) : (
        <View style={[styles.content, contentStyle]}>{children}</View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scroll: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
    padding: spacing.lg,
    gap: spacing.md,
  },
});
