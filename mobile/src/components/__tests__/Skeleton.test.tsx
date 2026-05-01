/**
 * Skeleton render tests — basic shape coverage.
 *
 * The component is presentational with a single Reanimated effect; we
 * smoke-test it by rendering each common variant (rect / circle / wide
 * percentage block) and asserting that the dimensions land on the
 * underlying View.
 */

import React from 'react';
import { render } from '@testing-library/react-native';

import { Skeleton } from '../Skeleton';

function flattenStyle(style: unknown): Record<string, unknown> {
  if (!style) return {};
  if (Array.isArray(style)) {
    return style.reduce<Record<string, unknown>>(
      (acc, s) => Object.assign(acc, flattenStyle(s)),
      {},
    );
  }
  return style as Record<string, unknown>;
}

// Skeleton sets accessibilityElementsHidden / importantForAccessibility,
// so RNTL v12's default queries skip it. Opt in with the deep flag.
const DEEP = { includeHiddenElements: true } as const;

describe('Skeleton', () => {
  test('renders a fixed-size rectangle', () => {
    const { getByTestId } = render(
      <Skeleton width={120} height={32} radius={4} testID="sk" />,
    );
    const node = getByTestId('sk', DEEP);
    const flat = flattenStyle(node.props.style);
    expect(flat.width).toBe(120);
    expect(flat.height).toBe(32);
    expect(flat.borderRadius).toBe(4);
  });

  test('renders a circle when radius matches half of side', () => {
    const { getByTestId } = render(
      <Skeleton width={48} height={48} radius={24} testID="sk" />,
    );
    const node = getByTestId('sk', DEEP);
    const flat = flattenStyle(node.props.style);
    expect(flat.width).toBe(48);
    expect(flat.height).toBe(48);
    expect(flat.borderRadius).toBe(24);
  });

  test('renders a percentage-width block', () => {
    const { getByTestId } = render(
      <Skeleton width="80%" height={16} testID="sk" />,
    );
    const node = getByTestId('sk', DEEP);
    const flat = flattenStyle(node.props.style);
    expect(flat.width).toBe('80%');
    expect(flat.height).toBe(16);
    // Default radius is 6 when none is provided.
    expect(flat.borderRadius).toBe(6);
  });

  test('hides itself from screen readers', () => {
    const { getByTestId } = render(
      <Skeleton width={100} height={20} testID="sk" />,
    );
    const node = getByTestId('sk', DEEP);
    expect(node.props.accessibilityElementsHidden).toBe(true);
    expect(node.props.importantForAccessibility).toBe('no-hide-descendants');
  });
});
