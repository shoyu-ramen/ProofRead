/**
 * Icon render tests.
 *
 * The full SF Symbols vs. Feather routing is covered by the
 * `_iconResolveFor` helper (deterministic — given a platform string,
 * returns the engine + symbol that the renderer would pick). The
 * platform-default test exercises the actual rendered tree on the
 * jest-expo default platform (iOS) so we get real-component coverage
 * too.
 */

import React from 'react';
import { Platform } from 'react-native';
import { render } from '@testing-library/react-native';

import { Icon, _iconResolveFor } from '../Icon';

describe('Icon._iconResolveFor', () => {
  test('iOS returns SF Symbol metadata', () => {
    const r = _iconResolveFor('chevron-right', 'ios');
    expect(r.engine).toBe('sf');
    expect(r.symbol).toBe('chevron.right');
  });

  test('Android falls back to Feather', () => {
    const r = _iconResolveFor('chevron-right', 'android');
    expect(r.engine).toBe('feather');
    expect(r.symbol).toBe('chevron-right');
  });

  test.each([
    ['camera', 'camera.viewfinder', 'camera'],
    ['alert-circle', 'exclamationmark.circle.fill', 'alert-circle'],
    ['check-circle', 'checkmark.circle.fill', 'check-circle'],
    ['inbox', 'tray', 'inbox'],
    ['x', 'xmark', 'x'],
  ] as const)('%s maps to SF=%s and Feather=%s', (name, sf, feather) => {
    const ios = _iconResolveFor(name, 'ios');
    const android = _iconResolveFor(name, 'android');
    expect(ios.symbol).toBe(sf);
    expect(android.symbol).toBe(feather);
  });
});

describe('Icon component', () => {
  test('renders without throwing on the default platform', () => {
    const { toJSON } = render(<Icon name="alert-circle" size={20} />);
    expect(toJSON()).toBeTruthy();
  });

  test('exposes its accessibility label when provided', () => {
    const { getByLabelText } = render(
      <Icon
        name="alert-circle"
        accessibilityLabel="Validation error"
      />,
    );
    expect(getByLabelText('Validation error')).toBeTruthy();
  });

  test('hides itself from screen readers when no label is provided', () => {
    const { toJSON } = render(<Icon name="check-circle" />);
    const tree = toJSON();
    // The Icon's wrapping View applies accessibilityElementsHidden when
    // there's no label; we walk the tree to verify the prop landed.
    function findHidden(node: ReturnType<typeof toJSON>): boolean {
      if (!node) return false;
      if (Array.isArray(node)) return node.some(findHidden);
      if (typeof node !== 'object') return false;
      const props = (node as { props?: Record<string, unknown> }).props ?? {};
      if (props.accessibilityElementsHidden === true) return true;
      const children = (node as { children?: unknown }).children;
      if (Array.isArray(children)) return children.some(findHidden);
      return false;
    }
    expect(findHidden(tree)).toBe(true);
  });

  test('current jest platform is iOS by default', () => {
    // jest-expo defaults Platform.OS to 'ios'; the Icon component uses
    // that to choose its rendering engine. Asserting it here keeps the
    // expectations in the tests above honest if the preset ever
    // changes its default.
    expect(Platform.OS).toBe('ios');
  });
});
