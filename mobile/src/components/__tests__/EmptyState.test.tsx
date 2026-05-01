/**
 * EmptyState render tests — confirms each variant renders the expected
 * surface (title + optional description + optional action).
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';

import { EmptyState } from '../EmptyState';

describe('EmptyState', () => {
  test('renders title only when description and action are omitted', () => {
    const { getByText, queryByRole } = render(
      <EmptyState title="No scans yet" />,
    );
    expect(getByText('No scans yet')).toBeTruthy();
    // No CTA should be present.
    expect(queryByRole('button')).toBeNull();
  });

  test('renders title and description', () => {
    const { getByText } = render(
      <EmptyState
        title="No scans yet"
        description="Scan your first label to see it here."
      />,
    );
    expect(getByText('No scans yet')).toBeTruthy();
    expect(getByText('Scan your first label to see it here.')).toBeTruthy();
  });

  test('renders an action button and fires its handler on press', () => {
    const onPress = jest.fn();
    const { getByText } = render(
      <EmptyState
        title="No scans yet"
        action={{ label: 'Start your first scan', onPress }}
      />,
    );
    const cta = getByText('Start your first scan');
    fireEvent.press(cta);
    expect(onPress).toHaveBeenCalledTimes(1);
  });

  test('exposes a composed accessibility label', () => {
    const { getByLabelText } = render(
      <EmptyState
        title="No scans yet"
        description="Scan your first label to see it here."
      />,
    );
    expect(
      getByLabelText('No scans yet. Scan your first label to see it here.'),
    ).toBeTruthy();
  });
});
