/**
 * ErrorState render tests — covers the canonical "load failed" surface.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';

import { ErrorState } from '../ErrorState';

describe('ErrorState', () => {
  test('renders title, description, and a retry CTA', () => {
    const retry = jest.fn();
    const { getByText } = render(
      <ErrorState
        title="Couldn't load history"
        description="Check your connection and try again."
        retry={retry}
      />,
    );
    expect(getByText("Couldn't load history")).toBeTruthy();
    expect(getByText('Check your connection and try again.')).toBeTruthy();
    fireEvent.press(getByText('Retry'));
    expect(retry).toHaveBeenCalledTimes(1);
  });

  test('honors a custom retryLabel', () => {
    const retry = jest.fn();
    const { getByText, queryByText } = render(
      <ErrorState
        title="Report unavailable"
        retry={retry}
        retryLabel="Retry submission"
      />,
    );
    expect(getByText('Retry submission')).toBeTruthy();
    expect(queryByText('Retry')).toBeNull();
  });

  test('renders an optional secondaryAction below the retry CTA', () => {
    const retry = jest.fn();
    const onSecondary = jest.fn();
    const { getByText } = render(
      <ErrorState
        title="Report unavailable"
        retry={retry}
        secondaryAction={{ label: 'Back to home', onPress: onSecondary }}
      />,
    );
    fireEvent.press(getByText('Back to home'));
    expect(onSecondary).toHaveBeenCalledTimes(1);
    expect(retry).not.toHaveBeenCalled();
  });

  test('exposes role=alert for screen readers', () => {
    const { getByRole } = render(
      <ErrorState title="Couldn't load" retry={() => {}} />,
    );
    expect(getByRole('alert')).toBeTruthy();
  });
});
