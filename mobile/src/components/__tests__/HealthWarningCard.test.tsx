/**
 * HealthWarningCard render tests.
 *
 * Covers each of the four card states the report screen can produce:
 *   - compliant (rule.status === 'pass')
 *   - non_compliant (rule.status === 'fail') with a diff against the
 *     canonical text
 *   - unverified (rule.status === 'advisory')
 *   - not detected (rule absent or rule.finding empty)
 *
 * The card is mostly presentational; we exercise the public surface
 * through rendered text rather than reaching into private styles, so
 * later layout changes don't break these tests.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Linking } from 'react-native';

import {
  HealthWarningCard,
  CANONICAL_HEALTH_WARNING,
} from '../HealthWarningCard';
import type { RuleResultDTO } from '@src/api/types';

function makeRule(overrides: Partial<RuleResultDTO> = {}): RuleResultDTO {
  return {
    rule_id: 'beer.health_warning.exact_text',
    rule_version: 1,
    citation: '27 CFR 16.21',
    status: 'pass',
    finding: CANONICAL_HEALTH_WARNING,
    expected: CANONICAL_HEALTH_WARNING,
    fix_suggestion: null,
    bbox: null,
    surface: 'panorama',
    explanation: null,
    ...overrides,
  };
}

// The status pill is rendered inside an `accessibilityElementsHidden`
// parent (the parent View carries a composed accessibility label that
// already covers "Government Warning compliance: COMPLIANT"); RNTL v12's
// queries skip hidden elements by default, so we opt in with
// `includeHiddenElements: true` when we want to assert on the visible-
// but-hidden-from-AX badge text directly.
const DEEP = { includeHiddenElements: true } as const;

describe('HealthWarningCard', () => {
  test('renders compliant state when the rule passed', () => {
    const { getByText, queryByText } = render(
      <HealthWarningCard rule={makeRule({ status: 'pass' })} />,
    );
    expect(getByText('COMPLIANT', DEEP)).toBeTruthy();
    expect(getByText('Government Warning per 27 CFR 16.21')).toBeTruthy();
    // No "How to fix" surface for compliant.
    expect(queryByText('How to fix')).toBeNull();
  });

  test('renders non-compliant state and surfaces fix_suggestion', () => {
    const rule = makeRule({
      status: 'fail',
      finding: 'GOVERNMENT WARNING: women should not drink during pregnancy.',
      fix_suggestion: 'Use the verbatim §16.21 statement.',
    });
    const { getByText } = render(<HealthWarningCard rule={rule} />);
    expect(getByText('NON-COMPLIANT', DEEP)).toBeTruthy();
    expect(getByText('How to fix')).toBeTruthy();
    expect(getByText('Use the verbatim §16.21 statement.')).toBeTruthy();
  });

  test('renders unverified state for advisory rule', () => {
    const rule = makeRule({ status: 'advisory', finding: null });
    const { getByText } = render(<HealthWarningCard rule={rule} />);
    expect(getByText('UNVERIFIED', DEEP)).toBeTruthy();
  });

  test('renders not-detected message when finding is empty', () => {
    const rule = makeRule({ status: 'fail', finding: '' });
    const { getByText } = render(<HealthWarningCard rule={rule} />);
    expect(getByText('NON-COMPLIANT', DEEP)).toBeTruthy();
    expect(getByText('Not detected on this label.')).toBeTruthy();
  });

  test('renders unverified when no rule is provided', () => {
    const { getByText } = render(<HealthWarningCard rule={null} />);
    expect(getByText('UNVERIFIED', DEEP)).toBeTruthy();
    expect(getByText('Not detected on this label.')).toBeTruthy();
  });

  test('citation tap opens eCFR via the override hook', () => {
    const onOpenCitation = jest.fn();
    const { getByLabelText } = render(
      <HealthWarningCard
        rule={makeRule({ status: 'pass' })}
        onOpenCitation={onOpenCitation}
      />,
    );
    const link = getByLabelText('Open 27 CFR Part 16 on eCFR.gov');
    fireEvent.press(link);
    expect(onOpenCitation).toHaveBeenCalledTimes(1);
  });

  test('citation tap defaults to Linking.openURL when no override given', () => {
    const openSpy = jest
      .spyOn(Linking, 'openURL')
      .mockImplementation(() => Promise.resolve(true));
    const { getByLabelText } = render(
      <HealthWarningCard rule={makeRule({ status: 'pass' })} />,
    );
    const link = getByLabelText('Open 27 CFR Part 16 on eCFR.gov');
    fireEvent.press(link);
    expect(openSpy).toHaveBeenCalledWith(
      expect.stringContaining('ecfr.gov'),
    );
    openSpy.mockRestore();
  });
});
