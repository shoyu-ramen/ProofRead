/**
 * HealthWarningCard — hero compliance card for the Government Warning rule.
 *
 * Surfaced ABOVE the rule list on the report screen because, of all the
 * TTB requirements, the §16.21 Government Warning is the highest-stakes
 * label-art rule: missing or altered text is a per-unit recall risk for
 * the brand, and the wording must match verbatim. Treating it as just
 * another row in the failed/passed list buried the most important
 * verdict; this card lifts it to the top of the screen.
 *
 * Card content (per the v1 brief):
 *   - Status pill: ✓ COMPLIANT / ✕ NON-COMPLIANT / ! UNVERIFIED
 *   - "Government Warning per 27 CFR 16.21" headline
 *   - Side-by-side: extracted text vs canonical text. Diffs are
 *     highlighted character-level when status is non-compliant.
 *   - Tappable citation link → eCFR Title 27 Part 16
 *   - Fix suggestion when present (non-compliant only)
 *
 * Visual weight: matches the existing report HeaderCard primitive but
 * with slightly larger padding and a semantic accent border (status
 * color) so the eye lands here first. Uses only existing theme tokens.
 *
 * Accessibility: each section has descriptive labels; the diff highlight
 * is reinforced by labeling the columns "Found on label" and "Required
 * text", so users who can't see the highlight color still get the
 * comparison through screen-reader narration.
 */

import React, { useMemo } from 'react';
import {
  Linking,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { colors, radius, spacing, typography } from '@src/theme';
import type { RuleResultDTO, RuleStatus } from '@src/api/types';

/**
 * Canonical Government Warning text per 27 CFR 16.21. Mirrored from
 * `backend/app/canonical/health_warning.txt` — kept in sync there.
 * Surfaced inline rather than fetched from the backend so the card can
 * render even when the report response doesn't include the canonical
 * field (older API rows). The wording is statutory, not configurable —
 * the regulation prescribes it verbatim.
 */
export const CANONICAL_HEALTH_WARNING =
  'GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and may cause health problems.';

/**
 * eCFR canonical URL for 27 CFR Part 16 (Alcoholic Beverage Health
 * Warning Statement). Hard-coded because the citation never changes
 * and we don't want to depend on `ecfrUrlForCitation` returning the
 * right thing for a free-form citation string.
 */
export const HEALTH_WARNING_ECFR_URL =
  'https://www.ecfr.gov/current/title-27/chapter-I/subchapter-A/part-16';

export type HealthWarningStatus = 'compliant' | 'non_compliant' | 'unverified';

export interface HealthWarningCardProps {
  /**
   * The `health_warning.exact_text` rule result, when present in the
   * report. The card derives its status from this; if not provided
   * (rule wasn't evaluated), the card renders in the `unverified` state.
   */
  rule?: RuleResultDTO | null;
  /**
   * Optional override for the canonical text — primarily for tests and
   * for forward-compat if a later regulation revision lands. Defaults to
   * the 27 CFR 16.21 statutory text.
   */
  canonicalText?: string;
  /**
   * Tap handler for the citation link. Defaults to opening the eCFR URL
   * with `Linking.openURL`. Override for tests or analytics.
   */
  onOpenCitation?: () => void;
}

/**
 * Map a rule result's status (or absence) to the card's three-way
 * verdict. The rule's `pass` becomes ✓ COMPLIANT, `fail` becomes
 * ✕ NON-COMPLIANT, and `advisory` (or missing rule) becomes ! UNVERIFIED.
 * Advisory specifically means the extractor couldn't read the warning
 * with confidence — we don't know whether it's compliant or not, so the
 * honest verdict is "unverified" rather than a softer fail.
 */
function statusForRule(rule: RuleResultDTO | null | undefined): HealthWarningStatus {
  if (!rule) return 'unverified';
  switch (rule.status) {
    case 'pass':
      return 'compliant';
    case 'fail':
      return 'non_compliant';
    case 'advisory':
    default:
      return 'unverified';
  }
}

interface StatusVisual {
  label: string;
  glyph: string;
  fg: string;
  bg: string;
  border: string;
  accent: string;
}

function visualFor(status: HealthWarningStatus): StatusVisual {
  switch (status) {
    case 'compliant':
      return {
        label: 'COMPLIANT',
        glyph: '✓',
        fg: colors.pass,
        bg: 'rgba(61,220,151,0.12)',
        border: colors.pass,
        accent: colors.pass,
      };
    case 'non_compliant':
      return {
        label: 'NON-COMPLIANT',
        glyph: '✕',
        fg: colors.fail,
        bg: 'rgba(255,107,107,0.12)',
        border: colors.fail,
        accent: colors.fail,
      };
    case 'unverified':
    default:
      return {
        label: 'UNVERIFIED',
        glyph: '!',
        fg: colors.advisory,
        bg: 'rgba(244,184,96,0.12)',
        border: colors.advisory,
        accent: colors.advisory,
      };
  }
}

export function HealthWarningCard({
  rule,
  canonicalText = CANONICAL_HEALTH_WARNING,
  onOpenCitation,
}: HealthWarningCardProps): React.ReactElement {
  const status = statusForRule(rule ?? null);
  const visual = visualFor(status);
  const extracted = rule?.finding ?? '';
  const trimmedExtracted = extracted.trim();
  const hasExtracted = trimmedExtracted.length > 0;

  // Only do the expensive char-level diff for non_compliant. For
  // compliant we know it matches; for unverified the user sees the raw
  // canonical and we just label the extracted column with a "not
  // detected" message rather than a noisy red strikethrough.
  const diffSegments = useMemo(() => {
    if (status !== 'non_compliant' || !hasExtracted) return null;
    return diffStrings(trimmedExtracted, canonicalText);
  }, [status, hasExtracted, trimmedExtracted, canonicalText]);

  const handleCitationPress = () => {
    if (onOpenCitation) {
      onOpenCitation();
      return;
    }
    try {
      void Linking.openURL(HEALTH_WARNING_ECFR_URL);
    } catch {
      // Linking can throw on platforms without a registered http handler;
      // swallow rather than crash the report screen.
    }
  };

  return (
    <View
      style={[
        styles.card,
        { borderLeftColor: visual.accent, borderColor: visual.border },
      ]}
      accessible
      accessibilityRole="summary"
      accessibilityLabel={`Government Warning compliance: ${visual.label}`}
    >
      <View style={styles.headerRow}>
        <View
          style={[
            styles.statusPill,
            { backgroundColor: visual.bg, borderColor: visual.border },
          ]}
          accessibilityElementsHidden
          importantForAccessibility="no"
        >
          <Text style={[styles.statusGlyph, { color: visual.fg }]}>
            {visual.glyph}
          </Text>
          <Text style={[styles.statusLabel, { color: visual.fg }]}>
            {visual.label}
          </Text>
        </View>
      </View>

      <Text style={styles.headline}>Government Warning per 27 CFR 16.21</Text>

      <View style={styles.compareRow}>
        <View style={styles.compareCol}>
          <Text style={styles.compareLabel}>Found on label</Text>
          <View
            style={[
              styles.compareBox,
              !hasExtracted && styles.compareBoxEmpty,
            ]}
          >
            {hasExtracted ? (
              status === 'non_compliant' && diffSegments ? (
                <Text style={styles.compareText}>
                  {diffSegments.found.map((seg, idx) => (
                    <Text
                      key={`f-${idx}`}
                      style={
                        seg.changed
                          ? [
                              styles.diffChanged,
                              {
                                color: colors.fail,
                                backgroundColor: 'rgba(255,107,107,0.18)',
                              },
                            ]
                          : undefined
                      }
                    >
                      {seg.text}
                    </Text>
                  ))}
                </Text>
              ) : (
                <Text style={styles.compareText}>{trimmedExtracted}</Text>
              )
            ) : (
              <Text style={styles.notDetected}>Not detected on this label.</Text>
            )}
          </View>
        </View>
        <View style={styles.compareCol}>
          <Text style={styles.compareLabel}>Required text</Text>
          <View style={[styles.compareBox, styles.compareBoxCanonical]}>
            <Text style={styles.compareText}>
              {status === 'non_compliant' && diffSegments
                ? diffSegments.canonical.map((seg, idx) => (
                    <Text
                      key={`c-${idx}`}
                      style={
                        seg.changed
                          ? [
                              styles.diffChanged,
                              {
                                color: colors.pass,
                                backgroundColor: 'rgba(61,220,151,0.18)',
                              },
                            ]
                          : undefined
                      }
                    >
                      {seg.text}
                    </Text>
                  ))
                : canonicalText}
            </Text>
          </View>
        </View>
      </View>

      <Pressable
        onPress={handleCitationPress}
        accessibilityRole="link"
        accessibilityLabel="Open 27 CFR Part 16 on eCFR.gov"
        hitSlop={8}
        style={({ pressed }) => [
          styles.citationLink,
          pressed && { opacity: 0.7 },
        ]}
      >
        <Text style={styles.citationLinkText}>27 CFR Part 16 on eCFR.gov</Text>
        <Text style={styles.citationLinkIcon}>↗</Text>
      </Pressable>

      {status === 'non_compliant' && rule?.fix_suggestion ? (
        <View style={styles.fixBox}>
          <Text style={styles.fixLabel}>How to fix</Text>
          <Text style={styles.fixBody}>{rule.fix_suggestion}</Text>
        </View>
      ) : null}
    </View>
  );
}

interface DiffSegment {
  text: string;
  changed: boolean;
}

interface DiffPair {
  found: DiffSegment[];
  canonical: DiffSegment[];
}

/**
 * Character-level diff between the extracted and canonical strings.
 *
 * Implementation: longest-common-subsequence (LCS) so we don't get
 * false-positive mismatches when the extractor inserts or drops a
 * single character. We're capped at ~512 chars on either side (the
 * canonical statement is ~330 chars, longest plausible OCR slop is
 * comparable), so the O(n*m) memory cost is bounded. For inputs over
 * the cap, we fall back to a naive char-by-char compare so we still
 * surface *something* rather than spending budget on a degenerate diff.
 */
function diffStrings(found: string, canonical: string): DiffPair {
  const MAX = 512;
  if (found.length > MAX || canonical.length > MAX) {
    return diffNaive(found, canonical);
  }
  return diffLcs(found, canonical);
}

function diffNaive(found: string, canonical: string): DiffPair {
  const f: DiffSegment[] = [];
  const c: DiffSegment[] = [];
  const len = Math.max(found.length, canonical.length);
  for (let i = 0; i < len; i += 1) {
    const fc = i < found.length ? found[i] : '';
    const cc = i < canonical.length ? canonical[i] : '';
    f.push({ text: fc, changed: fc !== cc });
    c.push({ text: cc, changed: fc !== cc });
  }
  return { found: mergeAdjacent(f), canonical: mergeAdjacent(c) };
}

function diffLcs(found: string, canonical: string): DiffPair {
  const m = found.length;
  const n = canonical.length;
  // dp[i][j] = LCS length of found[0..i] vs canonical[0..j]
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    new Array<number>(n + 1).fill(0),
  );
  for (let i = 1; i <= m; i += 1) {
    for (let j = 1; j <= n; j += 1) {
      if (found[i - 1] === canonical[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  // Walk backward, emitting per-char segments.
  const fOut: DiffSegment[] = [];
  const cOut: DiffSegment[] = [];
  let i = m;
  let j = n;
  while (i > 0 && j > 0) {
    if (found[i - 1] === canonical[j - 1]) {
      fOut.unshift({ text: found[i - 1], changed: false });
      cOut.unshift({ text: canonical[j - 1], changed: false });
      i -= 1;
      j -= 1;
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      // Char in `found` was deleted relative to canonical → mark in found col.
      fOut.unshift({ text: found[i - 1], changed: true });
      i -= 1;
    } else {
      // Char in canonical missing from `found` → mark in canonical col.
      cOut.unshift({ text: canonical[j - 1], changed: true });
      j -= 1;
    }
  }
  while (i > 0) {
    fOut.unshift({ text: found[i - 1], changed: true });
    i -= 1;
  }
  while (j > 0) {
    cOut.unshift({ text: canonical[j - 1], changed: true });
    j -= 1;
  }
  return { found: mergeAdjacent(fOut), canonical: mergeAdjacent(cOut) };
}

/**
 * Coalesce runs of segments with the same `changed` flag into a single
 * segment. Reduces the rendered Text-element count from N (one per
 * char) to roughly the number of edit clusters, which is what RN's text
 * layout actually wants — RN's nested-Text path is fine with a few dozen
 * elements but slows down dramatically at hundreds.
 */
function mergeAdjacent(segs: DiffSegment[]): DiffSegment[] {
  if (segs.length === 0) return segs;
  const out: DiffSegment[] = [{ ...segs[0] }];
  for (let i = 1; i < segs.length; i += 1) {
    const last = out[out.length - 1];
    if (segs[i].changed === last.changed) {
      last.text += segs[i].text;
    } else {
      out.push({ ...segs[i] });
    }
  }
  return out;
}

/**
 * Predicate used by the report screen — does this rule result feed the
 * HealthWarningCard? We surface the `health_warning.exact_text` rule
 * (the verbatim wording check); the related `health_warning.presence`
 * rule remains a normal row in the rule list.
 */
export function isHealthWarningRule(rule: RuleResultDTO): boolean {
  return rule.rule_id === 'beer.health_warning.exact_text';
}

/**
 * Helper for the rule list rendering — returns a flag indicating that
 * this rule is also being surfaced in the hero card. Call sites use
 * this to add a visual hint (e.g. dimming or an "also shown above"
 * annotation) without removing the row.
 */
export function ruleIsSurfacedByHero(rule: RuleResultDTO): boolean {
  return isHealthWarningRule(rule);
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    // Slightly larger padding than the standard HeaderCard to give the
    // hero card visual weight without introducing a new size token.
    padding: spacing.lg + spacing.xs,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderLeftWidth: 4,
    gap: spacing.sm,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: 999,
    borderWidth: 1,
  },
  statusGlyph: {
    fontSize: 14,
    fontWeight: '700',
    lineHeight: 16,
  },
  statusLabel: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.6,
    lineHeight: 14,
  },
  headline: {
    ...typography.heading,
    color: colors.text,
  },
  compareRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  compareCol: {
    flex: 1,
    gap: spacing.xs,
  },
  compareLabel: {
    ...typography.caption,
    color: colors.textMuted,
    fontWeight: '700',
    letterSpacing: 0.4,
  },
  compareBox: {
    backgroundColor: colors.surfaceAlt,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.sm,
    minHeight: 96,
  },
  compareBoxEmpty: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  compareBoxCanonical: {
    backgroundColor: colors.surfaceAlt,
  },
  compareText: {
    ...typography.caption,
    color: colors.text,
    lineHeight: 18,
  },
  diffChanged: {
    fontWeight: '700',
  },
  notDetected: {
    ...typography.caption,
    color: colors.textMuted,
    fontStyle: 'italic',
    textAlign: 'center',
  },
  citationLink: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    alignSelf: 'flex-start',
  },
  citationLinkText: {
    ...typography.caption,
    color: colors.primary,
    textDecorationLine: 'underline',
  },
  citationLinkIcon: {
    ...typography.caption,
    color: colors.primary,
  },
  fixBox: {
    backgroundColor: colors.surfaceAlt,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.xs,
  },
  fixLabel: {
    ...typography.caption,
    color: colors.textMuted,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  fixBody: {
    ...typography.body,
    color: colors.text,
  },
});
