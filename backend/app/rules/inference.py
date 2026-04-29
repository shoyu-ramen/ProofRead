"""Cross-field inferences applied to the rule context before evaluation.

The user's POST claim ("imported: false") and the label's actual content
can disagree. SPEC §0.5 fail-honestly: when the label says "Product of
Germany" but the user forgot to tick the imported box, we'd rather apply
the country-of-origin rule and surface the divergence than silently skip
the rule and let a non-compliant label pass.

The functions here are pure — no side effects on the context. Callers
swap in the returned value and emit the divergence as an advisory rule
result (see `RuleEngine.evaluate`).
"""

from __future__ import annotations

from app.rules.types import ExtractedField


def infer_is_imported(
    extracted_fields: dict[str, ExtractedField],
    claimed_imported: bool,
) -> tuple[bool, str | None]:
    """Reconcile the user's `is_imported` claim against the extracted label.

    Returns ``(effective_imported, divergence)``:

      * ``effective_imported`` — the value the rule engine should use when
        evaluating ``applies_if: is_imported == True`` guards. The
        user's claim wins unless the label clearly indicates imported
        (a non-empty ``country_of_origin`` field) and they claimed
        domestic — in which case we flip to True so the rule fires.
      * ``divergence`` — a short reason code when we flipped the flag,
        ``None`` otherwise. The orchestrator surfaces this as a
        ``claim_consistency`` ADVISORY so the user sees why the rule was
        evaluated despite their claim.

    We deliberately do NOT flip the other direction (claimed imported
    but no country field on the label). The country-of-origin field can
    legitimately be unreadable from the front surface alone, and demoting
    a user's positive claim could silently skip a rule the user meant to
    enforce — a worse outcome than running the rule and letting it fail
    on a missing field.
    """
    coo = extracted_fields.get("country_of_origin")
    has_country = (
        coo is not None
        and coo.value is not None
        and coo.value.strip() != ""
    )
    if has_country and not claimed_imported:
        return True, "label_indicates_imported"
    return claimed_imported, None
