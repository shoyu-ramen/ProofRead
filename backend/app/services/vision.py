"""Claude vision extractor — single VLM call to read TTB-relevant fields.

Replaces the OCR + regex-extractor combination for the /v1/verify path. The
system prompt defines the JSON output shape and is marked for prompt caching
since it's static across requests.

The extractor prefers `unreadable: true` over guessing on poor images — that
loops back through the rule engine's confidence-aware degradation so an
agent never sees a verdict that was guessed from a smudge.

SPEC §0.5 wiring:

  * The verify path runs `assess_capture_quality()` before this extractor
    is called. The resulting CaptureQualityReport is passed in via
    `capture_quality=` and inlined into the user prompt as an objective
    prior — same shape the scan-path extractor uses (label bbox, glare
    blobs, motion direction, sensor tier). The model decides per-region
    whether it can trust each part of the label.

  * The producer record (the user's COLA-style submission) is passed in
    via `producer_record=` and shown to the model as context, NOT as
    ground truth. The contract is unchanged: the extractor READS, the
    rule engine JUDGES. Knowing the claim only helps the model
    disambiguate ambiguous readings (e.g. choose between two volume
    statements on the same can).

  * The model returns its own image_quality verdict alongside per-field
    confidence/note/bbox; the verify orchestrator merges this with the
    sensor verdict pessimistically.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from app.config import settings
from app.rules.types import ExtractedField
from app.services.anthropic_client import (
    DEFAULT_VISION_TIMEOUT_S,
    ExtractorUnavailable,
    build_client,
    call_with_resilience,
)
from app.services.sensor_briefing import format_capture_quality

logger = logging.getLogger(__name__)

# Fields the extractor is asked to return. Mirrors the keys the rule engine
# (and producer_record cross-references) expect.
#
# The seven base fields are extracted on every label regardless of beverage
# type. Three additional fields are beverage-type-conditional and only
# requested when relevant:
#   * `sulfite_declaration` (wine only) — 27 CFR 4.32 declaration when SO₂
#     ≥ 10 ppm.
#   * `organic_certification` (wine only) — 7 CFR 205 optional claim.
#   * `age_statement` (spirits only) — 27 CFR 5.40 (required for some
#     whiskey classes; advisory otherwise).
# Asking the model for fields that don't apply (e.g. sulfites on a beer
# panel) wastes output tokens and biases extraction toward false positives,
# so the prompt + parser route fields per beverage_type.
EXTRACTOR_BASE_FIELDS = (
    "brand_name",
    "class_type",
    "alcohol_content",
    "net_contents",
    "name_address",
    "country_of_origin",
    "health_warning",
)

EXTRACTOR_BEVERAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "wine": ("sulfite_declaration", "organic_certification"),
    "spirits": ("age_statement",),
    "beer": (),
}

# All fields the parser knows about, in extraction-prompt order. The parser
# walks this tuple to consume any field the model returns; the per-beverage
# allowed-set keeps a model from polluting an extraction with off-type
# fields when it ignored the prompt's routing.
EXTRACTOR_FIELDS = EXTRACTOR_BASE_FIELDS + tuple(
    f for fields in EXTRACTOR_BEVERAGE_FIELDS.values() for f in fields
)


# Per-field minimum confidence floor — applied at the extractor BEFORE the
# field reaches the rule engine. The default global floor
# (`settings.low_confidence_threshold`, 0.6) treats a "partial read; some
# characters uncertain" as usable evidence. That works for ABV/IBU/brand
# where a partial read still carries signal, but it's wrong for proper-
# noun-heavy fields where a "partial read" of a city/state/country is
# *literally* a hallucinated location.
#
# Sonnet 4.6 in production was caught returning
# `name_address = "BREWED & PACKAGED IN OREGON"` at 0.72 confidence on a
# Three Notch'd (Virginia) can — the verb phrase was readable, the model
# fabricated a plausible-sounding state to fill the tail. A 0.85 floor on
# these two fields pushes such partial reads into `unreadable`, which
# downgrades the matching rules to ADVISORY rather than serving a
# confident wrong-pass (the SPEC §0.5 worst-case outcome).
#
# Other fields keep the global default — they're either already clamped
# by the rule engine (alcohol_content / net_contents have their own
# format checks) or low entropy enough that a partial read is genuinely
# useful evidence.
FIELD_CONFIDENCE_FLOORS: dict[str, float] = {
    "name_address": 0.85,
    "country_of_origin": 0.85,
}


def fields_for_beverage(beverage_type: str | None) -> tuple[str, ...]:
    """Return the ordered field list expected for a given beverage_type.

    Falls back to the seven base fields when beverage_type is None or
    unknown — the model still gets a coherent schema and the rule engine's
    own filters handle anything off-spec.
    """
    if beverage_type is None:
        return EXTRACTOR_BASE_FIELDS
    extra = EXTRACTOR_BEVERAGE_FIELDS.get(beverage_type, ())
    return EXTRACTOR_BASE_FIELDS + extra


# Anti-leakage note: the previous version of this prompt embedded specific
# fictional brand/class strings (e.g. "OLD TOM DISTILLERY",
# "Kentucky Straight Bourbon Whiskey", "45% Alc./Vol. (90 Proof)") as
# per-field examples. Those strings are *exactly* what the four bundled
# sample labels print, so the model could "pass" the bundled samples by
# regurgitating its own prompt. Examples below are deliberately abstract —
# format rules and structural shapes only — so the model can only succeed
# by actually reading the label.
SYSTEM_PROMPT = """You are an OCR assistant for U.S. alcohol beverage labels. \
Read a label image and return a JSON object describing what is on the label.

Return ONLY a single JSON object — no Markdown fences (```), no commentary, \
no leading or trailing prose. The downstream parser is strict about that.

Return text VERBATIM as it appears on the label. Preserve the original case \
(uppercase, mixed-case, title-case), punctuation, and whitespace. The downstream \
compliance engine cares about exactness, so do not normalise, expand abbreviations, \
or "tidy up" the wording.

If a field is genuinely not present, or you cannot read it confidently, set \
`unreadable: true` and leave `value` as null. Prefer `unreadable: true` over \
guessing — the agent reviewing this label needs to trust your output.

CRITICAL: Read what is on THIS label. Field descriptions in this prompt are \
*format rules* only. Do NOT copy phrases from this prompt into your output; do \
NOT pattern-match on the user's claimed metadata if any is shown in the user \
message. The user message provides claim/sensor context as a *prior*, not \
ground truth.

Output shape (per field):

  value:        verbatim text from the label, or null if unreadable/absent
  confidence:   0.0–1.0
  unreadable:   include this key ONLY when the field is unreadable (set true).
                Omit it when value was successfully read; the downstream
                parser treats absent `unreadable` as false.
  note:         OMIT this key unless confidence < 0.85, in which case give a
                one-line reason (e.g. "partial occlusion lower-right").

Do NOT emit a `bbox` field — downstream consumers do not use it and emitting \
it costs latency. The schema permits absent bbox; do not fabricate one.

Top-level fields (always present):

  image_quality:           "good" | "degraded" | "unreadable" — your honest
                           assessment of whether the label is fully readable.
  image_quality_notes:     one or two sentences naming the specific reason
                           (e.g. "specular highlight covers right third of
                           label", "hand occludes lower text"). If the
                           sensor pre-check in the user message flagged
                           something, agree or disagree explicitly.
  beverage_type_observed:  "beer" | "wine" | "spirits" | "unknown" — what
                           the label itself indicates (e.g. "Brewing Co.",
                           "IBU", "Ale" → beer; "Vintage 2020", "AVA" →
                           wine; "Distilled", "Bourbon", "Gin" → spirits).
                           If the user's claimed beverage_type disagrees,
                           record the conflict in image_quality_notes.

Field definitions (FORMAT GUIDANCE — do not copy any wording from this list):

  brand_name        — the most prominent brand text on the front face,
                      transcribed verbatim. Usually large and stylised.

  class_type        — the TTB class/type designation (typically a noun
                      phrase like "<style> Lager" / "<varietal>" /
                      "<spirit subtype>"). Verbatim, original case.

  alcohol_content   — alcohol-content statement verbatim. Format usually
                      "<number>% ABV" or "<number>% Alc./Vol." sometimes
                      followed by "(<number> Proof)" in parentheses.
                      The "%" sign and any "ABV" / "Alc./Vol." marker
                      are PART OF THE VALUE — read them together with
                      the number. The downstream format check looks for
                      "%", so a digit alone fails compliance even when
                      the digit is correct.
                        good:  "5.5% ABV"     "4.8% Alc./Vol."
                               "45% Alc./Vol. (90 Proof)"
                        bad:   "5.5"          "4.8"
                      If glare or focus genuinely makes the "%" sign
                      unreadable, lower confidence (≤0.6) and explain in
                      `note` — but still capture whatever marker text
                      you CAN see ("4.8% AB" with one tail letter
                      glared is more useful than "4.8" alone).

  net_contents      — net-contents statement verbatim, INCLUDING the
                      unit. The unit is part of the value, not a
                      separate field — capture the number and the unit
                      together. May include both metric and U.S.
                      customary in parentheses.
                        good:  "12 FL OZ"     "12 fl. oz."
                               "355 mL"       "0.75 L"
                               "12 FL. OZ. (355 mL)"
                        bad:   "12"           "355"
                      If the unit text is genuinely glared or cropped
                      out, lower confidence and capture whatever you
                      can see ("12 fl" or "355 m") rather than dropping
                      to digits alone.

  name_address      — bottler / producer / brewer / importer statement
                      verbatim. Typically begins with a present-tense
                      verb phrase: "Bottled by", "Brewed by", "Distilled
                      by", "Imported by", "Produced by".
                      ANTI-FABRICATION RULE: the city / state / country
                      portion of this statement is the single most
                      hallucination-prone substring on a beverage label.
                      "Brewed in Oregon" is a plausible-sounding default
                      that you must NEVER substitute when the actual
                      location text is glared, cropped, or rotated past
                      legibility. If you can read the verb phrase
                      ("BREWED & PACKAGED BY", "Brewed by") but the
                      proper-noun tail (company name, city, state,
                      country) is unreadable, return ONLY the readable
                      fragment verbatim with confidence ≤0.5 and a note
                      naming the missing portion (e.g. value="BREWED &
                      PACKAGED BY", note="company name and location
                      illegible"). Do NOT guess a state, a country, or a
                      brewery name from typography, color, or visual
                      style. A partial fragment is correct; a fabricated
                      location is the worst possible answer.

  country_of_origin — ONLY if the label EXPLICITLY declares a country of
                      origin (e.g. "Product of <country>", "Imported
                      from <country>"). For domestic U.S. labels with no
                      such statement, set unreadable: true.
                      ANTI-FABRICATION RULE: do NOT infer a country from
                      the language of the label, the brand styling, the
                      apparent origin of the producer, or any other
                      contextual cue. Only return a value when the
                      country name itself is printed and legible. If you
                      can see "Product of" but the country name is
                      unreadable, set unreadable: true rather than
                      guessing.

  health_warning    — the full Government Warning paragraph verbatim,
                      INCLUDING the prefix exactly as it appears. If the
                      label uses "GOVERNMENT WARNING:" in capitals,
                      return it that way. If it uses a different case
                      (e.g. "Government Warning:"), return it that way.
                      The compliance check is character-exact.
                      Recall is critical for THIS field — the warning
                      is the most legally consequential element on the
                      label and the costliest to wrongly mark missing.
                      If glare or specular highlights hide part of the
                      paragraph, return whatever fragment you can read
                      verbatim ("GOVERNMENT WARNING:" header alone,
                      "Surgeon General", "(1) ... pregnancy ...", or
                      any unambiguous chunk) with confidence dropped
                      accordingly — partial text still sets value with
                      unreadable=false. Only set unreadable=true when
                      you have inspected the entire label end-to-end
                      and there is genuinely no warning paragraph
                      anywhere on it.
                      When you can SEE a block of fine print in a
                      typical warning location (back, side, lower
                      edge) but glare or smudging makes the characters
                      themselves unreadable, set unreadable=true AND
                      ensure image_quality_notes explicitly mentions
                      both the warning region and the obstruction
                      (e.g. "warning paragraph visible but glared
                      out", "fine-print block obscured by specular
                      highlight"). The downstream cross-check parses
                      those notes for the obstruction cue and uses it
                      to refuse a confident "warning missing" verdict.

Beverage-type-conditional fields (only requested for the matching
beverage_type — see the per-request "Fields to extract" list in the
user message below):

  sulfite_declaration  — WINE ONLY (27 CFR 4.32). Verbatim sulfite
                         declaration text. Required by TTB when total
                         SO₂ ≥ 10 ppm; nearly all U.S. wines carry it.
                         Format usually a short statement near the
                         government warning or net-contents block.
                           good:  "CONTAINS SULFITES"
                                  "Contains sulfites"
                                  "CONTAINS SULFITES (USED AS A
                                   PRESERVATIVE)"
                         If you cannot find a sulfite declaration on
                         the label, set unreadable: true with value
                         null. The rule engine treats absence as
                         non-compliance — do NOT invent the phrase
                         from a beverage that doesn't show it.

  organic_certification — WINE ONLY (7 CFR 205). OPTIONAL field.
                         Verbatim organic claim if present (USDA
                         Organic seal text, "Made with Organic Grapes",
                         etc.). Many wines have NO organic claim;
                         that is the normal case. If the label does
                         NOT carry an organic statement, set
                         unreadable: true with value null — this is
                         not a compliance failure, it just means the
                         producer did not make an organic claim.
                           good:  "USDA ORGANIC"
                                  "Made with Organic Grapes"
                                  "Certified Organic by [agent]"

  age_statement         — SPIRITS ONLY (27 CFR 5.40). The age
                         declaration if present (verbatim). Required
                         for some whiskey classes (e.g. straight
                         whiskey aged less than 4 years), advisory
                         otherwise — the rule engine handles which
                         applies based on the class_type you read.
                         Capture the format INCLUDING the unit.
                           good:  "Aged 4 Years"
                                  "AGED 18 MONTHS"
                                  "Aged a minimum of 6 years"
                         If no age declaration appears on the label,
                         set unreadable: true with value null.

Confidence scale:

  1.0  — text is large, sharp, and unambiguous
  0.85 — readable but smaller or partially stylised
  0.6  — partial read; some characters uncertain
  <0.6 — very uncertain; consider unreadable: true instead
"""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class _FieldOut(BaseModel):
    value: str | None = Field(
        None, description="Verbatim text as printed on the label, or null."
    )
    unreadable: bool = Field(
        False, description="True if the field cannot be read confidently."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int] | None = Field(
        None, description="[x, y, w, h] in image pixels, or null if unlocated."
    )
    note: str | None = Field(
        None, description="Required if confidence < 0.85."
    )


class VerifyLabelExtraction(BaseModel):
    """Structured output of one verify-path label inspection.

    Mirrors `extractors/claude_vision.LabelExtraction` but trimmed to the
    verify-path fields. The seven base fields are always present; the
    three beverage-type-conditional fields (sulfite_declaration,
    organic_certification, age_statement) are optional — populated only
    when the requested beverage_type asks for them.
    """

    image_quality: Literal["good", "degraded", "unreadable"] = "good"
    image_quality_notes: str = ""
    beverage_type_observed: Literal["beer", "wine", "spirits", "unknown"] = "unknown"
    brand_name: _FieldOut = Field(default_factory=_FieldOut)
    class_type: _FieldOut = Field(default_factory=_FieldOut)
    alcohol_content: _FieldOut = Field(default_factory=_FieldOut)
    net_contents: _FieldOut = Field(default_factory=_FieldOut)
    name_address: _FieldOut = Field(default_factory=_FieldOut)
    country_of_origin: _FieldOut = Field(default_factory=_FieldOut)
    health_warning: _FieldOut = Field(default_factory=_FieldOut)
    # Beverage-type-conditional fields. Default to None so a beer or
    # spirits scan that didn't request `sulfite_declaration` doesn't
    # surface it as an extracted field; the `_from_pydantic` walker
    # skips None entries. Routing lives in `EXTRACTOR_BEVERAGE_FIELDS`.
    sulfite_declaration: _FieldOut | None = None
    organic_certification: _FieldOut | None = None
    age_statement: _FieldOut | None = None


# ---------------------------------------------------------------------------
# Extractor surface
# ---------------------------------------------------------------------------


@dataclass
class VisionExtraction:
    fields: dict[str, ExtractedField]
    unreadable: list[str]
    raw_response: str
    image_quality: str | None = None
    image_quality_notes: str | None = None
    beverage_type_observed: str | None = None


class VisionExtractor(Protocol):
    def extract(
        self,
        image_bytes: bytes,
        media_type: str = ...,
        *,
        capture_quality: Any | None = ...,
        producer_record: dict[str, Any] | None = ...,
        beverage_type: str | None = ...,
        container_size_ml: int | None = ...,
        is_imported: bool = ...,
    ) -> VisionExtraction: ...


def _format_producer_record(record: dict[str, Any] | None) -> str:
    """Render the producer claim as a context block.

    Shown to the model as a *prior*, not ground truth. The compliance
    engine still does the actual claim-vs-label cross-check.
    """
    if not record:
        return "No producer record was provided.\n"
    keys = (
        "brand_name",
        "class_type",
        "alcohol_content",
        "net_contents",
        "name_address",
        "country_of_origin",
    )
    lines = ["Producer record (claim from the submitter — context, not ground truth):"]
    seen_any = False
    for k in keys:
        v = record.get(k)
        if v in (None, ""):
            continue
        seen_any = True
        lines.append(f"  {k:18s} = {v!r}")
    if not seen_any:
        return "No producer record was provided.\n"
    return "\n".join(lines) + "\n"


def _format_claim_header(
    *,
    beverage_type: str | None,
    container_size_ml: int | None,
    is_imported: bool,
) -> str:
    parts: list[str] = []
    if beverage_type:
        parts.append(f"Beverage type (claimed): {beverage_type}")
    if container_size_ml:
        parts.append(f"Container size (claimed): {container_size_ml} mL")
    parts.append(f"Imported: {is_imported}")
    return "\n".join(parts) + "\n"


def _format_fields_to_extract(beverage_type: str | None) -> str:
    """Tell the model which beverage-type-conditional fields to populate.

    The static system prompt describes ALL fields (cached for free), but
    asking the model to extract `sulfite_declaration` from a beer panel
    or `age_statement` from a wine panel wastes output tokens and biases
    the extractor toward false positives. The per-request fields list
    routes which conditional fields apply.
    """
    extra = EXTRACTOR_BEVERAGE_FIELDS.get(beverage_type or "", ())
    lines = ["Fields to extract on this label (set unreadable: true if absent):"]
    lines.extend(f"  - {f}" for f in EXTRACTOR_BASE_FIELDS)
    if extra:
        lines.append(
            f"Beverage-type-specific fields ({beverage_type}) — "
            "extract if present on the label, leave unreadable: true otherwise:"
        )
        lines.extend(f"  - {f}" for f in extra)
    else:
        lines.append(
            "No beverage-type-specific fields apply to this scan; do NOT "
            "emit `sulfite_declaration`, `organic_certification`, or "
            "`age_statement` keys."
        )
    return "\n".join(lines) + "\n"


def _build_user_text(
    *,
    capture_quality: Any | None,
    producer_record: dict[str, Any] | None,
    beverage_type: str | None,
    container_size_ml: int | None,
    is_imported: bool,
) -> str:
    """Compose the text portion of the user message: claim header, producer
    record, sensor briefing, and the inspection instruction."""
    text_blocks: list[str] = [
        _format_claim_header(
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
        ),
        _format_fields_to_extract(beverage_type),
        _format_producer_record(producer_record),
        format_capture_quality(capture_quality),
        (
            "Inspect this label and return the JSON object described above. "
            "Be honest about confidence — a low confidence is the correct "
            "answer when the image is degraded; the rule engine will downgrade "
            "those rules to advisory rather than guess pass/fail."
        ),
    ]
    return "\n".join(b for b in text_blocks if b)


class ClaudeVisionExtractor:
    """Production extractor backed by Claude Sonnet with vision input.

    Uses the centralised `build_client` factory so per-call timeout and
    retry budget come from one place; transient SDK errors are translated
    to `ExtractorUnavailable` so the pipeline can fall back to OCR.

    No extended thinking on this call path. The output schema (per-field
    value + confidence + unreadable flag) already disciplines the model
    into honest under-claiming on degraded labels; thinking buys little
    on a transcription task while costing a meaningful fraction of the
    end-to-end latency budget. The redundant Government-Warning second-
    pass — run concurrently in `verify()` — provides the fail-honestly
    redundancy SPEC §0.5 calls for instead.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        # 1024 is comfortable headroom for the seven-field structured
        # output: a typical label JSON measures ~350 output tokens
        # (verified on the bundled samples), and even a long Government-
        # Warning paragraph plus per-field notes lands well inside 700.
        # The previous 4096 was sized for a much chattier prompt; cutting
        # it now caps tail-latency on weird labels where the model would
        # otherwise spin generating filler before stopping.
        max_tokens: int = 1024,
        timeout: float = DEFAULT_VISION_TIMEOUT_S,
    ) -> None:
        self._client = client or build_client(timeout=timeout)
        self._model = model or settings.anthropic_model
        self._max_tokens = max_tokens

    def extract(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        *,
        capture_quality: Any | None = None,
        producer_record: dict[str, Any] | None = None,
        beverage_type: str | None = None,
        container_size_ml: int | None = None,
        is_imported: bool = False,
    ) -> VisionExtraction:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        user_text = _build_user_text(
            capture_quality=capture_quality,
            producer_record=producer_record,
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
        )
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            },
            {"type": "text", "text": user_text},
        ]

        # Anthropic's `messages.parse` (structured output) rejects this
        # ~38-property schema with "Schema is too complex." The system
        # prompt already constrains the JSON shape, so we use plain
        # `messages.create` and parse the response with the same
        # tolerant parser the Qwen fallback uses.
        # temperature=0 makes the transcription deterministic — same
        # bytes, same fields, same JSON. The cache layer relies on
        # determinism for hit-rate, and a temperature>0 sample on the
        # same image would bust caching upstream of us too.
        response = call_with_resilience(
            self._client.messages.create,
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return _parse_vision_response(text)


class MockVisionExtractor:
    """Deterministic extractor for tests and offline demos.

    Pass a fixture dict shaped like the JSON the real extractor returns:

        {"brand_name": {"value": "...", "confidence": 0.95}, ...}

    or shorthand:

        {"brand_name": "...", ...}

    Top-level keys `image_quality`, `image_quality_notes`, and
    `beverage_type_observed` are also honoured. The mock records the
    last context kwargs passed to `extract()` so tests can assert that
    the briefing wiring is intact.
    """

    def __init__(self, fixture: dict[str, Any]) -> None:
        self._fixture = fixture
        self.last_call: dict[str, Any] = {}

    def extract(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        *,
        capture_quality: Any | None = None,
        producer_record: dict[str, Any] | None = None,
        beverage_type: str | None = None,
        container_size_ml: int | None = None,
        is_imported: bool = False,
    ) -> VisionExtraction:
        self.last_call = {
            "media_type": media_type,
            "capture_quality": capture_quality,
            "producer_record": producer_record,
            "beverage_type": beverage_type,
            "container_size_ml": container_size_ml,
            "is_imported": is_imported,
        }
        normalised: dict[str, Any] = {}
        for name in EXTRACTOR_FIELDS:
            entry = self._fixture.get(name)
            if entry is None:
                continue
            if isinstance(entry, dict):
                normalised[name] = entry
            else:
                normalised[name] = {"value": str(entry), "unreadable": False, "confidence": 0.95}
        for top in ("image_quality", "image_quality_notes", "beverage_type_observed"):
            if top in self._fixture:
                normalised[top] = self._fixture[top]
        return _parse_vision_response(json.dumps(normalised))


def _from_pydantic(result: VerifyLabelExtraction) -> VisionExtraction:
    """Convert structured-output Pydantic result to the dataclass the verify
    orchestrator already consumes."""
    fields: dict[str, ExtractedField] = {}
    unreadable: list[str] = []
    for name in EXTRACTOR_FIELDS:
        fe: _FieldOut | None = getattr(result, name, None)
        if fe is None:
            # Beverage-type-conditional fields default to None when the
            # extractor wasn't asked for them. Don't add them to either
            # `fields` or `unreadable` — they're simply not on this scan.
            continue
        if (
            fe.unreadable
            or fe.value is None
            or (isinstance(fe.value, str) and not fe.value.strip())
        ):
            unreadable.append(name)
            continue
        try:
            confidence = max(0.0, min(1.0, float(fe.confidence)))
        except (TypeError, ValueError):
            confidence = 0.85
        # Per-field anti-fabrication floor — see FIELD_CONFIDENCE_FLOORS.
        # Drop the read into `unreadable` rather than letting a 0.7-confident
        # location fragment reach the rule engine as a confident pass.
        floor = FIELD_CONFIDENCE_FLOORS.get(name)
        if floor is not None and confidence < floor:
            unreadable.append(name)
            continue
        fields[name] = ExtractedField(
            value=fe.value,
            bbox=fe.bbox,
            confidence=confidence,
            source_image_id="front",
        )
    raw = result.model_dump_json()
    return VisionExtraction(
        fields=fields,
        unreadable=unreadable,
        raw_response=raw,
        image_quality=result.image_quality,
        image_quality_notes=result.image_quality_notes or None,
        beverage_type_observed=result.beverage_type_observed,
    )


def _parse_vision_response(text: str) -> VisionExtraction:
    """Parse a free-form JSON response (used by the Qwen fallback, which
    speaks an OpenAI-compatible API and can't bind to a Pydantic schema).

    Tolerates `image_quality` / `image_quality_notes` / `beverage_type_observed`
    at the top level — when missing, falls back to None so the verify
    orchestrator's existing inference rules apply unchanged.
    """
    cleaned = re.sub(r"^\s*```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Same defence as the second-pass parser: Haiku occasionally
        # appends qualifying prose ("**Note:** the warning text was at an
        # angle…") after the JSON object on degraded labels, even though
        # the system prompt forbids it. The end-of-string fence regex
        # above doesn't catch that, so the JSON parse fails and the
        # whole verify request blows up at 500 — discarding a perfectly
        # valid extraction. Recover by walking out the first balanced
        # JSON object from the response and parsing that instead.
        from app.services.health_warning_second_pass import (
            _extract_first_json_object,
        )

        recovered = _extract_first_json_object(cleaned)
        if recovered is None:
            raise ValueError(
                f"Vision extractor returned non-JSON output: {text[:200]!r}"
            ) from exc
        try:
            data = json.loads(recovered)
        except json.JSONDecodeError as exc2:
            raise ValueError(
                f"Vision extractor returned non-JSON output: {text[:200]!r}"
            ) from exc2

    if not isinstance(data, dict):
        raise ValueError(
            f"Vision extractor returned a non-object JSON value: {type(data).__name__}"
        )

    fields: dict[str, ExtractedField] = {}
    unreadable: list[str] = []

    for name in EXTRACTOR_FIELDS:
        entry = data.get(name)
        if not isinstance(entry, dict):
            continue
        if entry.get("unreadable"):
            unreadable.append(name)
            continue
        value = entry.get("value")
        if value is None or (isinstance(value, str) and not value.strip()):
            unreadable.append(name)
            continue
        confidence = entry.get("confidence", 0.85)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.85
        clamped_confidence = max(0.0, min(1.0, confidence))
        # Per-field anti-fabrication floor — see FIELD_CONFIDENCE_FLOORS.
        # Mirrors the same check in `_from_pydantic` so the Qwen fallback
        # path enforces the same hallucination guard as the primary path.
        floor = FIELD_CONFIDENCE_FLOORS.get(name)
        if floor is not None and clamped_confidence < floor:
            unreadable.append(name)
            continue
        bbox = entry.get("bbox")
        bbox_tuple: tuple[int, int, int, int] | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                bbox_tuple = (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                )
            except (TypeError, ValueError):
                bbox_tuple = None
        fields[name] = ExtractedField(
            value=str(value),
            bbox=bbox_tuple,
            confidence=clamped_confidence,
            source_image_id="front",
        )

    image_quality_raw = data.get("image_quality")
    if (
        isinstance(image_quality_raw, str)
        and image_quality_raw in {"good", "degraded", "unreadable"}
    ):
        image_quality: str | None = image_quality_raw
    else:
        image_quality = None
    image_quality_notes = data.get("image_quality_notes")
    if not isinstance(image_quality_notes, str) or not image_quality_notes.strip():
        image_quality_notes = None
    beverage_type_observed = data.get("beverage_type_observed")
    if not isinstance(beverage_type_observed, str):
        beverage_type_observed = None

    return VisionExtraction(
        fields=fields,
        unreadable=unreadable,
        raw_response=text,
        image_quality=image_quality,
        image_quality_notes=image_quality_notes,
        beverage_type_observed=beverage_type_observed,
    )


def get_default_extractor() -> VisionExtractor:
    """Build the configured verify-path extractor.

    When `enable_qwen_fallback=True` AND a Qwen3-VL endpoint is configured,
    Claude is wrapped in a fallback chain — Claude first, Qwen second.
    Both `ExtractorUnavailable` from the Anthropic client and HTTP errors
    against the Qwen server are caught by the chain so a transient outage
    on either side keeps `/v1/verify` working as long as one extractor is
    reachable.
    """
    if settings.vision_extractor != "claude":
        raise RuntimeError(
            f"Unknown vision_extractor {settings.vision_extractor!r}; "
            "expected 'claude'."
        )
    if not settings.anthropic_api_key:
        raise ExtractorUnavailable(
            "ANTHROPIC_API_KEY is not set; required for vision_extractor=claude. "
            "Set the env var or override VISION_EXTRACTOR for tests."
        )

    primary = ClaudeVisionExtractor()
    if settings.enable_qwen_fallback and settings.qwen_vl_base_url:
        from app.services.qwen_vl import QwenVLExtractor
        from app.services.vision_chain import ChainedVerifyExtractor

        return ChainedVerifyExtractor([primary, QwenVLExtractor()])
    return primary
