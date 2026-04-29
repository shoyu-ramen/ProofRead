"""AI-generated per-rule explanations.

Adds a plain-language sentence to each failed rule so users see WHY their
label tripped the check, contextualized to THIS scan's extracted values
and image quality — not the static fix_suggestion shipped in rule YAML.

Generation runs on the verify cold path (so warm hits are free) and
batches all failed rules in one Claude call to amortize the overhead.
The explanation is purely additive — generation failures never block the
verdict; the response simply omits the field for unexplained rules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.services.anthropic_client import (
    DEFAULT_SECOND_PASS_TIMEOUT_S,
    ExtractorUnavailable,
    build_client,
    call_with_resilience,
)

logger = logging.getLogger(__name__)


# Bound the per-rule output. 25 words / ~150 chars is what we ask the
# model for; 200 tokens of headroom per rule covers JSON quoting + the
# wrapper object without ever truncating mid-sentence on Haiku.
_TOKENS_PER_RULE = 200
# Floor + ceiling on the total max_tokens so a single-rule call doesn't
# under-budget and a six-rule call doesn't blow the response budget.
_MIN_OUTPUT_TOKENS = 400
_MAX_OUTPUT_TOKENS = 2048


SYSTEM_PROMPT = """You are explaining U.S. TTB alcohol-beverage label \
compliance findings to the producer who just submitted the label. They \
care about ONE thing: a clear sentence saying WHY their label failed each \
specific check, grounded in what was actually read off THEIR label.

You will receive:
- The beverage type (beer/wine/spirits), container size, and import status
- The overall image quality of the scan
- A list of failed rules — for each one: rule_id, citation, finding, \
expected, the static fix-it text from the rule book, and the relevant \
extracted field values from this label

Your job: produce ONE short sentence per rule — plain language, no jargon, \
no markdown, no bullet points, no greetings. Reference the actual extracted \
value when it sharpens the explanation (e.g. "your 5.5% ABV beer in a \
355 mL can needs..."). Each sentence MUST be ≤ 25 words.

Output ONLY a JSON object mapping rule_id to its explanation string. No \
prose before or after. No markdown fences. No commentary on rules you \
weren't given. Example shape:

{"beer.health_warning.exact_text": "Your label's warning paragraph differs \
from the statutory text by 47 characters; rewrite it verbatim.", \
"beer.alcohol_content.format": "TTB requires '% alcohol by volume' or \
'% alc/vol' — your label shows '5.5% ABV' which isn't a permitted form."}"""


@dataclass(frozen=True)
class RuleExplanationInput:
    rule_id: str
    rule_status: str  # "fail" | "advisory"
    citation: str
    finding: str | None
    expected: str | None
    # Static text from rule YAML — passed in for grounding so the model
    # rephrases rather than re-deriving it.
    fix_suggestion: str | None
    # Relevant extracted fields keyed by field id, e.g.
    # {"alcohol_content": "5.5% ABV", "container_size_ml": "355"}.
    field_values: dict[str, str | None]


async def explain_rules(
    rules: list[RuleExplanationInput],
    *,
    beverage_type: str,
    container_size_ml: int,
    is_imported: bool,
    image_quality: str,
    timeout_s: float = 6.0,
) -> dict[str, str]:
    """Return a mapping of rule_id -> one-sentence explanation.

    Returns an empty dict on any failure (timeout, API error, malformed
    response). Never raises — explanation is additive UX, not a verdict
    input.

    Calls Claude Haiku with a single batch prompt: ALL failed rules
    explained in one round-trip, returned as a JSON object the caller can
    map back. This is dramatically cheaper than one call per rule when
    a label fails on 3-5 checks (the common case).

    Single-rule case still goes through the same JSON path for shape
    consistency.
    """
    # Disabled at the settings layer — short-circuit so tests can assert
    # no API call ever happens, and prod can flip the kill-switch without
    # restarting the service.
    if not settings.explanation_enabled:
        return {}

    if not rules:
        return {}

    # Cap the batch so a label with a long tail of advisories doesn't
    # blow the prompt budget. The N most-severe rules are passed in by
    # the caller; we just trim defensively.
    capped = rules[: settings.explanation_max_rules]
    expected_ids = {r.rule_id for r in capped}

    model = settings.explanation_model or settings.anthropic_model
    max_tokens = max(
        _MIN_OUTPUT_TOKENS, min(_MAX_OUTPUT_TOKENS, _TOKENS_PER_RULE * len(capped))
    )
    prompt = _build_prompt(
        capped,
        beverage_type=beverage_type,
        container_size_ml=container_size_ml,
        is_imported=is_imported,
        image_quality=image_quality,
    )

    try:
        client = build_client(timeout=DEFAULT_SECOND_PASS_TIMEOUT_S)
    except ExtractorUnavailable:
        # No API key, no client. Quietly drop — explanation is additive.
        logger.debug("Explanation skipped: anthropic client unavailable")
        return {}

    def _invoke() -> Any:
        return call_with_resilience(
            client.messages.create,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(_invoke),
            timeout=timeout_s,
        )
    except TimeoutError:
        logger.warning(
            "Explanation generation timed out after %.1fs (rules=%d)",
            timeout_s,
            len(capped),
        )
        return {}
    except ExtractorUnavailable as exc:
        logger.warning("Explanation generation unavailable: %s", exc)
        return {}
    except Exception as exc:  # noqa: BLE001  — additive UX, never re-raise
        logger.warning("Explanation generation failed unexpectedly: %s", exc)
        return {}

    text = "".join(
        getattr(block, "text", "")
        for block in getattr(response, "content", []) or []
        if getattr(block, "type", None) == "text"
    )
    return _parse_response(text, expected_ids)


def _build_prompt(
    rules: list[RuleExplanationInput],
    *,
    beverage_type: str,
    container_size_ml: int,
    is_imported: bool,
    image_quality: str,
) -> str:
    """Assemble the user-message payload.

    The system prompt carries the "what to do" and is cached; the user
    prompt is the per-request specifics — beverage profile, image
    quality, and the per-rule finding/expected/field-values bundle.
    Kept as a single text blob (not a structured tool input) so Haiku's
    JSON-following behaviour is what we measure, not its tool-use
    behaviour, which is a separate failure mode.
    """
    import_label = "imported" if is_imported else "domestic"
    header = (
        f"Beverage: {beverage_type}\n"
        f"Container size: {container_size_ml} mL\n"
        f"Origin: {import_label}\n"
        f"Image quality: {image_quality}\n"
        f"\n"
        f"Failed rules ({len(rules)}):"
    )

    blocks: list[str] = [header]
    for r in rules:
        # Render the field values as a compact list, omitting None to
        # avoid feeding the model "alcohol_content: None" noise that
        # would tempt it to invent context.
        fields_lines = [
            f"  - {k}: {v}" for k, v in r.field_values.items() if v is not None
        ]
        fields_block = (
            "\n".join(fields_lines) if fields_lines else "  (no extracted fields)"
        )
        block = (
            f"\n---\n"
            f"rule_id: {r.rule_id}\n"
            f"status: {r.rule_status}\n"
            f"citation: {r.citation}\n"
            f"finding: {r.finding or '(none)'}\n"
            f"expected: {r.expected or '(none)'}\n"
            f"static fix-it text: {r.fix_suggestion or '(none)'}\n"
            f"extracted fields:\n{fields_block}"
        )
        blocks.append(block)

    blocks.append(
        "\n---\n"
        'Output ONLY a JSON object: {"<rule_id>": "<one sentence>", ...}. '
        "No markdown fences. No prose. One sentence per rule, ≤25 words, "
        "plain language, no jargon."
    )
    return "\n".join(blocks)


def _parse_response(raw: str, expected_rule_ids: set[str]) -> dict[str, str]:
    """Parse Claude's JSON response. Tolerant: extracts the JSON object
    from possibly-prefixed markdown, validates each entry is a string,
    only returns rule_ids in expected_rule_ids."""
    if not raw or not raw.strip():
        return {}

    # Strip Markdown code fences if Haiku wrapped the JSON despite the
    # system prompt forbidding it. Same defence pattern as
    # health_warning_second_pass._parse_response — Haiku follows the rule
    # ~95% of the time and we recover the other 5% rather than throwing.
    cleaned = re.sub(r"^\s*```(?:json)?", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(cleaned)
        if extracted is None:
            return {}
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError:
            return {}

    if not isinstance(data, dict):
        return {}

    out: dict[str, str] = {}
    for rule_id, value in data.items():
        if rule_id not in expected_rule_ids:
            # Hallucinated rule_id (model invented one we never asked
            # about). Drop silently — the response shape is otherwise
            # valid.
            continue
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        out[rule_id] = text
    return out


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced top-level JSON object in `text`, or None.

    String-literal aware so quoted braces don't confuse the depth counter.
    Mirrors the recovery used in health_warning_second_pass: explanation
    payloads are simple flat string-valued objects, but if Haiku wraps
    them in commentary we want to fish the JSON back out cheaply.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
