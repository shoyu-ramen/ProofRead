"""Tests for AI-generated per-rule explanations.

The explanation service is a cold-path enrichment that adds one
plain-language sentence per failed rule. It must be:

  * Quiet on the failure path — timeouts, API errors, malformed JSON,
    and a kill-switch in settings all produce an empty dict, not an
    exception.
  * Strict on the response surface — only return rule_ids the caller
    asked about (no model hallucinations leaking through), only return
    string values, drop empty strings.
  * Frugal with the API — a single batched call covers all failed rules
    in one round-trip; an empty input list never invokes the API at all.

These guarantees are what let the verify path bolt on the call
unconditionally without worrying about user-visible regressions.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import settings
from app.services.anthropic_client import ExtractorUnavailable
from app.services.explanation import (
    RuleExplanationInput,
    _build_prompt,
    _parse_response,
    explain_rules,
)


def _input(
    rule_id: str,
    *,
    status: str = "fail",
    citation: str = "27 CFR 7.22",
    finding: str | None = "label finding text",
    expected: str | None = "expected text",
    fix_suggestion: str | None = "static fix-it text",
    field_values: dict[str, str | None] | None = None,
) -> RuleExplanationInput:
    return RuleExplanationInput(
        rule_id=rule_id,
        rule_status=status,
        citation=citation,
        finding=finding,
        expected=expected,
        fix_suggestion=fix_suggestion,
        field_values=field_values or {"alcohol_content": "5.5% ABV"},
    )


def _scripted_response(payload: dict[str, str]) -> SimpleNamespace:
    """Mimic the SDK's response object: an object with `.content`, where
    each block has `.type == "text"` and `.text` carrying the JSON."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))]
    )


def _scripted_text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


# ---------------------------------------------------------------------------
# explain_rules — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_rules_list_returns_empty_dict_no_api_call():
    """Zero failed rules → never invoke the API. Saves the round-trip on
    the (common) all-pass label and prevents wasted spend."""

    with patch(
        "app.services.explanation.build_client"
    ) as build_mock, patch(
        "app.services.explanation.call_with_resilience"
    ) as call_mock:
        out = await explain_rules(
            [],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )
    assert out == {}
    assert build_mock.call_count == 0
    assert call_mock.call_count == 0


@pytest.mark.asyncio
async def test_single_failed_rule_explanation_returned():
    """Single-rule case still uses the JSON envelope so the response
    shape never bifurcates by batch size."""

    payload = {
        "beer.alcohol_content.format": (
            "Your label shows '5.5% ABV' but TTB requires '% alcohol by "
            "volume' or '% alc/vol'."
        )
    }
    response = _scripted_response(payload)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch(
        "app.services.explanation.call_with_resilience", return_value=response
    ) as call_mock:
        out = await explain_rules(
            [_input("beer.alcohol_content.format")],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )

    assert out == payload
    assert call_mock.call_count == 1


@pytest.mark.asyncio
async def test_three_failed_rules_explained_in_one_batch():
    """All three rules go in ONE call. The whole point of the batch is
    to amortise the API overhead — one call per rule would 3x the cost
    on a typical 'label fails on 3-5 checks' submission."""

    payload = {
        "beer.alcohol_content.format": "Sentence one.",
        "beer.health_warning.exact_text": "Sentence two.",
        "beer.brand_name.match": "Sentence three.",
    }
    response = _scripted_response(payload)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch(
        "app.services.explanation.call_with_resilience", return_value=response
    ) as call_mock:
        out = await explain_rules(
            [
                _input("beer.alcohol_content.format"),
                _input("beer.health_warning.exact_text"),
                _input("beer.brand_name.match"),
            ],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )

    assert out == payload
    assert call_mock.call_count == 1


# ---------------------------------------------------------------------------
# explain_rules — failure paths (must never raise)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_timeout_returns_empty_dict():
    """A slow Claude call (longer than the per-call timeout) must NOT
    propagate. The verify response has already been computed; we drop
    the explanations silently rather than 500ing on a working verdict."""

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))

    def _slow(*args, **kwargs):
        # Block long enough that asyncio.wait_for trips. The thread
        # call is what wait_for cancels — we just need it to not return
        # within the test's timeout budget.
        import time

        time.sleep(2.0)
        return _scripted_response({})

    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch("app.services.explanation.call_with_resilience", side_effect=_slow):
        out = await explain_rules(
            [_input("rule.x")],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
            timeout_s=0.05,
        )
    assert out == {}


@pytest.mark.asyncio
async def test_extractor_unavailable_returns_empty_dict():
    """The resilience wrapper translates transient SDK failures to
    ExtractorUnavailable. We must catch that here and drop quietly —
    explanation is additive UX."""

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch(
        "app.services.explanation.call_with_resilience",
        side_effect=ExtractorUnavailable("upstream 503"),
    ):
        out = await explain_rules(
            [_input("rule.x")],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )
    assert out == {}


@pytest.mark.asyncio
async def test_arbitrary_exception_returns_empty_dict():
    """Defensive: anything unexpected from the SDK or our wrapper must
    NOT leak. The whole point of the additive UX contract is that the
    verify path doesn't have to think about the explanation call."""

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch(
        "app.services.explanation.call_with_resilience",
        side_effect=RuntimeError("totally unexpected"),
    ):
        out = await explain_rules(
            [_input("rule.x")],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )
    assert out == {}


@pytest.mark.asyncio
async def test_malformed_json_response_returns_empty_dict():
    """Haiku occasionally goes off-script and returns prose instead of
    JSON. Better to drop everything than to surface a partial parse —
    the static fix_suggestion is still on the rule, so the user UI is
    not broken, only un-enhanced."""

    response = _scripted_text_response(
        "Sure! I think these rules failed because the label is messy."
    )
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch(
        "app.services.explanation.call_with_resilience", return_value=response
    ):
        out = await explain_rules(
            [_input("rule.x")],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )
    assert out == {}


@pytest.mark.asyncio
async def test_response_with_extra_rule_ids_keeps_only_requested_ones():
    """Hallucinated rule_ids (model invented one we didn't ask about)
    must be dropped, not surfaced. Otherwise the UI would render an
    explanation against a rule_id that doesn't exist in the rule_results
    list."""

    payload = {
        "beer.alcohol_content.format": "asked-for explanation",
        "beer.totally.invented": "model hallucination",
    }
    response = _scripted_response(payload)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch(
        "app.services.explanation.call_with_resilience", return_value=response
    ):
        out = await explain_rules(
            [_input("beer.alcohol_content.format")],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )

    assert out == {"beer.alcohol_content.format": "asked-for explanation"}
    assert "beer.totally.invented" not in out


@pytest.mark.asyncio
async def test_response_missing_some_requested_ids_returns_present_ones():
    """If Haiku skipped a rule we asked about, return what we got. The
    UI falls back to the static fix_suggestion for the missing ones —
    no need to hold the whole batch hostage to a partial response."""

    payload = {
        "beer.alcohol_content.format": "explanation A",
        # beer.health_warning.exact_text intentionally absent
    }
    response = _scripted_response(payload)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch(
        "app.services.explanation.call_with_resilience", return_value=response
    ):
        out = await explain_rules(
            [
                _input("beer.alcohol_content.format"),
                _input("beer.health_warning.exact_text"),
            ],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )

    assert out == {"beer.alcohol_content.format": "explanation A"}


@pytest.mark.asyncio
async def test_explanation_disabled_setting_short_circuits(monkeypatch):
    """Kill-switch test: setting the toggle to False must skip the API
    entirely. Lets ops disable explanations under cost/incident pressure
    without redeploying code."""

    monkeypatch.setattr(settings, "explanation_enabled", False)

    with patch(
        "app.services.explanation.build_client"
    ) as build_mock, patch(
        "app.services.explanation.call_with_resilience"
    ) as call_mock:
        out = await explain_rules(
            [_input("rule.x")],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )
    assert out == {}
    assert build_mock.call_count == 0
    assert call_mock.call_count == 0


@pytest.mark.asyncio
async def test_no_api_key_returns_empty_dict_quietly(monkeypatch):
    """No ANTHROPIC_API_KEY → build_client raises ExtractorUnavailable.
    The service must catch that at the construction step too, not only
    the call step. Ensures local dev / CI without a key still gets a
    working verify path with no explanations attached."""

    fake_client_err = ExtractorUnavailable("no api key")

    with patch(
        "app.services.explanation.build_client", side_effect=fake_client_err
    ), patch("app.services.explanation.call_with_resilience") as call_mock:
        out = await explain_rules(
            [_input("rule.x")],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )
    assert out == {}
    assert call_mock.call_count == 0


# ---------------------------------------------------------------------------
# Prompt assembly — ensure the contextual signals reach the model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_includes_beverage_type_citation_expected_field_values():
    """Spy on the actual call kwargs to confirm the prompt the model sees
    contains the load-bearing per-scan context. Regression bait: a
    refactor that drops one of these silently degrades explanation
    quality without breaking any behavioural assertion otherwise."""

    captured: dict = {}

    def _capture(_callable, **kwargs):
        captured.update(kwargs)
        return _scripted_response({"beer.alcohol_content.format": "."})

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    rule = _input(
        "beer.alcohol_content.format",
        citation="27 CFR 7.65",
        expected="must include '% alcohol by volume' or '% alc/vol'",
        finding="found '5.5% ABV', not a permitted form",
        field_values={
            "alcohol_content": "5.5% ABV",
            "container_size_ml": "355",
        },
    )
    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch(
        "app.services.explanation.call_with_resilience", side_effect=_capture
    ):
        await explain_rules(
            [rule],
            beverage_type="beer",
            container_size_ml=355,
            is_imported=False,
            image_quality="good",
        )

    # The user message is a plain text string in the messages list.
    [user_msg] = captured["messages"]
    user_text = user_msg["content"]
    assert "Beverage: beer" in user_text
    assert "Container size: 355" in user_text
    assert "domestic" in user_text  # is_imported=False
    assert "Image quality: good" in user_text
    # Per-rule context
    assert "27 CFR 7.65" in user_text
    assert "must include '% alcohol by volume'" in user_text
    assert "5.5% ABV" in user_text
    assert "beer.alcohol_content.format" in user_text

    # The model is the Haiku default (or whatever the override resolves to).
    assert captured["model"] == (
        settings.explanation_model or settings.anthropic_model
    )

    # System prompt is cached so the prefix is paid once per process.
    [system_block] = captured["system"]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert "JSON object" in system_block["text"]


@pytest.mark.asyncio
async def test_imported_flag_is_surfaced_in_prompt():
    """Symmetric coverage of the imported branch — the prompt must
    distinguish an imported label from a domestic one so the model can
    contextualise rules that key off COLA-vs-import status."""

    captured: dict = {}

    def _capture(_callable, **kwargs):
        captured.update(kwargs)
        return _scripted_response({"rule.x": "."})

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    with patch(
        "app.services.explanation.build_client", return_value=fake_client
    ), patch(
        "app.services.explanation.call_with_resilience", side_effect=_capture
    ):
        await explain_rules(
            [_input("rule.x")],
            beverage_type="wine",
            container_size_ml=750,
            is_imported=True,
            image_quality="degraded",
        )

    user_text = captured["messages"][0]["content"]
    assert "imported" in user_text
    assert "Image quality: degraded" in user_text


# ---------------------------------------------------------------------------
# _parse_response — direct unit tests of the tolerant parser
# ---------------------------------------------------------------------------


def test_parse_response_clean_json():
    raw = json.dumps({"rule.a": "explanation a"})
    out = _parse_response(raw, expected_rule_ids={"rule.a"})
    assert out == {"rule.a": "explanation a"}


def test_parse_response_strips_markdown_fences():
    raw = '```json\n{"rule.a": "explanation a"}\n```'
    out = _parse_response(raw, expected_rule_ids={"rule.a"})
    assert out == {"rule.a": "explanation a"}


def test_parse_response_recovers_json_with_trailing_prose():
    raw = (
        '{"rule.a": "explanation a"}\n\n'
        "Note: I picked the most actionable framing for this finding."
    )
    out = _parse_response(raw, expected_rule_ids={"rule.a"})
    assert out == {"rule.a": "explanation a"}


def test_parse_response_drops_non_string_values():
    """Defensive: if the model returns a number or null instead of a
    sentence, drop that entry rather than letting a non-string into the
    DTO."""
    raw = json.dumps({"rule.a": "ok", "rule.b": 42, "rule.c": None})
    out = _parse_response(raw, expected_rule_ids={"rule.a", "rule.b", "rule.c"})
    assert out == {"rule.a": "ok"}


def test_parse_response_drops_empty_strings():
    """Empty strings serve no UI purpose — drop them so the caller can
    fall back to the static fix_suggestion."""
    raw = json.dumps({"rule.a": "ok", "rule.b": "", "rule.c": "   "})
    out = _parse_response(raw, expected_rule_ids={"rule.a", "rule.b", "rule.c"})
    assert out == {"rule.a": "ok"}


def test_parse_response_handles_garbage_input():
    out = _parse_response("not json at all", expected_rule_ids={"rule.a"})
    assert out == {}


def test_parse_response_handles_empty_string():
    out = _parse_response("", expected_rule_ids={"rule.a"})
    assert out == {}


def test_parse_response_rejects_non_object_top_level():
    raw = json.dumps(["not", "an", "object"])
    out = _parse_response(raw, expected_rule_ids={"rule.a"})
    assert out == {}


# ---------------------------------------------------------------------------
# _build_prompt — direct sanity check on the user message structure
# ---------------------------------------------------------------------------


def test_build_prompt_lists_each_rule_block_with_separator():
    rules = [
        _input("rule.a", citation="cite A", finding="finding A"),
        _input("rule.b", citation="cite B", finding="finding B"),
    ]
    prompt = _build_prompt(
        rules,
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        image_quality="good",
    )
    assert prompt.count("rule_id:") == 2
    assert "rule.a" in prompt
    assert "rule.b" in prompt
    assert "cite A" in prompt
    assert "cite B" in prompt
    assert "finding A" in prompt
    assert "finding B" in prompt
    # Output instruction is appended at the bottom so the last thing the
    # model sees is the JSON shape contract.
    assert prompt.rfind("JSON object") > prompt.rfind("rule_id:")


def test_build_prompt_omits_none_field_values():
    """Don't feed the model 'alcohol_content: None' lines — they're
    pure noise and tempt the model to invent context. Drop None fields
    and emit a stub when nothing's left so the block is still valid."""
    prompt = _build_prompt(
        [
            _input(
                "rule.a",
                field_values={"alcohol_content": None, "brand_name": "Foo Ale"},
            )
        ],
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        image_quality="good",
    )
    assert "alcohol_content: None" not in prompt
    assert "brand_name: Foo Ale" in prompt
