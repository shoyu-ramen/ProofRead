"""Unit tests for the TTB COLA Registry adapter.

These tests pin the adapter's contract:

  * Empty-result HTML → None
  * One result, exact brand match → confidence ≥ 0.9
  * Multiple results → returns highest confidence
  * 5xx response → None (no exception escapes)
  * Timeout → None
  * Confidence below 0.4 floor → None
  * `ExternalMatch.to_dict()` ↔ `ExternalMatch.from_dict()` round-trip
  * Adapter registry: register / get / list

`respx` is not in the project's dev deps, so we use `httpx.MockTransport`
the way the verify-latency probe tests already do. That keeps test
isolation hermetic — no live network, no respx dep — while exercising
the adapter's actual httpx call path end-to-end.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from app.services.external import (
    ExternalLookupAdapter,
    ExternalMatch,
    get_adapter,
    list_adapters,
    register_adapter,
)
from app.services.external.adapter import _clear_registry_for_tests
from app.services.external.ttb_cola import TTBColaAdapter

# ---------------------------------------------------------------------------
# Test HTML fixtures
# ---------------------------------------------------------------------------


# Minimal search-form GET response that mirrors the live page's
# structurally-relevant pieces: a <form name="searchForm"> with a hidden
# input the adapter must echo back on POST. We deliberately keep the
# rest of the page bare so the parser is exercised on edge-case inputs.
_FORM_HTML = """
<html>
  <body>
    <form name="searchForm" action="/colasonline/publicSearchColasBasic.do" method="post">
      <input type="hidden" name="formId" value="basic-search" />
      <input type="hidden" name="csrfToken" value="abc123" />
      <input type="text" name="searchCriteria.brandName" />
    </form>
  </body>
</html>
"""


def _results_html(rows: list[dict[str, str]]) -> str:
    """Build a minimal results-table HTML page from `rows`.

    Each row dict maps header label → cell text. The first row in the
    table is the header row; subsequent rows are the data. Each TTB
    ID cell wraps the value in an <a href> so the adapter can pick up
    a detail link.
    """
    headers = ["TTB ID", "Brand Name", "Fanciful Name", "Class/Type", "Approval Date"]
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for r in rows:
        ttb_id = r.get("TTB ID", "")
        detail = (
            f"/colasonline/viewColaDetails.do"
            f"?action=publicFormDisplay&ttbid={ttb_id}"
        )
        cells = [
            f'<td><a href="{detail}">{ttb_id}</a></td>',
            f'<td>{r.get("Brand Name", "")}</td>',
            f'<td>{r.get("Fanciful Name", "")}</td>',
            f'<td>{r.get("Class/Type", "")}</td>',
            f'<td>{r.get("Approval Date", "")}</td>',
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
    <html>
      <body>
        <table>
          <tr>{header_html}</tr>
          {''.join(body)}
        </table>
      </body>
    </html>
    """


_NO_RESULTS_HTML = """
<html>
  <body>
    <p>No records found matching your criteria.</p>
  </body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_adapter(handler) -> TTBColaAdapter:
    """Return a TTBColaAdapter wired to a MockTransport `handler`.

    Each test injects its own handler so the test reads top-down: HTML
    fixtures + handler at the top, assertions at the bottom.
    """
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        base_url="https://www.ttbonline.gov",
        transport=transport,
        headers={"User-Agent": "test"},
    )
    return TTBColaAdapter(client=client)


def _form_then_results(results_html: str):
    """Build a 2-call handler: GET → form HTML, POST → results HTML."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_FORM_HTML)
        return httpx.Response(200, text=results_html)

    return handler


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test starts with an empty adapter registry.

    The TTB module's `_maybe_register_default_adapter` only registers
    when settings.ttb_cola_lookup_enabled is True (default False), so
    importing the module is safe — but other tests may register/replace
    entries, and we want a clean slate per test.
    """
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


# ---------------------------------------------------------------------------
# Lookup behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_results_returns_none():
    """An HTML page with no result table is treated as no-match."""
    adapter = _build_adapter(_form_then_results(_NO_RESULTS_HTML))
    result = await adapter.lookup(brand="UnknownBrand", beverage_type="beer")
    assert result is None


@pytest.mark.asyncio
async def test_exact_brand_match_yields_high_confidence():
    """Exact case-insensitive brand match → confidence ≥ 0.9."""
    rows = [
        {
            "TTB ID": "20-42-001",
            "Brand Name": "ANYTOWN ALE",
            "Fanciful Name": "PUMPKIN ALE",
            "Class/Type": "MALT BEVERAGE",
            "Approval Date": "04/15/2023",
        }
    ]
    adapter = _build_adapter(_form_then_results(_results_html(rows)))
    result = await adapter.lookup(brand="anytown ale", beverage_type="beer")

    assert result is not None
    assert result.source == "ttb_cola"
    assert result.source_id == "20-42-001"
    assert result.brand == "ANYTOWN ALE"
    assert result.fanciful_name == "PUMPKIN ALE"
    assert result.class_type == "MALT BEVERAGE"
    assert result.approval_date == date(2023, 4, 15)
    assert result.confidence >= 0.9
    assert result.source_url is not None
    assert "20-42-001" in result.source_url


@pytest.mark.asyncio
async def test_exact_brand_and_fanciful_yields_higher_confidence():
    """Brand + fanciful exact match should outscore brand-only match."""
    rows = [
        {
            "TTB ID": "20-42-002",
            "Brand Name": "ANYTOWN ALE",
            "Fanciful Name": "PUMPKIN ALE",
            "Class/Type": "MALT BEVERAGE",
            "Approval Date": "04/15/2023",
        }
    ]
    adapter = _build_adapter(_form_then_results(_results_html(rows)))
    result = await adapter.lookup(
        brand="ANYTOWN ALE",
        beverage_type="beer",
        fanciful_name="pumpkin ale",
    )
    assert result is not None
    assert result.confidence >= 0.95


@pytest.mark.asyncio
async def test_multiple_results_returns_highest_confidence():
    """Adapter must surface the best-scoring row even when it isn't first."""
    rows = [
        # Prefix-only match — confidence 0.6
        {
            "TTB ID": "20-42-100",
            "Brand Name": "ANYTOWN",
            "Fanciful Name": "",
            "Class/Type": "MALT BEVERAGE",
            "Approval Date": "01/05/2020",
        },
        # Exact match — confidence 0.9
        {
            "TTB ID": "20-42-200",
            "Brand Name": "ANYTOWN ALE",
            "Fanciful Name": "",
            "Class/Type": "MALT BEVERAGE",
            "Approval Date": "06/01/2022",
        },
    ]
    adapter = _build_adapter(_form_then_results(_results_html(rows)))
    result = await adapter.lookup(brand="ANYTOWN ALE", beverage_type="beer")
    assert result is not None
    assert result.source_id == "20-42-200"
    assert result.confidence >= 0.9


@pytest.mark.asyncio
async def test_5xx_response_returns_none():
    """A server error must NOT propagate as an exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_FORM_HTML)
        return httpx.Response(503, text="Service Unavailable")

    adapter = _build_adapter(handler)
    result = await adapter.lookup(brand="ANYTOWN ALE", beverage_type="beer")
    assert result is None


@pytest.mark.asyncio
async def test_timeout_returns_none():
    """An httpx timeout must be caught and reported as no-match."""

    def handler(request: httpx.Request) -> httpx.Response:
        # MockTransport surfaces a raised exception as the call result;
        # `TimeoutException` derives from `httpx.HTTPError` so the
        # adapter's catch-all suppresses it cleanly.
        raise httpx.ConnectTimeout("simulated timeout")

    adapter = _build_adapter(handler)
    result = await adapter.lookup(brand="ANYTOWN ALE", beverage_type="beer")
    assert result is None


@pytest.mark.asyncio
async def test_below_confidence_floor_returns_none():
    """A row that only matches under the 0.4 floor must not surface."""
    rows = [
        # Brand "OTHER BREWERY" against target "ANYTOWN ALE" — no
        # substring overlap, no prefix overlap → score 0.0, below floor.
        {
            "TTB ID": "20-99-999",
            "Brand Name": "OTHER BREWERY",
            "Fanciful Name": "",
            "Class/Type": "MALT BEVERAGE",
            "Approval Date": "01/01/2020",
        }
    ]
    adapter = _build_adapter(_form_then_results(_results_html(rows)))
    result = await adapter.lookup(brand="ANYTOWN ALE", beverage_type="beer")
    assert result is None


@pytest.mark.asyncio
async def test_no_brand_short_circuits_without_http():
    """No brand → no HTTP call at all (we have nothing to search)."""
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(200, text=_FORM_HTML)

    adapter = _build_adapter(handler)
    assert await adapter.lookup(brand=None, beverage_type="beer") is None
    assert await adapter.lookup(brand="", beverage_type="beer") is None
    assert await adapter.lookup(brand="   ", beverage_type="beer") is None
    assert state["calls"] == 0


@pytest.mark.asyncio
async def test_unsupported_beverage_type_returns_none_without_http():
    """Beverage type not in the class-code map → no HTTP, no match."""
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(200, text=_FORM_HTML)

    adapter = _build_adapter(handler)
    result = await adapter.lookup(brand="WHATEVER", beverage_type="cider")
    assert result is None
    assert state["calls"] == 0


@pytest.mark.asyncio
async def test_class_code_routing_for_beer_vs_spirits():
    """Beer → class 001, spirits → class 041 in the form payload."""
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_FORM_HTML)
        body = request.content.decode("utf-8")
        captured.append(body)
        return httpx.Response(200, text=_NO_RESULTS_HTML)

    adapter = _build_adapter(handler)
    await adapter.lookup(brand="X", beverage_type="beer")
    await adapter.lookup(brand="X", beverage_type="spirits")
    assert any("productClassTypeCodeStr=001" in b for b in captured)
    assert any("productClassTypeCodeStr=041" in b for b in captured)


# ---------------------------------------------------------------------------
# ExternalMatch round-trip
# ---------------------------------------------------------------------------


def test_external_match_to_dict_from_dict_roundtrip():
    """Lossless serialization of every populated field."""
    original = ExternalMatch(
        source="ttb_cola",
        source_id="20-42-001",
        brand="ANYTOWN ALE",
        fanciful_name="PUMPKIN ALE",
        class_type="MALT BEVERAGE",
        approval_date=date(2023, 4, 15),
        label_image_url="https://example.com/label.jpg",
        confidence=0.95,
        source_url="https://www.ttbonline.gov/colasonline/viewColaDetails.do",
    )
    payload = original.to_dict()
    # JSON-safe payload check: dates must be strings, not date objects.
    assert payload["approval_date"] == "2023-04-15"
    assert isinstance(payload["confidence"], float)
    restored = ExternalMatch.from_dict(payload)
    assert restored == original


def test_external_match_roundtrip_with_nulls():
    """Nullable fields round-trip as None without losing identity keys."""
    original = ExternalMatch(
        source="ttb_cola",
        source_id="20-42-002",
        brand=None,
        fanciful_name=None,
        class_type=None,
        approval_date=None,
        label_image_url=None,
        confidence=0.4,
        source_url=None,
    )
    restored = ExternalMatch.from_dict(original.to_dict())
    assert restored == original


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


class _StubAdapter(ExternalLookupAdapter):
    name = "stub"

    async def lookup(
        self,
        *,
        brand: str | None,
        beverage_type: str,
        fanciful_name: str | None = None,
        timeout_s: float = 4.0,
    ) -> ExternalMatch | None:
        return None


def test_register_and_get_adapter():
    """register_adapter + get_adapter round-trips a single instance."""
    adapter = _StubAdapter()
    register_adapter(adapter)
    assert get_adapter("stub") is adapter
    assert get_adapter("nonexistent") is None


def test_list_adapters_preserves_insertion_order():
    """list_adapters yields adapters in the order they were registered."""

    class _A(_StubAdapter):
        name = "a"

    class _B(_StubAdapter):
        name = "b"

    class _C(_StubAdapter):
        name = "c"

    a, b, c = _A(), _B(), _C()
    register_adapter(a)
    register_adapter(b)
    register_adapter(c)
    names = [adapter.name for adapter in list_adapters()]
    assert names == ["a", "b", "c"]


def test_register_adapter_replaces_same_name():
    """Re-registering under the same name swaps the instance in place."""
    first = _StubAdapter()
    second = _StubAdapter()
    register_adapter(first)
    register_adapter(second)
    assert get_adapter("stub") is second
    assert len(list_adapters()) == 1


def test_register_adapter_rejects_empty_name():
    """An adapter with no `name` is a configuration bug — refuse it."""

    class _Nameless(_StubAdapter):
        name = ""

    with pytest.raises(ValueError):
        register_adapter(_Nameless())


def test_ttb_adapter_has_canonical_name():
    """The shipped adapter must use the documented `ttb_cola` identifier."""
    assert TTBColaAdapter.name == "ttb_cola"
