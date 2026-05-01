"""TTB COLA Registry public-search adapter.

The federal Alcohol and Tobacco Tax and Trade Bureau (TTB) publishes
every approved Certificate of Label Approval (COLA) at
https://www.ttbonline.gov/colasonline/publicSearchColasBasic.do . A
hit here gives a /v1/verify caller massive value — "your label
matches approved COLA #20-42-001 from 2023-04-15" — for free, since
the catalog is already public.

Three things this module owns end-to-end:

  1. **Form discovery.** TTB's basic search is a classic server-rendered
     HTML form with a CSRF-style hidden field set we have to mirror on
     POST. We fetch the form once per adapter instance and cache the
     hidden inputs alongside the action URL; subsequent searches reuse
     them so we pay the GET cost once.

  2. **Polite rate limiting.** The TTB site is small and government-
     hosted; even a modest concurrency burst from us could earn an IP
     block. An `asyncio.Semaphore(2)` plus a 600 ms inter-request
     floor keeps us well under any reasonable rate budget.

  3. **Defensive parsing.** The HTML structure is keyed off column
     header strings (TTB has historically reordered columns without
     warning). We map header text to indices on each parse pass rather
     than relying on positions; missing-column → skip the row, which is
     the right failure mode for a heuristic enrichment layer.

The adapter is registered into the global registry only at module
import time when `settings.ttb_cola_lookup_enabled` is true, so a
stale `from app.services.external import ttb_cola` import in a deploy
without the flag set does not silently route traffic.

Failure model — same as `anthropic_client.call_with_resilience`: every
transient failure (timeout, network, 5xx, parse) returns None and logs
a warning. The adapter NEVER raises into the verify orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import UTC, date, datetime
from typing import Any

import httpx

from app.config import settings
from app.services.external.adapter import (
    ExternalLookupAdapter,
    register_adapter,
)
from app.services.external.types import ExternalMatch

logger = logging.getLogger(__name__)


# Endpoint constants. Resolved against the configured base URL so a
# test environment can swap the host without touching the adapter
# code; production points at ttbonline.gov directly.
_TTB_BASE_URL = "https://www.ttbonline.gov"
_SEARCH_PATH = "/colasonline/publicSearchColasBasic.do"
_DETAIL_PATH = "/colasonline/viewColaDetails.do"


# Beverage-type → TTB product class code. The codes here match the
# "Class/Type Code" the TTB form sends in the
# `searchCriteria.productClassTypeCodeStr` parameter. We keep the map
# narrow (the v1 verify path only handles beer + spirits); future types
# should slot in here rather than at the call site.
_BEVERAGE_TYPE_TO_CLASS_CODE: dict[str, str] = {
    "beer": "001",  # Malt Beverage (most beer products)
    "spirits": "041",  # Distilled Spirits (the umbrella class)
}


# Min wall-clock between successive HTTP calls from this adapter. The
# TTB public site does not publish a rate limit but is small enough
# that a tight loop could trip protective measures. 600 ms (≈ 1.6 RPS)
# is the polite floor we ship.
_MIN_REQUEST_INTERVAL_S = 0.6


# Token-bucket concurrency. Two simultaneous in-flight calls is enough
# for a single /v1/verify request that concurrently looks up brand
# alone and brand+fanciful, but low enough that we don't burst across
# many concurrent verify requests.
_CONCURRENCY_LIMIT = 2


# Confidence floor — matches below this score never come back. The
# verify orchestrator may apply its own ceiling; this is the adapter's
# self-honesty filter so we don't surface "fuzzy guess" hits to users.
_CONFIDENCE_FLOOR = 0.4


# Heuristic threshold for "the TTB site is down right now" — tripped
# when this many consecutive HTTP/parse failures have happened with no
# intervening success. Surfaced via the admin endpoint as
# ``circuit_open`` so the on-call dashboard sees the state at a
# glance. The adapter itself does NOT short-circuit when the flag is
# set — a per-call attempt is cheap (single HTTP round-trip with a
# 4 s timeout), and a wedged adapter that refuses to retry would
# delay recovery once TTB came back.
_CIRCUIT_OPEN_AFTER_ERRORS = 3


# Confidence heuristics. Each tier is documented at the call site
# (`_score_match`); the constants live here so a tuning change is one
# value swap. The numbers are calibrated against the spec's documented
# tiers (exact brand → 0.9, exact brand + fanciful → 0.95, prefix →
# 0.6, fuzzy → 0.4).
_SCORE_EXACT_BRAND = 0.9
_SCORE_EXACT_BRAND_AND_FANCIFUL = 0.95
_SCORE_PREFIX_BRAND = 0.6
_SCORE_FUZZY_BRAND = 0.4


# Regex used to normalize whitespace in HTML cell text before
# comparison. The TTB site occasionally emits a single &nbsp; or
# tab inside cell text; collapsing to a single space keeps comparison
# stable.
_WS_RE = re.compile(r"\s+")


def _normalize_text(s: str | None) -> str:
    """Collapse whitespace + strip; used for both header and cell text."""
    if s is None:
        return ""
    return _WS_RE.sub(" ", s).strip()


class TTBColaAdapter(ExternalLookupAdapter):
    """Public-search adapter for the TTB COLA Registry.

    Constructor accepts an optional `client` so tests can inject a
    pre-built `httpx.AsyncClient` wired to a `MockTransport`; the
    production path lazily creates a client per `lookup` call so the
    adapter has no long-lived sockets to manage across application
    lifecycle events.
    """

    name = "ttb_cola"

    def __init__(
        self,
        *,
        base_url: str = _TTB_BASE_URL,
        timeout_s: float | None = None,
        user_agent: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = (
            timeout_s if timeout_s is not None else settings.ttb_cola_timeout_s
        )
        self._user_agent = user_agent or settings.ttb_cola_user_agent
        self._client = client
        # Form metadata cache: filled on first successful GET so we
        # only pay the discovery cost once per adapter instance.
        self._form_state: dict[str, str] | None = None
        self._form_action: str | None = None
        # Politeness primitives. The semaphore bounds concurrency
        # within this adapter instance; the lock + last-call timestamp
        # together implement the per-instance min-delay floor.
        self._semaphore = asyncio.Semaphore(_CONCURRENCY_LIMIT)
        self._rate_lock = asyncio.Lock()
        self._last_request_at: float = 0.0
        # Observability counters surfaced via the admin cache-health
        # endpoint. ``_last_request_wall`` is wall-clock (datetime), kept
        # alongside ``_last_request_at`` (monotonic, used by the rate
        # limiter) so the dashboard sees a human-readable timestamp
        # without breaking the rate-limiter's comparison contract. The
        # circuit-open hint trips after `_CIRCUIT_OPEN_AFTER_ERRORS`
        # consecutive failures and resets on the next success — gives
        # operators a fast eyeball on "the TTB site is down right now"
        # without imposing real flow control on the adapter (failures
        # already fail-open per-call).
        self._request_count = 0
        self._error_count = 0
        self._last_request_wall: datetime | None = None
        self._consecutive_errors = 0

    async def lookup(
        self,
        *,
        brand: str | None,
        beverage_type: str,
        fanciful_name: str | None = None,
        timeout_s: float = 4.0,
    ) -> ExternalMatch | None:
        """Search the TTB COLA Registry for the best match.

        Returns None on any of:
          * no brand to search by
          * unsupported beverage type
          * empty result set
          * transient HTTP/parse failure
          * best-match confidence below `_CONFIDENCE_FLOOR`
        """
        if not brand or not brand.strip():
            return None
        normalized_brand = brand.strip()

        class_code = _BEVERAGE_TYPE_TO_CLASS_CODE.get(beverage_type)
        if class_code is None:
            logger.debug(
                "ttb_cola: skipping unsupported beverage_type=%r", beverage_type
            )
            return None

        # Honor the per-call timeout if it's tighter than our default;
        # operators may set a global budget but a particular call site
        # (e.g. a degraded /v1/verify path) may want to compress
        # further.
        effective_timeout = min(self._timeout_s, timeout_s)

        try:
            client_owned, client = self._acquire_client(effective_timeout)
        except RuntimeError as exc:
            # Optional dep missing or env misconfigured — the adapter
            # cannot operate, but the verify path still must.
            logger.warning("ttb_cola: cannot build client: %s", exc)
            return None

        try:
            async with self._semaphore:
                await self._respect_rate_limit()
                self._record_request_start()
                rows = await self._search(
                    client=client,
                    brand=normalized_brand,
                    fanciful_name=fanciful_name,
                    class_code=class_code,
                )
                self._record_request_success()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            # httpx.HTTPError is the parent of TimeoutException + most
            # network errors; we keep TimeoutException explicit for
            # readability and so the warning message is greppable.
            self._record_request_failure()
            logger.warning("ttb_cola: HTTP failure: %s", exc)
            return None
        except (ValueError, KeyError, IndexError, AttributeError) as exc:
            # Defensive: any HTML schema drift bubbles up as one of
            # these. A None return is the right failure mode for a
            # heuristic enrichment layer.
            self._record_request_failure()
            logger.warning("ttb_cola: parse failure: %s", exc)
            return None
        finally:
            if client_owned:
                await client.aclose()

        if not rows:
            return None

        # Score every parsed row, drop sub-threshold ones, and pick
        # the top match. Ties are broken by approval_date (newer wins)
        # — labels are reissued and the freshest is the most likely
        # to be currently sold.
        best: tuple[float, ExternalMatch] | None = None
        for row in rows:
            score = _score_match(
                target_brand=normalized_brand,
                target_fanciful=fanciful_name,
                row=row,
            )
            if score < _CONFIDENCE_FLOOR:
                continue
            match = _row_to_match(
                row=row, confidence=score, base_url=self._base_url
            )
            if best is None or score > best[0]:
                best = (score, match)
            elif score == best[0]:
                # Tie: prefer newer approval_date (None sorts oldest).
                cur_date = best[1].approval_date or date.min
                new_date = match.approval_date or date.min
                if new_date > cur_date:
                    best = (score, match)
        return best[1] if best is not None else None

    def _acquire_client(
        self, timeout_s: float
    ) -> tuple[bool, httpx.AsyncClient]:
        """Return (owned_by_us, client). Owner closes the client."""
        if self._client is not None:
            return False, self._client
        client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout_s,
            headers={"User-Agent": self._user_agent},
            # Don't follow redirects automatically — the TTB site uses
            # a 302 to an error page when a search rejects parameters,
            # and we want to surface that cleanly rather than parse the
            # error page as if it were results.
            follow_redirects=True,
        )
        return True, client

    async def _respect_rate_limit(self) -> None:
        """Sleep just enough to honor `_MIN_REQUEST_INTERVAL_S`.

        Held under `_rate_lock` so concurrent callers in the same
        adapter instance see a serialized "last request" timestamp.
        The sleep happens *inside* the lock so the next caller already
        sees the post-sleep timestamp when it runs.
        """
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_at
            if elapsed < _MIN_REQUEST_INTERVAL_S:
                await asyncio.sleep(_MIN_REQUEST_INTERVAL_S - elapsed)
            self._last_request_at = time.monotonic()

    def _record_request_start(self) -> None:
        """Note that we are about to dispatch an HTTP request.

        Bumps the lifetime request counter and stamps the wall-clock
        timestamp surfaced via the admin endpoint. Called inside the
        rate-limit + semaphore so concurrency is bounded already; a
        plain attribute write under the GIL is sufficient.
        """
        self._request_count += 1
        self._last_request_wall = datetime.now(UTC)

    def _record_request_success(self) -> None:
        """Reset the consecutive-error counter on a clean response."""
        self._consecutive_errors = 0

    def _record_request_failure(self) -> None:
        """Bump the lifetime + consecutive-error counters."""
        self._error_count += 1
        self._consecutive_errors += 1

    def stats(self) -> dict[str, Any]:
        """Snapshot of adapter state for admin observability.

        Returns ``{"enabled", "last_request_at", "request_count",
        "error_count", "circuit_open"}`` — exactly the shape the admin
        cache-health endpoint surfaces. ``last_request_at`` is ISO-8601
        UTC (or ``None`` before the first request); ``circuit_open`` is
        a heuristic on consecutive failures, NOT a hard breaker (the
        adapter still attempts the next call so a recovered upstream is
        picked up immediately).
        """
        return {
            "enabled": settings.ttb_cola_lookup_enabled,
            "last_request_at": (
                self._last_request_wall.isoformat()
                if self._last_request_wall is not None
                else None
            ),
            "request_count": self._request_count,
            "error_count": self._error_count,
            "circuit_open": self._consecutive_errors
            >= _CIRCUIT_OPEN_AFTER_ERRORS,
        }

    async def _ensure_form_state(self, client: httpx.AsyncClient) -> None:
        """Fetch + cache the search form's hidden fields and action.

        TTB's basic-search form ships hidden inputs (typically struts
        token + flow id) that the server validates on POST. We mirror
        whatever the form ships — defensively, by parsing the GET HTML
        and lifting every <input type="hidden"> into our payload base.
        """
        if self._form_state is not None and self._form_action is not None:
            return
        response = await client.get(_SEARCH_PATH)
        response.raise_for_status()
        action, hidden = _parse_form_metadata(response.text)
        # Cache only on success so a transient blip doesn't poison
        # the adapter for the rest of the process lifetime.
        self._form_state = hidden
        self._form_action = action

    async def _search(
        self,
        *,
        client: httpx.AsyncClient,
        brand: str,
        fanciful_name: str | None,
        class_code: str,
    ) -> list[dict[str, str]]:
        """Issue the POST and return raw row dicts (header → text).

        Returns an empty list on no-results; raises on HTTP / parse
        errors so the outer `lookup` can normalize them.
        """
        await self._ensure_form_state(client)
        action = self._form_action or _SEARCH_PATH
        # Start with whatever hidden fields the live form requires,
        # then layer our search criteria on top. The field names mirror
        # the form's actual `name` attributes so a CSRF token or flow
        # id passes through unchanged.
        payload: dict[str, str] = dict(self._form_state or {})
        payload["searchCriteria.searchPrintFlag"] = "S"
        payload["searchCriteria.brandName"] = brand
        if fanciful_name:
            payload["searchCriteria.fancifulName"] = fanciful_name
        payload["searchCriteria.productClassTypeCodeStr"] = class_code

        response = await client.post(action, data=payload)
        response.raise_for_status()
        return _parse_results_table(response.text)


def _parse_form_metadata(html: str) -> tuple[str, dict[str, str]]:
    """Parse the basic-search form's action URL + hidden inputs.

    Returns (action_url, hidden_fields). Hidden fields with no `name`
    attribute are skipped; `value` defaults to empty string when
    absent. We deliberately do not raise on a missing form: a
    no-form page indicates a TTB outage or a layout change, and the
    caller's `httpx.HTTPError`/`ValueError` catch will see the
    eventual empty payload as a graceful no-results.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    # Heuristic: the basic-search form has a name attribute we look
    # for; if missing, fall back to the first <form> on the page.
    form = soup.find("form", attrs={"name": "searchForm"})
    if form is None:
        form = soup.find("form")
    if form is None:
        raise ValueError("ttb_cola: search form not found in page HTML")

    action = form.get("action") or _SEARCH_PATH
    if isinstance(action, list):
        action = action[0] if action else _SEARCH_PATH

    hidden: dict[str, str] = {}
    for inp in form.find_all("input", attrs={"type": "hidden"}):
        name = inp.get("name")
        if not name:
            continue
        value = inp.get("value", "")
        hidden[str(name)] = str(value)
    return str(action), hidden


def _parse_results_table(html: str) -> list[dict[str, str]]:
    """Extract result rows as a list of header→text dicts.

    The TTB results table has been reordered historically; binding to
    column-header text rather than positional index keeps us stable
    across those reorders. Returns an empty list when:
      * no <table> with the expected header set is present (no-results)
      * the table is structurally broken in ways we can't recover from
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    # The results table is the first one that contains "Brand Name" and
    # "TTB ID" headers. Iterate every <table> rather than relying on a
    # cosmetic id/class which has changed in past site refreshes.
    target_table = None
    target_headers: list[str] = []
    for table in soup.find_all("table"):
        headers_row = table.find("tr")
        if headers_row is None:
            continue
        cells = headers_row.find_all(["th", "td"])
        if not cells:
            continue
        text_cells = [_normalize_text(c.get_text()) for c in cells]
        lower = {c.lower() for c in text_cells}
        if "ttb id" in lower and "brand name" in lower:
            target_table = table
            target_headers = text_cells
            break
    if target_table is None:
        return []

    # Map header label → column index for value lookup. Lowercased so
    # case drift in the markup doesn't break the binding.
    header_index: dict[str, int] = {
        h.lower(): i for i, h in enumerate(target_headers)
    }

    rows: list[dict[str, str]] = []
    body_rows = target_table.find_all("tr")[1:]  # Skip header row.
    for tr in body_rows:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        # Lift the per-row anchor href (if any) so we can build a
        # detail URL — the link sits on the TTB ID cell typically.
        detail_href: str | None = None
        for cell in cells:
            anchor = cell.find("a", href=True)
            if anchor is not None:
                detail_href = str(anchor["href"])
                break
        row_dict: dict[str, str] = {}
        for header, idx in header_index.items():
            if idx >= len(cells):
                continue
            row_dict[header] = _normalize_text(cells[idx].get_text())
        if detail_href is not None:
            row_dict["__detail_href"] = detail_href
        # Skip header echoes / completely-empty rows the parser can
        # latch onto when a table is malformed.
        if not row_dict.get("ttb id") and not row_dict.get("brand name"):
            continue
        rows.append(row_dict)
    return rows


def _score_match(
    *,
    target_brand: str,
    target_fanciful: str | None,
    row: dict[str, str],
) -> float:
    """Heuristic confidence for a single search-result row.

    Tiers:
      * exact brand + exact fanciful (both case-insensitive)  → 0.95
      * exact brand                                           → 0.90
      * brand prefix (target startswith row, or vice versa)   → 0.60
      * fuzzy (substring or fold-equal-after-collapse)        → 0.40
      * none of the above                                     → 0.0
    """
    row_brand = row.get("brand name", "").strip()
    if not row_brand:
        return 0.0
    target_lower = target_brand.casefold()
    row_lower = row_brand.casefold()

    if row_lower == target_lower:
        if target_fanciful:
            row_fanciful = row.get("fanciful name", "").strip()
            if (
                row_fanciful
                and row_fanciful.casefold() == target_fanciful.strip().casefold()
            ):
                return _SCORE_EXACT_BRAND_AND_FANCIFUL
        return _SCORE_EXACT_BRAND

    # Prefix match — symmetric, so a label brand "ANYTOWN" hits a TTB
    # row "ANYTOWN ALE" and vice versa.
    if row_lower.startswith(target_lower) or target_lower.startswith(row_lower):
        return _SCORE_PREFIX_BRAND

    # Fuzzy fallback: substring containment after whitespace collapse.
    # Avoids the rapidfuzz dep here — a label-extraction brand close
    # enough to substring-match is usually a layout artifact (extra
    # word, ™ symbol) rather than a different product.
    if target_lower in row_lower or row_lower in target_lower:
        return _SCORE_FUZZY_BRAND

    return 0.0


def _row_to_match(
    *,
    row: dict[str, str],
    confidence: float,
    base_url: str,
) -> ExternalMatch:
    """Convert a parsed row + score into an `ExternalMatch`."""
    ttb_id = row.get("ttb id", "").strip()
    brand = row.get("brand name", "").strip() or None
    fanciful = row.get("fanciful name", "").strip() or None
    class_type = row.get("class/type", "").strip() or row.get("class", "").strip() or None
    approval_date = _parse_date(row.get("approval date"))

    # The detail link is sometimes relative ("/colasonline/viewColaDetails.do?...")
    # and sometimes a fragment-only anchor; build an absolute URL where
    # we can, otherwise leave None.
    detail_href = row.get("__detail_href")
    source_url: str | None = None
    if detail_href:
        if detail_href.startswith("http://") or detail_href.startswith("https://"):
            source_url = detail_href
        elif detail_href.startswith("/"):
            source_url = f"{base_url}{detail_href}"
        else:
            source_url = f"{base_url}/colasonline/{detail_href}"
    # If we have no anchor at all but DO have a TTB ID, fall back to a
    # constructed canonical detail URL so the consumer always has a
    # clickable target.
    if source_url is None and ttb_id:
        source_url = f"{base_url}{_DETAIL_PATH}?action=publicFormDisplay&ttbid={ttb_id}"

    return ExternalMatch(
        source="ttb_cola",
        source_id=ttb_id,
        brand=brand,
        fanciful_name=fanciful,
        class_type=class_type,
        approval_date=approval_date,
        # The basic search response does not embed label image URLs;
        # callers who need the image follow source_url.
        label_image_url=None,
        confidence=confidence,
        source_url=source_url,
    )


def _parse_date(raw: Any | None) -> date | None:
    """Parse the TTB-formatted approval date.

    The site renders dates as "MM/DD/YYYY". We try that first, then
    ISO-8601 as a fallback (in case TTB ever rationalizes the layout
    or a future caller injects a parsed value already). None on any
    non-string or unparseable input — date is not load-bearing for
    the match itself.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _maybe_register_default_adapter() -> None:
    """Register a TTBColaAdapter into the global registry if enabled.

    Gating on `settings.ttb_cola_lookup_enabled` keeps the verify
    orchestrator's default behaviour unchanged for operators who have
    not explicitly opted in. Tests that want the adapter wired call
    `register_adapter(TTBColaAdapter(...))` directly.
    """
    if settings.ttb_cola_lookup_enabled:
        register_adapter(TTBColaAdapter())


_maybe_register_default_adapter()
