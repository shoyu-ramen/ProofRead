"""Adapter base class + module-level registry.

Three goals:

  1. **Uniform call shape.** Every external source — TTB COLA, OFF,
     Untappd, etc. — accepts the same `(brand, beverage_type,
     fanciful_name, timeout_s)` and returns at most one
     `ExternalMatch`. The verify orchestrator can iterate the registry
     and merge results without per-source branching.

  2. **Self-registration.** New adapters call `register_adapter` at
     module import time. The orchestrator looks them up by `name`
     instead of importing concrete classes, so the verify path stays
     decoupled from the adapter implementation set.

  3. **Test isolation.** The registry is a module-level dict, but
     tests can clear it (or temporarily replace entries) via the public
     surface — no monkeypatching of internals required.

Adapters MUST NOT raise on transient failures (network, timeout,
parsing). They return `None` and log a warning instead, so a
single-source outage does not break /v1/verify for every other source.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.services.external.types import ExternalMatch

logger = logging.getLogger(__name__)


class ExternalLookupAdapter(ABC):
    """Abstract interface every external-source adapter implements.

    Concrete subclasses set `name` to a short identifier matching the
    `ExternalMatch.source` they emit (so a downstream consumer can
    associate a match back to its adapter). The constructor signature
    is intentionally not constrained here — adapters typically take a
    settings-driven config plus an optional pre-built httpx client for
    test injection.
    """

    name: str

    @abstractmethod
    async def lookup(
        self,
        *,
        brand: str | None,
        beverage_type: str,
        fanciful_name: str | None = None,
        timeout_s: float = 4.0,
    ) -> ExternalMatch | None:
        """Search the source for the best match.

        Args:
            brand: Brand name as extracted from the label, or None if
                the vision pass could not read it confidently.
            beverage_type: One of "beer" | "spirits" | "wine" |
                (future). Adapters MAY return None for types they don't
                support; that is the documented happy-path "no match"
                result.
            fanciful_name: Optional sub-brand / SKU-level name used to
                disambiguate brand families (e.g. "PUMPKIN ALE").
            timeout_s: Per-call wall-clock budget. The adapter MAY
                shorten it internally if it knows requests typically
                take less; it MUST NOT exceed it.

        Returns:
            The single best match the adapter found, or None if no
            confident match exists or a transient failure occurred.

        Implementations MUST NOT raise on transient errors.
        """


# Module-level registry. We use a plain dict rather than a class so the
# adapter set is process-global (matching the import-time-side-effect
# style of `register_adapter`) without forcing callers to thread a
# registry instance through their code.
_registry: dict[str, ExternalLookupAdapter] = {}


def register_adapter(adapter: ExternalLookupAdapter) -> None:
    """Add `adapter` to the registry, replacing any same-name entry.

    Replacement (instead of error-on-duplicate) is the right default
    for tests, which routinely re-register a fresh instance per test;
    a production deploy registers once at import time, so a duplicate
    would be a configuration bug surfaced via the warning log.
    """
    if not adapter.name:
        raise ValueError(
            "ExternalLookupAdapter.name must be a non-empty identifier; "
            "got an instance with empty/None name."
        )
    if adapter.name in _registry:
        logger.warning(
            "External adapter %r already registered; replacing.",
            adapter.name,
        )
    _registry[adapter.name] = adapter


def get_adapter(name: str) -> ExternalLookupAdapter | None:
    """Return the adapter registered under `name`, or None if absent.

    Returning None (rather than raising) lets the verify orchestrator
    feature-flag adapters off cleanly: a `get_adapter("ttb_cola")` call
    when the TTB adapter is not registered is a non-fatal no-op.
    """
    return _registry.get(name)


def list_adapters() -> list[ExternalLookupAdapter]:
    """Return all currently-registered adapters in insertion order.

    The orchestrator iterates this for "fan out across all enabled
    sources" — the order is stable so a downstream consumer can rely
    on it for deterministic merging.
    """
    return list(_registry.values())


def _clear_registry_for_tests() -> None:
    """Drop every registered adapter.

    Test-only helper: production code never wants this. Exposed via a
    name that grep'ing for "for_tests" makes obvious so a stray
    production caller is caught at code review.
    """
    _registry.clear()
