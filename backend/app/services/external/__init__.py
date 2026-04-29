"""External label-data sources tier.

This package owns the pluggable adapter framework that lets the verify
path enrich a vision extraction with hits from real external label
catalogs (TTB COLA Registry today; Open Food Facts, Untappd, etc. in
future workstreams). Adapters are registered at module import time and
discovered by name; the verify orchestrator iterates the registry and
folds each adapter's best match into the L3 cache.

Re-exports the public surface so callers can write a single import:

    from app.services.external import (
        ExternalLookupAdapter,
        ExternalMatch,
        get_adapter,
        list_adapters,
        register_adapter,
    )

Concrete adapter modules (e.g. ``ttb_cola``) are not auto-imported here
to avoid pulling optional HTTP dependencies into callers that only need
the type definitions.
"""

from __future__ import annotations

from app.services.external.adapter import (
    ExternalLookupAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
)
from app.services.external.types import ExternalMatch

__all__ = [
    "ExternalLookupAdapter",
    "ExternalMatch",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
