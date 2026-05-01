"""Admin observability surface — cache + external-source health.

``GET /v1/admin/cache-health`` returns a JSON snapshot of every tier in
the verify-path cache stack plus the TTB COLA adapter's request /
error counters. Intended for the on-call dashboard and ad-hoc
``curl``-from-laptop debugging; not on the hot path of any user-facing
flow.

Auth model (deliberately stricter than the existing
``/v1/verify/_stats`` route):

  * ``X-Admin-Token`` header MUST match
    ``settings.admin_api_token``. A wrong token returns 401.
  * If ``settings.admin_api_token`` is unset, the endpoint returns
    503 (``admin_disabled``) — NOT 200, NOT 401. An unset token must
    not silently fall through to "anyone can hit it" and must not
    lie to the caller about an authentication problem when the real
    issue is configuration.

The returned shape mirrors the SPEC:

  {
    "l1": {"size", "hits", "misses", "max_entries"},
    "l2": {"size", "hits", "misses", "max_entries", "hamming_threshold"},
    "l3": {
      "rows", "total_hits",
      "top_hit_counts": [{"signature_prefix", "hit_count", "last_seen_at"}, ...]
    },
    "ttb_cola": {
      "enabled", "last_request_at", "request_count", "error_count",
      "circuit_open"
    }
  }

Each tier degrades to a ``null`` value (or ``"unavailable"`` payload)
when its cache instance hasn't been built yet (the in-process LRUs
are lazy) or when the DB is unreachable. The endpoint never 500s on
a tier failure — admin tooling should be able to tell us which tier
is broken without the endpoint itself going down.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException

from app.api import verify as verify_api
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


_ADMIN_DISABLED_CODE = "admin_disabled"
_ADMIN_UNAUTHORIZED_CODE = "admin_unauthorized"


def _check_admin_auth(x_admin_token: str | None) -> None:
    """Reject the request unless ``X-Admin-Token`` matches the env var.

    Two distinct rejection modes:

      * ``ADMIN_API_TOKEN`` unset → 503 (``admin_disabled``). Treats
        the unset state as "this endpoint is not available in this
        deploy" rather than as an authentication failure, so an
        operator forgetting to set the token in Railway Variables
        sees a clean configuration signal instead of believing their
        token is wrong.
      * Token present but wrong / missing in the request → 401.

    Constant-time comparison via ``secrets.compare_digest`` so a
    timing-attack-aware probe doesn't leak the token byte by byte.
    """
    expected = settings.admin_api_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail={
                "code": _ADMIN_DISABLED_CODE,
                "message": (
                    "Admin endpoint disabled: ADMIN_API_TOKEN is not set."
                ),
            },
        )
    if not x_admin_token:
        raise HTTPException(
            status_code=401,
            detail={
                "code": _ADMIN_UNAUTHORIZED_CODE,
                "message": "Missing X-Admin-Token header.",
            },
        )
    import secrets

    if not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(
            status_code=401,
            detail={
                "code": _ADMIN_UNAUTHORIZED_CODE,
                "message": "Invalid X-Admin-Token.",
            },
        )


def _l1_payload() -> dict[str, Any] | None:
    """Snapshot of the byte-exact verify cache.

    Returns ``None`` when the cache hasn't been constructed yet
    (lazy init: never built means it has never been needed) so the
    response distinguishes "no requests yet" from "0 hits".
    """
    cache = verify_api.get_verify_cache()
    if cache is None:
        return None
    cs = cache.stats()
    return {
        "size": cs.size,
        "hits": cs.hits,
        "misses": cs.misses,
        "max_entries": cs.max_entries,
    }


def _l2_payload() -> dict[str, Any] | None:
    """Snapshot of the perceptual reverse-lookup cache."""
    rcache = verify_api.get_reverse_lookup_cache()
    if rcache is None:
        return None
    rs = rcache.stats()
    return {
        "size": rs.size,
        "hits": rs.hits,
        "misses": rs.misses,
        "max_entries": rs.max_entries,
        "hamming_threshold": rs.hamming_threshold,
    }


async def _l3_payload() -> dict[str, Any] | None:
    """Snapshot of the durable Postgres-backed perceptual cache.

    Returns ``{"enabled": False}`` when the persisted-cache feature
    flag is off — operators flipping the flag mid-deploy see the
    state explicitly. Returns ``{"unavailable": true, "error": ...}``
    when the DB query itself blows up (e.g. migration not yet run,
    DATABASE_URL unreachable). The endpoint never 500s on a tier
    failure — that would defeat the whole point of an admin
    observability surface.
    """
    persisted = verify_api.get_persisted_label_cache()
    if persisted is None:
        return {"enabled": False}
    try:
        stats = await persisted.stats()
        top = await persisted.top_hit_entries(limit=10)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("admin: l3 stats query failed: %s", exc)
        return {"enabled": True, "unavailable": True, "error": str(exc)}
    return {
        "enabled": True,
        "rows": stats.total_entries,
        "total_hits": stats.total_hits,
        "top_hit_counts": top,
    }


def _ttb_cola_payload() -> dict[str, Any]:
    """Snapshot of the TTB COLA adapter's request / error counters.

    When the feature flag is off the adapter is not registered, so
    we fall back to a fixed payload that only carries the
    ``enabled=False`` signal — the verify path doesn't issue calls,
    so the per-call counters are necessarily zero.
    """
    from app.services.external import get_adapter

    adapter = get_adapter("ttb_cola")
    if adapter is None or not hasattr(adapter, "stats"):
        return {
            "enabled": settings.ttb_cola_lookup_enabled,
            "last_request_at": None,
            "request_count": 0,
            "error_count": 0,
            "circuit_open": False,
        }
    return adapter.stats()


@router.get("/cache-health")
async def cache_health(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict[str, Any]:
    """Snapshot every cache tier + the TTB COLA adapter.

    See module docstring for the response schema and auth model.
    """
    _check_admin_auth(x_admin_token)

    return {
        "l1": _l1_payload(),
        "l2": _l2_payload(),
        "l3": await _l3_payload(),
        "ttb_cola": _ttb_cola_payload(),
    }
