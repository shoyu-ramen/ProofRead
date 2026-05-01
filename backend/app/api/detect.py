"""POST /v1/detect-container — pre-capture beverage-container gate.

Single-image multipart upload. Returns a `ContainerDetection` payload that
the mobile / web UI uses to decide whether to start the cylindrical-scan
flow. The point of this route is to refuse to start a 10.8 s scan when
the camera is pointed at a face, a hand, or a wall — the user feels
every second of that scan, so a cheap pre-flight check pays for itself
many times over.

Mirrors the multipart shape and error envelope of /v1/verify so the
client only has one error handler for both endpoints. See
`app/api/verify.py:330` for the shape this is patterned on.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.services.anthropic_client import ExtractorUnavailable
from app.services.container_check import (
    ContainerDetection,
    detect_container,
)

router = APIRouter(prefix="/detect-container", tags=["detect"])


# 503 payload code — same shape as the verify route's `vision_unavailable`
# so the client maps a single transient failure mode for both endpoints.
_SERVICE_UNAVAILABLE_CODE = "vision_unavailable"
# 413 payload code for oversize uploads. Same envelope shape as verify.
_IMAGE_TOO_LARGE_CODE = "image_too_large"


@router.post("", response_model=ContainerDetection)
async def detect_container_endpoint(
    image: UploadFile = File(...),
) -> ContainerDetection:
    """Inspect a single image and decide whether a beverage container is
    the primary subject.

    Multipart shape mirrors `/v1/verify`'s single-image legacy field but
    with no producer-record / beverage-type / container-size form
    parameters — this is a pre-flight gate, not a compliance check.
    """
    declared = getattr(image, "size", None)
    max_bytes = settings.max_image_bytes
    if declared is not None and declared > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "code": _IMAGE_TOO_LARGE_CODE,
                "message": (
                    f"Image upload is {declared} bytes; the maximum is "
                    f"{max_bytes} bytes. Resize before submitting."
                ),
            },
        )
    image_bytes = await image.read()
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "code": _IMAGE_TOO_LARGE_CODE,
                "message": (
                    f"Image upload is {len(image_bytes)} bytes; the maximum "
                    f"is {max_bytes} bytes. Resize before submitting."
                ),
            },
        )
    if not image_bytes:
        raise HTTPException(400, "image upload is empty")
    media_type = image.content_type or "image/jpeg"
    if not media_type.startswith("image/"):
        raise HTTPException(
            400,
            f"image content_type must be image/*, got {media_type!r}",
        )

    # `detect_container` is synchronous (issues a blocking Anthropic SDK
    # call). Run it off the event loop so a single in-flight detection
    # doesn't park every other request on this worker.
    try:
        result = await asyncio.to_thread(
            detect_container, image_bytes, media_type
        )
    except ExtractorUnavailable as exc:
        # Pre-capture gate is best-effort. 503 (not 500) so the client
        # treats this as a transient/retryable condition; the UI can
        # then either prompt the user to retry or fall through to the
        # scan path with a "couldn't pre-check" warning.
        raise HTTPException(
            status_code=503,
            detail={
                "code": _SERVICE_UNAVAILABLE_CODE,
                "message": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    return result
