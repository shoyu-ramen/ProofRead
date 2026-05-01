"""Shadow-mode inference seam for detect-container.

Slot for an eventual brand-recognition / first-frame model
(MODEL_INTEGRATION_PLAN §3.2 + §1.c). Shadow inference runs alongside
the real detect-container VLM call but its output is logged ONLY —
never reflected in the user-facing response. This lets us measure
agreement-with-Claude over real traffic before flipping a model into
the live decision path.

v1 ships a no-op so the call site can land before the model does. The
no-op returns ``skipped=True`` and never raises.

Two contracts:

  1. **Logging-only.** The detect-container response shape is
     determined upstream of this call; the prediction never reaches the
     client. Even when the model is live, the only visible effect of
     ``shadow_predict()`` is a log line a daily SQL cron joins against
     the verify logs to compute agreement rate.

  2. **Fire-and-forget at the call site.** The prediction runs on a
     daemon thread (see MODEL_INTEGRATION_PLAN §3.2 example) so a slow
     or hung model can never widen the detect-container latency. The
     daemon-thread approach is appropriate ONLY in shadow mode; if/when
     we promote to A/B (Phase 2, §4.3), the call must become
     synchronous and its latency budgeted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ShadowPrediction:
    """What the custom model would have predicted (for logging only).

    ``predicted_label`` matches the ``brand_name_normalized`` shape Task
    #1 stamps on ``LabelCacheEntry`` so the agreement metric is a
    string-equality compare against the VLM's brand. ``confidence`` is
    the model's softmax-equivalent score in [0.0, 1.0]. ``latency_ms``
    is wall-clock for the inference call, for the latency-budget panel
    (MODEL_INTEGRATION_PLAN §5.3). ``skipped=True`` means the no-op (or
    a future feature-flag-off path) didn't run; the agreement metric
    excludes skipped rows so a flag flip doesn't show as drift.
    """

    model_version: str = "noop"
    predicted_label: str | None = None
    confidence: float = 0.0
    latency_ms: int = 0
    skipped: bool = True


def shadow_predict(
    first_frame_bytes: bytes, beverage_type: str | None
) -> ShadowPrediction:
    """Run shadow inference on the detect-container first frame.

    v1 no-op: always returns ``skipped=True``. When a brand-recognition
    or quality model ships in shadow mode (MODEL_INTEGRATION_PLAN
    §4.2), this is replaced with the real ONNX inference call. Output
    is never reflected in the detect-container response — logging
    only.

    Must never raise: a model error returns the default
    ``ShadowPrediction()`` rather than propagating up to the caller. A
    daemon-thread caller (see MODEL_INTEGRATION_PLAN §3.2) silently
    drops a raised exception, but defending here keeps the contract
    explicit: shadow mode never disrupts the request path, period.
    """
    return ShadowPrediction()


def _emit_shadow_telemetry(
    prediction: ShadowPrediction, vlm_brand: str | None
) -> None:
    """Structured-log telemetry for the shadow prediction.

    Shape per MODEL_INTEGRATION_PLAN §5.1: a single info line keyed by
    ``shadow_*`` fields plus the VLM-side ``vlm_brand`` so a daily SQL
    join can compute agreement rate without a second query. The
    agreement field is left ``null`` in no-op mode (no prediction =
    no comparison); when the real model ships, agreement is computed
    here and surfaced inline.

    Image bytes are never logged. ``predicted_label`` and ``vlm_brand``
    are already-normalized brand strings (lower-cased + stripped) and
    are not PII per SPEC's threat model — they're public label text.
    """
    if prediction.skipped:
        agreement: str = "null"
    else:
        agreement = (
            "true"
            if (prediction.predicted_label or "") == (vlm_brand or "")
            else "false"
        )
    logger.info(
        "detect_container.shadow "
        "shadow_model_version=%s "
        "shadow_predicted_brand=%s "
        "shadow_confidence=%.3f "
        "shadow_latency_ms=%d "
        "shadow_skipped=%s "
        "vlm_brand=%s "
        "vlm_agreement=%s",
        prediction.model_version,
        prediction.predicted_label or "null",
        prediction.confidence,
        prediction.latency_ms,
        "true" if prediction.skipped else "false",
        vlm_brand or "null",
        agreement,
    )


def _timed_shadow_predict(
    first_frame_bytes: bytes, beverage_type: str | None
) -> ShadowPrediction:
    """Run ``shadow_predict`` and stamp ``latency_ms`` on the result.

    Catches every exception and returns the default
    ``ShadowPrediction()`` so the daemon-thread caller in
    ``app/api/detect.py`` never unwinds with an uncaught error. The
    permissive default has ``skipped=True`` which the agreement metric
    excludes — same semantics as a no-op return.
    """
    started = time.monotonic()
    try:
        prediction = shadow_predict(first_frame_bytes, beverage_type)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("shadow_predict raised; dropping: %s", exc)
        prediction = ShadowPrediction()
    prediction.latency_ms = int((time.monotonic() - started) * 1000)
    return prediction
