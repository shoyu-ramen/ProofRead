"""Pre-VLM panorama quality gate (forward-compat seam).

Slot for the future image-quality classifier (MODEL_INTEGRATION_PLAN §3.1
+ CUSTOM_MODEL_RESEARCH §2.d). v1 ships a permissive no-op so the call
site can land before the model does — when 2.d trains and ships, the
no-op `quality_gate()` body is replaced with the MobileNetV3 regressor
call and nothing else changes.

SPEC §0.5 fail-honestly: the gate can ONLY downgrade verdicts to
``advisory``, never to ``fail``. A model that's confident the panorama
is unreadable still cannot produce a hard fail — the user is told to
rescan, not given a guess. The no-op returns ``advisory=False``
unconditionally so the production critical path is unchanged until the
real model lands.

The gate must never raise: a model error returns the permissive
``QualityGateVerdict()`` rather than propagating up. Callers
(``app.services.verify.verify``) wire the call site at the slot point
described in MODEL_INTEGRATION_PLAN §3.1 and trust the contract.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QualityGateVerdict:
    """Pre-VLM gate decision.

    ``score`` is P("Claude can extract a 95%+ accurate health warning
    from this panorama"), in [0.0, 1.0]. ``advisory=True`` recommends
    the caller short-circuit to an "advisory" overall verdict (not
    "fail" — see module docstring). ``reasons`` is a human-readable
    list surfaced as ``image_quality_notes`` when the gate fires.
    ``model_version`` tags telemetry so dashboards can split agreement
    rate by model release; ``"noop"`` until 2.d ships.
    """

    score: float = 1.0
    advisory: bool = False
    reasons: list[str] = field(default_factory=list)
    model_version: str = "noop"


def quality_gate(panorama_bytes: bytes) -> QualityGateVerdict:
    """Return a verdict on whether the panorama is Claude-extractable.

    v1 no-op: always returns ``score=1.0, advisory=False`` so the call
    site fires no advisory short-circuit. When the quality classifier
    ships (MODEL_INTEGRATION_PLAN §4.2 → §4.3), this function loads the
    ONNX artifact, runs inference on ``panorama_bytes`` plus the 12-d
    sensor features, and produces an ``advisory=True`` verdict when
    the score falls below the production threshold.

    Must never raise. A future inference error must produce the
    permissive ``QualityGateVerdict()`` rather than propagating up to
    the verify orchestrator — fail-open here is correct because the
    rule engine + Claude verdict downstream are still authoritative.
    """
    return QualityGateVerdict()


def _emit_quality_gate_telemetry(
    verdict: QualityGateVerdict, elapsed_ms: int
) -> None:
    """Structured-log telemetry for the gate decision.

    Shape per MODEL_INTEGRATION_PLAN §5.1: a single info line with
    ``key=value`` pairs that downstream dashboards (and a daily SQL
    cron in v1) parse for the agreement-rate / latency views. The
    log line is emitted on every gate call — even no-op — so the
    dashboard's "% of verify calls with a gate decision" metric is
    monotonic with traffic and a regression that breaks the call site
    is immediately visible.

    No image bytes; no PII. The verdict's ``reasons`` are joined into
    one comma-separated field so log parsers don't have to handle
    nested structures.
    """
    reasons = ",".join(verdict.reasons) if verdict.reasons else "-"
    logger.info(
        "verify.quality_gate "
        "quality_gate_score=%.3f "
        "quality_gate_advisory=%s "
        "quality_gate_version=%s "
        "quality_gate_latency_ms=%d "
        "quality_gate_reasons=%s",
        verdict.score,
        "true" if verdict.advisory else "false",
        verdict.model_version,
        elapsed_ms,
        reasons,
    )


def _timed_quality_gate(panorama_bytes: bytes) -> tuple[QualityGateVerdict, int]:
    """Run the gate and return (verdict, elapsed_ms).

    Helper so the verify call site doesn't repeat the wall-clock
    measurement boilerplate. Catches any exception and returns the
    permissive verdict — a model crash MUST NOT take down the verify
    pipeline.
    """
    started = time.monotonic()
    try:
        verdict = quality_gate(panorama_bytes)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("quality_gate raised; using permissive verdict: %s", exc)
        verdict = QualityGateVerdict()
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return verdict, elapsed_ms
