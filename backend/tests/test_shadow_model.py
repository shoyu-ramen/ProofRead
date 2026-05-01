"""Tests for the no-op shadow_model seam.

Per MODEL_INTEGRATION_PLAN §3.2: the v1 implementation is a forward-
compat seam that always returns ``skipped=True``. The detect-container
call site fires the prediction on a daemon thread; this module's
contract is that the prediction itself never raises and the telemetry
emitter handles every shape consistently.
"""

from __future__ import annotations

import logging

from app.services.shadow_model import (
    ShadowPrediction,
    _emit_shadow_telemetry,
    _timed_shadow_predict,
    shadow_predict,
)


def test_noop_returns_skipped_prediction():
    """v1 must hardcode skipped=True. A change here means the no-op
    started reflecting real signal — the entire point of the seam is
    that the response shape never changes until the model artifact
    ships."""
    pred = shadow_predict(b"any-bytes", "beer")
    assert pred.skipped is True
    assert pred.predicted_label is None
    assert pred.confidence == 0.0
    assert pred.model_version == "noop"


def test_noop_handles_none_beverage_type():
    """The detect-container response can have a null beverage_type
    when the model declined to classify. The shadow predictor must
    handle that without raising."""
    pred = shadow_predict(b"x", None)
    assert pred.skipped is True


def test_noop_does_not_raise_on_empty_bytes():
    pred = shadow_predict(b"", "wine")
    assert pred.skipped is True


def test_timed_helper_swallows_exceptions(monkeypatch):
    """A future model that raises must not propagate up through the
    daemon thread. The timed helper catches and returns the default —
    `skipped=True` matches the no-op semantics, so the agreement
    metric correctly excludes the row."""
    import app.services.shadow_model as sm

    def _raise(_b: bytes, _bev: str | None) -> ShadowPrediction:
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(sm, "shadow_predict", _raise)
    pred = _timed_shadow_predict(b"any-bytes", "beer")
    assert pred.skipped is True
    assert pred.predicted_label is None
    assert pred.latency_ms >= 0


def test_timed_helper_stamps_latency(monkeypatch):
    """Even on the no-op, the helper must populate `latency_ms`. The
    detect-container telemetry uses this for the latency-budget panel
    (MODEL_INTEGRATION_PLAN §5.3)."""
    pred = _timed_shadow_predict(b"any-bytes", "beer")
    assert isinstance(pred.latency_ms, int)
    assert pred.latency_ms >= 0


def test_emit_telemetry_logs_skipped_with_null_agreement(caplog):
    """When `skipped=True`, the agreement field must be `null` (not
    `false`). A skipped row is excluded from the agreement-rate
    denominator; if the metric had to distinguish skipped-vs-disagree,
    a `null` token does the job."""
    caplog.set_level(logging.INFO, logger="app.services.shadow_model")
    _emit_shadow_telemetry(ShadowPrediction(), vlm_brand="ipa-co")
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "shadow_skipped=true" in msg
    assert "vlm_agreement=null" in msg
    assert "vlm_brand=ipa-co" in msg
    assert "shadow_predicted_brand=null" in msg


def test_emit_telemetry_logs_agreement_true(caplog):
    """Live-model row, predicted brand matches VLM → agreement=true."""
    caplog.set_level(logging.INFO, logger="app.services.shadow_model")
    pred = ShadowPrediction(
        model_version="brand-v1.0",
        predicted_label="ipa-co",
        confidence=0.91,
        latency_ms=23,
        skipped=False,
    )
    _emit_shadow_telemetry(pred, vlm_brand="ipa-co")
    msg = caplog.records[0].getMessage()
    assert "shadow_skipped=false" in msg
    assert "vlm_agreement=true" in msg
    assert "shadow_predicted_brand=ipa-co" in msg
    assert "shadow_confidence=0.910" in msg


def test_emit_telemetry_logs_agreement_false(caplog):
    """Live-model row, predicted brand differs from VLM → agreement=false.
    The model's confidence is logged regardless so dashboards can
    bucket disagreements by confidence."""
    caplog.set_level(logging.INFO, logger="app.services.shadow_model")
    pred = ShadowPrediction(
        model_version="brand-v1.0",
        predicted_label="lager-bros",
        confidence=0.42,
        latency_ms=18,
        skipped=False,
    )
    _emit_shadow_telemetry(pred, vlm_brand="ipa-co")
    msg = caplog.records[0].getMessage()
    assert "vlm_agreement=false" in msg
    assert "shadow_predicted_brand=lager-bros" in msg
    assert "vlm_brand=ipa-co" in msg


def test_emit_telemetry_handles_null_vlm_brand(caplog):
    """The detect-container response may have `brand_name=null` when
    the VLM didn't read the brand. Telemetry must handle that without
    raising."""
    caplog.set_level(logging.INFO, logger="app.services.shadow_model")
    _emit_shadow_telemetry(ShadowPrediction(), vlm_brand=None)
    msg = caplog.records[0].getMessage()
    assert "vlm_brand=null" in msg
