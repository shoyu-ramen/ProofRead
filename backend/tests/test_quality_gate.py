"""Tests for the no-op quality_gate seam.

Per MODEL_INTEGRATION_PLAN §3.1: the v1 implementation is a forward-
compat seam that always returns ``advisory=False`` so the production
critical path is unchanged until the real model lands. Two contracts:

  1. The no-op return is permissive (score=1.0, advisory=False).
  2. The function never raises — even on garbage input.
  3. The verify call site does NOT short-circuit when the no-op is in
     play (a regression in the call-site wiring would surface as the
     verify pipeline silently bypassing the VLM).
"""

from __future__ import annotations

from app.services.quality_gate import (
    QualityGateVerdict,
    _emit_quality_gate_telemetry,
    _timed_quality_gate,
    quality_gate,
)


def test_noop_returns_permissive_verdict():
    """v1 must hardcode score=1.0, advisory=False, version=noop.

    A change here means the no-op started reflecting real signal,
    which is exactly what we forbid in v1 — the call site needs to be
    invisible until the model artifact ships.
    """
    verdict = quality_gate(b"any-bytes")
    assert verdict.score == 1.0
    assert verdict.advisory is False
    assert verdict.model_version == "noop"
    assert verdict.reasons == []


def test_noop_does_not_raise_on_empty_bytes():
    """An empty payload must produce the permissive verdict, not an
    error — the verify orchestrator's caller has already validated
    inputs but defending here keeps the contract explicit."""
    verdict = quality_gate(b"")
    assert verdict.advisory is False


def test_noop_does_not_raise_on_garbage_bytes():
    """Random bytes (not even a valid image header) must not trip
    the gate. The call site is upstream of any validation — the
    sensor pre-check is what would catch malformed images."""
    verdict = quality_gate(b"\x00\x01\x02not-an-image" * 100)
    assert verdict.advisory is False
    assert verdict.score == 1.0


def test_timed_helper_returns_verdict_and_elapsed():
    """`_timed_quality_gate` is the verify-call-site helper. Returns
    a (verdict, elapsed_ms) tuple. Elapsed must be a non-negative int."""
    verdict, elapsed_ms = _timed_quality_gate(b"any-bytes")
    assert isinstance(verdict, QualityGateVerdict)
    assert isinstance(elapsed_ms, int)
    assert elapsed_ms >= 0


def test_timed_helper_swallows_exceptions(monkeypatch):
    """A future model that raises must not propagate up. The helper
    catches and returns the permissive default. SPEC §0.5 fail-open
    on the gate is correct because the rule engine + VLM downstream
    are still authoritative."""

    def _raise(_b: bytes) -> QualityGateVerdict:
        raise RuntimeError("simulated model failure")

    import app.services.quality_gate as qg

    monkeypatch.setattr(qg, "quality_gate", _raise)
    verdict, elapsed_ms = _timed_quality_gate(b"any-bytes")
    assert verdict.advisory is False
    assert verdict.score == 1.0
    assert verdict.model_version == "noop"
    assert elapsed_ms >= 0


def test_emit_telemetry_logs_structured_fields(caplog):
    """Telemetry must emit a single info line with the expected
    `quality_gate_*` keys. Parse the line back to confirm the shape
    a daily SQL cron will see."""
    import logging

    caplog.set_level(logging.INFO, logger="app.services.quality_gate")
    verdict = QualityGateVerdict(
        score=0.42, advisory=True, reasons=["glare", "blur"], model_version="v1.0.0"
    )
    _emit_quality_gate_telemetry(verdict, elapsed_ms=12)

    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "quality_gate_score=0.420" in msg
    assert "quality_gate_advisory=true" in msg
    assert "quality_gate_version=v1.0.0" in msg
    assert "quality_gate_latency_ms=12" in msg
    assert "quality_gate_reasons=glare,blur" in msg


def test_emit_telemetry_handles_empty_reasons(caplog):
    """Empty `reasons` should be logged as `-` so log parsers don't
    have to handle a trailing key with no value."""
    import logging

    caplog.set_level(logging.INFO, logger="app.services.quality_gate")
    _emit_quality_gate_telemetry(QualityGateVerdict(), elapsed_ms=1)
    msg = caplog.records[0].getMessage()
    assert "quality_gate_reasons=-" in msg


def test_verify_call_site_does_not_short_circuit_under_noop(
    monkeypatch, synthetic_label_png
):
    """Integration check: the wire-up in verify.py must NOT downgrade
    a normal verify run when the no-op gate is in play. A regression
    that flipped the call-site polarity (`if not advisory` → `if advisory`)
    would silently route every request to the advisory pathway.

    We exercise the verify orchestrator with a stub extractor and
    confirm the run reaches it (i.e. the gate's advisory short-circuit
    didn't fire).
    """
    from app.services.verify import VerifyInput, verify
    from app.services.vision import VisionExtraction

    class _StubExtractor:
        called = False

        def extract(
            self, image_bytes: bytes, media_type: str = "image/png", **_: object
        ) -> VisionExtraction:
            type(self).called = True
            return VisionExtraction(
                fields={},
                unreadable=[],
                raw_response="",
                image_quality="good",
                image_quality_notes=None,
            )

    inp = VerifyInput(
        image_bytes=synthetic_label_png(),
        media_type="image/png",
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        application={"producer_record": {}},
    )
    extractor = _StubExtractor()
    report = verify(inp, extractor=extractor)
    # The extractor was reached → the gate did not short-circuit.
    assert _StubExtractor.called is True
    # And the report is not the gate's "advisory + degraded" payload.
    assert report.image_quality != "degraded" or (
        report.image_quality_notes != "Quality gate: rescan recommended"
    )


def test_verify_call_site_short_circuits_when_gate_fires(
    monkeypatch, synthetic_label_png
):
    """When a future model returns `advisory=True`, the call site must
    route to an advisory `VerifyReport` with `image_quality="degraded"`
    and SHOULD NOT call the VLM. The reasons list flows through to
    `image_quality_notes`. SPEC §0.5: never escalates to fail.
    """
    import app.services.quality_gate as qg
    from app.services.verify import VerifyInput, verify
    from app.services.vision import VisionExtraction

    monkeypatch.setattr(
        qg,
        "quality_gate",
        lambda _b: qg.QualityGateVerdict(
            score=0.10,
            advisory=True,
            reasons=["panorama_blurry", "glare_on_warning"],
            model_version="v1.0.0",
        ),
    )

    class _ShouldNotBeCalledExtractor:
        called = False

        def extract(
            self, image_bytes: bytes, media_type: str = "image/png", **_: object
        ) -> VisionExtraction:
            type(self).called = True
            raise AssertionError(
                "extractor must not run when the gate fires"
            )

    inp = VerifyInput(
        image_bytes=synthetic_label_png(),
        media_type="image/png",
        beverage_type="beer",
        container_size_ml=355,
        is_imported=False,
        application={"producer_record": {}},
    )
    report = verify(inp, extractor=_ShouldNotBeCalledExtractor())
    assert _ShouldNotBeCalledExtractor.called is False
    assert report.overall == "advisory"
    assert report.image_quality == "degraded"
    assert "panorama_blurry" in (report.image_quality_notes or "")
    assert "glare_on_warning" in (report.image_quality_notes or "")
