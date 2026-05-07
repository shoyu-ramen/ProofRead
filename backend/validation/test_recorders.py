"""Recorder smoke tests.

Three complementary tests:

  1. `test_synth_recorder_round_trip` — exercises the no-network path.
     Drives `record_one(..., mode="synth")` against a fixture and
     re-reads the resulting `recorded_extraction.json` through the
     loader + replay extractor. Verifies the synth recorder still
     produces a payload the harness can consume.

  2. `test_anthropic_cost_gate` — exit-2 contract on `--mode anthropic`
     without `--i-know-this-costs-money`. Guards the $0 budget.

  3. `test_live_qwen_recorder_round_trip` — `live_recorder` mark; auto-
     skipped without `QWEN_VL_BASE_URL`. Hits the local model server,
     produces a real recording, confirms the harness loop still closes.

These intentionally avoid the network in (1) and (2) so CI runs are
fast and reproducible.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from validation.measure import measure
from validation.real_corpus import load_corpus
from validation.replay_extractor import ReplayVisionExtractor
from validation.scripts.record_extraction import record_one

_FIXTURES_ROOT = Path(__file__).resolve().parent / "tests_fixtures"


def _replay_factory(item):
    return ReplayVisionExtractor.from_payload(
        item.recorded_extraction, source=item.id
    )


@pytest.fixture
def temp_wine_item(tmp_path: Path) -> Path:
    """Copy the wine fixture into a tmp dir so we can rewrite recordings."""
    src = _FIXTURES_ROOT / "lbl-9001"
    if not src.exists():
        pytest.skip(f"wine fixture missing at {src}")
    dst = tmp_path / "lbl-9001"
    shutil.copytree(src, dst)
    # Drop the existing recording so `record_one` writes a fresh one.
    rec = dst / "recorded_extraction.json"
    if rec.exists():
        rec.unlink()
    return dst


def test_synth_recorder_round_trip(temp_wine_item: Path) -> None:
    """`mode=synth` produces a recording the harness consumes cleanly."""
    out = record_one(temp_wine_item, mode="synth", force=False)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["model_provider"] == "synth_from_truth"
    assert "sulfite_declaration" in payload["fields"]

    items = load_corpus(
        root=temp_wine_item.parent,
        beverage_type="wine",
        require_recording=True,
    )
    assert len(items) == 1
    report = measure(
        items,
        vision_extractor_factory=_replay_factory,
        skip_capture_quality=True,
    )
    assert report.items_evaluated == 1
    # Synth recording reproduces the truth, so both wine rules score 1.0.
    for rule_id, score in report.rule_scores.items():
        assert score.precision == 1.0 and score.recall == 1.0, (
            f"{rule_id}: synth round-trip lost signal "
            f"(precision={score.precision:.3f}, recall={score.recall:.3f})"
        )


def test_anthropic_cost_gate(tmp_path: Path) -> None:
    """`--mode anthropic` without the cost-gate flag must exit 2.

    Run the script as a subprocess so we exercise the actual CLI guard
    rather than the in-process function (which only validates payload
    construction). Asserts:

      * non-zero exit
      * the error message names the cost-gate flag so the user knows
        what to type if they really want to use Anthropic
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.scripts.record_extraction",
            "--mode",
            "anthropic",
            str(tmp_path),  # placeholder path, never reached
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, (
        f"expected exit 2, got {proc.returncode}; stderr={proc.stderr}"
    )
    assert "--i-know-this-costs-money" in proc.stderr


@pytest.mark.live_recorder
def test_live_qwen_recorder_round_trip(temp_wine_item: Path) -> None:
    """Hit a real local Qwen3-VL server and confirm the recording is well-formed.

    Auto-skipped when `QWEN_VL_BASE_URL` is unset (handled in
    `validation/conftest.py`). Doesn't assert per-rule precision/recall
    — the model can disagree with the truth for legitimate reasons (its
    own image_quality verdict, character-level miscalibration on the
    Health Warning, etc.). The point of this test is to confirm the
    recorder writes a valid payload that the harness can consume.
    """
    out = record_one(temp_wine_item, mode="qwen", force=False)
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["model_provider"] == "qwen3_vl_local"
    assert isinstance(payload["fields"], dict) and payload["fields"]
    # The recording must round-trip through the loader + replay
    # extractor without raising.
    items = load_corpus(
        root=temp_wine_item.parent,
        beverage_type="wine",
        require_recording=True,
    )
    assert len(items) == 1
    measure(
        items,
        vision_extractor_factory=_replay_factory,
        skip_capture_quality=True,
    )
