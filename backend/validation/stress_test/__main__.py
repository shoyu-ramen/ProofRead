"""CLI entry: ``python -m validation.stress_test``.

Walks ``artwork/labels/`` for the four canonical PNGs, runs the matrix,
writes ``REPORT.md`` next to this file. Honours ``ANTHROPIC_API_KEY``
for opt-in vision sampling.
"""

from __future__ import annotations

import logging
from pathlib import Path

from validation.stress_test.report import write_report
from validation.stress_test.runner import run_stress_matrix

LABELS_DIR = Path(__file__).resolve().parents[3] / "artwork" / "labels"
REPORT_DIR = Path(__file__).resolve().parent
EXPECTED_LABELS = (
    "01_pass_old_tom_distillery.png",
    "02_warn_stones_throw_gin.png",
    "03_fail_mountain_crest_ipa.png",
    "04_unreadable_heritage_vineyards.png",
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    label_paths = [LABELS_DIR / name for name in EXPECTED_LABELS]
    for p in label_paths:
        if not p.exists():
            raise FileNotFoundError(p)
    matrix = run_stress_matrix(label_paths=label_paths)
    out = write_report(matrix, REPORT_DIR / "REPORT.md")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
