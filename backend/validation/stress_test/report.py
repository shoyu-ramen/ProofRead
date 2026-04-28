"""Render a StressMatrix as a markdown report.

The report has three sections:

  1. **Matrix** — rows are degradations, columns are (label, severity).
     Each cell shows the sensor verdict, abbreviated to one letter (so
     the table fits in a normal Github render):

        G = good · D = degraded · U = unreadable · ! = error

  2. **Worst breakdowns** — every (label, condition, severity) where the
     baseline label was acceptable but the degraded variant came back
     ``unreadable``. These are the conditions the pre-check correctly
     rejected at the strongest severity.

  3. **Probably-too-mild conditions** — conditions for which no severity
     dropped any label below ``acceptable``. Either the synthetic
     degradation isn't aggressive enough, or sensor_check is genuinely
     blind to that signature.

When ``ANTHROPIC_API_KEY`` was set during the run we also include a
**Vision-extractor sample** section showing per-condition field counts
and call latency. Skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from validation.stress_test.runner import StressMatrix, StressResult

# Compact verdict glyphs. We deliberately avoid emojis so the table
# renders cleanly in any terminal / markdown viewer.
_VERDICT_GLYPH = {
    "good": "G",
    "degraded": "D",
    "unreadable": "U",
    "error": "!",
}


def render_markdown(matrix: StressMatrix) -> str:
    """Produce the markdown body. Pure function — no I/O."""
    lines: list[str] = []
    lines.append("# Stress-test report — SPEC §0.5 capture conditions")
    lines.append("")
    lines.append(
        f"Labels: {len(matrix.labels)} · "
        f"Conditions: {len(matrix.conditions)} · "
        f"Severities: {len(matrix.severities)} · "
        f"Total runs: {len(matrix.results)}"
    )
    lines.append("")
    lines.append("Verdict glyph: G = good · D = degraded · U = unreadable · ! = error")
    lines.append("")

    # ------------------------------------------------------------------
    # Matrix table
    # ------------------------------------------------------------------
    lines.append("## Matrix")
    lines.append("")

    by_key: dict[tuple[str, str, str], StressResult] = {
        (r.label, r.condition, r.severity): r for r in matrix.results
    }

    header = ["condition"]
    for label in matrix.labels:
        for sev in matrix.severities:
            header.append(f"{_short_label(label)}/{sev[:1].upper()}")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for condition in matrix.conditions:
        row = [condition]
        for label in matrix.labels:
            for sev in matrix.severities:
                r = by_key.get((label, condition, sev))
                row.append(_VERDICT_GLYPH.get(r.sensor_verdict, "?") if r else "·")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    # Worst breakdowns
    # ------------------------------------------------------------------
    lines.append("## Worst breakdowns — acceptable -> unreadable")
    lines.append("")
    breakdowns = _find_breakdowns(matrix)
    if not breakdowns:
        lines.append("_No condition pushed any label all the way to `unreadable`._")
    else:
        lines.append(
            "| label | condition | severity | confidence | top issue |"
        )
        lines.append("|---|---|---|---|---|")
        for r in breakdowns:
            top = r.issues[0] if r.issues else ""
            top = top.replace("|", "\\|")
            lines.append(
                f"| {r.label} | {r.condition} | {r.severity} | "
                f"{r.sensor_confidence:.2f} | {top} |"
            )
    lines.append("")

    # ------------------------------------------------------------------
    # Probably-too-mild
    # ------------------------------------------------------------------
    lines.append("## Probably-too-mild conditions")
    lines.append("")
    too_mild = _find_too_mild(matrix)
    if not too_mild:
        lines.append(
            "_Every condition reduced at least one label's verdict at some severity._"
        )
    else:
        for cond in too_mild:
            lines.append(f"- `{cond}` — no severity dropped any label below `good`.")
    lines.append("")

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------
    errors = [r for r in matrix.results if r.error]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for r in errors:
            lines.append(
                f"- `{r.label}` × `{r.condition}` × `{r.severity}` — "
                f"{r.error.splitlines()[0] if r.error else ''}"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Vision samples (if any)
    # ------------------------------------------------------------------
    if matrix.vision_samples:
        lines.append("## Vision-extractor sample")
        lines.append("")
        lines.append(
            f"Sampled label: `{matrix.vision_samples[0].label}`, "
            f"severity: `{matrix.vision_samples[0].severity}`"
        )
        lines.append("")
        lines.append("| condition | fields | unreadable | elapsed (s) | error |")
        lines.append("|---|---|---|---|---|")
        for s in matrix.vision_samples:
            err = (s.error or "").replace("|", "\\|")
            lines.append(
                f"| {s.condition} | {s.fields_returned} | "
                f"{len(s.unreadable_fields)} | {s.elapsed_s:.1f} | {err} |"
            )
        lines.append("")
    else:
        lines.append("## Vision-extractor sample")
        lines.append("")
        lines.append(
            "_Skipped — ANTHROPIC_API_KEY was not set, or sampling was disabled._"
        )
        lines.append("")

    if matrix.artifact_dir:
        lines.append(f"_Degraded artifacts retained at: `{matrix.artifact_dir}`_")
        lines.append("")

    return "\n".join(lines)


def write_report(matrix: StressMatrix, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = render_markdown(matrix)
    dest.write_text(text, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _short_label(label: str) -> str:
    """Shrink `01_pass_old_tom_distillery` -> `01-pass`. The first two
    underscore-separated tokens carry the index + intended verdict, which
    is all that matters for the column header."""
    parts = label.split("_")
    return "-".join(parts[:2]) if len(parts) >= 2 else label


def _find_breakdowns(matrix: StressMatrix) -> list[StressResult]:
    return sorted(
        (r for r in matrix.results if r.sensor_verdict == "unreadable"),
        key=lambda r: (r.label, r.condition, r.severity),
    )


def _find_too_mild(matrix: StressMatrix) -> list[str]:
    """Conditions where no severity reduced any label below `good`."""
    by_condition: dict[str, list[str]] = {}
    for r in matrix.results:
        by_condition.setdefault(r.condition, []).append(r.sensor_verdict)
    return [
        c
        for c, verdicts in by_condition.items()
        if all(v == "good" for v in verdicts)
    ]
