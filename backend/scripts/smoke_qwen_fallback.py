"""End-to-end smoke test for the Qwen3-VL fallback.

Run against the live OpenAI-compatible endpoint configured by
`QWEN_VL_BASE_URL` / `QWEN_VL_API_KEY` / `QWEN_VL_MODEL`. The Anthropic
key is intentionally NOT used — this script exercises only the local-
fallback leg of the chain so a green run confirms the fallback would
serve traffic if Claude went down.

Usage (from repo root, with the Railway environment loaded):

    railway run -- python backend/scripts/smoke_qwen_fallback.py \\
        artwork/labels/01_pass_old_tom_distillery.png

Exits 0 on a structured response, non-zero (and prints the failure)
otherwise. Costs a few cents per run on OpenRouter.
"""

from __future__ import annotations

import os
import pathlib
import sys

# Make `app.*` importable without installing the package — mirrors what
# `pytest` does via conftest.
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from app.config import settings  # noqa: E402
from app.services.qwen_vl import QwenVLExtractor  # noqa: E402


def _redact(value: str | None, keep: int = 4) -> str:
    if not value:
        return "<unset>"
    if len(value) <= keep + 3:
        return "***"
    return value[:keep] + "…(" + str(len(value)) + " chars)"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: smoke_qwen_fallback.py <image-path>", file=sys.stderr)
        return 2
    path = pathlib.Path(argv[1])
    if not path.exists():
        print(f"image not found: {path}", file=sys.stderr)
        return 2

    print("=== Qwen3-VL fallback smoke test ===")
    print(f"  base_url   = {settings.qwen_vl_base_url or '<unset>'}")
    print(f"  model      = {settings.qwen_vl_model}")
    print(f"  api_key    = {_redact(settings.qwen_vl_api_key)}")
    print(f"  timeout_s  = {settings.qwen_vl_timeout_s}")
    print(f"  enabled    = {settings.enable_qwen_fallback}")
    print(f"  image      = {path} ({path.stat().st_size} bytes)")
    print()

    if not settings.qwen_vl_base_url:
        print("FAIL: QWEN_VL_BASE_URL is unset", file=sys.stderr)
        return 1
    if not settings.enable_qwen_fallback:
        print(
            "WARN: ENABLE_QWEN_FALLBACK is false — extractor will still run "
            "for this smoke test, but production traffic would skip the "
            "fallback chain.",
            file=sys.stderr,
        )

    image_bytes = path.read_bytes()
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    extractor = QwenVLExtractor()
    try:
        result = extractor.extract(image_bytes, media_type=media)
    except Exception as exc:  # noqa: BLE001 — surface every error verbatim
        print(f"FAIL: extractor raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("=== Result ===")
    print(f"  image_quality       = {result.image_quality!r}")
    print(f"  image_quality_notes = {result.image_quality_notes!r}")
    print(f"  beverage_observed   = {result.beverage_type_observed!r}")
    print(f"  unreadable_fields   = {result.unreadable!r}")
    print("  fields:")
    for name, field in result.fields.items():
        print(f"    {name:18s} value={field.value!r:60s} conf={field.confidence}")

    expected = {
        "brand_name",
        "class_type",
        "alcohol_content",
        "net_contents",
        "name_address",
        "country_of_origin",
        "health_warning",
    }
    missing = expected - set(result.fields.keys()) - set(result.unreadable)
    if missing:
        print(f"FAIL: response missing expected fields: {sorted(missing)}", file=sys.stderr)
        return 1

    print()
    print("PASS: Qwen3-VL fallback returned a structured extraction.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
