"""End-to-end failover test: deliberately broken Anthropic key + live Qwen.

Builds the same `ChainedVerifyExtractor` the production deploy uses, but
injects an obviously-invalid `ANTHROPIC_API_KEY` so the primary leg
fails with `AuthenticationError`. Asserts that the chain falls through
to the Qwen leg (live OpenRouter Nemotron) and returns a structured
extraction.

This proves the failover path works against real credentials without
touching the deployed container's env vars or risking a prod outage.

Usage:
    railway run -- python backend/scripts/smoke_failover.py \\
        artwork/labels/01_pass_old_tom_distillery.png
"""

from __future__ import annotations

import pathlib
import sys

from app.config import settings
from app.services.anthropic_client import ExtractorUnavailable, build_client
from app.services.qwen_vl import QwenVLExtractor
from app.services.vision import ClaudeVisionExtractor
from app.services.vision_chain import ChainedVerifyExtractor


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: smoke_failover.py <image-path>", file=sys.stderr)
        return 2
    path = pathlib.Path(argv[1])
    if not path.exists():
        print(f"image not found: {path}", file=sys.stderr)
        return 2

    print("=== Failover smoke test ===")
    print("  primary  = ClaudeVisionExtractor with INVALID api_key (forces failover)")
    print(f"  fallback = QwenVLExtractor → {settings.qwen_vl_base_url}")
    print(f"  model    = {settings.qwen_vl_model}")
    print(f"  image    = {path} ({path.stat().st_size} bytes)")
    print()

    bad_client = build_client(api_key="sk-ant-invalid-deliberately", max_retries=0)
    primary = ClaudeVisionExtractor(client=bad_client)
    fallback = QwenVLExtractor()
    chain = ChainedVerifyExtractor([primary, fallback])

    image_bytes = path.read_bytes()
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    try:
        result = chain.extract(image_bytes, media_type=media, beverage_type="spirits")
    except ExtractorUnavailable as exc:
        print(f"FAIL: chain exhausted — {exc}", file=sys.stderr)
        return 1

    print("=== Result ===")
    print(f"  image_quality       = {result.image_quality!r}")
    print(f"  beverage_observed   = {result.beverage_type_observed!r}")
    print(f"  unreadable_fields   = {result.unreadable!r}")
    print(f"  fields_extracted    = {len(result.fields)}")
    print()
    for name, field in result.fields.items():
        val = field.value
        if isinstance(val, str) and len(val) > 80:
            val = val[:77] + "…"
        print(f"  {name:18s} {val!r}  conf={field.confidence}")

    expected = {"brand_name", "alcohol_content", "net_contents", "health_warning"}
    missing = expected - set(result.fields.keys())
    if missing:
        print(f"FAIL: response missing critical fields: {sorted(missing)}", file=sys.stderr)
        return 1

    print()
    print("PASS: primary failed (invalid key) and chain fell through to Qwen,")
    print("      which returned a structured extraction. Failover works end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
