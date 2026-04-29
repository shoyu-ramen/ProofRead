"""Latency probe for `/v1/verify`.

Fires one cold request followed by `n - 1` warm requests at a configurable
concurrency, then prints a side-by-side summary so a Sonnet+crop deploy can
be compared directly against the prior Opus baseline. All requests use the
same image bytes + form payload, so the first should be a cache miss and
every subsequent one should hit the in-process verify cache (`cache_hit=true`,
elapsed_ms < 50) — which doubles as a quick correctness check on the
caching layer in addition to the latency measurement.

Usage:

    python backend/scripts/verify_latency.py \\
        --base https://proofread-rk.fly.dev \\
        --image backend/app/static/samples/01_pass_old_tom_distillery.png \\
        --n 10 --concurrency 5

The script never modifies state on the target — it just POSTs the same
bytes, reads the response, and times wall-clock + server-reported elapsed
separately so we can see network overhead in isolation. Safe to point at
production; pair `--n` with caution if you're charged by API call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class CallResult:
    status: int
    client_ms: float
    server_ms: int | None
    cache_hit: bool
    overall: str | None
    error: str | None = None


def _percentile(values: list[float], pct: float) -> float:
    """`statistics.quantiles` would do this, but it requires n >= 2 and
    chokes on small samples. Manual percentile keeps the script honest at
    n=2 (one cold, one warm) which is the minimum useful run."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


async def _one_call(
    client: httpx.AsyncClient,
    *,
    base: str,
    image_bytes: bytes,
    image_name: str,
    media_type: str,
    beverage_type: str,
    container_size_ml: int,
    is_imported: bool,
) -> CallResult:
    """Single POST + parse. Errors are captured into the result rather
    than raised so a flaky middle request doesn't kill the whole run."""
    files = {"image": (image_name, image_bytes, media_type)}
    data = {
        "beverage_type": beverage_type,
        "container_size_ml": str(container_size_ml),
        "is_imported": "true" if is_imported else "false",
        "application": json.dumps({"producer_record": {}}),
    }
    t0 = time.monotonic()
    try:
        res = await client.post(f"{base}/v1/verify", data=data, files=files)
        client_ms = (time.monotonic() - t0) * 1000.0
    except httpx.HTTPError as exc:
        return CallResult(
            status=0,
            client_ms=(time.monotonic() - t0) * 1000.0,
            server_ms=None,
            cache_hit=False,
            overall=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    body: dict | None = None
    try:
        body = res.json()
    except Exception:
        body = None

    if res.status_code != 200 or not isinstance(body, dict):
        detail = None
        if isinstance(body, dict):
            detail = body.get("detail")
        return CallResult(
            status=res.status_code,
            client_ms=client_ms,
            server_ms=None,
            cache_hit=False,
            overall=None,
            error=str(detail) if detail else res.text[:200],
        )

    return CallResult(
        status=res.status_code,
        client_ms=client_ms,
        server_ms=body.get("elapsed_ms"),
        cache_hit=bool(body.get("cache_hit")),
        overall=body.get("overall"),
    )


async def _run(
    *,
    base: str,
    image_bytes: bytes,
    image_name: str,
    media_type: str,
    beverage_type: str,
    container_size_ml: int,
    is_imported: bool,
    n: int,
    concurrency: int,
    timeout: float,
) -> tuple[CallResult, list[CallResult]]:
    """Cold call first (sequentially, so the first warm call really is
    warm — otherwise N concurrent cold calls all miss). Then fire the
    remaining `n - 1` warm calls at the requested concurrency."""
    limits = httpx.Limits(max_connections=concurrency * 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        cold = await _one_call(
            client,
            base=base,
            image_bytes=image_bytes,
            image_name=image_name,
            media_type=media_type,
            beverage_type=beverage_type,
            container_size_ml=container_size_ml,
            is_imported=is_imported,
        )

        warm_count = max(0, n - 1)
        if warm_count == 0:
            return cold, []

        sem = asyncio.Semaphore(concurrency)

        async def _bounded() -> CallResult:
            async with sem:
                return await _one_call(
                    client,
                    base=base,
                    image_bytes=image_bytes,
                    image_name=image_name,
                    media_type=media_type,
                    beverage_type=beverage_type,
                    container_size_ml=container_size_ml,
                    is_imported=is_imported,
                )

        warm = await asyncio.gather(*[_bounded() for _ in range(warm_count)])
        return cold, list(warm)


def _format_ms(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if value < 1000:
        return f"{value:.0f} ms"
    return f"{value / 1000.0:.2f} s"


def _summary_payload(cold: CallResult, warm: list[CallResult]) -> dict:
    """Machine-readable payload for `--json` consumers (CI, dashboards).

    Mirrors the human summary one-for-one so the same regression checks
    can run in either mode. `cold` is reported as a single dict; `warm`
    contains aggregate stats (no per-call records — those would balloon
    the artifact for high-N runs and aren't actionable).
    """
    successful = [w for w in warm if w.status == 200 and w.error is None]
    client_ms = [w.client_ms for w in successful]
    server_ms = [w.server_ms for w in successful if w.server_ms is not None]
    cache_hits = sum(1 for w in successful if w.cache_hit)
    verdicts = Counter(w.overall for w in successful)

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "p50": None, "p95": None, "p99": None, "min": None, "max": None, "mean": None}
        return {
            "n": len(values),
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "p99": _percentile(values, 99),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }

    return {
        "cold": {
            "status": cold.status,
            "client_ms": cold.client_ms,
            "server_ms": cold.server_ms,
            "cache_hit": cold.cache_hit,
            "overall": cold.overall,
            "error": cold.error,
        },
        "warm": {
            "requested": len(warm),
            "ok": len(successful),
            "client_ms": _stats(client_ms),
            "server_ms": _stats([float(s) for s in server_ms]),
            "cache_hits": cache_hits,
            "cache_hit_rate": (cache_hits / len(successful)) if successful else 0.0,
            "verdicts": dict(verdicts),
            "errors": [
                {"status": w.status, "error": w.error}
                for w in warm
                if w not in successful
            ],
        },
    }


def _print_summary(cold: CallResult, warm: list[CallResult]) -> None:
    print("Cold call (1):")
    print(f"  status     = {cold.status}")
    print(f"  client_ms  = {_format_ms(cold.client_ms)}")
    print(f"  server_ms  = {_format_ms(cold.server_ms)}")
    print(f"  cache_hit  = {cold.cache_hit}")
    print(f"  overall    = {cold.overall}")
    if cold.error:
        print(f"  error      = {cold.error}")
    print()

    if not warm:
        print("No warm calls requested.")
        return

    successful = [w for w in warm if w.status == 200 and w.error is None]
    if not successful:
        print(f"Warm calls (n={len(warm)}):")
        print("  every warm call errored — check `error` lines below.")
        for i, w in enumerate(warm):
            print(f"  [{i}] status={w.status} error={w.error}")
        return

    client_ms = [w.client_ms for w in successful]
    server_ms = [w.server_ms for w in successful if w.server_ms is not None]
    cache_hits = sum(1 for w in successful if w.cache_hit)
    verdicts = Counter(w.overall for w in successful)

    print(f"Warm calls (n={len(warm)}, ok={len(successful)}):")
    print(
        f"  client p50  = {_format_ms(_percentile(client_ms, 50))}    "
        f"client p95  = {_format_ms(_percentile(client_ms, 95))}    "
        f"client p99  = {_format_ms(_percentile(client_ms, 99))}"
    )
    if server_ms:
        print(
            f"  server p50  = {_format_ms(_percentile(server_ms, 50))}    "
            f"server p95  = {_format_ms(_percentile(server_ms, 95))}    "
            f"server p99  = {_format_ms(_percentile(server_ms, 99))}"
        )
    print(
        f"  client mean = {_format_ms(statistics.fmean(client_ms))}    "
        f"client min  = {_format_ms(min(client_ms))}    "
        f"client max  = {_format_ms(max(client_ms))}"
    )
    print(f"  cache hits  = {cache_hits}/{len(successful)}")
    verdicts_str = ", ".join(f"{v}×{c}" for v, c in verdicts.most_common())
    print(f"  verdicts    = {verdicts_str}")

    if cache_hits != len(successful):
        # The whole point of the warm phase is to exercise the in-process
        # verify cache. Anything below 100% is a correctness-of-caching
        # signal worth flagging — likely a process restart between
        # requests or a cache-key change broke determinism.
        print()
        print("⚠  Warm cache hit rate < 100%. Either the target restarted")
        print("   mid-run or the cache-key inputs aren't stable across requests.")

    errored = [w for w in warm if w not in successful]
    if errored:
        print()
        print(f"  {len(errored)} warm call(s) errored:")
        for i, w in enumerate(errored):
            print(f"    [{i}] status={w.status} error={w.error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        help="Base URL of the deployed instance, e.g. https://proofread-rk.fly.dev",
    )
    parser.add_argument(
        "--image",
        default=str(
            Path(__file__).resolve().parents[1]
            / "app"
            / "static"
            / "samples"
            / "01_pass_old_tom_distillery.png"
        ),
        help="Path to the image to send. Defaults to the bundled bourbon sample.",
    )
    parser.add_argument(
        "--beverage-type",
        default="spirits",
        choices=["beer", "wine", "spirits"],
    )
    parser.add_argument("--container-size-ml", type=int, default=750)
    parser.add_argument("--imported", action="store_true")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Total number of requests (1 cold + n-1 warm).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max in-flight warm calls. Cold call always runs alone first.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request HTTP timeout (seconds).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit a machine-readable JSON object on stdout in addition to "
            "the human summary. Use for CI thresholding / dashboards."
        ),
    )
    parser.add_argument(
        "--expect-overall",
        default=None,
        help=(
            "Fail (exit 3) if either the cold call's `overall` verdict or "
            "any warm call's verdict differs from this value. Use to catch "
            "accuracy regressions while measuring speed (e.g. "
            "--expect-overall=pass for the bourbon sample)."
        ),
    )
    parser.add_argument(
        "--max-cold-ms",
        type=int,
        default=None,
        help=(
            "Fail (exit 4) if the cold-call client_ms exceeds this. "
            "Latency-regression alarm; pick a threshold ~25%% above the "
            "current p95 baseline so normal jitter doesn't trip it."
        ),
    )
    parser.add_argument(
        "--max-warm-p95-ms",
        type=int,
        default=None,
        help=(
            "Fail (exit 4) if warm-call client p95 exceeds this. The warm "
            "path is the cache; a regression here usually means the cache "
            "key inputs lost determinism."
        ),
    )
    parser.add_argument(
        "--require-warm-cache-hits",
        action="store_true",
        help=(
            "Fail (exit 5) if the warm-cache-hit rate is below 100%%. The "
            "verify cache should serve every warm call after the cold one — "
            "anything less is a determinism bug, not a transient miss."
        ),
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f"image not found: {image_path}")
    image_bytes = image_path.read_bytes()
    media_type = (
        "image/png"
        if image_path.suffix.lower() == ".png"
        else "image/jpeg"
        if image_path.suffix.lower() in (".jpg", ".jpeg")
        else "image/png"
    )

    base = args.base.rstrip("/")

    print(
        f"Target: {base}/v1/verify  ·  image: {image_path.name} "
        f"({len(image_bytes) // 1024} KiB, {media_type})"
    )
    print(
        f"Plan:   1 cold, {max(0, args.n - 1)} warm at concurrency={args.concurrency}, "
        f"timeout={args.timeout:.0f}s"
    )
    print()

    cold, warm = asyncio.run(
        _run(
            base=base,
            image_bytes=image_bytes,
            image_name=image_path.name,
            media_type=media_type,
            beverage_type=args.beverage_type,
            container_size_ml=args.container_size_ml,
            is_imported=args.imported,
            n=args.n,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )
    )
    _print_summary(cold, warm)

    payload = _summary_payload(cold, warm)
    if args.json:
        # Emit on a fresh line, prefixed so a CI step can grep for it
        # without picking up stray text from inside the human summary.
        print()
        print("=== JSON ===")
        print(json.dumps(payload, indent=2))

    # Exit-code contract (lowest-numbered failure wins, so callers can map
    # exit codes to specific check failures):
    #   0  everything green
    #   2  cold call failed outright (transport / 5xx / timeout)
    #   3  --expect-overall mismatch on cold or any warm call
    #   4  latency threshold exceeded (cold or warm p95)
    #   5  --require-warm-cache-hits failed
    if cold.status != 200 or cold.error:
        return 2

    successful_warm = [w for w in warm if w.status == 200 and w.error is None]

    if args.expect_overall is not None:
        if cold.overall != args.expect_overall:
            print(
                f"\n✘ cold overall={cold.overall!r} != expected "
                f"{args.expect_overall!r}",
                file=sys.stderr,
            )
            return 3
        bad = [w for w in successful_warm if w.overall != args.expect_overall]
        if bad:
            print(
                f"\n✘ {len(bad)}/{len(successful_warm)} warm calls overall "
                f"!= {args.expect_overall!r}",
                file=sys.stderr,
            )
            return 3

    if args.max_cold_ms is not None and cold.client_ms > args.max_cold_ms:
        print(
            f"\n✘ cold client_ms {cold.client_ms:.0f} > "
            f"--max-cold-ms {args.max_cold_ms}",
            file=sys.stderr,
        )
        return 4

    if args.max_warm_p95_ms is not None:
        warm_p95 = payload["warm"]["client_ms"]["p95"]
        if warm_p95 is not None and warm_p95 > args.max_warm_p95_ms:
            print(
                f"\n✘ warm client p95 {warm_p95:.0f} > "
                f"--max-warm-p95-ms {args.max_warm_p95_ms}",
                file=sys.stderr,
            )
            return 4

    if args.require_warm_cache_hits and successful_warm:
        if payload["warm"]["cache_hit_rate"] < 1.0:
            print(
                f"\n✘ warm cache_hit_rate "
                f"{payload['warm']['cache_hit_rate']:.0%} < 100%",
                file=sys.stderr,
            )
            return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())
