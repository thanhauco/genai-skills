"""
analyze_results.py — turn raw load-test samples into a baseline report.

Pairs with async_load_test.py (run it with --out samples.json first). Kept
separate so you can re-analyze captured runs, compare configs, etc.

Run:
    python async_load_test.py --out samples.json
    python analyze_results.py samples.json
    python analyze_results.py samples.json --gpu-rate 12.25   # add $/1M tokens

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", help="JSON file written by async_load_test.py --out")
    ap.add_argument("--gpu-rate", type=float, default=None, help="$/hr to compute $/1M tokens")
    ap.add_argument("--wall", type=float, default=None, help="wall seconds (for throughput/cost)")
    args = ap.parse_args()

    try:
        rows = load(args.samples)
    except FileNotFoundError:
        print(f"file not found: {args.samples}\nRun: python async_load_test.py --out {args.samples}")
        sys.exit(1)

    ok = [r for r in rows if r["ok"]]
    fails = [r for r in rows if not r["ok"]]
    ttfts = [r["ttft_s"] * 1000 for r in ok]
    e2es = [r["e2e_s"] * 1000 for r in ok]
    out_tokens = sum(r["out_tokens"] for r in ok)

    print(f"=== analysis of {args.samples} ===")
    print(f"n={len(rows)}  ok={len(ok)}  failed={len(fails)}  error_rate={len(fails)/max(1,len(rows)):.1%}")
    print(f"TTFT ms : p50={pct(ttfts,50):.1f}  p90={pct(ttfts,90):.1f}  p95={pct(ttfts,95):.1f}  p99={pct(ttfts,99):.1f}")
    print(f"E2E  ms : p50={pct(e2es,50):.1f}  p95={pct(e2es,95):.1f}  p99={pct(e2es,99):.1f}")
    print(f"total output tokens = {out_tokens:,}")

    if args.wall:
        tok_s = out_tokens / args.wall
        print(f"throughput = {tok_s:,.0f} tok/s (wall={args.wall:.1f}s)")
        if args.gpu_rate:
            cost = args.gpu_rate / (tok_s * 3600) * 1_000_000 if tok_s else float("inf")
            print(f"cost = ${cost:.3f} / 1M output tokens at ${args.gpu_rate:.2f}/hr")
    elif args.gpu_rate:
        print("(pass --wall <seconds> to compute throughput and $/1M tokens)")

    # SLO check example
    slo_ttft_p95 = 250.0
    verdict = "PASS" if pct(ttfts, 95) <= slo_ttft_p95 else "FAIL"
    print(f"\nSLO check: p95 TTFT <= {slo_ttft_p95:.0f}ms ? -> {verdict} (p95={pct(ttfts,95):.0f}ms)")
    print("Use this as a regression gate: re-run on every model/quant/config change.")


if __name__ == "__main__":
    main()
