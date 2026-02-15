"""
metrics.py — the quality + system metrics a Field Engineer reports.

Why this matters:
  "Production-quality metrics, not just benchmark scores." You must compute and
  communicate BOTH the quality metrics (accuracy, pass@k) AND the system metrics
  (p50/p95/p99 latency, throughput, $/1M tokens) — because the customer buys on
  the joint trade-off.

Run:
    python metrics.py
Stdlib only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


# ----------------------------- quality metrics --------------------------------
def accuracy(correct: Sequence[bool]) -> float:
    return sum(correct) / len(correct) if correct else 0.0


def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021). Probability that at least
    one of k samples is correct, given n_correct correct out of n_samples."""
    if n_samples - n_correct < k:
        return 1.0
    # 1 - C(n-c, k)/C(n, k)
    return 1.0 - (math.comb(n_samples - n_correct, k) / math.comb(n_samples, k))


def wilson_interval(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (so you don't over-claim on small sets)."""
    if n == 0:
        return (0.0, 0.0)
    p = passed / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ----------------------------- system metrics ---------------------------------
def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def cost_per_million_tokens(
    gpu_dollars_per_hour: float,
    throughput_tokens_per_s: float,
) -> float:
    """$/1M output tokens from a GPU hourly rate and measured throughput."""
    if throughput_tokens_per_s <= 0:
        return float("inf")
    tokens_per_hour = throughput_tokens_per_s * 3600
    return gpu_dollars_per_hour / tokens_per_hour * 1_000_000


def main() -> None:
    print("=== quality metrics ===")
    correct = [True, True, False, True, True, False, True, True, True, False]
    p = accuracy(correct)
    lo, hi = wilson_interval(sum(correct), len(correct))
    print(f"accuracy = {p:.0%}  (95% CI: {lo:.0%}-{hi:.0%}, n={len(correct)})")
    print(f"  -> small n means a WIDE interval; don't over-claim from 10 examples")

    print("\npass@k for a code task (10 samples, 3 correct):")
    for k in (1, 3, 5):
        print(f"  pass@{k} = {pass_at_k(10, 3, k):.2%}")

    print("\n=== system metrics ===")
    # pretend these are measured TTFT samples (ms) from a load test
    ttft = [120, 130, 140, 150, 160, 175, 190, 210, 260, 420]
    print(f"TTFT  p50={percentile(ttft,50):.0f}ms  p95={percentile(ttft,95):.0f}ms  p99={percentile(ttft,99):.0f}ms")

    print("\n$/1M tokens at various throughputs (H100 ~ $3/hr on-demand-ish):")
    for tps in (1000, 3000, 8000):
        print(f"  throughput={tps:5d} tok/s -> ${cost_per_million_tokens(3.0, tps):6.3f} / 1M tokens")

    print(
        "\nThe sentence that wins deals:\n"
        "  'At your p95 SLO of 200ms TTFT, config X holds 3,000 tok/s, which is\n"
        "   $0.28 / 1M tokens. Quantizing to fp8 raises throughput to ~6,000 tok/s\n"
        "   (-> ~$0.14 / 1M) and the eval accuracy held at 92% (CI 88-95%).'"
    )


if __name__ == "__main__":
    main()
