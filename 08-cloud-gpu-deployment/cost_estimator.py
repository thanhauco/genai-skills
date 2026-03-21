"""
cost_estimator.py — turn GPU hourly rates into $/1M tokens (the number that sells).

Why this matters:
  Customers don't buy "GPU hours" — they buy cost per token at a latency SLO. You
  convert a measured throughput + an instance price into $/1M tokens and compare
  configs / clouds / managed-vs-self-hosted on ONE axis.

    $/1M tokens = hourly_rate / (throughput_tok_per_s * 3600) * 1e6

PRICES BELOW ARE ILLUSTRATIVE PLACEHOLDERS. Always confirm current pricing; the
method is the point, not the exact numbers.

Run:
    python cost_estimator.py
Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Instance:
    name: str
    cloud: str
    gpus: int
    gpu_type: str
    usd_per_hour: float   # ILLUSTRATIVE on-demand-ish placeholder


# Illustrative instances (verify real prices!). Names are representative.
INSTANCES = [
    Instance("p5.48xlarge (8xH100)", "AWS", 8, "H100", 98.0),
    Instance("p4d.24xlarge (8xA100)", "AWS", 8, "A100", 32.0),
    Instance("g6e.12xlarge (4xL40S)", "AWS", 4, "L40S", 10.5),
    Instance("ND-H100-v5 (8xH100)", "Azure", 8, "H100", 96.0),
    Instance("a3-highgpu-8g (8xH100)", "GCP", 8, "H100", 88.0),
    Instance("a2-ultragpu (8xA100-80)", "GCP", 8, "A100", 40.0),
]


def per_gpu_hour(inst: Instance) -> float:
    return inst.usd_per_hour / inst.gpus


def cost_per_million(usd_per_hour: float, throughput_tok_s: float) -> float:
    if throughput_tok_s <= 0:
        return float("inf")
    return usd_per_hour / (throughput_tok_s * 3600) * 1_000_000


def main() -> None:
    print("NOTE: prices are ILLUSTRATIVE placeholders. Confirm real pricing.\n")

    print("=== per-GPU effective hourly rate ===")
    for i in sorted(INSTANCES, key=per_gpu_hour):
        print(f"  {i.cloud:5s} {i.name:26s} {i.gpu_type:5s}  ${per_gpu_hour(i):6.2f}/GPU-hr")

    print("\n=== $/1M tokens vs measured throughput (whole instance) ===")
    print("Plug in the tok/s YOUR load test measured for the config on that instance.\n")
    throughputs = [2000, 6000, 15000, 40000]
    header = "instance".ljust(28) + "".join(f"{t:>10}" for t in throughputs)
    print("  " + header + "   (tok/s ->)")
    for i in INSTANCES:
        row = f"{i.cloud} {i.name}".ljust(28)
        row += "".join(f"{cost_per_million(i.usd_per_hour, t):>10.3f}" for t in throughputs)
        print("  " + row)

    print("\n=== self-hosted break-even vs a managed $/1M price ===")
    managed_price = 0.20  # e.g., a managed per-1M-token price (illustrative)
    inst = INSTANCES[0]   # 8xH100
    print(f"Managed price assumed: ${managed_price:.2f}/1M tokens")
    print(f"Self-host instance   : {inst.name} @ ${inst.usd_per_hour:.0f}/hr")
    # throughput needed so self-hosted cost <= managed price
    needed = inst.usd_per_hour / (managed_price / 1_000_000) / 3600
    print(f"-> You must sustain ~{needed:,.0f} tok/s on that instance to beat the managed price.")
    print("   Below that throughput, MANAGED (e.g., Fireworks) is cheaper AND no ops.")

    print(
        "\nHow to use this with a customer:\n"
        "  1) load-test (module 09) -> sustained tok/s at their p95 SLO\n"
        "  2) compute $/1M for each candidate config + cloud here\n"
        "  3) compare to managed; factor in ENGINEER TIME + reliability, not just $\n"
        "  4) recommend the cheapest option that holds the SLO and the eval bar"
    )


if __name__ == "__main__":
    main()
