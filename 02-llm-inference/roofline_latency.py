"""
roofline_latency.py — back-of-envelope TPOT and prefill/decode intuition.

Why this matters:
  You should be able to estimate decode latency (TPOT) from GPU memory bandwidth,
  and explain WHY decode is bandwidth-bound while prefill is compute-bound. This
  gives you a sanity check on benchmark numbers and a story for the customer.

Decode roofline (single request, weight-only memory traffic dominates):
    bytes_moved_per_token ~= model_weight_bytes   (each token reloads the weights)
    TPOT_floor ~= bytes_moved_per_token / mem_bandwidth
  Batching amortizes the weight read across B requests:
    TPOT_per_request_in_batch ~= TPOT_floor (shared)  -> throughput ~ B / TPOT_floor

Run:
    python roofline_latency.py
Stdlib only. Numbers are approximations to build intuition, not exact.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1024**3
TB = 1000**4  # bandwidth usually quoted in decimal TB/s


@dataclass
class GPU:
    name: str
    bw_tbs: float       # memory bandwidth, TB/s
    flops_tf: float     # ~ dense bf16 TFLOP/s (for prefill intuition)


H100 = GPU("H100", 3.35, 990.0)
A100 = GPU("A100", 2.04, 312.0)
L40S = GPU("L40S", 0.86, 362.0)


def decode_tpot_ms(weight_gb: float, gpu: GPU, batch: int = 1, mfu: float = 0.7) -> float:
    """Approx time per output token (ms). Decode reloads weights from HBM each step."""
    bytes_per_step = weight_gb * GB  # weights read once per decode step (shared by batch)
    eff_bw = gpu.bw_tbs * TB * mfu
    seconds = bytes_per_step / eff_bw
    return seconds * 1000  # per step; per-request TPOT ~ same, throughput ~ batch/step


def prefill_ms(prompt_tokens: int, n_params_b: float, gpu: GPU, mfu: float = 0.5) -> float:
    """Approx prefill time (ms). ~2*N FLOPs per token (fwd pass), compute-bound."""
    flops = 2 * n_params_b * 1e9 * prompt_tokens
    eff_flops = gpu.flops_tf * 1e12 * mfu
    return flops / eff_flops * 1000


def main() -> None:
    print("=== Decode TPOT floor (memory-bandwidth-bound) ===")
    print("Weights reloaded from HBM every token; bigger model -> slower decode.\n")
    for weight_gb, label in [(16.0, "8B fp16"), (8.0, "8B fp8"), (140.0, "70B fp16"), (35.0, "70B int4")]:
        for gpu in (H100, A100, L40S):
            tpot = decode_tpot_ms(weight_gb, gpu)
            print(f"  {label:10s} on {gpu.name:5s}: TPOT~{tpot:6.2f} ms  (~{1000/tpot:5.0f} tok/s @ batch1)")
        print()

    print("=== Prefill latency (compute-bound, grows with prompt length) ===\n")
    for ptoks in (256, 2048, 8192):
        t = prefill_ms(ptoks, 8.0, H100)
        print(f"  8B, prompt={ptoks:5d} tok on H100: prefill~{t:6.1f} ms (this is most of TTFT)")

    print("\n=== Batching amortizes the weight read (throughput scaling) ===")
    print("Same per-step weight read serves the whole batch -> throughput ~ linear in batch:\n")
    step_ms = decode_tpot_ms(16.0, H100)
    for b in (1, 8, 32, 128):
        print(f"  8B fp16 on H100, batch={b:3d}: ~{b*1000/step_ms:7.0f} tok/s aggregate "
              f"(per-request TPOT still ~{step_ms:.2f} ms until KV/compute saturates)")

    print(
        "\nInterview line:\n"
        "  'Decode is bandwidth-bound: each token reloads the weights, so TPOT floor\n"
        "   = weight_bytes / HBM_bandwidth. That's why fp8/int4 speeds up decode AND\n"
        "   why batching is free throughput until you run out of KV cache or compute.'"
    )


if __name__ == "__main__":
    main()
