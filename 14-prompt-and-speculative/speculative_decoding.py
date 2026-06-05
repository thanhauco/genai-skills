"""
speculative_decoding.py — the draft-and-verify mechanism + speedup math.

Why this matters:
  Speculative decoding cuts decode latency (TPOT) WITHOUT changing the output: a
  cheap draft model proposes k tokens, the big target model verifies them in ONE
  forward pass, and you accept the longest correct prefix. You should be able to
  explain it and estimate the speedup from the acceptance rate.

This simulates the accept/reject loop with a toy "agreement" probability and
computes expected tokens-per-verification-step (the speedup driver).

Run:
    python speculative_decoding.py
Stdlib only.
"""

from __future__ import annotations

import random


def simulate_one_step(k: int, accept_prob: float, rng: random.Random) -> int:
    """Draft k tokens; accept a contiguous prefix while each is 'agreed'.
    Returns the number of TARGET tokens produced this verification step.

    Standard spec-decoding guarantees >=1 token per step: if the first draft
    token is rejected, the target's own correction token is emitted instead.
    So accepted tokens range from 1..k+1."""
    accepted = 0
    for _ in range(k):
        if rng.random() < accept_prob:
            accepted += 1
        else:
            break
    # +1 for the target's bonus/correction token (always at least 1 produced)
    return accepted + 1


def expected_tokens_per_step(k: int, accept_prob: float) -> float:
    """Closed form: E[tokens] = 1 + sum_{i=1..k} p^i  (geometric prefix + bonus)."""
    return 1.0 + sum(accept_prob**i for i in range(1, k + 1))


def speedup_estimate(k: int, accept_prob: float, draft_cost_frac: float) -> dict:
    """Approximate wall-time speedup vs plain decoding.

    Plain: 1 target forward per token.
    Spec : per step = 1 target forward (verify k) + draft overhead (~k*draft_cost_frac
           of a target forward). Produces E[tokens] per step.
    speedup ~= E[tokens] / (1 + k*draft_cost_frac)
    """
    tps = expected_tokens_per_step(k, accept_prob)
    cost = 1.0 + k * draft_cost_frac
    return {"exp_tokens_per_step": tps, "rel_cost_per_step": cost, "speedup": tps / cost}


def main() -> None:
    rng = random.Random(0)
    k = 4
    draft_cost_frac = 0.15  # draft model ~15% the cost of the target per token

    print(f"draft length k={k}, draft cost ~{draft_cost_frac:.0%} of target per token\n")
    print(f"{'accept_p':>9s} {'E[tok/step]':>12s} {'speedup':>9s}  regime")
    for p in (0.9, 0.7, 0.5, 0.3, 0.1):
        est = speedup_estimate(k, p, draft_cost_frac)
        regime = "big win" if est["speedup"] > 1.6 else ("win" if est["speedup"] > 1.05 else "NOT worth it")
        print(f"{p:9.2f} {est['exp_tokens_per_step']:12.2f} {est['speedup']:9.2f}x  {regime}")

    # Monte-Carlo sanity check at p=0.7
    p = 0.7
    trials = 20000
    total = sum(simulate_one_step(k, p, rng) for _ in range(trials))
    print(f"\nMonte-Carlo @ p={p}: avg tokens/step = {total/trials:.3f} "
          f"(closed form = {expected_tokens_per_step(k, p):.3f})")

    print(
        "\nTakeaways:\n"
        "  - Speedup is driven by ACCEPTANCE RATE: high agreement -> many tokens/step.\n"
        "  - Low acceptance (creative / high-temp) -> draft+verify overhead can LOSE.\n"
        "  - It cuts TPOT (decode), NOT TTFT (prefill).\n"
        "  - It's distribution-preserving: identical output quality to the target.\n"
        "  - Pick the draft model + k to maximize acceptance for the workload;\n"
        "    measure it with a load test (module 09) before committing."
    )


if __name__ == "__main__":
    main()
