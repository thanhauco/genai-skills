"""
batching_simulation.py — why continuous batching wins.

Why this matters:
  Naive ("static") batching waits for the slowest request in a batch before
  starting the next batch -> the GPU idles and short requests wait behind long
  ones (head-of-line blocking). Continuous (in-flight) batching admits new
  requests as soon as slots free up, keeping the GPU saturated. This is the core
  reason vLLM/SGLang deliver high throughput.

This is a discrete-event simulation (no GPU needed) that contrasts the two and
reports throughput + latency.

Run:
    python batching_simulation.py
Stdlib only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Req:
    rid: int
    arrival: float
    gen_tokens: int     # how many decode steps it needs
    start: float = -1.0
    finish: float = -1.0

    @property
    def latency(self) -> float:
        return self.finish - self.arrival


def make_workload(n: int, seed: int = 0) -> list[Req]:
    rng = random.Random(seed)
    reqs = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(20.0)  # ~20 req/s arrivals
        gen = rng.choice([16, 32, 64, 128, 256])  # mixed output lengths
        reqs.append(Req(i, t, gen))
    return reqs


def simulate_static(reqs: list[Req], batch_size: int, step_s: float = 0.01) -> list[Req]:
    """Static batching: form a batch, run ALL to completion, then next batch."""
    reqs = [Req(r.rid, r.arrival, r.gen_tokens) for r in reqs]
    clock = 0.0
    i = 0
    while i < len(reqs):
        batch = reqs[i : i + batch_size]
        clock = max(clock, batch[0].arrival)
        # batch runs until the LONGEST member finishes (head-of-line blocking)
        steps = max(r.gen_tokens for r in batch)
        for r in batch:
            r.start = clock
            r.finish = clock + steps * step_s  # all finish together (idle for short ones)
        clock += steps * step_s
        i += batch_size
    return reqs


def simulate_continuous(reqs: list[Req], max_slots: int, step_s: float = 0.01) -> list[Req]:
    """Continuous batching: at each decode step, run all active reqs one token;
    finished reqs free their slot immediately and waiting reqs are admitted."""
    reqs = [Req(r.rid, r.arrival, r.gen_tokens) for r in reqs]
    pending = sorted(reqs, key=lambda r: r.arrival)
    active: list[Req] = []
    remaining: dict[int, int] = {}
    clock = 0.0
    pi = 0

    while pi < len(pending) or active:
        # admit arrivals up to capacity
        while pi < len(pending) and len(active) < max_slots and pending[pi].arrival <= clock:
            r = pending[pi]
            r.start = clock
            active.append(r)
            remaining[r.rid] = r.gen_tokens
            pi += 1

        if not active:
            clock = pending[pi].arrival  # jump to next arrival
            continue

        # one decode step for every active request
        clock += step_s
        done = []
        for r in active:
            remaining[r.rid] -= 1
            if remaining[r.rid] <= 0:
                r.finish = clock
                done.append(r)
        for r in done:
            active.remove(r)

    return reqs


def report(name: str, reqs: list[Req]) -> None:
    lats = sorted(r.latency for r in reqs)
    makespan = max(r.finish for r in reqs) - min(r.arrival for r in reqs)
    total_tokens = sum(r.gen_tokens for r in reqs)

    def pct(p):
        return lats[min(len(lats) - 1, int(p / 100 * len(lats)))]

    print(
        f"{name:22s} makespan={makespan:6.2f}s  "
        f"throughput={total_tokens/makespan:7.1f} tok/s  "
        f"p50={pct(50):5.2f}s p95={pct(95):5.2f}s p99={pct(99):5.2f}s"
    )


def main() -> None:
    work = make_workload(300, seed=3)
    print("Same workload, two scheduling strategies (8 concurrent slots):\n")
    report("static batching", simulate_static(work, batch_size=8))
    report("continuous batching", simulate_continuous(work, max_slots=8))
    print(
        "\nWhy continuous wins:\n"
        "  - No head-of-line blocking: a 256-token request doesn't trap seven\n"
        "    16-token requests in its batch.\n"
        "  - The GPU stays saturated: freed slots are refilled every step.\n"
        "  - This is the default in vLLM/SGLang; it's why they beat naive serving."
    )


if __name__ == "__main__":
    main()
