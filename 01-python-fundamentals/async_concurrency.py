"""
async_concurrency.py — bounded concurrent calls to a (simulated) LLM endpoint.

Why this matters for a Field Engineer:
  LLM calls are I/O-bound. To benchmark or batch-process a customer's traffic you
  fan out many requests, but you must BOUND concurrency (rate limits, fairness,
  memory) and handle timeouts/failures gracefully.

Run:
    python async_concurrency.py

Stdlib only — simulates an endpoint with random latency so it runs anywhere.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass


@dataclass
class Result:
    request_id: int
    ok: bool
    latency_s: float
    error: str | None = None


async def fake_llm_call(request_id: int, *, fail_rate: float = 0.1) -> str:
    """Simulate an inference request: variable latency, occasional failure."""
    # Decode-heavy requests take longer; model that with a skewed distribution.
    await asyncio.sleep(random.uniform(0.05, 0.40))
    if random.random() < fail_rate:
        raise TimeoutError(f"request {request_id} timed out upstream")
    return f"response-{request_id}"


async def call_with_guardrails(
    request_id: int,
    sem: asyncio.Semaphore,
    *,
    timeout_s: float = 0.5,
) -> Result:
    """One request: bounded by a semaphore, wrapped in a timeout, never raises."""
    start = time.perf_counter()
    async with sem:  # backpressure: at most `sem._value` requests in flight
        try:
            async with asyncio.timeout(timeout_s):  # py3.11+
                await fake_llm_call(request_id)
            return Result(request_id, ok=True, latency_s=time.perf_counter() - start)
        except (TimeoutError, asyncio.TimeoutError) as e:
            return Result(request_id, ok=False, latency_s=time.perf_counter() - start, error=str(e))


async def run_batch(n: int, concurrency: int) -> list[Result]:
    """Fan out n requests, at most `concurrency` at a time."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(call_with_guardrails(i, sem)) for i in range(n)]
    return await asyncio.gather(*tasks)


def summarize(results: list[Result]) -> None:
    oks = [r for r in results if r.ok]
    fails = [r for r in results if not r.ok]
    lats = sorted(r.latency_s for r in oks)

    def pct(p: float) -> float:
        if not lats:
            return 0.0
        idx = min(len(lats) - 1, int(p / 100 * len(lats)))
        return lats[idx]

    print(f"total={len(results)}  ok={len(oks)}  failed={len(fails)}")
    if lats:
        print(f"latency  p50={pct(50)*1000:6.1f}ms  p95={pct(95)*1000:6.1f}ms  p99={pct(99)*1000:6.1f}ms")
    if fails:
        print(f"sample error: {fails[0].error}")


async def main() -> None:
    random.seed(7)
    n, concurrency = 200, 16
    t0 = time.perf_counter()
    results = await run_batch(n, concurrency)
    wall = time.perf_counter() - t0

    print(f"Fanned out {n} requests at concurrency={concurrency} in {wall:.2f}s")
    summarize(results)

    # Demonstrate the throughput/concurrency relationship: more concurrency,
    # lower wall time — until you hit the endpoint's limits (here: none, it's fake).
    print("\nSweeping concurrency to show the throughput curve:")
    for c in (1, 4, 16, 64):
        t0 = time.perf_counter()
        await run_batch(100, c)
        print(f"  concurrency={c:3d}  wall={time.perf_counter()-t0:5.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
