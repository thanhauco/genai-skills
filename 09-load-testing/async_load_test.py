"""
async_load_test.py — a real LLM load generator + metrics, with an offline mode.

Why this matters:
  The JD wants you to "run load tests and establish latency, throughput, and cost
  baselines." This produces the exact report you'd hand a customer: TTFT, TPOT,
  E2E p50/p95/p99, aggregate tokens/sec, error rate — and optionally $/1M tokens.

Modes:
  - OFFLINE (default): simulates an endpoint (realistic prompt/output lengths,
    prefill+decode timing) so it runs with zero setup.
  - LIVE: --base-url points at any OpenAI-compatible server (module 03 mock,
    vLLM, SGLang, Fireworks). Streams responses and times real TTFT/TPOT.

Run:
    python async_load_test.py                                   # offline sim
    python async_load_test.py --concurrency 32 --requests 400   # heavier sim
    python async_load_test.py --base-url http://localhost:8000/v1 --model mock/llama-v3p1-8b-instruct
    python async_load_test.py --out samples.json                # write raw samples

Stdlib only (asyncio + urllib). LIVE mode runs blocking HTTP in a thread pool.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import urllib.request
from dataclasses import asdict, dataclass


@dataclass
class Sample:
    ok: bool
    ttft_s: float
    e2e_s: float
    out_tokens: int
    error: str | None = None

    @property
    def tpot_s(self) -> float:
        return (self.e2e_s - self.ttft_s) / max(1, self.out_tokens - 1)


# ----------------------------- workload model ---------------------------------
def sample_lengths(rng: random.Random) -> tuple[int, int]:
    """Realistic-ish prompt/output length distributions (NOT uniform)."""
    prompt = int(rng.lognormvariate(5.5, 0.6))      # ~ a few hundred tokens, skewed
    output = rng.choice([32, 64, 128, 128, 256, 512])  # mixed, weighted toward mid
    return max(8, prompt), output


# ----------------------------- offline simulator ------------------------------
async def simulate_request(prompt_len: int, out_len: int) -> Sample:
    """Model TTFT (prefill, grows with prompt) + TPOT (decode, ~constant)."""
    ttft = 0.08 + prompt_len * 0.00004 + random.uniform(0, 0.05)   # prefill
    tpot = 0.012 + random.uniform(0, 0.006)                         # decode/token
    e2e = ttft + (out_len - 1) * tpot
    # don't actually sleep the full time *serially*; scale down so the demo is quick
    await asyncio.sleep(min(e2e, 0.4))
    # ~2% simulated failures under load
    if random.random() < 0.02:
        return Sample(False, ttft, e2e, 0, error="simulated 503")
    return Sample(True, ttft, e2e, out_len)


# ----------------------------- live request -----------------------------------
def _live_stream_blocking(base_url: str, model: str, prompt: str, api_key: str | None) -> Sample:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "stream": True,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    start = time.perf_counter()
    ttft, n = 0.0, 0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                if not delta:
                    continue
                if n == 0:
                    ttft = time.perf_counter() - start
                n += 1
        return Sample(True, ttft, time.perf_counter() - start, n)
    except Exception as e:  # noqa: BLE001
        return Sample(False, 0.0, time.perf_counter() - start, 0, error=str(e))


# ----------------------------- driver -----------------------------------------
async def worker(name: int, queue: asyncio.Queue, results: list[Sample], args, rng: random.Random):
    while True:
        try:
            _ = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        prompt_len, out_len = sample_lengths(rng)
        if args.base_url:
            prompt = "x " * prompt_len
            s = await asyncio.to_thread(_live_stream_blocking, args.base_url, args.model, prompt, args.api_key)
        else:
            s = await simulate_request(prompt_len, out_len)
        results.append(s)
        queue.task_done()


async def run(args) -> list[Sample]:
    rng = random.Random(42)
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(args.requests):
        queue.put_nowait(i)
    results: list[Sample] = []
    workers = [asyncio.create_task(worker(i, queue, results, args, rng)) for i in range(args.concurrency)]
    t0 = time.perf_counter()
    await asyncio.gather(*workers)
    args._wall = time.perf_counter() - t0
    return results


# ----------------------------- reporting --------------------------------------
def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def report(results: list[Sample], wall_s: float, gpu_rate: float | None) -> None:
    ok = [r for r in results if r.ok]
    fails = [r for r in results if not r.ok]
    ttfts = [r.ttft_s * 1000 for r in ok]
    e2es = [r.e2e_s * 1000 for r in ok]
    tpots = [r.tpot_s * 1000 for r in ok]
    total_out = sum(r.out_tokens for r in ok)
    agg_tok_s = total_out / wall_s if wall_s else 0.0

    print("\n================= LOAD TEST BASELINE =================")
    print(f"requests={len(results)}  ok={len(ok)}  failed={len(fails)}  "
          f"error_rate={len(fails)/max(1,len(results)):.1%}")
    print(f"wall={wall_s:.2f}s  RPS={len(results)/wall_s:.1f}  aggregate_throughput={agg_tok_s:,.0f} tok/s")
    print(f"TTFT  ms : p50={pct(ttfts,50):7.1f}  p95={pct(ttfts,95):7.1f}  p99={pct(ttfts,99):7.1f}")
    print(f"TPOT  ms : p50={pct(tpots,50):7.1f}  p95={pct(tpots,95):7.1f}")
    print(f"E2E   ms : p50={pct(e2es,50):7.1f}  p95={pct(e2es,95):7.1f}  p99={pct(e2es,99):7.1f}")
    if gpu_rate:
        cost = gpu_rate / (agg_tok_s * 3600) * 1_000_000 if agg_tok_s else float("inf")
        print(f"COST     : ${cost:.3f} / 1M output tokens  (at ${gpu_rate:.2f}/hr)")
    if fails:
        print(f"sample error: {fails[0].error}")
    print("=====================================================")
    print("Reminder: SLOs live at p95/p99, not the mean. Report the tail.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="", help="OpenAI-compatible base; omit for offline sim")
    ap.add_argument("--model", default="mock/llama-v3p1-8b-instruct")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--gpu-rate", type=float, default=3.0, help="$/hr for $/1M-token estimate (sim only)")
    ap.add_argument("--out", default="", help="write raw samples to this JSON file")
    args = ap.parse_args()

    mode = "LIVE" if args.base_url else "OFFLINE-SIM"
    print(f"mode={mode}  concurrency={args.concurrency}  requests={args.requests}")
    results = asyncio.run(run(args))
    report(results, args._wall, None if args.base_url else args.gpu_rate)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f)
        print(f"\nwrote {len(results)} samples -> {args.out}  (analyze with analyze_results.py)")


if __name__ == "__main__":
    main()
