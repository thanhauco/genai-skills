# 09 — Load Testing & Benchmarking

The JD: "Run load tests and establish latency, throughput, and cost baselines against realistic customer traffic profiles, and tune deployments to hit those targets." This is the deliverable that turns a vibe into a number an exec will sign off on.

## Files

| File | What it teaches |
| --- | --- |
| [`async_load_test.py`](async_load_test.py) | An async load generator for an OpenAI-compatible endpoint: TTFT, TPOT, p50/p95/p99, tok/s, errors — with an offline simulator so it runs anywhere |
| [`locustfile.py`](locustfile.py) | A Locust scenario for LLM endpoints (open-model load testing tool) |
| [`analyze_results.py`](analyze_results.py) | Turn raw per-request samples into a baseline report + $/1M tokens |
| [`load_testing_guide.md`](load_testing_guide.md) | How to design a realistic test; metrics glossary; tuning loop |

## Run

```bash
# Offline simulation (no server needed) — see the full metric report:
python 09-load-testing/async_load_test.py

# Against the mock server from module 03:
python 03-serving-frameworks/mock_inference_server.py --port 8000      # terminal A
python 09-load-testing/async_load_test.py --base-url http://localhost:8000/v1 --concurrency 20 --requests 200

# Analyze a samples file (the load test can write one):
python 09-load-testing/async_load_test.py --out samples.json
python 09-load-testing/analyze_results.py samples.json

# Locust (if installed): real-time web UI
pip install locust
locust -f 09-load-testing/locustfile.py --host http://localhost:8000
```

## The metrics you report (glossary)

| Metric | Meaning | Why it matters |
| --- | --- | --- |
| **TTFT** | Time To First Token | Chat/agent UX; the user-felt latency |
| **TPOT / ITL** | Time Per Output Token | Streaming smoothness; total = TTFT + (out-1)×TPOT |
| **E2E latency** | full request time | SLA target |
| **Throughput** | output tokens/sec (aggregate) | capacity / cost driver |
| **RPS** | requests/sec served | concurrency planning |
| **p50/p95/p99** | latency percentiles | SLOs live at the tail, not the mean |
| **error rate** | failed / total | saturation + reliability |
| **$/1M tokens** | cost from throughput + GPU rate | the number execs decide on |

## The golden rule

> **Report the tail, not the mean.** A 120ms average TTFT with a 900ms p99 is a bad experience for 1% of every user's requests. SLOs are p95/p99.

## Designing a *realistic* test (not a synthetic lie)

- **Match the traffic shape**: real prompt-length and output-length distributions, not all-256-token uniform.
- **Match the arrival pattern**: bursty/Poisson, not a perfectly even drip.
- **Match concurrency**: ramp to the customer's expected peak; find the knee where p95 breaks.
- **Warm up**: discard the first N requests (cold cache / autoscale lag).
- **Measure long enough**: steady state, through at least one autoscale cycle.
- **Vary one thing at a time** when tuning (TP, quant, max-num-seqs).

## Interview Q&A

1. **How do you establish a latency/throughput/cost baseline?**
   - Replay a realistic traffic profile (prompt/output length + arrival distribution) at increasing concurrency, record TTFT/TPOT/E2E percentiles + tok/s + errors, find the concurrency where p95 still meets the SLO, then convert sustained tok/s + GPU rate into $/1M tokens. See `async_load_test.py` + `analyze_results.py`.

2. **Throughput looks great but p99 latency is awful — what's happening?**
   - You're batching too aggressively / saturating: high `max-num-seqs` raises throughput but queues long requests, blowing the tail. Back off batch size, or separate prefill/decode, or add replicas. Tune to the **p95 SLO**, not peak tok/s.

3. **How do you find the saturation point?**
   - Ramp concurrency; plot throughput and p95 latency. Throughput rises then plateaus while latency climbs — the knee is your max safe concurrency. Past it, error rate/queue time spikes.

4. **What makes a load test misleading?**
   - Uniform prompt/output lengths, even arrival, no warmup, too short a run, testing a warm prefix cache that production won't have, or ignoring the tail. Each inflates the result.

5. **How does this feed the deployment decision?**
   - The baseline picks the config (TP/quant/replicas) that holds the SLO at the lowest $/1M, and it becomes the **regression gate**: re-run on any change before production.
