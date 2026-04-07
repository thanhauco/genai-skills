# Load testing guide — design, metrics, and the tuning loop

How to produce a latency/throughput/cost baseline a customer's exec will trust, and how to tune a deployment to hit targets.

## Metrics glossary

| Metric | Definition | Notes |
| --- | --- | --- |
| **TTFT** | Time To First Token | Dominated by prefill; grows with prompt length. The latency users *feel* in chat. |
| **TPOT / ITL** | Time Per Output Token / Inter-Token Latency | Decode cost per token; streaming smoothness. |
| **E2E latency** | Full request wall time | `≈ TTFT + (out_tokens − 1) × TPOT`. |
| **Throughput** | Aggregate output tokens/sec | Capacity + the basis for cost. |
| **RPS** | Requests/sec | Concurrency planning. |
| **Goodput** | Requests/sec **meeting the SLO** | Better than raw RPS — counts only "good" responses. |
| **p50/p95/p99** | Latency percentiles | SLOs live here, not at the mean. |
| **Error rate** | failed / total | Spikes at saturation. |
| **$/1M tokens** | `gpu_$/hr ÷ (tok/s × 3600) × 1e6` | The number execs decide on. |

## Designing a realistic test

1. **Traffic shape**: sample real prompt-length and output-length distributions. Uniform 256/256 is a lie that flatters the system.
2. **Arrival pattern**: Poisson/bursty, not an even drip. Real load clusters.
3. **Concurrency ramp**: step users up to the expected peak; find the **knee** where p95 breaks the SLO.
4. **Warm-up**: discard the first N requests (cold cache, autoscale lag).
5. **Duration**: run to steady state, through at least one autoscale cycle.
6. **Prefix realism**: if production won't share a prefix, don't test with a warm shared prefix cache (and vice-versa for RAG/agents that DO share one).
7. **One variable at a time** when tuning, so you can attribute the change.

## The saturation curve (what you're looking for)

```
 throughput                          p95 latency
 (tok/s)                             (ms)
   |            ____________            |                 /
   |          /                         |               /
   |        /                           |            __/
   |      /                             |       ____/
   |    /                               |  ____/
   |  /                                 |_/
   +-------------------- concurrency    +-------------------- concurrency
        ^ knee: throughput plateaus          ^ knee: latency starts climbing
   Max SAFE concurrency = just before both knees (SLO still met).
```

Past the knee: throughput stops rising, latency + queue time + error rate spike. Don't run production there.

## The tuning loop (field workflow)

```
 1. baseline (fp16, modest max-num-seqs)
 2. load test -> record p50/p95/p99 TTFT, TPOT, tok/s, errors, $/1M
 3. turn ONE knob: TP | quant | max-num-seqs | prefix cache | replicas
 4. re-test -> keep it only if p95 stays within SLO AND it helps tok/s / $
 5. eval-gate (module 05) so quality didn't regress
 6. repeat until target hit; codify the winning shape as a repeatable pattern
```

Knob → effect cheat sheet:
- `max-num-seqs` ↑ → throughput ↑, tail latency ↑ (batch contention).
- `quantization` (fp8/int4) → more KV cache + faster decode → throughput ↑ (eval-gate quality).
- `tensor-parallel-size` ↑ → fits bigger models, lowers latency, costs more GPUs.
- `enable-prefix-caching` → big TTFT win for shared-prefix (RAG/agent) workloads.
- `replicas` ↑ (HPA) → linear RPS scale-out; watch cold-start lag.

## Reporting template (hand this to the customer)

```
Workload: <model> <quant> on <N×GPU> | prompts ~<p50>/<p95> tok, outputs ~<p50>/<p95> tok
Traffic : <RPS target>, <arrival pattern>, peak concurrency <C>

Results @ concurrency C (SLO: p95 TTFT <= 250ms):
  TTFT   p50 / p95 / p99 = ___ / ___ / ___ ms     [PASS/FAIL]
  TPOT   p50 / p95       = ___ / ___ ms
  E2E    p50 / p95 / p99 = ___ / ___ / ___ ms
  Throughput             = ___ tok/s (aggregate)
  Error rate             = ___ %
  Cost                   = $___ / 1M output tokens

Recommendation: <config> hits the SLO at the lowest $/1M. Next lever: <quant/replicas>.
```

## Tools

- `async_load_test.py` — zero-dependency async generator + metrics (offline sim or live).
- `analyze_results.py` — re-analyze captured samples, compute $/1M, SLO gate.
- `locustfile.py` — Locust scenario with a live web UI + custom TTFT metric.
- Others you might mention: **vLLM's `benchmark_serving.py`**, **k6**, **wrk**, **NVIDIA GenAI-Perf** for production-grade LLM benchmarking.
