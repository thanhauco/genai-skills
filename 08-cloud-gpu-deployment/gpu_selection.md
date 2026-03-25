# GPU selection guide

Pick the smallest/cheapest GPU configuration that fits the model and holds the latency SLO. This is a core sizing decision the JD calls out ("determining optimal shapes").

## The GPUs you'll actually discuss

| GPU | HBM | Rough role | Notes |
| --- | --- | --- | --- |
| **H200** | 141 GB | Biggest models, long context, max throughput | More HBM than H100 → more KV cache |
| **H100** | 80 GB | Throughput king, fp8, low latency | The default for serious serving |
| **A100** | 40 / 80 GB | Great value, broad availability | No native fp8; still excellent |
| **L40S** | 48 GB | Cost-effective mid-size serve + tune | Good perf/$, widely available |
| **L4** | 24 GB | Small models, cheap high-volume, edge | Low power, low cost |
| **AMD MI300X** | 192 GB | Huge HBM → fewer GPUs for big models | ROCm software path |

## Mapping to instances per cloud (representative)

| Cloud | H100 | A100 | L40S / L4 |
| --- | --- | --- | --- |
| **AWS** | p5 (8×H100) | p4d/p4de (8×A100) | g6e (L40S), g6 (L4) |
| **Azure** | ND H100 v5 | ND A100 v4 | NV-series (L4/L40S-class) |
| **GCP** | a3 (8×H100) | a2 (A100) | g2 (L4) |

(Names/SKUs change — confirm current offerings. The mapping intuition is what matters.)

## How to choose (the algorithm)

1. **Will the weights fit?** `params_B × bytes_per_param`. If not on one GPU → tensor-parallel across GPUs **or** quantize. (Use `02-llm-inference/kv_cache_calculator.py`.)
2. **How much KV cache do you need?** Driven by context length × concurrency. More HBM (H200/MI300X) or quantization buys concurrency.
3. **Latency SLO?** Tighter TTFT/TPOT → more compute + bandwidth (H100/H200), fewer requests per GPU.
4. **Budget / volume?** High volume + cost-sensitive → push quantization, consider L40S/L4 for small models, or managed.
5. **Availability / region?** H100 can be scarce; A100/L40S often easier to get. Region must satisfy latency + data residency.

## Single-GPU vs multi-GPU

```
 fits on 1 GPU      -> simplest; one pod, one GPU
 too big for 1 GPU  -> tensor-parallel (TP) across GPUs on ONE node (NVLink)
 too big for 1 node -> add pipeline parallel (PP) across nodes (adds latency)
 high throughput    -> data-parallel replicas (many copies behind a load balancer)
```

Rule of thumb: **TP within a node** (needs NVLink bandwidth), **replicas/DP for scale-out**, **PP only when a model won't fit in a node**.

## Quantization interplay (ties to module 02)

- fp16 70B ≈ 140GB → needs ~2×H100 (TP=2) just for weights.
- fp8 70B ≈ 70GB → fits closer to 1×H100; frees KV budget on 2×.
- int4 70B ≈ 35GB → single H100 feasible; eval-gate quality.
- Bigger HBM (H200 141GB, MI300X 192GB) can collapse a multi-GPU deploy into fewer GPUs.

## Don't forget the non-GPU costs

- **Networking/egress**: cross-AZ/region and data egress add up; keep traffic local.
- **Storage**: fast disk / cache for model weights (cold starts).
- **Idle**: a GPU you reserved but don't saturate is pure waste → autoscale or go managed.
- **Engineer time**: running GPU infra has a real human cost — factor it against managed.

## The recommendation sentence

> "Your 70B at a 200ms p95 TTFT and ~6k tok/s fits best on 2×H100 with fp8: weights ~70GB leave healthy KV headroom for your context length, and it lands around $X/1M tokens. If volume is bursty, managed (Fireworks) likely beats self-hosting once you price in idle GPUs and ops time."
