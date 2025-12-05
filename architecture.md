# Architecture & Mental Models

ASCII diagrams of the systems an AI Field Engineer reasons about daily. Use these to anchor whiteboard conversations.

---

## 1. The end-to-end LLM serving stack

```
                         ┌────────────────────────────────────────────┐
   Customer app /        │                 CLIENT                      │
   agent / pipeline ────►│  OpenAI-compatible SDK, streaming (SSE)     │
                         └───────────────────┬────────────────────────┘
                                             │  HTTPS  /v1/chat/completions
                                             ▼
                         ┌────────────────────────────────────────────┐
                         │              API GATEWAY / LB               │
                         │  auth, rate-limit, routing, retries         │
                         └───────────────────┬────────────────────────┘
                                             ▼
              ┌──────────────────────────────────────────────────────────────┐
              │                   INFERENCE SERVER  (per replica)             │
              │                                                               │
              │   ┌──────────────┐   ┌───────────────┐   ┌────────────────┐   │
              │   │  Scheduler   │──►│  Continuous   │──►│   Model exec    │  │
              │   │ (admission,  │   │  batching     │   │  (prefill +     │  │
              │   │  preemption) │   │  (in-flight)  │   │   decode loop)  │  │
              │   └──────────────┘   └───────────────┘   └────────┬───────┘   │
              │                                                    │           │
              │            ┌──────────────────────────────────────▼────────┐  │
              │            │   KV CACHE (PagedAttention / radix tree)       │  │
              │            │   GPU HBM — the real capacity bottleneck       │  │
              │            └────────────────────────────────────────────────┘ │
              └──────────────────────────────────────────────┬───────────────┘
                                                             ▼
                                   ┌──────────────────────────────────────┐
                                   │   GPU(s):  H100 / A100 / L40S / MI300 │
                                   │   TP/PP sharding across devices       │
                                   └──────────────────────────────────────┘
```

**Field-engineer takeaways**
- Throughput is gated by **KV-cache memory**, not just FLOPs. Batch size ↑ until KV cache is full.
- Latency has two regimes: **prefill** (compute-bound, scales with prompt length) and **decode** (memory-bandwidth-bound, one token at a time).
- The scheduler + continuous batching is *why* vLLM/SGLang beat naive serving — they keep the GPU busy.

---

## 2. A single request's life: prefill vs decode

```
 prompt = "Summarize this contract: ...."          generate 200 tokens
 ───────────────────────────────────────────►   ──────────────────────────►

 ┌───────────────────────────┐     ┌──┐┌──┐┌──┐┌──┐┌──┐ ...  ┌──┐
 │        PREFILL            │     │d1││d2││d3││d4││d5│      │dN│
 │  process all prompt       │     └──┘└──┘└──┘└──┘└──┘      └──┘
 │  tokens in parallel       │      each step = 1 forward pass, reuses KV
 │  (compute-bound, GPU hot) │      (memory-bandwidth-bound)
 └───────────────────────────┘
        │                              │
        ▼                              ▼
   TTFT (time to first token)     TPOT (time per output token)
   ↑ grows with prompt length     ↑ ~constant per token; total = TPOT × out_len

 Total latency ≈ TTFT + (output_tokens − 1) × TPOT
```

**Key metrics to say out loud in an interview**
- **TTFT** — Time To First Token (UX-critical for chat/agents).
- **TPOT / ITL** — Time Per Output Token / Inter-Token Latency.
- **Throughput** — total output tokens/sec across all concurrent requests.
- The **latency vs throughput** tension: bigger batches raise throughput but can raise TTFT.

---

## 3. KV cache memory — the capacity math

```
 KV bytes per token = 2 (K and V)
                    × num_layers
                    × num_kv_heads × head_dim     (= hidden size for MHA)
                    × bytes_per_elem (fp16=2, fp8=1)

 Per-request KV = KV_bytes_per_token × (prompt_len + output_len)

 Max concurrent ≈ (GPU_HBM − weights − activations) / Per-request KV
```

```
 GPU HBM (e.g. 80 GB H100)
 ┌───────────────────────────────────────────────────────────┐
 │ model weights (e.g. 13B fp16 ≈ 26 GB)                      │
 ├───────────────────────────────────────────────────────────┤
 │ activations / overhead                                     │
 ├───────────────────────────────────────────────────────────┤
 │ KV CACHE  ◄── everything left over = your concurrency      │
 │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
 └───────────────────────────────────────────────────────────┘
```

> Quantizing weights (fp16→fp8/int4) frees HBM for **more KV cache → higher concurrency → higher throughput**. This is the lever you pull for the customer. See `02-llm-inference/kv_cache_calculator.py`.

---

## 4. Parallelism: how a big model spans GPUs

```
 Tensor Parallel (TP) — split each layer's matrices across GPUs (fast interconnect needed, NVLink)
 ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
 │ GPU0    │ │ GPU1    │ │ GPU2    │ │ GPU3    │   one layer, sharded 4 ways
 │ W[:,0:n]│ │ W[:,n:..│ │  ...    │ │  ...    │   all-reduce every layer
 └─────────┘ └─────────┘ └─────────┘ └─────────┘

 Pipeline Parallel (PP) — split layers into stages across GPUs (less bandwidth, adds latency bubbles)
 ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
 │ layers  │──►│ layers  │──►│ layers  │──►│ layers  │
 │ 0-7     │   │ 8-15    │   │ 16-23   │   │ 24-31   │
 └─────────┘   └─────────┘   └─────────┘   └─────────┘

 Rule of thumb: prefer TP within a node (NVLink), PP/DP across nodes.
```

---

## 5. Where a Field Engineer operates (the engagement loop)

```
  ┌──────────────┐    discovery     ┌──────────────┐   build POC/MVP    ┌──────────────┐
  │  CUSTOMER    │ ───────────────► │  FIELD ENG   │ ─────────────────► │  PRODUCTION  │
  │  (AI-native) │ ◄─────────────── │  (you)       │ ◄───────────────── │  DEPLOYMENT  │
  └──────────────┘   pain points    └──────┬───────┘   load test/tune   └──────────────┘
                                           │
                                           │ codify patterns, file product gaps
                                           ▼
                                  ┌────────────────────┐
                                  │ FIREWORKS PRODUCT  │  ◄── compress field→roadmap loop
                                  │  & ENGINEERING     │
                                  └────────────────────┘
```

---

## 6. Fireworks-flavored deployment (how the pieces map to the JD)

```
  Customer traffic profile  ──►  load test (09)  ──►  baselines: p95 TTFT, tok/s, $/1M
            │                                                   │
            ▼                                                   ▼
  Model selection (04/05)  ──►  shape: TP/quant (02/03)  ──►  tune to hit targets
            │                                                   │
            ▼                                                   ▼
  Fine-tune if needed (04)  ──►  eval gate (05)  ──►  deploy on GPU infra (07/08)
            │
            ▼
  Production integration in customer's codebase  ──►  feedback to product
```

Use this diagram to structure a system-design answer: start at the **traffic profile**, end at **$/1M tokens and a feedback loop**.
