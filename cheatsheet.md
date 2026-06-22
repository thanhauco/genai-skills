# Cheat sheet — last-minute review (one page)

The formulas, identities, and decision ladders to skim right before an interview. Everything here is expanded in the modules.

## The latency identity (say it cold)

```
 E2E latency ≈ TTFT + (output_tokens − 1) × TPOT
 TTFT  = prefill, COMPUTE-bound, grows with prompt length
 TPOT  = decode, MEMORY-BANDWIDTH-bound, ~constant per token
 Throughput = gated by KV-cache memory (how many requests fit) × GPU utilization
```

## KV-cache capacity math

```
 kv_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes   (GQA/MQA shrink n_kv_heads)
 per_request  = kv_per_token × (prompt_len + gen_len)
 max_concurrent ≈ (HBM − weights − overhead) / per_request
 weights_bytes = params_B × bytes_per_param   (fp16=2, fp8=1, int4=0.5)
```
**Lever:** quantize weights → free HBM → more KV cache → more concurrency → higher throughput → lower $/1M.

## Decode TPOT floor

```
 TPOT_floor ≈ weight_bytes / (HBM_bandwidth × utilization)
 batching amortizes the weight read → throughput ≈ batch / TPOT_floor (until KV/compute saturates)
```

## $/1M tokens

```
 $/1M = gpu_$per_hour / (throughput_tok_per_s × 3600) × 1e6
```

## Quantization quick-pick

```
 fp16/bf16  baseline, quality-critical
 fp8        ~lossless on H100/MI300, big memory+throughput win  ← default push
 int8       broad support, ~lossless weight-only
 int4 AWQ   memory/cost-bound, small quality drop → EVAL-GATE
 + KV-cache quant when context is long and concurrency-bound
```

## Fine-tuning ladder (stop at the first that clears the bar)

```
 Prompt → Few-shot → RAG → SFT(LoRA) → DPO → RFT/RLHF
 cheap/instant ........................... expensive/slow
 SFT = show right answers | DPO = pick better of two | RFT = try + score (checkable reward)
 RAG for KNOWLEDGE, fine-tune for BEHAVIOR. Always eval-gate.
```

## Serving framework quick-pick

```
 vLLM     default workhorse; broad models; continuous batching
 SGLang   shared-prefix / agents / structured (RadixAttention)
 TRT-LLM  max NVIDIA perf at stable high volume (build engines; high ops)
 Fireworks managed perf + fine-tune + function-calling, no GPU ops
 (all OpenAI-compatible → migrate by changing base_url)
```

## Latency levers (when asked "make it faster")

```
 TTFT ↓ : prefix caching, chunked prefill, smaller/quantized model, TP, prefill/decode disagg
 TPOT ↓ : speculative decoding, fp8/int4, continuous batching
 throughput ↑ : raise max-num-seqs to the p95 knee, quantize, TP, replicas (HPA)
 cost ↓ : quantize, right-size GPU, autoscale idle away, prefix cache, batch offline
```

## RAG debugging order

```
 1) retrieval first: recall@k / MRR  (right chunk in top-k?)
    low → fix chunking / embeddings / hybrid / reranker / k
 2) THEN generation: groundedness / correctness  (used the context?)
 A bad RAG answer is usually a RETRIEVAL problem, not the model.
```

## Agent robustness checklist

```
 step + token budget | loop detection (dedupe action+args) | validate args vs schema
 errors → observations (recover, don't crash) | eval the TRAJECTORY, not just the answer
 prefix-cache the static system+tools prompt | route cheap, act expensive
 SECURITY: tool output is untrusted (prompt injection); validate+whitelist; never eval()
```

## Kubernetes GPU triage

```
 get pods → describe pod (Events) → logs --previous
 Pending     → GPU scheduling (selector/taint/no free nvidia.com/gpu)
 CrashLoop   → CUDA OOM (TP/quant), /dev/shm for NCCL, driver mismatch, model auth
 Not-Ready   → slow model load → use a startupProbe
 nvidia.com/gpu: request == limit, whole devices, no oversubscription
 autoscale on QUEUE DEPTH / GPU util, NOT CPU%
```

## Load-test reporting

```
 report the TAIL (p95/p99), not the mean | warm up | realistic prompt/output dist + bursty arrivals
 find the knee: throughput plateaus while p95 climbs = max safe concurrency
 metrics: TTFT, TPOT, E2E p50/p95/p99, tok/s, error rate, $/1M, goodput
```

## Multimodal / structured output

```
 image = a block of TOKENS (resolution/detail → more tiles → more tokens → TTFT+$)
 structured output: constrained decoding (valid by construction) + validate + repair loop
 function calling IS structured output (tool name + JSON args)
```

## Discovery (don't pitch first)

```
 frame → context → PAIN (5 whys) → constraints (SLO/budget/data) → SUCCESS METRIC → quantify → next step
 end with a scoped POC + measurable bar + a date. SPIN / MEDDIC to structure.
```

## Fireworks talking points

```
 fastest LLM inference (benchmarked) | own function-calling + multimodal models
 Series C ~$4B | ex-Meta PyTorch + Google Vertex | partnered w/ Azure AI Foundry
 blogs: sparse-delta cross-region RL | open-source agents | "fine-tuning bottleneck = iteration speed, not the algorithm"
```

## The one sentence that frames every answer

> "Start at the customer's **traffic profile + success metric**, build the smallest thing that proves value **in their codebase**, **measure** it (eval + load test), tune to the **SLO and $/1M**, and feed what I learned back to **product**."
