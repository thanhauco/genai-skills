# Serving frameworks — comparison & decision guide

A Field Engineer picks the right serving stack for a customer's workload and justifies it. Here's the cheat sheet.

## At a glance

| Dimension | vLLM | SGLang | TensorRT-LLM | Fireworks (managed) |
| --- | --- | --- | --- | --- |
| Core idea | PagedAttention + continuous batching | RadixAttention (prefix-cache tree) + batching | Compiled NVIDIA engines | Tuned proprietary serving (FireAttention) + platform |
| Best at | General-purpose, broad model coverage, easy setup | Shared-prefix workloads, structured output, agents | Max perf on NVIDIA at scale | TRT-class perf w/o ops; fine-tune + deploy + scale |
| Setup effort | Low | Low–medium | High (build engines per model/shape/precision) | Lowest (managed) |
| Flexibility | High | High | Low (recompile to change) | High (managed configs) |
| Quantization | AWQ/GPTQ/fp8/int8, KV quant | fp8/int4, KV quant | int8/fp8/int4 via builder | Managed, tuned per model |
| API | OpenAI-compatible | OpenAI-compatible | Triton / OpenAI front-ends | OpenAI-compatible |
| Hardware | NVIDIA + others (ROCm, etc.) | NVIDIA-focused | NVIDIA only | Managed fleet |
| Ops burden | You run it | You run it | You run it (highest) | None (SaaS) |

## When to pick what

- **Default to vLLM** for most POCs: fast to stand up, huge model coverage, OpenAI-compatible, continuous batching out of the box.
- **Choose SGLang** when the workload has **large shared prefixes** (RAG with a big system prompt, few-shot, agent loops re-sending context) or needs **fast structured/constrained output**. RadixAttention auto-shares prefix KV across requests.
- **Choose TensorRT-LLM** when the customer has **stable, high volume on NVIDIA** and needs the last 20–40% of performance, and can absorb the engine-build/ops complexity.
- **Choose Fireworks** when the customer wants **production performance without running infra**, plus **fine-tuning, function calling, multimodal, and autoscaling** in one platform — i.e., they'd rather spend engineering time on their product than on serving ops.

## The trade-off triangle

```
            performance (tok/s, latency)
                       /\
                      /  \
                     /    \
   flexibility ─────/──────\───── ops simplicity
   (swap models,   TRT-LLM is here-ish: top perf,
    quant, fast     low flexibility, high ops.
    iteration)      Managed (Fireworks) buys perf + simplicity,
                    trading some low-level control.
```

## Key tuning knobs (apply across vLLM/SGLang)

- **Tensor parallelism** — fit larger models / cut latency (NVLink within a node).
- **max-num-seqs / max-num-batched-tokens** — throughput vs latency; raise until p95 SLO breaks.
- **max-model-len** — cap context to avoid over-allocating KV cache.
- **quantization** — fp8/int8/int4 to grow KV budget (eval-gate it).
- **prefix caching / RadixAttention** — reuse shared-prefix KV.
- **gpu-memory-utilization** — how aggressively to claim HBM for KV.
- **speculative decoding** — draft model to cut TPOT for accept-heavy workloads.
- **chunked prefill / prefill-decode disaggregation** — keep long prefills from stalling decodes.

## How this maps to the Fireworks pitch

> "We benchmark your traffic profile, pick the optimal shape (TP + quant), and either tune an open framework with you or move you onto Fireworks' managed serving so you get the performance without owning the ops — and we keep fine-tuning + scaling in the same platform."
