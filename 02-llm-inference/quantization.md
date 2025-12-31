# Quantization — the Field Engineer's cheat sheet

Quantization reduces the numerical precision of weights (and sometimes activations / KV cache) to save memory and bandwidth. For inference, this is one of your biggest levers: **smaller weights → more KV cache → more concurrency → lower $/token**, and **less memory traffic → faster decode (lower TPOT)**.

## Precision formats

| Format | Bytes/param | Typical use | Quality impact |
| --- | --- | --- | --- |
| FP32 | 4 | training reference | baseline |
| FP16 / BF16 | 2 | default serving precision | ~lossless (BF16 better dynamic range) |
| FP8 (E4M3/E5M2) | 1 | H100+/MI300, weight+activation | usually near-lossless w/ calibration |
| INT8 | 1 | weight-only or W8A8 | small, often negligible with good scales |
| INT4 (GPTQ/AWQ) | 0.5 | memory-constrained, weight-only | small-to-moderate; eval-gate it |

> Rule of thumb for weight memory: `params_B × bytes_per_param` GB. 70B @ fp16 ≈ 140GB; @ fp8 ≈ 70GB; @ int4 ≈ 35GB.

## Methods you should be able to name

- **Weight-only PTQ (post-training quantization):**
  - **GPTQ** — second-order, per-layer error compensation; great int4 quality.
  - **AWQ** (Activation-aware Weight Quantization) — protects salient weight channels using activation stats; popular for int4, fast kernels.
  - **bitsandbytes NF4** — used for QLoRA fine-tuning (4-bit base + LoRA adapters).
- **Weight + activation (W8A8 / FP8):** quantize activations too for more speed; needs calibration. FP8 on Hopper/MI300 is the sweet spot now.
- **KV-cache quantization (fp8/int8 KV):** shrinks the KV cache itself → even more concurrency / longer context. Watch quality on long generations.
- **SmoothQuant** — migrates activation outliers into weights so W8A8 is feasible.

## What changes when you quantize

```
                 memory    decode speed   quality        when to use
 fp16/bf16        1.0x        1.0x         baseline       default; quality-critical
 fp8 (W8A8)       ~0.5x       faster       ~lossless*     H100/MI300, throughput push
 int8 weight-only ~0.5x       faster       ~lossless*     broad GPU support
 int4 AWQ/GPTQ    ~0.25x      faster       small drop*    memory-bound, cost-sensitive
   * always verify on the customer's eval — see module 05.
```

## The decision framework (say this in an interview)

1. **Start fp16/bf16** to establish a quality baseline + the production metric.
2. **Try fp8** (if hardware supports) — usually free quality, big memory/throughput win.
3. **Go int4 (AWQ)** only if memory-bound or cost-critical, and **gate with an eval** on the customer's real task + a latency/throughput load test.
4. **Consider KV-cache quant** when context is long and concurrency is the bottleneck.
5. Re-run the **load test** (module 09) to confirm p95 latency and $/1M tokens actually improved.

## Common traps

- "Benchmark MMLU didn't drop" ≠ "the customer's structured-output task didn't drop." Eval on **their** metric.
- int4 can hurt **long-form coherence** and **tool-call / JSON formatting** more than short QA.
- Quantizing the KV cache can degrade long-context recall — test at the context lengths the customer actually uses.
- Throughput gains only materialize if the framework has efficient kernels for that format on that GPU.

## Quick mapping to Fireworks-style conversations

- "We can move you from fp16 to fp8 on H100, roughly halving weight memory. That frees KV cache for ~2× concurrency at the same latency SLO, which lowers your $/1M tokens — and we'll gate it with an eval on your task before flipping it on."
