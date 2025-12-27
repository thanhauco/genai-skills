# 02 — LLM Inference Concepts

This is the core of the Field Engineer role: "architect the inference foundations… size deployments… run load tests and establish latency, throughput, and cost baselines… determine optimal shapes, quantization configs, and serving patterns."

You must be able to do the **capacity and latency math on a whiteboard**.

## Files

| File | What it teaches |
| --- | --- |
| [`tokenization_demo.py`](tokenization_demo.py) | Tokens vs words/chars; why pricing & context are per-token |
| [`kv_cache_calculator.py`](kv_cache_calculator.py) | KV-cache memory math → max batch size / concurrency on a GPU |
| [`batching_simulation.py`](batching_simulation.py) | Static vs continuous batching; the throughput/latency trade-off |
| [`roofline_latency.py`](roofline_latency.py) | Prefill (compute-bound) vs decode (bandwidth-bound); back-of-envelope TPOT |
| [`quantization.md`](quantization.md) | fp16/bf16/fp8/int8/int4, GPTQ/AWQ; quality vs memory vs speed |

## Run

```bash
python 02-llm-inference/tokenization_demo.py
python 02-llm-inference/kv_cache_calculator.py
python 02-llm-inference/batching_simulation.py
python 02-llm-inference/roofline_latency.py
```

`tokenization_demo.py` uses `tiktoken` if installed, else falls back to a word/char approximation. The rest are stdlib-only.

## The mental model (memorize this)

```
Total latency ≈ TTFT + (output_tokens − 1) × TPOT
TTFT  ← prefill, compute-bound, grows with prompt length
TPOT  ← decode, memory-bandwidth-bound, ~constant per token
Throughput ← gated by KV-cache memory (how many requests fit) and GPU utilization
```

## Capacity math (the one calculation to nail)

KV cache bytes per token:
```
kv_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
```
Per request: `kv_per_token × (prompt_len + gen_len)`.
Max concurrent requests ≈ `(HBM − weights − overhead) / per_request_kv`.

**Why you care:** quantizing weights frees HBM → more KV cache → more concurrent requests → higher throughput → lower $/1M tokens. That's the lever you sell.

## Interview Q&A

1. **Why is decode memory-bandwidth-bound but prefill compute-bound?**
   - Prefill processes all prompt tokens at once → large matmuls → compute-bound. Decode generates one token per step, reloading weights from HBM each step with tiny matmuls → you're bottlenecked moving weights, not on FLOPs. This is why batching helps decode (amortize the weight read over many requests).

2. **A customer wants lower TTFT for a RAG app with 8k-token prompts. Options?**
   - Prefix caching / KV reuse for the shared system+context prefix; chunked prefill; speculative decoding helps TPOT not TTFT; smaller/quantized model; TP to add compute; separate prefill/decode (disaggregation) so long prefills don't block decodes.

3. **They want higher throughput at fixed latency SLO. Levers?**
   - Continuous batching (already in vLLM/SGLang), quantization to grow KV cache, raise max-num-seqs until you hit the SLO p95, TP across GPUs, right-size max context, and route by request shape.

4. **How does quantization affect quality and when is it safe?**
   - fp8/int8 weight-only is usually near-lossless for many models; int4 (AWQ/GPTQ) trades a little quality for big memory wins. Always **gate with an eval** on the customer's task (module 05) — benchmark scores ≠ their production metric.

5. **Estimate: 13B model, fp16, on one 80GB H100 — roughly how many concurrent 2k-token requests?**
   - Weights ≈ 26GB. Leaves ~50GB for KV after overhead. With kv_per_token ≈ ~1MB-scale depending on arch, you get tens of concurrent long requests. Run `kv_cache_calculator.py` to make this concrete and to show you reason with the formula, not a memorized number.
