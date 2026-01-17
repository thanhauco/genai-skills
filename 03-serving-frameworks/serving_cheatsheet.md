# Serving cheat sheet — launch commands & knobs

Quick reference for standing up and tuning open-source serving. Commands are illustrative; check the version's docs for exact flags.

## vLLM — OpenAI-compatible server

```bash
pip install vllm

# Single GPU, 8B
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching

# 70B across 4 GPUs with fp8
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192
```

Key vLLM knobs:
- `--tensor-parallel-size N` — shard model across N GPUs (fit + latency).
- `--pipeline-parallel-size N` — stage layers across GPUs/nodes.
- `--max-num-seqs` — max concurrent sequences (throughput ↔ latency).
- `--max-num-batched-tokens` — token budget per step (prefill batching).
- `--max-model-len` — cap context to control KV-cache allocation.
- `--quantization {fp8,awq,gptq,...}` — shrink weights, grow KV budget.
- `--enable-prefix-caching` — reuse shared prompt prefixes.
- `--gpu-memory-utilization 0.0–1.0` — HBM fraction for the KV cache.
- `--kv-cache-dtype fp8` — quantize KV cache for more concurrency.
- `--speculative-model / --num-speculative-tokens` — speculative decoding.

## SGLang — OpenAI-compatible server

```bash
pip install "sglang[all]"

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 \
  --tp 1 \
  --enable-torch-compile

# 70B, TP=4, fp8
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp 4 \
  --quantization fp8 \
  --mem-fraction-static 0.85
```

SGLang highlights:
- **RadixAttention** automatically shares prefix KV across requests (great for RAG/agents/few-shot).
- Strong **structured/constrained decoding** (JSON, regex, grammar) support.
- `--tp`, `--quantization`, `--mem-fraction-static`, `--context-length`.

## TensorRT-LLM — compiled engines (NVIDIA only)

```bash
# 1) Build an engine (per model + precision + shape) — the heavy step
trtllm-build \
  --checkpoint_dir ./ckpt/llama3-8b \
  --gemm_plugin auto \
  --max_batch_size 64 \
  --max_input_len 4096 \
  --max_seq_len 8192 \
  --output_dir ./engines/llama3-8b-fp8

# 2) Serve (often behind Triton or the TRT-LLM OpenAI server)
```

TRT-LLM reality check:
- Engines are **shape- and precision-specific**: changing TP, max batch, context, or quant means a **rebuild**.
- Highest perf on NVIDIA, highest ops cost. Worth it at stable high volume.

## Smoke test any of them (same client)

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"stream":true}'
```

## Tuning loop (the field workflow)

1. **Baseline** fp16, single shape, `--max-num-seqs` modest.
2. **Load test** (module 09) against the customer's traffic profile → record p50/p95 TTFT, TPOT, tok/s, $/1M.
3. **Turn one knob at a time**: TP, quant, max-num-seqs, prefix cache.
4. **Re-measure** + **eval-gate** (module 05). Keep what improves p95 within the SLO.
5. **Codify** the winning shape as a reusable deployment pattern (JD: "codify repeatable deployment patterns").
