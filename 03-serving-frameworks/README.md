# 03 — Serving Frameworks (vLLM, SGLang, TensorRT-LLM)

The JD explicitly lists: "Deploy and validate new model families on inference frameworks (vLLM, SGLang), determining optimal shapes, quantization configs, and serving patterns." Preferred: "vLLM, SGLang, TensorRT-LLM and tuning deployments for real workloads."

You don't need to memorize every flag — you need to **reason about the trade-offs** and **speak the common API** (they're all OpenAI-compatible).

## Files

| File | What it teaches |
| --- | --- |
| [`openai_compatible_client.py`](openai_compatible_client.py) | The one client that talks to Fireworks, vLLM, SGLang, TGI — with a mock fallback |
| [`comparison.md`](comparison.md) | vLLM vs SGLang vs TensorRT-LLM vs Fireworks — when to pick what |
| [`serving_cheatsheet.md`](serving_cheatsheet.md) | Launch commands, key knobs (TP, max-num-seqs, quant, prefix cache) |
| [`mock_inference_server.py`](mock_inference_server.py) | A tiny stdlib HTTP server that mimics `/v1/chat/completions` (stream + non-stream) for local testing |

## Run

```bash
# Terminal A: start the mock server (stdlib only, no GPU/model)
python 03-serving-frameworks/mock_inference_server.py

# Terminal B: hit it with the OpenAI-compatible client
python 03-serving-frameworks/openai_compatible_client.py --base-url http://localhost:8000/v1
```

The client also runs **without** a server (built-in mock) so you can study the shape offline.

## The one API to know (OpenAI-compatible)

vLLM, SGLang, TGI, and Fireworks all expose `/v1/chat/completions`. Learn it once:

```python
client.chat.completions.create(
    model="...",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=256,
    temperature=0.2,
    stream=True,         # SSE token stream
    tools=[...],         # function calling (module 06)
)
```

## How to pick (the 30-second version)

- **vLLM** — the default open-source workhorse. PagedAttention + continuous batching, huge model coverage, easy to stand up, OpenAI-compatible server. Great first choice for most customers.
- **SGLang** — RadixAttention (automatic prefix-cache sharing across requests), excellent for **structured output, agents, and heavy shared-prefix** workloads; strong throughput. Reach for it when prompts share large prefixes (RAG, few-shot, agent loops).
- **TensorRT-LLM** — NVIDIA's compiled-kernel engine. Highest performance on NVIDIA when you invest in building engines per model/shape/precision; less flexible, more ops overhead. Reach for it at scale when squeezing the last 20–40% on NVIDIA matters.
- **Fireworks** — managed, tuned serving (FireAttention etc.), fast paths for function-calling/multimodal, fine-tuning + deployment in one platform. The pitch: customer gets TRT-class performance without the ops burden, plus fine-tuning and scaling.

## Interview Q&A

1. **A customer runs an agent with a 4k-token shared system prompt across thousands of calls. Framework?**
   - SGLang's RadixAttention auto-shares that prefix's KV across requests → big TTFT + throughput win. vLLM has prefix caching too; enable it. Quantify with a load test (module 09).

2. **What knobs do you tune first on vLLM?**
   - `--tensor-parallel-size` (fit + speed), `--max-num-seqs` and `--max-num-batched-tokens` (throughput vs latency), `--quantization`, `--max-model-len` (don't over-allocate KV), `--enable-prefix-caching`, `--gpu-memory-utilization`.

3. **When is TensorRT-LLM worth the complexity?**
   - High, stable volume on NVIDIA where the per-engine build cost amortizes and you need max tok/s or lowest latency. Not for fast-moving experimentation across many models.

4. **How do you validate a new model family on a framework?**
   - Stand it up, smoke-test correctness, pick a shape (TP/quant), then run the load test against the customer's traffic profile and gate on an eval. Document the optimal shape as a repeatable pattern (the JD's "codify repeatable deployment patterns").

5. **Same client code across all of these — why does that matter to a customer?**
   - OpenAI-compatible APIs mean migration is low-risk: point the base URL at Fireworks, keep the code. You de-risk the switch and can A/B in production.
