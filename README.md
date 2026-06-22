# Fireworks AI Field Engineer — Interview Prep Repo

A hands-on, runnable repository to prepare for the **AI Engineer (AI Native segment)** role. Every skill in the job description is mapped to a module with working code, study notes, and interview drills.

> Field Engineers are "the technical tip of the spear" — hands-on-keyboard building POCs/MVPs and production integrations, while holding executive-level architecture conversations. This repo is structured to make you fluent on **both** axes.

## How to use this repo

1. Read [`plan.md`](plan.md) for the study schedule and skill→module mapping.
2. Read [`architecture.md`](architecture.md) for the mental model (ASCII diagrams of the whole LLM-serving stack).
3. Work each module in order. Each has a `README.md` (concepts + interview Q&A) and runnable `.py` files.
4. Run the drills in [`10-interview-scenarios/`](10-interview-scenarios/) out loud.
5. Skim [`cheatsheet.md`](cheatsheet.md) right before an interview, and narrate [`capstone.md`](capstone.md) end-to-end.

## Skill → module map

| JD requirement | Module |
| --- | --- |
| Strong Python, debugging production code | [`01-python-fundamentals/`](01-python-fundamentals/) |
| LLM stack, inference trade-offs, model serving | [`02-llm-inference/`](02-llm-inference/) |
| Inference frameworks (vLLM, SGLang, TensorRT-LLM) | [`03-serving-frameworks/`](03-serving-frameworks/) |
| Fine-tuning workflows (SFT, DPO, RFT) | [`04-fine-tuning/`](04-fine-tuning/) |
| Evaluation methodology, production-quality metrics | [`05-evaluation/`](05-evaluation/) |
| Agentic systems, tool-use chains, function calling | [`06-agentic-systems/`](06-agentic-systems/) |
| Kubernetes, infrastructure engineering | [`07-kubernetes-infra/`](07-kubernetes-infra/) |
| Cloud (AWS/Azure/GCP), GPU deployment | [`08-cloud-gpu-deployment/`](08-cloud-gpu-deployment/) |
| Load tests, latency/throughput/cost baselines | [`09-load-testing/`](09-load-testing/) |
| Discovery calls, stakeholder mgmt, exec conversations | [`10-interview-scenarios/`](10-interview-scenarios/) |
| RAG / retrieval (knowledge, not fine-tuning) | [`11-rag/`](11-rag/) |
| Multimodal (vision/VLM) inference | [`12-multimodal/`](12-multimodal/) |
| Structured output / function-calling reliability | [`13-structured-output/`](13-structured-output/) |
| Prompt engineering + speculative decoding | [`14-prompt-and-speculative/`](14-prompt-and-speculative/) |

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt        # core (mostly stdlib; optional extras commented)
pip install -r requirements-dev.txt    # to run the test suite: pytest -q
```

Most modules are designed to run **without a GPU** and **without paid API keys** — they use simulations and mocks so you can study the patterns. Files that need heavyweight deps (torch, vllm) are clearly marked and degrade gracefully.

## Module index

- [`01-python-fundamentals/`](01-python-fundamentals/) — async, streaming, profiling, debugging
- [`02-llm-inference/`](02-llm-inference/) — tokenization, KV cache, batching, quantization
- [`03-serving-frameworks/`](03-serving-frameworks/) — vLLM, SGLang, OpenAI-compatible clients
- [`04-fine-tuning/`](04-fine-tuning/) — SFT (LoRA), DPO, RFT, data prep
- [`05-evaluation/`](05-evaluation/) — eval harness, LLM-as-judge, metrics
- [`06-agentic-systems/`](06-agentic-systems/) — function calling, ReAct, multi-tool agents
- [`07-kubernetes-infra/`](07-kubernetes-infra/) — GPU deployments, HPA, manifests
- [`08-cloud-gpu-deployment/`](08-cloud-gpu-deployment/) — AWS/Azure/GCP GPU serving
- [`09-load-testing/`](09-load-testing/) — async load generator, benchmark harness
- [`10-interview-scenarios/`](10-interview-scenarios/) — discovery, system design, behavioral
- [`11-rag/`](11-rag/) — chunking, vector search, retrieval metrics
- [`12-multimodal/`](12-multimodal/) — VLM serving, image-token cost
- [`13-structured-output/`](13-structured-output/) — JSON schema, constrained decoding
- [`14-prompt-and-speculative/`](14-prompt-and-speculative/) — prompt engineering, speculative decoding

## Top-level study aids

- [`capstone.md`](capstone.md) — one end-to-end mock engagement threading all 14 modules
- [`cheatsheet.md`](cheatsheet.md) — one-page formulas + decision ladders for last-minute review
- [`tests/`](tests/) — `pytest` unit + smoke tests (run `pytest -q`); CI in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

## Disclaimer

Code here is for **learning and interview prep**. It favors clarity over completeness and is not production-hardened.
