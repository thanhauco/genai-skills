# Study Plan — Fireworks AI Field Engineer

A focused plan to cover every skill in the job description. Adjust pacing to your timeline; the ordering builds dependencies bottom-up (language → inference → serving → tuning → eval → agents → infra → field).

## The role in one sentence

> Embed with high-velocity AI-native customers, build POCs/MVPs in **their** codebase, architect + tune inference deployments to hit latency/throughput/cost targets, guide model & fine-tuning strategy, and feed signals back into the product — all while holding executive conversations.

## Skill coverage checklist

- [ ] **Python** — async, streaming, profiling, debugging production code
- [ ] **LLM inference** — prefill/decode, KV cache, continuous batching, quantization, latency vs throughput
- [ ] **Serving frameworks** — vLLM, SGLang, TensorRT-LLM; shapes, TP/PP, quant configs
- [ ] **Fine-tuning** — SFT (LoRA/QLoRA), DPO, RFT/RLHF; data prep; when to tune vs prompt
- [ ] **Evaluation** — task metrics, LLM-as-judge, golden sets, production metrics (not just benchmarks)
- [ ] **Agentic systems** — function calling, tool-use chains, ReAct, multi-tool orchestration
- [ ] **Kubernetes** — GPU scheduling, deployments, HPA/KEDA, resource requests/limits
- [ ] **Cloud GPU** — AWS/Azure/GCP GPU instances, SageMaker/Foundry/Vertex, networking, cost
- [ ] **Load testing** — async load gen, p50/p95/p99, TTFT, TPOT, tokens/sec, $/1M tokens
- [ ] **Field skills** — discovery, stakeholder mgmt, exec comms, product feedback loops

## Suggested sequencing (10 work blocks)

| Block | Module | Goal you can demo |
| --- | --- | --- |
| 1 | `01-python-fundamentals` | Write an async streaming client; profile a hot loop |
| 2 | `02-llm-inference` | Explain KV cache memory math; estimate batch size for a GPU |
| 3 | `03-serving-frameworks` | Stand up a mental model of vLLM vs SGLang vs TRT-LLM trade-offs |
| 4 | `04-fine-tuning` | Decide SFT vs DPO vs RFT for a customer; sketch a LoRA run |
| 5 | `05-evaluation` | Design an eval harness for a customer's task |
| 6 | `06-agentic-systems` | Build a tool-use chain with function calling |
| 7 | `07-kubernetes-infra` | Write a GPU vLLM Deployment + HPA |
| 8 | `08-cloud-gpu-deployment` | Pick GPU + region + cloud for a workload and justify cost |
| 9 | `09-load-testing` | Run a load test, produce a latency/throughput/cost baseline |
| 10 | `10-interview-scenarios` | Run a mock discovery call + system design out loud |

## Daily rhythm (per block)

1. Read the module `README.md` (concepts + interview Q&A).
2. Run the code; break it, fix it, instrument it.
3. Answer the 5 interview questions out loud, timed (~2 min each).
4. Write one "customer story": a problem → your approach → the trade-off you made.

## The interview loop you're prepping for (typical FDE/AI-FE)

```
Recruiter screen ─► Hiring-manager / role fit ─► Technical: coding (Python)
        │                                              │
        ▼                                              ▼
  Technical: LLM/systems design ─► Customer-facing / discovery role-play ─► Values / final
```

Map each stage to a module set:
- **Coding** → `01`, `06`
- **Systems design** → `02`, `03`, `07`, `08`, `09`
- **ML depth** → `04`, `05`
- **Customer / discovery role-play** → `10`

## "Prove it" deliverables (build these to stand out)

1. A one-page **deployment sizing calculator** (extend `02-llm-inference/kv_cache_calculator.py`).
2. A **load-test report** for a mock endpoint with p50/p95/p99 + $/1M tokens (`09-load-testing`).
3. A **discovery → architecture** doc for a fictional AI-native customer (`10-interview-scenarios`).

## Progress log

Track commits per module; this repo commits incrementally after each module is completed.
