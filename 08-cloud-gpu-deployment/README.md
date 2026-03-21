# 08 — Cloud GPU Deployment (AWS / Azure / GCP)

The JD: "Experience with cloud infrastructure (AWS, Azure, GCP) and deploying models on GPU infrastructure" and (preferred) "Experience with hyperscaler AI platforms (Azure AI Foundry, AWS Bedrock/SageMaker, GCP Vertex)." Fireworks just **partnered with Microsoft Azure Foundry**, so this is timely.

## Files

| File | What it teaches |
| --- | --- |
| [`gpu_selection.md`](gpu_selection.md) | Pick the right GPU (H100/A100/L40S/L4/MI300) for a workload + the instance names per cloud |
| [`hyperscaler_platforms.md`](hyperscaler_platforms.md) | SageMaker/Bedrock vs Azure AI Foundry vs Vertex — what they are, when to use |
| [`cost_estimator.py`](cost_estimator.py) | Estimate $/1M tokens across GPUs/clouds from throughput; break-even vs serverless |
| [`deployment_patterns.md`](deployment_patterns.md) | Self-managed vs managed, networking, data residency, multi-region, the Fireworks fit |

## Run

```bash
python 08-cloud-gpu-deployment/cost_estimator.py
```

`cost_estimator.py` is stdlib-only. Prices are **illustrative placeholders** — always confirm current pricing; the point is the *method*, not the exact numbers.

## The decision an FDE makes

```
 workload (model size, traffic, latency SLO, budget, data rules)
        │
        ▼
 GPU choice ──► instance/region ──► self-managed vs managed ──► $/1M tokens
 (gpu_selection)   (per cloud)        (Fireworks vs DIY)        (cost_estimator)
```

## Quick GPU intuition (memorize)

| GPU | HBM | Sweet spot |
| --- | --- | --- |
| **H100 / H200** | 80 / 141 GB | Largest models, lowest latency, fp8; the throughput king |
| **A100** | 40 / 80 GB | Still great; cheaper than H100; broad availability |
| **L40S** | 48 GB | Cost-effective mid-size serving + fine-tuning |
| **L4** | 24 GB | Small models, high-volume cheap inference, edge |
| **AMD MI300X** | 192 GB | Huge HBM → fewer GPUs for big models; ROCm path |

More on multi-GPU + names per cloud in `gpu_selection.md`.

## Hyperscaler platforms in one line each

- **AWS Bedrock** — serverless managed models via API (no infra). **SageMaker** — train/deploy your own on managed GPU endpoints.
- **Azure AI Foundry** — model catalog + managed deployment + agents on Azure; **Fireworks partners here**.
- **GCP Vertex AI** — model garden + managed training/serving endpoints on GCP.
- **Fireworks** — managed, ultra-fast inference + fine-tuning across clouds; the pitch is *performance + simplicity* without owning GPU ops.

## Interview Q&A

1. **A customer on AWS wants to self-host Llama-70B with tight latency. What do you propose?**
   - p4d/p5 (A100/H100) instances, TP across GPUs on one node (NVLink), vLLM or TRT-LLM, fp8 to fit + speed up, EKS for orchestration. Then load-test to a $/1M baseline and compare against Fireworks managed — often Fireworks wins on perf-per-dollar without the ops.

2. **When managed (Bedrock/Foundry/Vertex/Fireworks) vs self-managed?**
   - Managed when the team wants to ship product, not run GPU ops; variable/bursty traffic; fast time-to-value. Self-managed when they need deep control, special models/kernels, data must stay in their VPC, or sustained high volume where owning hardware amortizes.

3. **How do data-residency / VPC requirements change the design?**
   - Keep inference in the customer's region/VPC, use private endpoints, no data egress; pick a provider/region that satisfies compliance. Fireworks supports dedicated/region options; for strict cases, deploy in their cloud.

4. **Spot/preemptible GPUs — when?**
   - Great for **fine-tuning / batch / RL rollouts** (checkpoint + resume). Risky for low-latency serving (preemption → dropped capacity) unless you design for it with on-demand baseline + spot burst.

5. **How do you turn a GPU hourly rate into a price the customer cares about?**
   - Load-test to get sustained throughput (tok/s), then `$/1M tokens = hourly_rate / (tok/s × 3600) × 1e6`. Compare configs/clouds/managed on that single number at the latency SLO. See `cost_estimator.py`.
