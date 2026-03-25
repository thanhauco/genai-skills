# Deployment patterns — self-managed vs managed, networking, multi-region

How to architect a GenAI deployment for a customer, and the trade-offs you'll defend in an exec conversation.

## Self-managed vs managed

```
 SELF-MANAGED (your GPUs, your cluster)        MANAGED (Fireworks / Bedrock / Foundry / Vertex)
 ──────────────────────────────────────        ────────────────────────────────────────────────
 + full control (model, kernels, quant)        + zero GPU ops, fast time-to-value
 + data stays in your VPC                       + elastic autoscaling, pay for use
 + cheapest at SUSTAINED high utilization       + provider tunes serving for you
 - you run GPU ops (drivers, scaling, oncall)   - less low-level control
 - idle GPUs = wasted money                      - data leaves your boundary (unless dedicated/region)
 - slow to provision GPU capacity                - per-token price at low/bursty volume
```

**Decision rule**: managed for bursty/variable traffic and teams that want to ship product; self-managed for sustained high volume, deep control, or strict data-in-VPC needs. Often **hybrid**: managed to start + prove value, self-host the steady-state high-volume core if economics favor it.

## Reference topology (managed inference behind the customer's app)

```
   customer app (their VPC)
        │ HTTPS (OpenAI-compatible)
        ▼
   API gateway / LB  ── auth, rate-limit, retries, routing
        │
        ▼
   inference (managed Fireworks  OR  self-hosted vLLM/SGLang on K8s)
        │
        ├── prefix cache (shared system prompt / RAG context)
        ├── autoscaling (queue-depth / GPU-util based)
        └── observability (latency, tokens, errors, cost)
        │
        ▼
   feedback loop -> eval gate (module 05) -> roadmap signals
```

## Networking & security (must-mention)

- **Private connectivity**: PrivateLink (AWS) / Private Endpoints (Azure) / Private Service Connect (GCP) so traffic doesn't traverse the public internet.
- **Keep traffic local**: same region/AZ as the app to cut latency + egress cost.
- **Secrets**: model/API keys in a secrets manager (not env files in git); least-privilege IAM.
- **Data residency / compliance**: pick region/provider that satisfies GDPR/HIPAA/SOC2; for strict cases run inference in the customer's cloud/region (dedicated deployments).
- **PII / logging**: be deliberate about what prompts/outputs you log; redact where required.

## Reliability & scale

- **Multi-AZ replicas** behind a load balancer; **multi-region** for DR / global latency (route by geo).
- **Autoscale on the right metric** (queue depth / GPU util, not CPU) and keep **warm headroom** because GPU pods cold-start slowly (module 07).
- **Graceful degradation**: fall back to a smaller/cheaper model or a managed endpoint under spike or outage.
- **Capacity planning**: H100 scarcity is real — reserve capacity or use managed elasticity for spikes.

## Cost levers (tie to cost_estimator.py)

1. **Quantization** (fp8/int4) → more KV cache → higher throughput → lower $/1M.
2. **Right-size the GPU** → don't run an 8B on 8×H100.
3. **Autoscale to traffic** → kill idle GPUs.
4. **Prefix caching** → cut repeated prompt cost (RAG/agents).
5. **Batch where latency allows** → offline jobs at high batch = cheapest tokens.
6. **Spot/preemptible** for fine-tuning/batch/RL rollouts (checkpoint + resume).
7. **Managed vs self-host** → compare on $/1M at the SLO **including** engineer time + idle.

## Fine-tuning / RL infra notes

- **Fine-tune jobs** are batch GPU workloads → great fit for spot + checkpointing.
- **RL rollouts** are expensive (lots of generation). Fireworks published cross-region rollouts with **98% sparse weight deltas** — only ship the policy delta to cut bandwidth/cost. Cite this as a production-systems insight for customers doing RL.
- Keep **fine-tune → eval → deploy** in one platform to compress iteration (the Fireworks thesis: the bottleneck is iteration speed, not the algorithm).

## The exec-level summary you should be able to give

> "We'll start you on managed inference to prove the latency and cost at your SLO with zero ops. We size the GPU + quantization to your traffic, autoscale on queue depth, and keep an eval gate so quality can't regress. If your steady-state volume grows enough that self-hosting wins on $/1M tokens — including ops time — we'll help you move the core in-house, keeping the same OpenAI-compatible API so nothing in your app changes."
