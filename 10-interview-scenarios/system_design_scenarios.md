# System design scenarios (LLM infra) — with model answers

Practice these out loud in 20–30 minutes each. Structure every answer the same way: **clarify → traffic profile → architecture → bottleneck math → tune to SLO → $/1M → eval + feedback loop.** Use the diagrams in [architecture.md](../architecture.md).

---

## Scenario 1 — "Make our RAG chatbot fast and cheap"

> A customer's support chatbot uses an 8k-token system+context prompt (RAG) per request. p95 TTFT is 1.5s; they want <400ms. Traffic: 50 RPS peak, ~200-token answers.

**Clarify**: shared prefix across requests? latency vs cost priority? current model/GPU/framework? quality bar?

**Diagnosis**: 8k-token prompts → prefill dominates TTFT. The system+context prefix is largely **shared** across requests.

**Plan**:
1. **Prefix caching / RadixAttention** (SGLang or vLLM `--enable-prefix-caching`) → reuse the shared-prefix KV → massive TTFT cut. This is the single biggest lever here.
2. **Chunked prefill** so long prompts don't block decodes.
3. **fp8** weights → faster + more KV headroom (eval-gate).
4. Consider a **smaller model** if eval allows; RAG often needs less raw capability.
5. **Load test** the real traffic; tune `max-num-seqs` to hold p95.

**Numbers**: convert to $/1M tokens at the SLO. Report p95 TTFT before/after, throughput, cost.

**Gate**: eval on their support task (groundedness + answer correctness) so caching/quant didn't hurt quality.

**Outcome sentence**: "Prefix caching the shared context plus fp8 gets p95 TTFT under 400ms at ~40% lower $/1M, quality held at X% on your eval."

---

## Scenario 2 — "Serve Llama-70B with tight latency on our AWS account"

> Self-hosted, p95 E2E target, moderate volume, must stay in their VPC.

**Clarify**: exact SLO, RPS + prompt/output lengths, budget, single vs multi-region, fine-tuned or base?

**Architecture**:
- **GPU/shape**: 70B fp16 ≈ 140GB → needs TP across GPUs. fp8 ≈ 70GB → 1–2×H100. Pick p5 (H100) for latency; TP within one node (NVLink).
- **Framework**: vLLM to start (fast to stand up); TRT-LLM if they need the last 20–40% and volume is stable.
- **Orchestration**: EKS, GPU node pool, the Deployment + HPA from module 07; private networking (PrivateLink), data stays in VPC.
- **Scale**: autoscale on queue depth; warm headroom for cold starts.

**Bottleneck math**: KV cache budget after weights → max concurrency (module 02). fp8 doubles KV headroom.

**Tune + price**: load test → p95 + tok/s → $/1M. Compare self-host vs **Fireworks managed** on $/1M *including ops time*. Be honest where managed wins.

**Gate**: eval + load test as the regression gate before prod.

---

## Scenario 3 — "Our agent is slow and expensive"

> A multi-step tool-using agent: 6–10 model calls per task, same big system+tools prompt every call.

**Diagnosis**: every step re-sends the identical system+tools prefix → repeated prefill cost; sequential tool calls add latency.

**Plan**:
1. **Prefix caching** for the static system+tools prompt (huge — it's identical every step).
2. **Tiered models**: cheap model for routing/simple turns, big model only when needed (module 06).
3. **Parallelize independent tool calls** (async, module 01).
4. **Bound context**: summarize history; cap steps + tokens (loop/budget guards).
5. **Cache deterministic tool results.**
6. Consider a **function-calling-tuned model** (Fireworks) for reliable, compact tool calls.

**Measure**: per-task latency, total tokens, $/task; eval the **trajectory** (right tools/order, recovery) not just final answer.

---

## Scenario 4 — "Should we fine-tune, and how?"

> Customer wants the model to reliably output their structured format and domain tone; quality is inconsistent with prompting.

**Walk the ladder** (module 04): prompt + constrained decoding → if still inconsistent → **SFT-LoRA** on a few hundred clean (input → correct output) pairs. DPO if it's a preference/tone gap with no gold answers. RFT only if correctness is checkable and value is high.

**Emphasize the real bottleneck**: **data quality + iteration speed**, not the algorithm (the Fireworks thesis). Show the data-prep + eval-gate workflow (modules 04/05).

**Deliverable**: a LoRA run + a frozen eval set + a CI gate; deploy the adapter on the serving stack (one base, swappable adapters).

---

## Scenario 5 — "Design a multi-tenant inference platform"

> Many customer apps, different models, varying SLAs, cost attribution.

**Key decisions**:
- **Isolation**: per-tenant namespaces/quotas; dedicated vs shared replicas by SLA tier.
- **Model multiplexing**: one base + **LoRA adapters per tenant** (cheap multi-tenancy); hot-swap adapters.
- **Routing**: gateway routes by tenant/model; rate-limit + auth per tenant.
- **Autoscaling**: per-pool on queue depth; bin-pack small models on L4/L40S, big ones on H100.
- **Cost attribution**: meter tokens per tenant → $/1M per tenant (module 09/08).
- **Fairness**: prevent one tenant's burst from starving others (per-tenant concurrency caps).
- **Observability**: per-tenant latency/error/cost dashboards; eval gates per model.

**Trade-off to articulate**: shared infra = better utilization + lower cost, but needs strong isolation + fairness; dedicated = simpler SLAs but more idle GPU.

---

## How you're graded (give yourself these points)

- [ ] Asked clarifying questions before designing.
- [ ] Started from the **traffic profile + SLO**, not the model.
- [ ] Did the **bottleneck math** (KV cache / prefill-decode), not hand-waving.
- [ ] Tuned to the **p95 SLO**, then expressed it as **$/1M tokens**.
- [ ] Added an **eval gate** and a **feedback-to-product** loop.
- [ ] Was **honest** about when managed (Fireworks) vs self-host wins.
