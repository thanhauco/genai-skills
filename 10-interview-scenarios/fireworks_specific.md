# Fireworks-specific prep

Know the company, their tech, and their public research well enough to have a peer conversation. Verify anything time-sensitive (valuation, partnerships, model lineup) before your interview — this reflects what's in the job posting and may have moved.

## The company (from the posting)

- **What they do**: generative AI **inference infrastructure** — highest-quality models with the fastest, most scalable inference. Independently benchmarked as the **leader in LLM inference speed**.
- **Stage**: **Series C, ~$4B valuation**; investors include **Benchmark, Sequoia, Lightspeed, Index, Evantic**.
- **Founders/team**: veterans of **Meta PyTorch** and **Google Vertex AI**.
- **Recent moves**: launched **Fireworks Training**; **partnered with Microsoft Azure AI Foundry**; publishing research from production systems.
- **Differentiated tech**: their own **function-calling** and **multimodal** models; optimized serving (e.g., FireAttention-style kernels).

## Their published research (be able to discuss each)

1. **"Frontier RL is cheaper than you think"** — cross-region RL rollouts using **98% sparse weight deltas**. Takeaway: the expensive part of RL is moving weights/data for rollouts; shipping only the sparse policy delta across regions slashes cost. Counters the "you need a mega-cluster" narrative. → Ties to module 04 (`rft_notes.md`) and module 08 (deployment patterns / RL infra).

2. **"Open source agents with frontier advisors"** — matching frontier agent performance via **training + harness engineering** on open models. Takeaway: harness/scaffolding + targeted training can close the gap to frontier on agentic tasks. → Ties to module 06 (agents) and module 04 (fine-tuning).

3. **"The fine-tuning bottleneck is not the algorithm"** — across dozens of customer engagements, **integration friction and iteration speed** stall teams, not the training algorithm. Takeaway: compress the data→train→eval→deploy loop. → Ties to module 04 (`decision_guide.md`, `data_prep.py`) and module 05 (eval).

> If you can connect each blog to a concrete thing you built in this repo, you'll sound like you already work there.

## How the repo maps to their stack

| Fireworks theme | Your prep |
| --- | --- |
| Fastest inference / serving | modules 02, 03, 09 (inference math, frameworks, load test) |
| Function-calling + agents | module 06 |
| Fine-tuning (SFT/DPO/RFT) + Training product | module 04 |
| Eval from production | module 05 |
| Deploy on GPUs / Azure Foundry partnership | modules 07, 08 |
| Field→roadmap loop | module 10 + architecture.md §5 |

## Talking points that resonate

- **OpenAI-compatible API** → low-risk migration: "point your base_url at Fireworks, keep your code" (module 03).
- **Perf-per-dollar** → you speak in **$/1M tokens at a p95 SLO**, not vibes (modules 08/09).
- **Iteration speed** → you optimize the **loop**, not just the model (module 04).
- **Honest positioning** → you know when managed beats self-host and vice-versa; you let the load test decide.
- **Production-systems mindset** → quantization, KV cache, prefix caching, continuous batching are tools you reach for by reflex.

## Smart questions to ask them

**Role / impact**
- "What does a great first 90 days look like for a Field Engineer in the AI-native segment?"
- "How embedded do you get — are we writing code in the customer's repo, or mostly advising?"

**Tech**
- "Where does FireAttention / your serving stack win most vs vLLM/TRT-LLM in practice?"
- "How do you decide what becomes a first-party model (function calling, multimodal) vs serving open weights?"
- "With the Azure AI Foundry partnership, how do enterprise engagements differ from AI-native ones?"

**Field → product**
- "How does a recurring field pain point actually become a roadmap item? What's the loop?"
- "Where do customer deployments most often stall, and how does the field team unblock them?"

**RL / research**
- "The sparse-weight-delta RL work — is that powering the Training product, and do FEs help customers run RFT?"

## Red flags to avoid in the interview

- Treating it as pure sales (it's hands-on-keyboard) **or** pure SWE (you must own the relationship).
- Quoting benchmark scores without the **customer-task eval** caveat.
- Recommending self-host or managed dogmatically instead of **measuring**.
- Not knowing the basic inference math (KV cache, prefill/decode) — that's table stakes here.

## Pre-interview checklist

- [ ] Re-skim all three blogs; have one connection from each to something you built here.
- [ ] Verify current facts (valuation, partnerships, latest models/launches).
- [ ] 6 STAR stories ready (behavioral_prep.md).
- [ ] Can whiteboard the serving stack + KV-cache math from memory.
- [ ] 5 sharp questions for them.
