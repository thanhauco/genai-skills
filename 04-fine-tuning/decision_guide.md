# Decision guide — prompt vs RAG vs SFT vs DPO vs RFT

A Field Engineer's job is to pick the **cheapest technique that hits the quality target**. Walk the ladder left-to-right and stop as soon as you meet the bar.

```
 Prompt eng. → Few-shot → RAG → SFT (LoRA) → DPO → RFT/RLHF
 cheap/instant ........................................ expensive/slow
```

## Decision table

| Symptom / goal | First try | If that fails |
| --- | --- | --- |
| Wrong format / structure | Prompt + constrained decoding | SFT on (input → correct format) |
| Missing fresh/proprietary facts | RAG (retrieval) | RAG + SFT on retrieval-augmented answers |
| Needs domain tone/behavior | Few-shot, system prompt | SFT |
| "Prefer style A over B" / safety/tone | DPO (preference pairs) | RFT |
| Correctness is checkable (code/math/tool) | SFT on traces | RFT with a verifiable reward |
| Latency/cost too high | Smaller model + SFT to recover quality | Distillation |
| Plateaued after SFT on a high-value task | DPO | RFT/RLHF |

## What each technique actually changes

- **Prompting / few-shot** — no weights change. Fastest iteration. Always start here.
- **RAG** — no weights change; injects context at inference. Best for *knowledge*, not *behavior*.
- **SFT** — changes weights to imitate gold input→output pairs. Best for *format + behavior + domain*.
- **DPO** — changes weights to prefer chosen over rejected. Best for *preferences/style/safety* when you lack gold answers.
- **RFT/RLHF** — changes weights to maximize a *reward*. Best when success is *measurable/checkable* or you need to push past SFT/DPO.

## Cost & iteration-speed reality (the Fireworks thesis)

> "The fine-tuning bottleneck is not the algorithm." The expensive parts are **data quality**, **iteration speed** (train→eval→inspect→repeat), and **integration** into the customer's pipeline + serving. Optimize the loop, not just the loss.

Rough relative cost / complexity:

```
 technique   data needed                 compute   infra      iteration speed
 prompt      none                         none      none       seconds
 RAG         a corpus + retriever         low       vector DB  minutes
 SFT-LoRA    100s-1000s gold pairs        low-med   1 GPU      minutes-hours
 DPO         100s-1000s preference pairs  med       1-N GPU    hours
 RFT/RLHF    reward model OR checker      high      N GPU      hours-days
```

## Model selection (the other half of "model strategy")

When advising on **which** model to tune/serve:
- **Capability vs cost vs latency**: pick the smallest model that clears the eval at the SLO.
- **Family fit**: instruction-tuned vs base; function-calling-tuned for agents; long-context for RAG; multimodal if needed.
- **License & data residency**: open-weight (Llama/Qwen/Mistral/DeepSeek) vs proprietary; where can data live.
- **Serve-ability**: does it have efficient kernels/quant support on the target GPU/framework?
- **Tune-ability**: LoRA support, adapter ecosystem, base availability.

## The FDE playbook for a fine-tuning engagement

1. **Define the metric first** (module 05). No metric → no fine-tuning.
2. **Baseline** the best prompt + RAG. Often good enough; document the gap.
3. **Curate data** (module: `data_prep.py`) — this is where most time goes.
4. **SFT-LoRA** → eval → inspect failures → iterate.
5. If preference/style gaps remain → **DPO**.
6. If correctness is checkable and value is high → **RFT**.
7. **Eval-gate + load-test** before production. Deploy adapter on the serving stack.
8. **Codify** the pipeline as a repeatable pattern; feed gaps to product.

---

## SFT vs DPO vs RFT — full side-by-side

| Dimension | **SFT** (Supervised Fine-Tuning) | **DPO** (Direct Preference Optimization) | **RFT / RLHF** (Reinforcement Fine-Tuning) |
| --- | --- | --- | --- |
| **What it optimizes** | Imitate gold outputs (max likelihood of the "right" answer) | Prefer chosen over rejected (relative log-prob, KL-anchored) | Maximize a reward signal (explore beyond demonstrations) |
| **Data needed** | (input → gold output) pairs | (prompt, chosen, rejected) preference triples | Prompts + a reward model **or** a programmatic checker |
| **Data volume (typical)** | 100s–1000s of clean examples | 100s–1000s of preference pairs | 1000s of prompts + a reliable reward/grader |
| **Where data comes from** | Human-written / curated / distilled from a stronger model | Human or AI A/B picks; thumbs up/down logs | Unit tests, schema/tool validators, math checkers, or a trained RM |
| **Compute / cost** | Low–med (LoRA: 1 GPU, minutes–hours) | Med (needs a reference model; 1–N GPU, hours) | High (rollouts/generation dominate; N GPU, hours–days) |
| **Infra complexity** | Lowest | Low–medium (no RL loop, no reward model) | Highest (sampling loop, reward, KL control, stability) |
| **Iteration speed** | Fast | Medium | Slow |
| **Stability / risk** | Very stable; can overfit/parrot if data is narrow | Stable; sensitive to `beta` and pair quality | Trickiest: reward hacking, collapse, KL drift |
| **Best for** | Format, structure, domain tone, behavior cloning | Style/tone/safety, "prefer A over B", subtle quality | Checkable correctness (code/math/tools), pushing past a plateau |
| **Can it exceed your demos?** | No — bounded by the gold data | Somewhat — sharpens preferences | Yes — RL can discover better strategies |
| **Main failure mode** | Garbage-in (bad/leaky data) → garbage-out | Noisy/contradictory preference pairs | Reward hacking; gaming the metric |
| **Key knob** | LR, epochs, LoRA rank | `beta` (how far policy may drift from reference) | Reward design + KL coefficient |
| **Fireworks product** | Fine-tuning (SFT) | Preference tuning | RFT (bring prompts + grader) |
| **Eval gate** | Required | Required | Required + watch for reward hacking |

> One-liner per method:
> - **SFT** = "show it the right answers."
> - **DPO** = "show it which of two answers is better."
> - **RFT** = "let it try, and score the result."

## Decision tree (run top-to-bottom, stop at the first match)

```
                       ┌───────────────────────────────────────────────┐
                       │  Have you exhausted prompting + RAG + few-shot?│
                       └───────────────┬───────────────────────────────┘
                                   no  │  yes
                       ┌───────────────┘
                       ▼
        Do that first (cheapest, fastest). ──────────────► still not good enough?
                                                                     │
                                                                     ▼
                          ┌──────────────────────────────────────────────────────┐
                          │ Is the gap KNOWLEDGE (facts) or BEHAVIOR (format/skill)?│
                          └───────────────┬──────────────────────┬────────────────┘
                              knowledge    │           behavior   │
                                           ▼                       ▼
                                  RAG / better retrieval   ┌──────────────────────────────┐
                                  (don't fine-tune facts)  │ Do you have GOLD outputs       │
                                                           │ (known-correct answers)?       │
                                                           └──────────┬─────────────────────┘
                                                            yes       │      no
                                                              ▼       │
                                                        ┌──────────┐  │
                                                        │   SFT    │  │
                                                        │ (LoRA)   │  │
                                                        └────┬─────┘  │
                                                             │        ▼
                                              still a style/ │   ┌─────────────────────────────┐
                                              preference gap?│   │ Do you have PREFERENCE signal│
                                                             ▼   │ (A chosen over B)?           │
                                                       ┌─────────┴───┐  yes      no             │
                                                       │ Is success   │   ▼        ▼             │
                                                       │ CHECKABLE    │ ┌─────┐  collect prefs / │
                                                       │ (tests/math/ │ │ DPO │  gold first ─────┘
                                                       │  schema)?    │ └─────┘
                                                       └──────┬───────┘
                                                       yes    │   no
                                                        ▼     ▼
                                                   ┌─────────┐  stick with SFT/DPO
                                                   │  RFT    │  (RFT needs a reward
                                                   │ (RLVR)  │   you can compute)
                                                   └─────────┘
```

## Worked examples (pick the method + justify)

| Customer ask | Method | Why |
| --- | --- | --- |
| "Always emit valid JSON for our schema" | Prompt + constrained decoding → **SFT** | Deterministic format; gold outputs are easy to write. RFT is overkill. |
| "Answer using our latest internal docs" | **RAG** (not fine-tuning) | It's a knowledge gap; tuning bakes in stale facts. |
| "Sound like our brand voice" | **SFT** then **DPO** | SFT for the baseline voice; DPO to prefer on-brand over off-brand when you lack a single gold answer. |
| "Be safer / refuse better" | **DPO** | You have (safe, unsafe) preference pairs; no single gold answer. |
| "Generate code that passes our tests" | **RFT (RLVR)** | Success is checkable (run the tests) → reward at scale; SFT on traces first. |
| "Solve our math/agent tasks better than demos" | **SFT → RFT** | SFT to bootstrap; RFT to explore beyond the demonstrations. |
| "Cheaper model, same quality" | **SFT** (+ distillation) | Tune a smaller model on the big model's outputs; eval-gate. |

## Gotchas to mention in an interview

- **Don't fine-tune for knowledge** — that's RAG's job. Tuning teaches *behavior*, not fresh facts.
- **DPO pair quality > quantity** — contradictory/noisy preferences poison it; `beta` controls drift from the reference model.
- **RFT reward design is the whole game** — vague rewards get hacked (length/format exploits). Mix correctness + format + safety, and inspect samples, not just the reward curve.
- **Always eval-gate** — benchmark scores ≠ the customer's production metric (module 05). And keep a held-out set RL can't game.
- **The real bottleneck** (Fireworks' thesis): data quality + iteration speed + integration, **not** the algorithm.
