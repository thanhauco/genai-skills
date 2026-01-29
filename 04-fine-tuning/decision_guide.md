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
