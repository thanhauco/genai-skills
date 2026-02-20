# Production evaluation — offline gates + online monitoring

Benchmarks tell you general capability. **Production evals** tell you whether the system works for *this* customer's task, at *their* latency/cost SLO, on *their* edge cases. This is the discipline that makes your technical recommendations credible.

## Offline vs online

```
 OFFLINE (pre-deploy gate)                 ONLINE (post-deploy monitoring)
 ─────────────────────────                 ──────────────────────────────
 frozen golden set                         sampled live traffic
 deterministic + LLM-judge scorers         lightweight judges + human spot checks
 runs in CI on every change                runs continuously
 BLOCKS deploy if it regresses             ALERTS on drift / quality drop
```

You need both. Offline catches regressions before they ship; online catches distribution shift and real-world failures the golden set didn't anticipate.

## Building a golden set (the asset that compounds)

- **Source from reality**: sample real or representative inputs, including the long tail and known failure cases.
- **Freeze + version it**: a moving target can't gate anything. Version it like code.
- **No leakage**: golden-set examples must never appear in fine-tuning data.
- **Label carefully**: gold outputs, or rubric + reference, or a checker (tests/schema).
- **Size for confidence**: report a confidence interval (see `metrics.py wilson_interval`). 20 examples ≈ a guess; hundreds give tight intervals.

## Scorer selection (prefer deterministic)

| Task type | Best scorer |
| --- | --- |
| Classification / extraction | exact / F1 |
| Structured output | JSON-schema validation |
| Code generation | run unit tests → pass@k |
| Math / factual short answer | exact / numeric match |
| Tool / function calling | does the call parse + validate + execute |
| Open-ended quality | LLM-as-judge (pairwise + rubric) + human calibration |
| Retrieval (RAG) | recall@k, MRR, faithfulness/groundedness |

## The CI eval gate (what you hand the customer)

```
 PR changes model / prompt / quant / adapter
            │
            ▼
   run golden-set eval  ──►  accuracy >= threshold?  ──no──►  BLOCK
            │                p95 latency <= SLO?      ──no──►  BLOCK
            │                cost/1M <= budget?       ──no──►  BLOCK
            ▼ yes
        deploy + log
```

This is how you stop quality regressions *after* you've moved on — the eval becomes the guardrail.

## Online monitoring & drift

- **Log everything**: inputs, outputs, latencies, token counts, model/version.
- **Sample + label**: a small % judged by LLM-as-judge and periodically by humans.
- **Watch for drift**: input distribution shift, quality decay, new failure clusters.
- **Feedback loop**: failures become new golden-set cases and product signals (the JD's "feed customer signals back into the product roadmap").

## RAG / agent eval specifics

- **RAG**: separate **retrieval** quality (recall@k, MRR) from **generation** quality (groundedness/faithfulness, answer correctness). A bad answer is often a retrieval problem, not a model problem.
- **Agents**: eval the **trajectory**, not just the final answer — did it pick the right tools, in the right order, recover from errors, and stay within step/cost budget? (See module 06.)

## Common eval anti-patterns (call these out)

- Reporting only the **mean** — always show p95/p99 and failure buckets.
- Evaluating on **benchmark** datasets instead of the customer's task.
- **Leakage** between eval and fine-tuning data → inflated, dishonest numbers.
- **LLM-judge without bias controls** (position/verbosity/self-preference).
- No **online** eval → silent drift after handoff.
- Optimizing quality while ignoring **latency and cost** — the customer buys the joint trade-off.
