# 05 — Evaluation Frameworks

The JD: "Design and implement evaluation frameworks that measure production-quality metrics, not just benchmark scores." This is the gate on every other decision — quantization, model choice, fine-tuning — only matters if you can *measure* whether quality held.

## Files

| File | What it teaches |
| --- | --- |
| [`eval_harness.py`](eval_harness.py) | A minimal task-eval harness: dataset → run → score → report, with exact/regex/semantic-ish scorers |
| [`llm_as_judge.py`](llm_as_judge.py) | Pairwise + rubric LLM-as-judge pattern (mock judge), with bias mitigations |
| [`metrics.py`](metrics.py) | Quality + system metrics: accuracy, pass@k, p50/p95/p99, cost/1M tokens |
| [`production_eval.md`](production_eval.md) | Offline vs online eval, golden sets, regression gates, drift |

## Run

```bash
python 05-evaluation/eval_harness.py
python 05-evaluation/llm_as_judge.py
python 05-evaluation/metrics.py
```

All stdlib-only (mocked model + mocked judge), so they run anywhere.

## The core idea: benchmark scores ≠ production quality

```
 MMLU / HELM / leaderboard  →  tells you general capability
 YOUR task eval             →  tells you if it works for THIS customer
                                (format, tools, latency, cost, edge cases)
```

A Field Engineer always builds the **second** one. The first is marketing; the second is the gate.

## Eval design checklist (say this in an interview)

1. **Define the production metric** with the customer (exact-match? JSON-valid? pass-tests? human-rated helpfulness? p95 latency?).
2. **Build a golden set** from real/representative inputs; freeze it; no train leakage.
3. **Pick scorers**: deterministic where possible (exact/regex/schema/unit-test), LLM-judge only where you must (open-ended quality).
4. **Report distribution, not just mean**: p50/p95/p99, failure buckets, confidence interval.
5. **Gate changes**: any model swap / quant / fine-tune must pass the eval + the load test before prod.
6. **Close the loop online**: log production traffic, sample + label, watch for drift.

## Scorer ladder (prefer deterministic)

```
 exact match  →  regex/structured  →  unit-test/checker  →  embedding similarity  →  LLM-as-judge
 cheap, objective ........................................................  expensive, subjective
```

## Interview Q&A

1. **A customer asks "is the smaller/cheaper model good enough?" How do you answer?**
   - Not with a leaderboard — with **their** eval. Run both models on the frozen golden set, compare the production metric at the latency/cost SLO, report p95 + failure buckets, and recommend the cheapest model that clears the bar.

2. **When do you use LLM-as-judge vs deterministic scoring?**
   - Deterministic whenever the task allows (classification, extraction, code that runs, JSON schema). LLM-judge for open-ended quality (helpfulness, tone), but with bias controls: randomize order, use a rubric, calibrate against human labels, and prefer pairwise over absolute scores.

3. **What are the failure modes of LLM-as-judge?**
   - Position bias (favoring the first answer), verbosity bias, self-preference, and rubric drift. Mitigate by swapping order and averaging, fixing a rubric, using a strong judge model, and spot-checking against humans.

4. **How do you make an eval that reflects production?**
   - Sample real inputs (including the long tail / edge cases), include the actual system prompt + tools, measure the metric the business cares about, and add latency/cost as first-class metrics — not just quality.

5. **How do you stop quality regressions after you hand off?**
   - A CI eval gate: the golden-set eval runs on every model/prompt/config change; ship only if it passes thresholds. Plus online monitoring with sampled human/LLM labels for drift.
