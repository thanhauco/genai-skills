# 04 — Fine-Tuning (SFT, DPO, RFT)

The JD: "Guide customers on model selection, fine-tuning strategy (SFT, DPO, RFT)… Build and run fine-tuning pipelines directly with customers, navigating trade-offs between model families, compute cost, and quality targets." SFT at minimum; DPO/RFT a strong plus.

Fireworks also published that "the fine-tuning bottleneck is not the algorithm" — **integration friction and iteration speed** stall teams. So this module emphasizes the *workflow and data*, not just the math.

## Files

| File | What it teaches |
| --- | --- |
| [`decision_guide.md`](decision_guide.md) | Prompt vs RAG vs SFT vs DPO vs RFT — when to use each |
| [`data_prep.py`](data_prep.py) | Build & validate an SFT chat dataset (JSONL), the #1 real bottleneck |
| [`sft_lora_example.py`](sft_lora_example.py) | LoRA/QLoRA SFT skeleton (TRL/PEFT) + a runnable no-dep math demo |
| [`dpo_example.py`](dpo_example.py) | Preference data format + the DPO loss, computed by hand in numpy/stdlib |
| [`rft_notes.md`](rft_notes.md) | RFT/RLHF (reward models, PPO/GRPO), reward design, when it's worth it |

## Run

```bash
python 04-fine-tuning/data_prep.py          # builds + validates a sample JSONL
python 04-fine-tuning/dpo_example.py        # DPO loss intuition, stdlib only
python 04-fine-tuning/sft_lora_example.py   # prints LoRA math + a runnable toy; real training is gated behind deps
```

`data_prep.py` and `dpo_example.py` are stdlib-only. `sft_lora_example.py` shows the real TRL/PEFT code but only *runs* the heavy path if `torch`/`peft`/`trl` are installed; otherwise it runs a tiny LoRA math demo so you still see the mechanism.

## The decision ladder (cheapest → most expensive)

```
Prompt eng.  →  Few-shot  →  RAG  →  SFT (LoRA)  →  DPO  →  RFT / RLHF
  cheap, fast  ............................................  expensive, slow
  Always exhaust the left before paying for the right.
```

- **SFT (Supervised Fine-Tuning):** teach format/behavior/domain from input→output examples. The workhorse. LoRA/QLoRA make it cheap.
- **DPO (Direct Preference Optimization):** align to *preferences* using (prompt, chosen, rejected) triples — no separate reward model, no RL loop. Great for "make it sound like X / prefer this style/safety."
- **RFT / RLHF (Reinforcement Fine-Tuning):** optimize against a **reward signal** (a reward model or a verifiable checker). Best when correctness is checkable (code passes tests, math is right, tool call validates). More compute + infra.

## Why "the bottleneck isn't the algorithm"

The hard parts in the field are:
1. **Data**: getting clean, well-formatted, deduped, leakage-free examples (`data_prep.py`).
2. **Iteration speed**: how fast can you train → eval → deploy → look at outputs → repeat.
3. **Integration**: does the tuned model actually slot into the customer's pipeline & serving?
4. **Eval**: do you have a metric that reflects production quality (module 05)?

Your value as an FDE: compress that loop.

## Interview Q&A

1. **Customer wants the model to always output valid JSON for their schema. SFT, DPO, or RFT?**
   - Start with **prompting + constrained decoding** (free). If still inconsistent, **SFT** on a few hundred good examples of (input → valid JSON). DPO/RFT is overkill unless you're trading off subtle quality.

2. **When DPO over SFT?**
   - When you have **preference signal** (humans/another model picked A over B) rather than gold answers, or you want to shift style/safety/tone. DPO is simpler than RLHF (no reward model, no PPO) and stable.

3. **When is RFT worth the cost?**
   - When success is **programmatically checkable** (unit tests, math equality, schema/tool validation) so you can generate a reward at scale, or when you've plateaued on SFT/DPO for a high-value task. Fireworks' RL blog: cross-region rollouts + sparse weight deltas make this cheaper than the mega-cluster narrative.

4. **LoRA vs full fine-tune?**
   - LoRA trains small low-rank adapters (≈0.1–1% of params): cheaper, faster, swappable, multi-tenant-friendly, near full-FT quality for most tasks. Full FT only when you need maximal capacity change or you're distilling broadly.

5. **A team says fine-tuning "isn't working." Your first questions?**
   - Show me the data (format, count, dedup, leakage), show me the eval (is the metric meaningful?), and show me the iteration loop (how long train→eval→inspect takes). Usually it's data or eval, not the algorithm.
