# 14 — Prompt Engineering & Speculative Decoding

Two cheap-but-high-leverage tools: **prompt engineering** (always the first thing to try, before RAG/fine-tuning) and **speculative decoding** (a serving trick that cuts decode latency for free quality). Both come up constantly in the field.

## Files

| File | What it teaches |
| --- | --- |
| [`prompt_engineering.md`](prompt_engineering.md) | Techniques, structure, few-shot/CoT, prompt caching, anti-patterns |
| [`speculative_decoding.py`](speculative_decoding.py) | The draft-and-verify mechanism + acceptance-rate → speedup math |
| [`speculative_notes.md`](speculative_notes.md) | How it works, when it helps, variants, trade-offs |

## Run

```bash
python 14-prompt-and-speculative/speculative_decoding.py
```

Stdlib only — simulates draft+verify and computes the expected speedup from acceptance rate.

## Why these two together

```
 PROMPTING                          SPECULATIVE DECODING
 ─────────                          ────────────────────
 free, instant, first lever         serving-side, cuts TPOT, no quality loss
 (quality / behavior)               (latency / throughput)
 module 04 ladder starts here       module 02/03 serving optimization
```

One improves *what the model does* for $0; the other improves *how fast it decodes* with identical outputs.

## Speculative decoding in one sentence

> A small **draft** model proposes several tokens cheaply; the big **target** model verifies them all in **one** forward pass and accepts the longest correct prefix — so you often get multiple tokens per expensive step, cutting TPOT **without changing the output distribution**.

## Interview Q&A

1. **What does speculative decoding optimize, and is it lossless?**
   - It cuts **TPOT** (decode latency) and raises tokens/sec. With proper verification (rejection sampling) it's **distribution-preserving** — the output is statistically identical to the target model. It doesn't help TTFT (prefill).

2. **When does it help vs hurt?**
   - Helps when the draft model agrees with the target often (high acceptance) — predictable text, code, low temperature. Hurts when acceptance is low (the wasted draft+verify costs more than it saves) — highly creative/high-temp output or a poorly matched draft. See the math in `speculative_decoding.py`.

3. **What's the acceptance rate and why does it matter?**
   - The fraction of drafted tokens the target accepts. Speedup ≈ tokens accepted per verification step. High acceptance → big speedup; low acceptance → little/negative. You pick the draft model + draft length to maximize it for the workload.

4. **Prompting first — why?**
   - It's free and instant; it's the top of the fine-tuning ladder (module 04). Many "we need to fine-tune" asks are solved by a better system prompt, few-shot examples, or constrained output. Always baseline the best prompt before paying for RAG/SFT.

5. **What's prompt caching and how does it relate to serving?**
   - Reusing the KV cache of a shared prompt prefix (system prompt, few-shot, RAG context) across requests → big TTFT + cost win for repeated prefixes. It's the prompt-side name for prefix caching / RadixAttention (modules 02/03/06).

## Ties to other modules

- Prompting as the first rung → module 04 decision ladder.
- Prompt caching / prefix caching → modules 02, 03, 06.
- Speculative decoding as a serving knob → modules 02, 03 (TPOT), 09 (measure it).
