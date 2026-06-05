# Speculative decoding — notes

A serving-side trick that cuts decode latency (TPOT) and raises throughput **without changing the output distribution**. One of the levers you'll reach for to hit a latency SLO (modules 02/03/09).

## How it works

```
 1. DRAFT: a small, cheap model proposes k tokens:  t1 t2 t3 t4
 2. VERIFY: the big TARGET model scores all k in ONE forward pass (parallel)
 3. ACCEPT: keep the longest prefix where the target "agrees"
            (formally: rejection sampling so the result == sampling from target)
 4. On the first disagreement, emit the target's own correction token.
 => you produce 1..k+1 target-quality tokens per single target forward pass.
```

The win: decode is **memory-bandwidth-bound** (module 02) — one target forward pass costs about the same whether it scores 1 token or k. So verifying k drafted tokens "for free" turns 1 expensive step into several accepted tokens.

## Why it's lossless

Proper speculative decoding uses **rejection sampling** during verification, which makes the output **distribution-identical** to sampling from the target model alone. You're trading the draft model's compute for latency, not quality.

## The speedup driver: acceptance rate

```
 expected tokens per step ≈ 1 + p + p^2 + ... + p^k    (p = per-token accept prob)
 speedup ≈ expected_tokens_per_step / (1 + draft_overhead)
```

- **High p** (0.7–0.9): predictable text, code, low temperature → big speedup (often 1.5–3×).
- **Low p** (<0.3): creative / high-temperature / mismatched draft → draft+verify overhead can make it **slower**. Don't use it there.

See `speculative_decoding.py` for the closed form + a Monte-Carlo check.

## Variants (name-drop-able)

- **Draft model** (classic) — a small model of the same family proposes tokens.
- **Self-speculative / Medusa** — extra decoding heads on the target predict multiple tokens; no separate draft model.
- **EAGLE / EAGLE-2** — feature-level autoregression for high acceptance; strong results.
- **Lookahead decoding** — n-gram/Jacobi-style parallel decoding, no draft model.
- **Prompt lookup decoding** — draft by copying spans from the prompt (great for summarization/RAG where output echoes input).

## What it does and doesn't help

| | Speculative decoding |
| --- | --- |
| TPOT (decode latency) | ✅ cuts it |
| Throughput (tok/s) | ✅ raises it (when acceptance is high) |
| TTFT (prefill) | ❌ no effect |
| Output quality | ➖ unchanged (distribution-preserving) |
| High-temp / creative output | ⚠️ low acceptance → may not help |
| Extra memory | ⚠️ draft model / heads use some HBM |

## When to recommend it

- Latency-sensitive decode (chat, agents) on **predictable** output (code, structured, low temp).
- You have a good small draft model of the same family, or use Medusa/EAGLE to avoid one.
- You've **measured** acceptance on the real workload (module 09) and it's high enough to win.

## The sentence that lands

> "Your decode latency is the bottleneck and your outputs are fairly predictable, so speculative decoding should give us multiple tokens per step at the same quality — it's distribution-preserving. We'll measure the acceptance rate on your traffic; if it's above ~60% we likely get a 1.5–2× TPOT win for free. It won't touch TTFT, so we pair it with prefix caching for the prompt side."
