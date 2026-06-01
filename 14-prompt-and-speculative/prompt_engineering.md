# Prompt engineering — the free first lever

Before RAG, before fine-tuning, before bigger GPUs: **fix the prompt**. It's instant, free, and the top rung of the fine-tuning decision ladder (module 04). A surprising share of "we need to fine-tune" requests are solved here.

## Anatomy of a good prompt

```
 ┌─ system ─────────────────────────────────────────────┐
 │ role + objective + constraints + output format        │   stable, cacheable
 │ "You are X. Do Y. Always Z. Output JSON matching ..."  │
 ├─ few-shot examples (optional) ────────────────────────┤   stable, cacheable
 │ input -> ideal output  (x2-5, covering edge cases)     │
 ├─ context (RAG, tools) ────────────────────────────────┤   per-request
 │ retrieved chunks / tool results                        │
 ├─ user request ───────────────────────────────────────-┤   per-request
 │ the actual task                                        │
 └────────────────────────────────────────────────────────┘
```

Put the **stable** parts first → they prefix-cache well (modules 02/03).

## Techniques (when to use)

| Technique | Use when | Note |
| --- | --- | --- |
| **Clear role + objective** | always | the cheapest quality win |
| **Explicit output format** | structured output needed | pair with constrained decoding (module 13) |
| **Few-shot examples** | format/behavior is hard to describe | 2–5 diverse examples > a long instruction |
| **Chain-of-thought / "think step by step"** | multi-step reasoning | let it reason *before* the final answer |
| **Decomposition** | complex task | break into sub-steps / multiple calls |
| **Delimiters / sections** | long/mixed input | fence context vs instructions (also anti-injection) |
| **Negative/positive constraints** | recurring mistakes | "do X; never Y" |
| **Self-consistency** | high-stakes reasoning | sample N, vote (costs tokens) |
| **Reference/grounding instruction** | RAG | "answer only from the context; cite sources" |

## Prompt caching (the serving connection)

Reusing the KV cache of a shared prompt **prefix** (system + few-shot + RAG context) across requests = big **TTFT** and **cost** savings. It's the prompt-side view of **prefix caching / RadixAttention** (modules 02/03/06). Design prompts so the big static part is a stable prefix.

## Parameters that matter

- **temperature / top_p** — low (0–0.3) for deterministic/structured/factual; higher for creative. Low temp also boosts speculative-decoding acceptance.
- **max_tokens** — bound output (latency + cost); the JD-relevant cost lever.
- **stop sequences** — clean termination for structured formats.
- **seed** — reproducibility for evals.

## Anti-patterns (what to fix in a customer's prompt)

- **Vague role / no objective** → inconsistent behavior.
- **One giant wall of instructions** → use structure + few-shot instead.
- **Format described in prose only** → use constrained decoding (module 13).
- **Instructions buried after huge context** → "lost in the middle"; put key instructions first/last.
- **No examples for a fiddly format** → 3 good examples beat 3 paragraphs.
- **Mixing untrusted data with instructions unfenced** → prompt-injection risk (module 06 security).
- **Over-prompting** → if the prompt is enormous and still failing, it's a RAG or fine-tune signal (module 04).

## The FDE move

> "Before we talk fine-tuning, let me tighten the prompt: a sharper system role, three few-shot examples covering your edge cases, constrained JSON output, and temperature down to 0.2. That's free and ships today. We'll measure it on your eval — if it clears the bar, we're done; if not, we now have the baseline and the gap to justify SFT."

This is the discovery→cheapest-fix→measure→escalate loop the role rewards.
