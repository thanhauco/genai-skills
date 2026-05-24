# 13 — Structured Output & Constrained Decoding

Reliable JSON / schema-conformant output is one of the most common customer needs (it's how LLMs plug into real software, and it underpins function calling in module 06). Fireworks supports structured output / JSON mode. You should know how it works and how to make it bulletproof.

## Files

| File | What it teaches |
| --- | --- |
| [`json_schema_demo.py`](json_schema_demo.py) | Validate model output against a schema + a "repair loop" for when it's malformed |
| [`constrained_decoding.md`](constrained_decoding.md) | How constrained decoding works (logit masking), tools, trade-offs |

## Run

```bash
python 13-structured-output/json_schema_demo.py
```

Stdlib only — a minimal schema validator + repair loop, no external `jsonschema` needed (though you'd use it / Pydantic in production).

## Two ways to get structured output

```
 1. PROMPT + VALIDATE + RETRY            2. CONSTRAINED DECODING (guaranteed)
 ───────────────────────────            ────────────────────────────────────
 ask for JSON -> parse -> validate       mask logits so only tokens that keep the
 -> if bad, feed the error back + retry   output VALID for the grammar/schema can
 (works anywhere; not guaranteed)         be sampled (JSON/regex/grammar).
                                          vLLM (outlines/xgrammar), SGLang, Fireworks
                                          JSON mode -> output is valid BY CONSTRUCTION.
```

Use constrained decoding when you need a **hard guarantee**; use prompt+validate+retry as a portable fallback or for richer semantic checks.

## How constrained decoding works (one sentence)

> At each decode step, the engine computes which next tokens would keep the output valid for the target grammar (JSON schema / regex / CFG) and **masks out** all the others before sampling — so the model literally cannot emit invalid structure.

## Interview Q&A

1. **A customer needs the model to always return JSON matching their schema. Approach?**
   - First: **constrained decoding / JSON mode** with their schema → valid by construction. Add a **validate + repair loop** as defense-in-depth and for semantic rules the grammar can't express. If still flaky, **SFT** on a few hundred examples of (input → valid JSON) (module 04). Gate with an eval (module 05).

2. **Constrained decoding vs prompt-and-retry — trade-offs?**
   - Constrained guarantees syntactic validity and cuts retries/cost, but needs engine support and can slightly constrain phrasing; very complex grammars add overhead. Prompt-and-retry is portable and can enforce semantic rules, but isn't guaranteed and burns tokens on retries. Best practice: constrained for structure + validate for semantics.

3. **Does forcing JSON hurt quality?**
   - It can if the schema fights the model's natural reasoning (e.g., forcing JSON before it "thinks"). Mitigate by letting it reason first then emit JSON (or a `reasoning` field), keeping schemas lean, and eval-gating. Function-calling-tuned models (Fireworks) handle this better.

4. **How does this relate to function calling?**
   - Function calling **is** constrained/structured output: the model emits a tool name + JSON args conforming to the tool schema. The reliability techniques here are exactly what make agents robust (module 06).

5. **Production hardening for structured output?**
   - Constrained decoding + schema validation + a bounded repair loop + semantic checks + telemetry on parse-failure rate. Treat a parse failure like any other error: log it, recover, and feed it into your eval set.

## Ties to other modules

- Function calling / agents → module 06 (same machinery).
- SFT to improve format reliability → module 04.
- Parse-failure rate as a production metric → module 05.
