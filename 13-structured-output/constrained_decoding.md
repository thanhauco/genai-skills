# Constrained decoding — how guaranteed structured output works

Constrained (a.k.a. guided / structured) decoding forces model output to conform to a grammar — JSON schema, regex, or a context-free grammar — so the result is **valid by construction**. This is the machinery behind reliable JSON mode and function calling.

## The core mechanism (logit masking)

```
 at each decode step:
   1. model produces logits over the whole vocabulary
   2. a grammar engine computes which next tokens keep the output VALID
      given what's been generated so far (the current parser state)
   3. invalid tokens are MASKED (set to -inf) before sampling
   4. sample only from the allowed set -> always-valid continuation
```

Because invalid tokens are removed *before* sampling, the model literally cannot emit a syntactically broken structure.

```
  logits:   {  "  n  a  m  e  "  :  ...  garbage_token  ...
 grammar:   ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓        ✗  (masked)
 sample from the ✓ set only
```

## What you can constrain to

- **JSON Schema** — objects/fields/types/enums (most common).
- **Regex** — phone numbers, IDs, dates, classification labels.
- **Context-free grammar (CFG)** — SQL, code, DSLs, custom formats.
- **Choice / enum** — restrict to a fixed set of options.

## Tools / where it lives

| Layer | Examples |
| --- | --- |
| Libraries | **Outlines**, **XGrammar**, **lm-format-enforcer**, **Guidance** |
| Servers | **vLLM** (guided_json/guided_regex via outlines/xgrammar), **SGLang** (built-in structured), **TGI** |
| Managed | **Fireworks JSON mode / structured output**, OpenAI structured outputs |
| App-side | **Pydantic** models → JSON schema; validate + repair |

## Trade-offs (say these)

**Pros**
- Syntactic validity **guaranteed** → no parse failures.
- **Fewer retries** → lower cost + latency variance.
- Makes small models usable for structured tasks.

**Cons / watch-outs**
- Can slightly **constrain phrasing**; forcing JSON too early can hurt reasoning quality → let the model reason first, then emit JSON (or include a `reasoning` field).
- Very **complex grammars** add per-step overhead.
- Guarantees **syntax, not semantics** — "valid JSON" can still be wrong content. Keep validation + evals.
- Engine support varies; it's a serving-side feature.

## Best-practice stack (defense in depth)

```
 constrained decoding (syntax guaranteed)
        +
 schema validation (Pydantic/jsonschema)        <- catches config drift, semantics
        +
 bounded repair loop (re-prompt with errors)    <- portable fallback
        +
 telemetry on parse-failure rate                <- production metric (module 05)
        +
 SFT on (input -> valid output) if still flaky  <- module 04
```

## Relationship to function calling (module 06)

Function calling **is** structured output: the model emits `{"name": ..., "arguments": {...}}` conforming to the tool schema. Everything here — schema constraint, validation, repair — is what makes tool-using agents reliable. A function-calling-tuned model (Fireworks) + constrained decoding = robust agents.

## The sentence that lands

> "We'll turn on structured output with your JSON schema so the response is valid by construction — no more parse failures or retry loops. We keep Pydantic validation for the semantic rules the grammar can't express, and we track parse-failure rate so any regression shows up before your users see it."
