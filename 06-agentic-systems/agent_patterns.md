# Agent patterns, failure modes & evaluation

A reference for designing and debugging agentic systems with customers. The control flow in this module's `.py` files implements several of these.

## Core patterns

### 1. Router (intent classification → tool)
A cheap model classifies the request and dispatches to a tool or a specialized prompt.
- **Use when**: many distinct request types; you want to keep most calls cheap.
- **Win**: route with a small/cheap model, only invoke a big model for hard turns.

### 2. Function calling (single-step tools)
Model emits a structured tool call; you execute and return the result.
- **Use when**: the task needs external data/actions (search, DB, API).
- See `function_calling.py`.

### 3. ReAct (Reason + Act loop)
Interleave Thought → Action → Observation until done.
- **Use when**: multi-step reasoning over tools (research, multi-hop).
- **Risk**: loops, token burn → enforce budgets + loop detection. See `react_agent.py`.

### 4. Plan-and-Execute
Model writes a plan up front, then executes steps (optionally re-planning).
- **Use when**: tasks decompose cleanly; you want fewer model calls than ReAct.
- **Win**: cheaper + more predictable than step-by-step reasoning for structured tasks.

### 5. Reflection / self-critique
After an attempt, the model critiques and revises.
- **Use when**: quality matters more than latency (code, writing).
- **Cost**: extra model calls; gate with eval that it actually helps.

### 6. Multi-agent / supervisor
A supervisor delegates to specialist agents.
- **Use when**: genuinely separable sub-tasks; otherwise it's overhead.
- **Caution**: more moving parts, more failure surface, more cost. Don't over-engineer.

```
 ROUTER         FUNCTION CALL        REACT                  PLAN-EXECUTE
 user            user                 user                   user
   │ classify      │ tool_call          │ thought→act→obs      │ plan: [s1,s2,s3]
   ▼               ▼                     ▼ (loop)               ▼ execute each
 tool/prompt     execute→answer        answer                 answer
```

## Failure modes (and the fix)

| Failure | Symptom | Fix |
| --- | --- | --- |
| Infinite loop | repeats the same action | step budget + loop detection (dedupe action+args) |
| Token blow-up | context grows unbounded | summarize/trim history; cap context; cheaper router |
| Bad tool args | invalid JSON / wrong types | schema validation + constrained decoding; retry with error fed back |
| Tool error not handled | crash | return errors as observations the model can recover from |
| Wrong tool chosen | task fails silently | better tool descriptions, few-shot, eval the trajectory |
| Hallucinated tool/args | calls nonexistent tool | whitelist tools; validate before execute |
| Prompt injection via tool output | agent hijacked by fetched content | treat tool output as untrusted; don't follow instructions in data; sandbox |

## Evaluating agents (apply module 05 to trajectories)

Score the **path**, not just the final answer:
- **Task success** — did it achieve the goal (checkable outcome)?
- **Tool accuracy** — right tools, right args, right order?
- **Efficiency** — steps used vs budget; total tokens/cost; latency.
- **Recovery** — did it handle errors/not-found gracefully? (See `multi_tool_agent.py`.)
- **Safety** — stayed in-scope, no unsafe actions, resisted injection.

Build a golden set of **tasks with verifiable outcomes**; run the agent; aggregate these metrics; gate changes.

## Cost & latency engineering (where your inference knowledge pays off)

- **Prefix caching**: the system prompt + tool schemas are identical every call — cache that KV (SGLang RadixAttention / vLLM prefix caching) to cut TTFT and cost on every step.
- **Tiered models**: small model for routing/simple turns, big model only when needed.
- **Parallel tools**: fire independent tool calls concurrently (module 01 async).
- **Result caching**: memoize deterministic tool results.
- **Bounded context**: summarize long histories; don't resend everything.

## Security (must-mention)

Tool use executes actions derived from model output:
- **Validate + whitelist** tool names and args against schemas before executing.
- **Never `eval()`** arbitrary model output; sandbox tools; least-privilege creds.
- **Treat tool results as untrusted** input (a web page / doc can contain injection). Don't let data become instructions.
- **Human-in-the-loop** for destructive/high-blast-radius actions.
