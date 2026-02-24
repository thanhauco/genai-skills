# 06 — Agentic Systems & Tool Use

The JD: "Experience building or integrating agentic systems, tool-use chains, or AI-native developer toolchains." Fireworks ships its **own function-calling model** and published "open source agents with frontier advisors." This is a core, differentiated skill.

## Files

| File | What it teaches |
| --- | --- |
| [`function_calling.py`](function_calling.py) | The function-calling loop: tool schemas → model picks a tool → execute → feed result back |
| [`react_agent.py`](react_agent.py) | ReAct (Reason + Act) loop with a step budget and a tool registry |
| [`multi_tool_agent.py`](multi_tool_agent.py) | Router + multiple tools + error recovery; trajectory logging for eval |
| [`agent_patterns.md`](agent_patterns.md) | Patterns (router, ReAct, plan-execute, reflection), failure modes, eval |

## Run

```bash
python 06-agentic-systems/function_calling.py
python 06-agentic-systems/react_agent.py
python 06-agentic-systems/multi_tool_agent.py
```

All use a **mock model** (a rule-based planner) so they run offline with no API key. The control flow is identical to a real OpenAI/Fireworks `tools=[...]` loop — swap the planner for a real model call.

## The function-calling loop (memorize this)

```
 1. Send user msg + tool SCHEMAS to the model
 2. Model returns either: final answer  OR  a tool_call (name + JSON args)
 3. If tool_call: validate args -> execute tool -> append result as a tool message
 4. Loop back to the model with the tool result
 5. Repeat until final answer OR step/cost budget exhausted
```

```
  user ──► [ MODEL ] ──tool_call(get_weather, {city:"SF"})──► [ TOOL ] ──result──┐
            ▲                                                                    │
            └──────────────── tool result appended to messages ◄────────────────┘
            (loop until the model emits a final answer or budget runs out)
```

## Why this matters for Fireworks specifically

- Fireworks has a **function-calling-tuned model** and fast structured output. You'll help customers wire tools reliably and cheaply.
- Agents are **latency- and cost-sensitive**: every step is a model call. Your inference knowledge (modules 02/03) directly improves agent economics (prefix caching for the repeated system prompt + tools, smaller models for routing, etc.).

## Interview Q&A

1. **Walk me through a function-calling loop.**
   - Tools as JSON schemas → model returns a structured tool call → you validate + execute → append the result → re-invoke the model → repeat until it answers or you hit the budget. Always validate args against the schema before executing (security + reliability).

2. **An agent loops forever / burns tokens. How do you make it robust?**
   - Hard **step + token budget**, detect repeated identical actions (loop detection), require progress, structured error messages back to the model so it can recover, and a fallback/final-answer path when the budget is hit. Log the trajectory for eval.

3. **How do you make tool calls reliable?**
   - Constrained/structured decoding for valid JSON, schema validation, a function-calling-tuned model, idempotent tools, timeouts + retries on the tool side, and few-shot examples for tricky tools.

4. **How do you evaluate an agent?**
   - Evaluate the **trajectory**, not just the final answer: right tools, right order, recovered from errors, within step/cost budget, and final-task success. Build a golden set of tasks with checkable outcomes (module 05).

5. **How do you cut agent cost/latency?**
   - Prefix-cache the static system+tools prompt (SGLang RadixAttention / vLLM prefix caching), use a small/cheap model for routing and a bigger one only when needed, parallelize independent tool calls, cap context, and cache tool results.

## Security note (you must say this)

Tool use = executing actions from model output. Treat model output as **untrusted**: validate/whitelist tool names + args, sandbox execution, never `eval()` arbitrary strings, scope credentials to least privilege, and guard against prompt injection from tool results (a fetched web page can try to hijack the agent). This module's tools are pure functions with validated args — no shell, no eval.
