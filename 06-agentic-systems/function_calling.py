"""
function_calling.py — the OpenAI/Fireworks function-calling loop, end to end.

Why this matters:
  Fireworks ships a function-calling model. The control flow below is EXACTLY the
  real `tools=[...]` loop; only the "model" is mocked (a rule-based planner) so it
  runs offline. Swap `mock_model_decide` for a real chat.completions call with
  tools= and you have a working agent.

Run:
    python function_calling.py
Stdlib only. No eval(), no shell — tools are pure functions with validated args.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass


# ----------------------------- tool definitions -------------------------------
# JSON-schema tool specs, identical shape to what you'd pass as tools=[...].
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the exchange rate from one currency to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "ISO currency, e.g. USD"},
                    "quote": {"type": "string", "description": "ISO currency, e.g. EUR"},
                },
                "required": ["base", "quote"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert",
            "description": "Convert an amount using a known rate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "rate": {"type": "number"},
                },
                "required": ["amount", "rate"],
            },
        },
    },
]

# Fake data so tools are deterministic.
_RATES = {("USD", "EUR"): 0.92, ("USD", "JPY"): 157.0, ("EUR", "USD"): 1.09}


def get_exchange_rate(base: str, quote: str) -> dict:
    rate = _RATES.get((base.upper(), quote.upper()))
    if rate is None:
        return {"error": f"no rate for {base}->{quote}"}
    return {"base": base.upper(), "quote": quote.upper(), "rate": rate}


def convert(amount: float, rate: float) -> dict:
    return {"result": round(amount * rate, 2)}


TOOLS: dict[str, Callable[..., dict]] = {"get_exchange_rate": get_exchange_rate, "convert": convert}


# ----------------------------- arg validation ---------------------------------
def validate_args(name: str, args: dict) -> tuple[bool, str]:
    """Validate tool args against the declared schema (required fields + types)."""
    spec = next((t["function"] for t in TOOL_SCHEMAS if t["function"]["name"] == name), None)
    if spec is None:
        return False, f"unknown tool {name!r}"
    params = spec["parameters"]
    for req in params.get("required", []):
        if req not in args:
            return False, f"missing required arg {req!r}"
    for k, v in args.items():
        decl = params["properties"].get(k)
        if decl is None:
            return False, f"unexpected arg {k!r}"
        if decl["type"] == "number" and not isinstance(v, (int, float)):
            return False, f"arg {k!r} must be number"
        if decl["type"] == "string" and not isinstance(v, str):
            return False, f"arg {k!r} must be string"
    return True, "ok"


# ----------------------------- the mock model ---------------------------------
@dataclass
class ToolCall:
    name: str
    args: dict


def mock_model_decide(messages: list[dict]) -> dict:
    """Stand-in for the model. Returns either {'tool_call': ToolCall} or {'final': str}.

    Logic: convert 100 USD to EUR. First fetch the rate, then convert, then answer.
    A real model emits this structure natively via function calling."""
    have_rate = any(m["role"] == "tool" and m["name"] == "get_exchange_rate" for m in messages)
    have_conv = any(m["role"] == "tool" and m["name"] == "convert" for m in messages)

    if not have_rate:
        return {"tool_call": ToolCall("get_exchange_rate", {"base": "USD", "quote": "EUR"})}
    if not have_conv:
        rate = json.loads(_last_tool_result(messages, "get_exchange_rate"))["rate"]
        return {"tool_call": ToolCall("convert", {"amount": 100, "rate": rate})}
    result = json.loads(_last_tool_result(messages, "convert"))["result"]
    return {"final": f"100 USD is about {result} EUR."}


def _last_tool_result(messages: list[dict], name: str) -> str:
    for m in reversed(messages):
        if m["role"] == "tool" and m["name"] == name:
            return m["content"]
    raise KeyError(name)


# ----------------------------- the agent loop ---------------------------------
def run_agent(user_msg: str, max_steps: int = 6) -> str:
    messages: list[dict] = [
        {"role": "system", "content": "You are a currency assistant. Use tools."},
        {"role": "user", "content": user_msg},
    ]
    for step in range(max_steps):
        decision = mock_model_decide(messages)

        if "final" in decision:
            print(f"[step {step}] model -> FINAL")
            return decision["final"]

        call: ToolCall = decision["tool_call"]
        ok, why = validate_args(call.name, call.args)
        print(f"[step {step}] model -> tool_call {call.name}({call.args})  valid={ok}")
        if not ok:
            # Feed the error back so the model can correct itself (don't crash).
            messages.append({"role": "tool", "name": call.name, "content": json.dumps({"error": why})})
            continue

        result = TOOLS[call.name](**call.args)  # safe: validated args, pure function
        messages.append({"role": "tool", "name": call.name, "content": json.dumps(result)})

    return "[budget exhausted] no final answer within step limit"


def main() -> None:
    print("Tools available:", [t["function"]["name"] for t in TOOL_SCHEMAS], "\n")
    answer = run_agent("Convert 100 USD to EUR.")
    print("\nANSWER:", answer)
    print(
        "\nThis is the real loop: schemas -> tool_call -> validate -> execute ->\n"
        "append result -> re-invoke -> final. Replace mock_model_decide with a\n"
        "Fireworks/OpenAI chat.completions(tools=...) call and it's production-shaped."
    )


if __name__ == "__main__":
    main()
