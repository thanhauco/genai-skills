"""
multi_tool_agent.py — router + multiple tools + error recovery + trajectory eval.

Why this matters:
  Real agents choose among many tools, recover from bad calls, and you must be
  able to EVALUATE whether they took a good path. This shows:
    - a lightweight intent ROUTER (cheap model picks the tool)
    - several tools with validation
    - error recovery (bad arg -> retry with a corrected call)
    - a trajectory + a simple trajectory scorer (module 05 idea applied to agents)

Run:
    python multi_tool_agent.py
Stdlib only. Mocked router/model; no eval/shell.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field


# ----------------------------- tools ------------------------------------------
_KB = {
    "refund policy": "Refunds within 30 days with order id. [doc:refunds-1]",
    "api limits": "Default 600 req/min; 429 on exceed; back off with jitter. [doc:limits-7]",
}
_ORDERS = {"ORD-12345": {"status": "shipped", "eta_days": 2}}


def kb_search(query: str) -> dict:
    for k, v in _KB.items():
        if k in query.lower():
            return {"answer": v}
    return {"answer": "no KB match"}


def order_status(order_id: str) -> dict:
    o = _ORDERS.get(order_id.upper())
    return o or {"error": f"order {order_id} not found"}


def escalate(reason: str) -> dict:
    return {"ticket": "TICK-9001", "reason": reason}


TOOLS: dict[str, Callable[..., dict]] = {
    "kb_search": kb_search,
    "order_status": order_status,
    "escalate": escalate,
}
REQUIRED_ARG = {"kb_search": "query", "order_status": "order_id", "escalate": "reason"}


# ----------------------------- router (mock) ----------------------------------
def route(user_msg: str) -> tuple[str, dict]:
    """A cheap intent classifier picks a tool + args. Real impl: small model."""
    m = user_msg.lower()
    if "ord-" in m:
        # extract an ORD-##### token, stripping surrounding punctuation
        oid = ""
        for tok in user_msg.split():
            cleaned = tok.strip(".,?!;:()'\"").upper()
            if cleaned.startswith("ORD-"):
                oid = cleaned
                break
        return "order_status", {"order_id": oid}
    if "refund" in m or "limit" in m or "429" in m:
        return "kb_search", {"query": user_msg}
    return "escalate", {"reason": user_msg}


# ----------------------------- trajectory + eval ------------------------------
@dataclass
class Turn:
    user: str
    tool: str
    args: dict
    result: dict
    recovered: bool = False


@dataclass
class Episode:
    turns: list[Turn] = field(default_factory=list)


def validate(tool: str, args: dict) -> bool:
    return REQUIRED_ARG[tool] in args and bool(args[REQUIRED_ARG[tool]])


def handle(user_msg: str) -> Turn:
    tool, args = route(user_msg)
    recovered = False
    if not validate(tool, args):
        # error recovery: route again / fall back to escalate
        print(f"  bad args for {tool}({args}); recovering -> escalate")
        tool, args, recovered = "escalate", {"reason": user_msg}, True
    result = TOOLS[tool](**args)
    if result.get("error"):
        print(f"  tool returned error: {result['error']}; recovering -> escalate")
        tool, args, result, recovered = "escalate", {"reason": user_msg}, escalate(user_msg), True
    return Turn(user_msg, tool, args, result, recovered)


def score_trajectory(ep: Episode, expected_tools: list[str]) -> dict:
    """Did the agent pick the right tool per turn? Did it recover when needed?"""
    correct = sum(1 for t, exp in zip(ep.turns, expected_tools) if t.tool == exp)
    recoveries = sum(1 for t in ep.turns if t.recovered)
    return {
        "tool_accuracy": correct / len(ep.turns) if ep.turns else 0.0,
        "recoveries": recoveries,
        "turns": len(ep.turns),
    }


def main() -> None:
    convo = [
        "What's your refund policy?",
        "Where is order ORD-12345?",
        "Where is order ORD-99999?",   # not found -> recover via escalate
        "My account was hacked!",      # no tool match -> escalate
    ]
    expected = ["kb_search", "order_status", "escalate", "escalate"]

    ep = Episode()
    for msg in convo:
        print(f"USER: {msg}")
        turn = handle(msg)
        print(f"  -> {turn.tool}({turn.args}) => {json.dumps(turn.result)}\n")
        ep.turns.append(turn)

    report = score_trajectory(ep, expected)
    print("=== trajectory eval (agents: score the PATH, not just the answer) ===")
    print(f"  tool_accuracy = {report['tool_accuracy']:.0%}")
    print(f"  recoveries    = {report['recoveries']} (handled bad args / not-found gracefully)")
    print(
        "\nField takeaways:\n"
        "  - route with a CHEAP model, act with a stronger one only when needed\n"
        "  - every tool call is validated; failures become recovery, not crashes\n"
        "  - log the trajectory so you can build an agent eval set (module 05)\n"
        "  - prefix-cache the static system+tools prompt to cut per-step cost"
    )


if __name__ == "__main__":
    main()
