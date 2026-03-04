"""
react_agent.py — ReAct (Reason + Act) loop with a step budget + loop detection.

Why this matters:
  ReAct interleaves Thought -> Action -> Observation until the agent answers. It's
  the canonical agent pattern. The FIELD value is making it ROBUST: budgets, loop
  detection, and graceful failure — the difference between a demo and production.

Run:
    python react_agent.py
Stdlib only; mock reasoner so it runs offline. No eval / no shell.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field


# ----------------------------- tools ------------------------------------------
def tool_calculator(expr: str) -> str:
    """Evaluate a SAFE arithmetic expression (digits, + - * / ( ) . only)."""
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return "error: only arithmetic allowed"
    try:
        # safe: restricted charset above; no names/builtins exposed
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 (charset-guarded)
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def tool_sqrt(x: str) -> str:
    try:
        return str(round(math.sqrt(float(x)), 4))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


TOOLS: dict[str, Callable[[str], str]] = {"calculator": tool_calculator, "sqrt": tool_sqrt}


# ----------------------------- trajectory -------------------------------------
@dataclass
class Step:
    thought: str
    action: str | None
    action_input: str | None
    observation: str | None


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)

    def signature(self) -> list[tuple]:
        return [(s.action, s.action_input) for s in self.steps]


# ----------------------------- mock reasoner ----------------------------------
def mock_reason(question: str, traj: Trajectory) -> Step:
    """Decide the next ReAct step. Task: compute sqrt((12+13)*4) = sqrt(100) = 10."""
    n = len(traj.steps)
    if n == 0:
        return Step("I need (12+13)*4 first.", "calculator", "(12+13)*4", None)
    if n == 1:
        val = traj.steps[0].observation
        return Step(f"Now sqrt of {val}.", "sqrt", val or "0", None)
    final = traj.steps[-1].observation
    return Step(f"The answer is {final}.", None, None, None)  # action=None -> finish


# ----------------------------- the loop ---------------------------------------
def run_react(question: str, max_steps: int = 6) -> str:
    traj = Trajectory()
    print(f"Q: {question}\n")
    for i in range(max_steps):
        step = mock_reason(question, traj)
        print(f"[{i}] Thought: {step.thought}")

        if step.action is None:  # model decided to answer
            print(f"[{i}] Finish")
            return step.thought

        # loop detection: same (action, input) already tried -> bail to avoid burn
        if (step.action, step.action_input) in traj.signature():
            print(f"[{i}] LOOP DETECTED on {step.action}({step.action_input}); stopping.")
            return "[stopped: repeated action without progress]"

        tool = TOOLS.get(step.action)
        if tool is None:
            step.observation = f"error: unknown tool {step.action}"
        else:
            step.observation = tool(step.action_input or "")
        print(f"[{i}] Action: {step.action}({step.action_input}) -> Observation: {step.observation}")
        traj.steps.append(step)

    return "[budget exhausted]"


def main() -> None:
    ans = run_react("What is the square root of (12+13)*4?")
    print("\nANSWER:", ans)
    print(
        "\nRobustness features that matter in the field:\n"
        "  - step budget (max_steps) so it can't run forever / burn tokens\n"
        "  - loop detection (same action+input twice -> stop)\n"
        "  - tool errors become observations the model can recover from\n"
        "  - full trajectory captured -> you can EVAL the path, not just the answer"
    )


if __name__ == "__main__":
    main()
