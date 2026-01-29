"""
dpo_example.py — Direct Preference Optimization, by hand.

Why this matters:
  DPO aligns a model to PREFERENCES using (prompt, chosen, rejected) triples,
  with no separate reward model and no RL loop. You should be able to explain the
  data shape and the loss.

DPO loss for one preference pair:
    L = -log sigmoid( beta * [ (logp_chosen - logp_chosen_ref)
                              - (logp_rejected - logp_rejected_ref) ] )
  Intuition: push the policy to raise the (relative) log-prob of the chosen
  response over the rejected one, anchored to a reference model so it doesn't
  drift too far (the beta * KL-ish term).

Run:
    python dpo_example.py
Stdlib only (uses math).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Preference dataset shape (this is what you collect / generate) -----------
PREFERENCE_DATA = [
    {
        "prompt": "Write a commit message for a bug fix in the auth retry logic.",
        "chosen": "fix(auth): add jitter to token refresh retries to avoid thundering herd",
        "rejected": "fixed stuff",
    },
    {
        "prompt": "Explain a 429 to a customer.",
        "chosen": "A 429 means you hit the rate limit. Back off with exponential jitter and retry.",
        "rejected": "error 429 bad",
    },
]


@dataclass
class LogProbs:
    """Stand-in for what a model would compute: sum log-prob of a response."""

    policy_chosen: float
    policy_rejected: float
    ref_chosen: float
    ref_rejected: float


def sigmoid(x: float) -> float:
    # numerically stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def dpo_loss(lp: LogProbs, beta: float = 0.1) -> float:
    """Single-pair DPO loss."""
    chosen_logratio = lp.policy_chosen - lp.ref_chosen
    rejected_logratio = lp.policy_rejected - lp.ref_rejected
    margin = beta * (chosen_logratio - rejected_logratio)
    return -math.log(sigmoid(margin) + 1e-12)


def implied_reward_margin(lp: LogProbs, beta: float = 0.1) -> float:
    """DPO's implicit reward = beta * (logp_policy - logp_ref). Return chosen-rejected."""
    r_chosen = beta * (lp.policy_chosen - lp.ref_chosen)
    r_rejected = beta * (lp.policy_rejected - lp.ref_rejected)
    return r_chosen - r_rejected


def main() -> None:
    print("=== DPO data shape: (prompt, chosen, rejected) ===\n")
    for d in PREFERENCE_DATA:
        print("prompt  :", d["prompt"])
        print("chosen  :", d["chosen"])
        print("rejected:", d["rejected"])
        print()

    print("=== DPO loss behaviour (lower loss when chosen >> rejected) ===\n")
    scenarios = {
        "well-separated (good)": LogProbs(policy_chosen=-5.0, policy_rejected=-9.0, ref_chosen=-6.0, ref_rejected=-7.0),
        "tied (model unsure)": LogProbs(policy_chosen=-7.0, policy_rejected=-7.0, ref_chosen=-7.0, ref_rejected=-7.0),
        "inverted (bad)": LogProbs(policy_chosen=-9.0, policy_rejected=-5.0, ref_chosen=-7.0, ref_rejected=-7.0),
    }
    for name, lp in scenarios.items():
        loss = dpo_loss(lp, beta=0.1)
        margin = implied_reward_margin(lp, beta=0.1)
        print(f"  {name:24s} loss={loss:6.3f}  implicit_reward_margin={margin:+.3f}")

    print(
        "\nReading the numbers:\n"
        "  - 'well-separated' -> low loss: policy already prefers chosen over rejected.\n"
        "  - 'tied' -> loss = -log(0.5) = 0.693: no preference learned yet.\n"
        "  - 'inverted' -> high loss: gradient pushes hard to flip the preference.\n"
        "\nWhen to use DPO:\n"
        "  - You have preference signal (human/AI picked A over B), not gold answers.\n"
        "  - You want to shift style/tone/safety. Simpler + more stable than PPO RLHF.\n"
        "  - beta controls how far the policy may drift from the reference model."
    )


if __name__ == "__main__":
    main()
