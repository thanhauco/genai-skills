# RFT / RLHF notes — reinforcement fine-tuning

RFT (Reinforcement Fine-Tuning) optimizes a model against a **reward signal** instead of imitating gold outputs. This covers RLHF (reward model from human prefs) and verifiable-reward RL (RLVR), and the algorithms PPO / GRPO / and DPO's RL-free cousin.

## When RFT is the right tool

- **Success is checkable at scale**: code passes unit tests, math equals the answer, the tool call validates against a schema, the agent reaches the goal state. You can mint reward automatically (RLVR) — no human labels per sample.
- **You've plateaued** on SFT + DPO for a high-value task and need to squeeze more.
- **You need exploration**: the model should discover better strategies than your demonstrations show (RL can exceed the demo distribution; SFT can't).

If success isn't measurable and you don't have preference data, **don't** reach for RFT — go back to SFT/DPO/prompting.

## The pieces

```
            ┌─────────────┐     sample      ┌──────────────┐
   prompts ─►│   POLICY    │ ───responses──► │   REWARD      │  reward model OR
            │  (the LLM)  │                 │   FUNCTION    │  verifier (tests,
            └──────┬──────┘ ◄──advantage────└──────────────┘  schema, math check)
                   │   gradient update (PPO/GRPO), KL-anchored to a reference
                   ▼
            updated policy
```

- **Reward model (RLHF):** train a model on human (chosen, rejected) prefs to score responses. Then RL the policy to maximize that score.
- **Verifiable reward (RLVR):** skip the reward model — use a programmatic checker (unit tests, exact-match, JSON/tool validation). Cheaper, less reward-hacking, great for code/math/agents.
- **KL anchor:** keep the policy close to a reference model so it doesn't collapse / reward-hack. (DPO bakes this in via beta.)

## Algorithms to name

- **PPO** — classic RLHF optimizer; needs value model + careful tuning; heavier infra.
- **GRPO** — group-relative; drops the value model by normalizing rewards within a group of sampled responses; popular for reasoning/code RL, simpler + cheaper than PPO.
- **DPO / IPO / KTO** — RL-free preference optimization (covered in `dpo_example.py`); a great first step before full RL.
- **RFT (as a product)** — vendors (incl. Fireworks) expose RFT where you bring prompts + a grader/checker and they run the rollouts + updates.

## Reward design — where projects succeed or fail

- **Specify exactly what "good" means** and make it cheap to compute. Vague rewards → reward hacking.
- **Guard against gaming**: length bias, format exploits, verifier loopholes. Add penalties / multiple checks.
- **Mix rewards**: correctness + format + safety, weighted. Normalize so one term doesn't dominate.
- **Hold out a real eval** (module 05) — RL can climb the reward while regressing on what you actually care about.

## Fireworks' published angle (talk to this in an interview)

> "Frontier RL is cheaper than the mega-cluster narrative suggests: cross-region rollouts using 98% sparse weight deltas." Takeaway: the expensive part of RL is moving weights/data around for rollouts; if you transmit only the **sparse delta** of the policy and run rollouts across regions, you cut cost dramatically. This is the kind of production-systems insight an AI Field Engineer brings to customers doing RL.

## FDE checklist for an RFT engagement

1. Is success **programmatically checkable**? If yes → RLVR with a grader. If no → reward model or step back to DPO.
2. Have you exhausted **SFT + DPO** first? RFT is the expensive last mile.
3. Is there a **held-out production eval** that RL can't game?
4. Can you afford the **rollout compute** (or use sparse-delta / cross-region tricks to cut it)?
5. Do you have **monitoring** for reward hacking (inspect samples, not just the reward curve)?
