"""
llm_as_judge.py — the pattern + the bias mitigations.

Why this matters:
  For open-ended outputs (helpfulness, tone, "which answer is better?") you can't
  use exact-match. LLM-as-judge scales human-like judgment, but it has well-known
  biases. You must show you know how to mitigate them.

This file:
  - Implements a PAIRWISE judge (A vs B) and a RUBRIC judge (score 1-5).
  - Uses a deterministic MOCK judge so it runs offline; swap in a real model call.
  - Demonstrates POSITION-BIAS mitigation: ask twice with A/B swapped and average.

Run:
    python llm_as_judge.py
Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    name: str
    text: str


# A mock "judge". In reality this is a strong model called with a careful prompt.
# Here we approximate quality with simple, transparent heuristics so output is
# stable and you can see the MECHANISM (not the model).
def mock_judge_pairwise(prompt: str, a: str, b: str) -> str:
    """Return 'A', 'B', or 'tie'. Mock: prefers the more specific, cited answer."""
    def quality(t: str) -> float:
        score = 0.0
        score += min(len(t.split()) / 20.0, 1.0)        # some length, capped
        score += 1.0 if any(c.isdigit() for c in t) else 0.0  # specificity
        score += 1.0 if "[doc:" in t or "http" in t else 0.0  # citation
        score -= 0.5 if len(t.split()) > 80 else 0.0     # penalize rambling
        return score

    qa, qb = quality(a), quality(b)
    if abs(qa - qb) < 0.25:
        return "tie"
    return "A" if qa > qb else "B"


def judge_with_position_swap(prompt: str, x: Candidate, y: Candidate) -> str:
    """Mitigate position bias: judge (x,y) AND (y,x), then reconcile."""
    r1 = mock_judge_pairwise(prompt, x.text, y.text)   # x is A
    r2 = mock_judge_pairwise(prompt, y.text, x.text)   # y is A
    # Translate both back into "winner name"
    w1 = x.name if r1 == "A" else y.name if r1 == "B" else "tie"
    w2 = y.name if r2 == "A" else x.name if r2 == "B" else "tie"
    if w1 == w2:
        return w1
    return "tie"  # judge is inconsistent across order -> call it a tie (honest)


def mock_judge_rubric(prompt: str, answer: str) -> dict:
    """Absolute 1-5 score on a fixed rubric. Returns score + reasons."""
    reasons = []
    score = 3
    if any(c.isdigit() for c in answer):
        score += 1
        reasons.append("+specific (contains concrete value)")
    if "[doc:" in answer or "http" in answer:
        score += 1
        reasons.append("+cited a source")
    if len(answer.split()) > 80:
        score -= 1
        reasons.append("-too verbose")
    if "i don't know" in answer.lower():
        score -= 2
        reasons.append("-non-answer")
    score = max(1, min(5, score))
    return {"score": score, "reasons": reasons}


def main() -> None:
    prompt = "Why am I getting a 429 and how do I fix it?"
    a = Candidate("modelA", "A 429 means you exceeded the rate limit. Back off with jitter and retry. [doc:limits-7]")
    b = Candidate("modelB", "It's an error, try again later.")

    print("=== Pairwise judge with position-swap (bias mitigation) ===")
    winner = judge_with_position_swap(prompt, a, b)
    print(f"  winner: {winner}")
    print(f"  (single-order would risk position bias; we swap A/B and require agreement)\n")

    print("=== Rubric judge (absolute 1-5) ===")
    for c in (a, b):
        r = mock_judge_rubric(prompt, c.text)
        print(f"  {c.name}: score={r['score']}  reasons={r['reasons']}")

    print(
        "\nLLM-as-judge bias controls to name in an interview:\n"
        "  - position bias -> swap order, average / require agreement\n"
        "  - verbosity bias -> penalize length in rubric; prefer concise+correct\n"
        "  - self-preference -> don't let a model grade its own family blindly\n"
        "  - calibrate the judge against a small set of HUMAN labels\n"
        "  - prefer PAIRWISE (A vs B) over absolute scores; it's more reliable"
    )


if __name__ == "__main__":
    main()
