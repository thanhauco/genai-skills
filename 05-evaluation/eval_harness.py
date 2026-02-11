"""
eval_harness.py — a minimal but real task-eval harness.

Pattern (the same shape as lm-eval-harness / promptfoo / your own):
    dataset[] -> run model -> score each -> aggregate -> report (with failures)

Why this matters:
  Every recommendation you make (model choice, quant, fine-tune) is only credible
  if you can MEASURE quality on the customer's task. This harness shows the bones:
  pluggable model fn + pluggable scorers + a report that surfaces FAILURES, not
  just an average.

Run:
    python eval_harness.py
Stdlib only; uses a mock model so it runs anywhere. Swap `mock_model` for a real
OpenAI-compatible call to evaluate a live endpoint.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field


# ----------------------------- dataset ----------------------------------------
@dataclass
class Example:
    id: str
    prompt: str
    expected: str
    kind: str  # "exact" | "regex" | "contains"


GOLDEN_SET = [
    Example("e1", "What HTTP status means rate limited?", "429", "exact"),
    Example("e2", "Return only the year ISO8601 starts: 0001 or 1970?", "1970", "contains"),
    Example("e3", "Give a regex-matchable order id like ORD-12345", r"ORD-\d{5}", "regex"),
    Example("e4", "What status means 'not found'?", "404", "exact"),
    Example("e5", "Name the attention memory structure reused across decode steps.", "kv cache", "contains"),
]


# ----------------------------- model under test -------------------------------
def mock_model(prompt: str) -> str:
    """Pretend to be an LLM. Deterministic-ish so the demo is stable.
    Note: e2 is intentionally wrong to show failure reporting."""
    time.sleep(0.005)
    p = prompt.lower()
    if "rate limited" in p:
        return "That would be a 429 Too Many Requests."
    if "iso8601 starts" in p:
        return "0001"  # WRONG on purpose -> should fail
    if "order id" in p:
        return "Sure: ORD-48217"
    if "not found" in p:
        return "404"
    if "attention memory structure" in p:
        return "The KV cache is reused across decode steps."
    return "I don't know."


# ----------------------------- scorers ----------------------------------------
def score_contains(pred: str, expected: str) -> bool:
    return expected.strip().lower() in pred.strip().lower()


def score_regex(pred: str, expected_pattern: str) -> bool:
    return re.search(expected_pattern, pred) is not None


SCORERS: dict[str, Callable[[str, str], bool]] = {
    "exact": score_contains,   # be lenient: accept if expected token present
    "contains": score_contains,
    "regex": score_regex,
}


# ----------------------------- runner -----------------------------------------
@dataclass
class Result:
    id: str
    passed: bool
    latency_ms: float
    pred: str
    expected: str


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def accuracy(self) -> float:
        return self.passed / self.n if self.n else 0.0

    def latency_pct(self, p: float) -> float:
        lat = sorted(r.latency_ms for r in self.results)
        return lat[min(len(lat) - 1, int(p / 100 * len(lat)))] if lat else 0.0

    def failures(self) -> list[Result]:
        return [r for r in self.results if not r.passed]


def run_eval(model: Callable[[str], str], dataset: list[Example]) -> Report:
    rep = Report()
    for ex in dataset:
        t0 = time.perf_counter()
        pred = model(ex.prompt)
        dt = (time.perf_counter() - t0) * 1000
        scorer = SCORERS[ex.kind]
        rep.results.append(Result(ex.id, scorer(pred, ex.expected), dt, pred, ex.expected))
    return rep


def main() -> None:
    rep = run_eval(mock_model, GOLDEN_SET)
    print("=== eval report ===")
    print(f"accuracy = {rep.passed}/{rep.n} = {rep.accuracy:.0%}")
    print(f"latency  p50={rep.latency_pct(50):.1f}ms  p95={rep.latency_pct(95):.1f}ms")
    print("\nfailures (this is the part you actually act on):")
    for r in rep.failures():
        print(f"  [{r.id}] expected~{r.expected!r}  got={r.pred!r}")

    print(
        "\nHow you'd extend this for a customer:\n"
        "  - swap mock_model() for an OpenAI-compatible call to their endpoint\n"
        "  - golden set = sampled real inputs incl. edge cases (no train leakage)\n"
        "  - add scorers: JSON-schema valid, unit-test pass, embedding similarity\n"
        "  - wire as a CI gate: block deploy if accuracy drops or p95 SLO breaks"
    )


if __name__ == "__main__":
    main()
