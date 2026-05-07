"""
retrieval_metrics.py — eval RETRIEVAL separately from generation.

Why this matters:
  A bad RAG answer is usually a retrieval problem. You must measure whether the
  right chunk made it into the top-k BEFORE blaming the model. These are the
  standard IR metrics.

Metrics:
  - hit@k        : did ANY relevant doc appear in the top-k?
  - recall@k     : fraction of all relevant docs found in the top-k
  - precision@k  : fraction of the top-k that are relevant
  - MRR          : mean reciprocal rank of the FIRST relevant doc (rank sensitivity)
  - nDCG@k       : rank-weighted gain (graded relevance)

Run:
    python retrieval_metrics.py
Stdlib only.
"""

from __future__ import annotations

import math


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = sum(1 for r in retrieved[:k] if r in relevant)
    return found / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    found = sum(1 for r in retrieved[:k] if r in relevant)
    return found / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    dcg = sum((1.0 / math.log2(i + 1)) for i, r in enumerate(retrieved[:k], 1) if r in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mean_reciprocal_rank(queries: list[tuple[list[str], set[str]]]) -> float:
    return sum(reciprocal_rank(ret, rel) for ret, rel in queries) / len(queries) if queries else 0.0


def main() -> None:
    # (retrieved doc ids in rank order, set of relevant doc ids) per query
    eval_set = [
        (["d2", "d5", "d1", "d9"], {"d1", "d2"}),   # both relevant retrieved, one high
        (["d7", "d3", "d4", "d1"], {"d1"}),         # relevant doc is last -> low MRR
        (["d8", "d6", "d0", "d5"], {"d2"}),         # miss entirely
    ]

    print(f"{'query':6s} {'hit@3':>6s} {'rec@3':>6s} {'prec@3':>7s} {'RR':>6s} {'nDCG@3':>7s}")
    for i, (ret, rel) in enumerate(eval_set):
        print(
            f"q{i:<5d} {hit_at_k(ret, rel, 3):6.2f} {recall_at_k(ret, rel, 3):6.2f} "
            f"{precision_at_k(ret, rel, 3):7.2f} {reciprocal_rank(ret, rel):6.2f} {ndcg_at_k(ret, rel, 3):7.2f}"
        )

    print(f"\nMRR (across queries) = {mean_reciprocal_rank(eval_set):.3f}")
    print(
        "\nHow to use this in the field:\n"
        "  1) Build a small (query -> known-relevant-chunk) gold set.\n"
        "  2) Measure recall@k FIRST. Low recall -> fix chunking/embeddings/k/reranker.\n"
        "  3) Only once recall is good do you debug GENERATION (groundedness, prompt).\n"
        "  4) MRR/nDCG tell you if the right doc is retrieved but ranked too low\n"
        "     (-> add a reranker). This is how you avoid 'fixing' the wrong stage."
    )


if __name__ == "__main__":
    main()
