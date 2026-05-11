"""
vector_search.py — embeddings + cosine similarity search, from scratch.

Why this matters:
  The retrieval core of RAG is: embed text -> compare vectors by cosine similarity
  -> return top-k. This implements it with ZERO dependencies using a toy
  bag-of-words embedding, so you can see the mechanism. In production you swap the
  embedder for a real model (e.g., an embedding endpoint) and the in-memory loop
  for a vector DB / ANN index (FAISS, pgvector, Pinecone, etc.).

Run:
    python vector_search.py
Stdlib only.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# A tiny knowledge base (chunks you'd have indexed).
CORPUS = [
    "Continuous batching keeps the GPU busy by admitting new requests every step.",
    "Prefix caching reuses the KV cache of shared prompt prefixes across requests.",
    "Quantization to fp8 reduces weight memory and frees HBM for more KV cache.",
    "Tensor parallelism shards a layer across GPUs connected by NVLink.",
    "A 429 status code means you are being rate limited; back off with jitter.",
    "LoRA trains small low-rank adapters on top of a frozen base model.",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(text: str, vocab: dict[str, int]) -> list[float]:
    """Toy embedding: L2-normalized term-frequency vector over a fixed vocab.
    Real systems use a learned embedding model; cosine search is identical."""
    counts = Counter(tokenize(text))
    vec = [0.0] * len(vocab)
    for term, c in counts.items():
        idx = vocab.get(term)
        if idx is not None:
            vec[idx] = float(c)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (already comparable)."""
    return sum(x * y for x, y in zip(a, b))


def build_vocab(corpus: list[str]) -> dict[str, int]:
    terms: list[str] = []
    seen: set[str] = set()
    for doc in corpus:
        for t in tokenize(doc):
            if t not in seen:
                seen.add(t)
                terms.append(t)
    return {t: i for i, t in enumerate(terms)}


@dataclass
class Hit:
    score: float
    doc: str
    index: int


class VectorIndex:
    """In-memory cosine index. Stand-in for a real vector DB."""

    def __init__(self, corpus: list[str]) -> None:
        self.corpus = corpus
        self.vocab = build_vocab(corpus)
        self.vectors = [embed(doc, self.vocab) for doc in corpus]

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = embed(query, self.vocab)
        scored = [Hit(cosine(q, v), self.corpus[i], i) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


def main() -> None:
    index = VectorIndex(CORPUS)
    print(f"indexed {len(CORPUS)} chunks, vocab size {len(index.vocab)}\n")

    for query in [
        "how do I get more KV cache?",
        "what does rate limiting look like?",
        "how to fine-tune cheaply?",
    ]:
        print(f"QUERY: {query}")
        for rank, hit in enumerate(index.search(query, k=2), 1):
            print(f"  {rank}. score={hit.score:.3f}  [{hit.index}] {hit.doc}")
        print()

    print(
        "Production swaps to make:\n"
        "  - embedder: toy BoW -> a real embedding model (better semantic match)\n"
        "  - index: linear scan -> ANN (FAISS/HNSW) or a vector DB for scale\n"
        "  - add HYBRID search (BM25 + vector) for recall on rare terms\n"
        "  - add a RERANKER (cross-encoder) over the top-N for precision\n"
        "  - measure retrieval with recall@k / MRR (see retrieval_metrics.py)"
    )


if __name__ == "__main__":
    main()
