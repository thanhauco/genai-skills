"""
chunking.py — text splitting strategies (the decision that makes-or-breaks RAG).

Why this matters:
  Chunking determines what can be retrieved. Too big -> diluted embeddings, wasted
  context, higher cost. Too small -> lost context, answers split across chunks.
  Overlap preserves context across boundaries. You must be able to reason about
  the trade-off and show a few strategies.

Run:
    python chunking.py
Stdlib only.
"""

from __future__ import annotations

import re

SAMPLE = (
    "Fireworks serves fast LLM inference. It uses continuous batching to keep GPUs busy. "
    "Prefix caching reuses shared prompt prefixes, which helps RAG and agents. "
    "Quantization to fp8 frees HBM for more KV cache. More KV cache means more concurrency. "
    "Higher concurrency lowers the cost per million tokens. Always gate changes with an eval."
)


def chunk_fixed(text: str, size: int) -> list[str]:
    """Fixed-size character chunks. Simple, but cuts mid-sentence (bad embeddings)."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def chunk_overlap(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window with overlap so context isn't lost at boundaries."""
    if overlap >= size:
        raise ValueError("overlap must be < size")
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step)]


def chunk_sentences(text: str, max_chars: int) -> list[str]:
    """Pack whole sentences up to max_chars — respects semantic boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur.strip())
    return chunks


def report(name: str, chunks: list[str]) -> None:
    sizes = [len(c) for c in chunks]
    print(f"\n{name}: {len(chunks)} chunks, sizes={sizes}")
    for i, c in enumerate(chunks):
        preview = c if len(c) <= 70 else c[:67] + "..."
        print(f"  [{i}] {preview!r}")


def main() -> None:
    print(f"source text: {len(SAMPLE)} chars")
    report("fixed(80) - note mid-word cuts", chunk_fixed(SAMPLE, 80))
    report("overlap(80, 20) - boundary safety", chunk_overlap(SAMPLE, 80, 20))
    report("sentences(<=120) - semantic", chunk_sentences(SAMPLE, 120))

    print(
        "\nField guidance:\n"
        "  - Prefer SEMANTIC boundaries (sentence/paragraph/markdown headings).\n"
        "  - Add OVERLAP (~10-20%) so answers spanning a boundary survive.\n"
        "  - Size chunks to the embedding model + the QUESTION granularity\n"
        "    (FAQ-style -> small; narrative reasoning -> larger).\n"
        "  - Store metadata (source, section) for citations + filtering.\n"
        "  - Chunking is the #1 lever on retrieval quality — tune it FIRST."
    )


if __name__ == "__main__":
    main()
