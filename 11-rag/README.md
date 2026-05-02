# 11 — Retrieval-Augmented Generation (RAG)

RAG is how most customers inject **fresh / proprietary knowledge** without fine-tuning. The decision guide (module 04) says it loudly: *don't fine-tune for facts — that's RAG's job.* As a Field Engineer you'll design, debug, and tune RAG pipelines constantly.

## Files

| File | What it teaches |
| --- | --- |
| [`chunking.py`](chunking.py) | Splitting strategies (fixed, overlap, sentence) and why chunking decides RAG quality |
| [`vector_search.py`](vector_search.py) | Build embeddings + cosine similarity search from scratch (no deps), top-k retrieval |
| [`retrieval_metrics.py`](retrieval_metrics.py) | recall@k, precision@k, MRR, hit-rate — eval **retrieval** separately from generation |
| [`rag_notes.md`](rag_notes.md) | Architecture, embeddings, rerankers, failure modes, the FDE tuning loop |

## Run

```bash
python 11-rag/chunking.py
python 11-rag/vector_search.py
python 11-rag/retrieval_metrics.py
```

All stdlib-only (a toy bag-of-words embedding so it runs offline). In production you'd swap the embedder for a real model and the in-memory index for a vector DB.

## The RAG pipeline (memorize)

```
 docs ──► chunk ──► embed ──► index (vector DB)        [INGEST, offline]
                                       │
 query ──► embed ──► search top-k ──► (rerank) ──► stuff into prompt ──► LLM ──► answer
                                       └──────────── [SERVE, per request] ───────────┘
```

## The most important insight to say out loud

> **A bad RAG answer is usually a retrieval problem, not a model problem.** Eval retrieval (recall@k, MRR) and generation (groundedness, answer correctness) **separately** — otherwise you'll tune the wrong thing.

## Interview Q&A

1. **A customer's RAG bot gives wrong answers. How do you debug?**
   - Split the pipeline: first measure **retrieval** (is the right chunk in the top-k? recall@k/MRR). If retrieval is bad → fix chunking/embeddings/reranker/k. If retrieval is good but the answer is wrong → it's a generation/prompt problem (groundedness, context ordering, model). Don't fine-tune until you've isolated it.

2. **How does chunking affect quality?**
   - Too big → diluted embeddings + wasted context + higher cost; too small → lost context, answer split across chunks. Overlap preserves context across boundaries. Chunk on semantic/sentence boundaries, size to the embedding model + the question granularity. See `chunking.py`.

3. **When RAG vs fine-tuning vs long-context?**
   - RAG for changing/large/proprietary **knowledge**. Fine-tune for **behavior/format**. Long-context when the corpus is small enough to stuff and recall is good — but it's expensive per token and recall degrades ("lost in the middle"). Often RAG + a little SFT for format.

4. **What's a reranker and when do you add one?**
   - A second-stage model (cross-encoder) that re-scores the top-N candidates for relevance before they hit the prompt. Add it when first-stage vector recall is decent but precision is low (right doc is retrieved but ranked low). Costs latency; worth it for quality-sensitive apps.

5. **How do you make RAG fast and cheap at scale?**
   - Cache the embedding of repeated queries, cache retrieved context, **prefix-cache** the shared system prompt (modules 02/03), tune `k` down to what's needed, use a smaller embed model + ANN index, and consider hybrid (BM25 + vector) for recall. Tie to the load test (module 09).

## Ties to other modules

- Retrieval/generation eval → module 05 (`production_eval.md` has the RAG-specific metrics).
- Prefix caching for the shared prompt → modules 02/03.
- Long shared-context latency → system design Scenario 1 (module 10).
