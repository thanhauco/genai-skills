# RAG architecture, tuning & failure modes

The reference for designing and debugging Retrieval-Augmented Generation with a customer. RAG is the default way to give a model **fresh / proprietary knowledge** — not fine-tuning.

## Architecture

```
 INGEST (offline)                          SERVE (per request)
 ───────────────                           ───────────────────
 documents                                 user query
    │ load + clean                            │ embed (same model!)
    ▼                                         ▼
 chunk (size + overlap + metadata)         vector search top-k  ◄── index
    │ embed                                   │
    ▼                                         ▼ (optional) rerank top-N -> top-k
 vector index (DB / ANN) ───────────────►  build prompt: system + retrieved context + query
                                              │
                                              ▼
                                           LLM ──► answer (+ citations)
```

## The components and the knobs

| Stage | Knobs | Failure if wrong |
| --- | --- | --- |
| **Chunking** | size, overlap, boundary (sentence/heading) | context split across chunks; diluted embeddings |
| **Embedding** | model quality, dimension, domain fit | semantically similar text doesn't match |
| **Index / search** | ANN params, `k`, hybrid (BM25+vector) | misses rare terms; recall too low |
| **Reranker** | cross-encoder, top-N → top-k | right doc retrieved but ranked low (no rerank) |
| **Prompt assembly** | order, dedup, max context, citation format | "lost in the middle"; hallucination; token bloat |
| **Generation** | model, temperature, grounding instructions | ignores context; not grounded |

## Eval RAG in two halves (the key discipline)

```
 RETRIEVAL quality                 GENERATION quality
 ─────────────────                 ──────────────────
 recall@k, precision@k             groundedness / faithfulness
 MRR, nDCG                         answer correctness
 (is the right chunk in top-k?)    (did it USE the context correctly?)
```

> Debug order: **retrieval first**. If recall@k is low, no model can save you. Only when the right chunk is reliably retrieved do you debug the prompt/model. See `retrieval_metrics.py` and module 05 `production_eval.md`.

## Common failure modes → fixes

- **"It makes things up"** → low retrieval recall (fix chunking/embeddings/k/hybrid) OR weak grounding instruction OR context too long ("lost in the middle" — trim + reorder, put key chunks first/last).
- **"Right info exists but it didn't find it"** → embedding mismatch or rare-term miss → add **hybrid search** (BM25 + vector) and/or a better embed model.
- **"Found it but answered from the wrong chunk"** → ranking issue → add a **reranker**; reduce `k` noise.
- **"Too slow / expensive"** → cache query embeddings + retrieved context, **prefix-cache** the shared system prompt (modules 02/03), shrink `k`, smaller embed model + ANN.
- **"Stale answers"** → re-index on a schedule; add freshness metadata + filtering.

## RAG vs fine-tuning vs long-context

| Need | Use |
| --- | --- |
| Changing / large / proprietary **knowledge** | **RAG** |
| **Behavior / format / tone** | Fine-tune (module 04) |
| Small corpus, want simplicity, recall is fine | **Long-context** (stuff it) — but $$ per token + "lost in the middle" |
| Both knowledge **and** format | RAG + a little SFT |

## Advanced patterns (name-drop-able)

- **Hybrid search** — BM25 (lexical) + dense (semantic), fused (e.g., RRF). Best recall.
- **Reranking** — cross-encoder re-scores candidates; big precision win.
- **Query rewriting / HyDE** — expand or hypothesize the query to improve recall.
- **Multi-hop / agentic RAG** — iterate retrieval with reasoning (ties to module 06).
- **Contextual / late chunking** — embed chunks with surrounding context for better vectors.
- **Metadata filtering** — restrict by source/date/permissions before vector search.

## FDE tuning loop for RAG

1. Build a **(query → relevant chunk)** gold set from real questions.
2. Measure **recall@k / MRR**; fix chunking/embeddings/hybrid/reranker until recall is high.
3. Then measure **groundedness + correctness**; fix prompt/order/model.
4. **Load-test** latency + cost (module 09); cache + prefix-cache to hit the SLO.
5. **Eval-gate** the whole pipeline; re-run on any change.
6. Codify the config as a repeatable pattern; feed gaps to product.

## The sentence that lands

> "Your bot isn't hallucinating because the model is bad — recall@5 on your eval is 40%, so the right chunk usually isn't even in context. Fix chunking + add hybrid retrieval and a reranker, get recall above ~90%, then we tune the prompt. Cheaper and faster than fine-tuning, and it stays current as your docs change."
