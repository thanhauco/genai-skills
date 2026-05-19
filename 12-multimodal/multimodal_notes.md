# Multimodal inference notes

Fireworks builds multimodal models. Customers want vision (image understanding, OCR, document AI, UI/screenshot analysis) and sometimes audio. Here's what you need to reason about deployments.

## How a vision-language model (VLM) works

```
  image ──► vision encoder (ViT) ──► projector/adapter ──► image tokens ─┐
                                                                         ├─► LLM decoder ─► text
  text  ───────────────────────────────────────────────► text tokens ───┘
```

- A **vision encoder** (usually a ViT) turns the image into embeddings.
- A **projector** maps those into the LLM's token space → a block of **image tokens**.
- The LLM decodes over **image tokens + text tokens** in one shared context.

Key consequence: **an image consumes context tokens** just like text. Resolution/detail → number of image tokens → cost + latency.

## The image-token cost model (why vision is prefill-heavy)

- Low-detail / thumbnail pass = small fixed token cost.
- High-detail = the image is **tiled** (e.g., 512px tiles); each tile adds tokens.
- A large screenshot or scanned page can be **1k–3k+ tokens** before any text.
- That's **prefill** work → higher **TTFT**, and it occupies **KV cache** → affects concurrency (module 02).

See `multimodal_message.py` for the request shape + a token estimate.

## Use cases you'll see

| Use case | Notes |
| --- | --- |
| Document AI / OCR + Q&A | invoices, forms, contracts; layout matters |
| Image understanding / VQA | describe, classify, find defects |
| UI / screenshot analysis | agents that "see" a screen (ties to module 06) |
| Charts / diagrams | extract structured data from visuals |
| Audio (ASR → LLM) | transcribe then reason; or native audio models |

## Architectural decision: end-to-end VLM vs pipeline

```
 VLM end-to-end                          OCR/ASR -> text LLM pipeline
 ──────────────                          ────────────────────────────
 + joint reasoning over image+text       + cheaper for clean text at scale
 + handles layout/handwriting/visuals    + each stage is separately eval-able
 + simpler pipeline                       + more control + easier debugging
 - image tokens cost $ + latency          - loses visual layout/context
 - can hallucinate visual details         - two systems to maintain
```

**Rule**: VLM for visually complex / mixed content; OCR(+layout) → text LLM for clean, high-volume text. **Benchmark both** on the customer's real documents: accuracy, **$/page**, latency.

## Serving trade-offs

- vLLM / SGLang support many VLMs; the **vision encoder adds compute** and **image tokens inflate KV cache** — include them in the concurrency/cost math.
- **Cache encoded images** when the same image repeats across requests.
- **Right-size resolution/detail** — the single biggest cost lever.
- Throughput per GPU is **lower** for vision than pure text (encoder + longer context).

## Evaluation

- Don't trust generic VQA benchmarks — **eval on the customer's images** (their forms, their UI, their defects).
- Metrics: task accuracy / field-extraction F1 / grounding; plus **$/page** and **p95 latency**.
- Watch for **hallucinated** values (a VLM may "read" a number that isn't there) — add validation/confidence checks for high-stakes extraction.

## The sentence that lands

> "Each page in high detail is ~1.5–3k image tokens, so your doc-AI cost is dominated by image tokens, not the answer. We'll test the lowest detail that still passes your extraction eval, and compare end-to-end VLM against OCR-plus-text-LLM on accuracy and $/page before we commit."
