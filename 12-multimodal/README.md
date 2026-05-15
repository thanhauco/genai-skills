# 12 — Multimodal Inference

Fireworks ships **multimodal models** (the JD calls out "multimodal models" as cutting-edge work). Customers want vision (image understanding, OCR, document AI), and sometimes audio. You should understand how VLMs are served, how they cost tokens, and the trade-offs.

## Files

| File | What it teaches |
| --- | --- |
| [`multimodal_message.py`](multimodal_message.py) | Build OpenAI-compatible multimodal messages (image_url) + estimate image token cost |
| [`multimodal_notes.md`](multimodal_notes.md) | VLM architecture, image-token cost, audio, use cases, serving trade-offs |

## Run

```bash
python 12-multimodal/multimodal_message.py
```

Stdlib only — constructs the request shape and does the image-token math (no model/API needed).

## VLM mental model (one diagram)

```
  image ──► vision encoder (ViT) ──► projector ──► image "tokens" ─┐
                                                                   ├──► LLM decoder ──► text
  text  ─────────────────────────────────────────► text tokens ───┘
```

- An image becomes a **block of tokens** that occupy the **same context window** as text → images cost context + latency + money.
- More image detail/resolution → more image tokens (often via tiling) → more cost. This is the key sizing fact.

## The cost fact to say out loud

> "An image isn't free — it's hundreds to thousands of tokens depending on resolution/detail. High-detail mode tiles the image into more patches → more tokens → higher TTFT and cost. For high-volume vision, pick the lowest detail that passes the eval."

## Interview Q&A

1. **How does an image affect latency and cost in a VLM call?**
   - The image is encoded into image tokens that share the context window. They add prefill work (TTFT) and token cost. Higher resolution / "high detail" tiling = more tokens. So vision requests are usually prefill-heavy; budget context and pick detail level by eval.

2. **A customer wants document understanding (OCR + Q&A) at scale. Trade-offs?**
   - VLM end-to-end is simpler and handles layout/handwriting, but costs image tokens per page and can hallucinate. A dedicated OCR + text-LLM pipeline can be cheaper and more controllable for clean docs. Benchmark both on their docs (accuracy + $/page + latency) and decide. Often: OCR for clean text, VLM for complex/visual layouts.

3. **How do you serve VLMs efficiently?**
   - vLLM/SGLang support many VLMs; the vision encoder adds compute and the image tokens inflate KV cache, so concurrency math (module 02) must include image-token length. Cache encoded images if the same image repeats. Right-size resolution/detail.

4. **When multimodal vs separate models?**
   - One multimodal model = simpler pipeline, joint reasoning over image+text. Separate (OCR/ASR → text LLM) = more control, often cheaper for narrow tasks, easier to eval each stage. Pick by accuracy/cost/latency on the real workload.

5. **Audio?**
   - Speech-to-text (e.g., Whisper-class) → text LLM is the common pattern; some models take audio directly. Same principle: audio becomes tokens/features that cost context + compute.

## Ties to other modules

- Image tokens → KV-cache + concurrency math (module 02).
- Cost per request/page → cost estimator + load test (modules 08/09).
- Eval the vision task on the customer's data (module 05) — benchmark VQA ≠ their documents.
