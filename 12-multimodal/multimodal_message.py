"""
multimodal_message.py — build OpenAI-compatible multimodal requests + cost them.

Why this matters:
  Vision-language models (VLMs) take images alongside text. The request shape is
  standardized (content is a LIST of parts: text + image_url). Critically, an
  image is NOT free: it becomes a block of TOKENS that share the context window,
  adding prefill latency (TTFT) and cost. You must be able to estimate that.

This builds the message payload and approximates image-token cost the way tiled
VLMs do (low detail = 1 tile; high detail = many tiles). Numbers are illustrative
to teach the SHAPE of the cost, not an exact vendor formula.

Run:
    python multimodal_message.py
Stdlib only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


def build_message(text: str, image_url: str, detail: str = "auto") -> dict:
    """OpenAI/Fireworks-compatible multimodal user message."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url, "detail": detail}},
        ],
    }


@dataclass
class ImageCost:
    width: int
    height: int
    detail: str
    tiles: int
    tokens: int


def estimate_image_tokens(
    width: int,
    height: int,
    detail: str = "high",
    *,
    base_tokens: int = 85,        # cost of the low-res thumbnail pass
    tokens_per_tile: int = 170,   # cost per high-detail tile
    tile_px: int = 512,
) -> ImageCost:
    """Approximate image-token cost for a tiled VLM.

    low  detail -> just the base thumbnail pass (cheap, fixed).
    high detail -> base + tokens_per_tile * (number of 512px tiles).
    """
    if detail == "low":
        return ImageCost(width, height, detail, tiles=0, tokens=base_tokens)
    tiles_w = math.ceil(width / tile_px)
    tiles_h = math.ceil(height / tile_px)
    tiles = tiles_w * tiles_h
    tokens = base_tokens + tokens_per_tile * tiles
    return ImageCost(width, height, detail, tiles, tokens)


def main() -> None:
    msg = build_message(
        "What's wrong with this architecture diagram?",
        "https://example.com/diagram.png",
        detail="high",
    )
    print("=== multimodal message payload (same shape vLLM/SGLang/Fireworks accept) ===")
    print(json.dumps(msg, indent=2))

    print("\n=== image-token cost grows with resolution + detail ===")
    print(f"{'image':14s} {'detail':6s} {'tiles':>5s} {'img_tokens':>11s}")
    cases = [
        (512, 512, "low"),
        (512, 512, "high"),
        (1024, 1024, "high"),
        (2048, 1536, "high"),   # a big screenshot / scanned page
    ]
    for w, h, d in cases:
        c = estimate_image_tokens(w, h, d)
        print(f"{w}x{h:<8d} {d:6s} {c.tiles:5d} {c.tokens:11d}")

    # what that means for a per-request budget
    big = estimate_image_tokens(2048, 1536, "high")
    prompt_text_tokens = 40
    max_output = 300
    total = big.tokens + prompt_text_tokens + max_output
    print(
        f"\nA 2048x1536 high-detail image ~= {big.tokens} tokens. With {prompt_text_tokens} text +\n"
        f"{max_output} output tokens, this single request occupies ~{total} context tokens —\n"
        "that's prefill work (TTFT) and KV cache (module 02) BEFORE any text reasoning."
    )
    print(
        "\nField guidance:\n"
        "  - Pick the LOWEST detail that passes the eval; high detail multiplies tokens.\n"
        "  - Downscale images client-side before sending if full res isn't needed.\n"
        "  - For high-volume doc AI, compare VLM end-to-end vs OCR + text-LLM on\n"
        "    accuracy AND $/page AND latency (see multimodal_notes.md).\n"
        "  - Include image tokens in your concurrency/cost math (modules 02/08/09)."
    )


if __name__ == "__main__":
    main()
