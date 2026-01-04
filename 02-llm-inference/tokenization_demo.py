"""
tokenization_demo.py — tokens are the unit of cost, context, and latency.

Why this matters:
  Pricing is per token. Context windows are in tokens. Latency scales with token
  counts. You must reason in tokens, not words/chars. A Field Engineer estimates
  "this prompt is ~1.3 tokens/word in English, more for code/JSON."

Uses tiktoken if available; otherwise falls back to a heuristic so it always runs.

Run:
    python tokenization_demo.py
"""

from __future__ import annotations

SAMPLES = {
    "english": "Fireworks delivers the fastest LLM inference in the industry.",
    "code": 'def add(a, b):\n    return a + b  # simple\n',
    "json": '{"model": "llama-v3p1-70b-instruct", "max_tokens": 512, "temperature": 0.2}',
    "repeated": "token " * 20,
}


def try_tiktoken():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s)), "tiktoken/cl100k_base"
    except Exception:
        # Heuristic fallback: ~1.3 tokens per whitespace word for English-ish text.
        def approx(s: str) -> int:
            words = max(1, len(s.split()))
            chars = len(s)
            # blend word- and char-based estimates (code/JSON skew higher)
            return int(round(max(words * 1.3, chars / 4)))

        return approx, "heuristic (install tiktoken for exact counts)"


def main() -> None:
    count, backend = try_tiktoken()
    print(f"tokenizer backend: {backend}\n")
    print(f"{'sample':10s} {'chars':>6s} {'words':>6s} {'tokens':>7s} {'tok/word':>9s}")
    for name, text in SAMPLES.items():
        toks = count(text)
        words = len(text.split())
        ratio = toks / max(1, words)
        print(f"{name:10s} {len(text):6d} {words:6d} {toks:7d} {ratio:9.2f}")

    print(
        "\nTakeaways:\n"
        "  - Code/JSON pack more tokens per word than prose (symbols, indentation).\n"
        "  - Cost & context budgeting must be done in tokens.\n"
        "  - For sizing: total_tokens = prompt_tokens + max_output_tokens, and KV\n"
        "    cache (module: kv_cache_calculator.py) scales with that sum."
    )


if __name__ == "__main__":
    main()
