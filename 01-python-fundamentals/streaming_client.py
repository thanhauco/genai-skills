"""
streaming_client.py — parse an SSE-style token stream and measure TTFT/TPOT.

Why this matters:
  Chat + agent UX depends on streaming. The two metrics customers feel are
  TTFT (time to first token) and TPOT (time per output token). You must be able
  to instrument a stream and report these.

This file simulates the byte stream an OpenAI-compatible server sends:
    data: {"choices":[{"delta":{"content":"Hel"}}]}
    data: {"choices":[{"delta":{"content":"lo"}}]}
    data: [DONE]

Run:
    python streaming_client.py
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field


def simulated_sse_stream(text: str, ttft_s: float = 0.20, tpot_s: float = 0.03) -> Iterator[bytes]:
    """Yield SSE lines like a real /v1/chat/completions stream=true endpoint."""
    # First chunk is delayed by prefill (TTFT). Subsequent chunks by TPOT (decode).
    chunks = _chunk(text, size=3)
    for i, piece in enumerate(chunks):
        time.sleep(ttft_s if i == 0 else tpot_s)
        payload = {"choices": [{"delta": {"content": piece}}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _chunk(s: str, size: int) -> list[str]:
    return [s[i : i + size] for i in range(0, len(s), size)]


@dataclass
class StreamStats:
    ttft_s: float = 0.0
    total_s: float = 0.0
    n_tokens: int = 0  # here: chunks, a proxy for tokens
    text: str = ""
    inter_token_s: list[float] = field(default_factory=list)

    @property
    def tpot_s(self) -> float:
        return sum(self.inter_token_s) / len(self.inter_token_s) if self.inter_token_s else 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.n_tokens / self.total_s if self.total_s else 0.0


def consume_stream(stream: Iterator[bytes]) -> StreamStats:
    """Parse SSE bytes, reconstruct text, and time TTFT / inter-token latency."""
    stats = StreamStats()
    start = time.perf_counter()
    last_token_t: float | None = None

    for raw in stream:
        line = raw.decode().strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break

        obj = json.loads(data)
        delta = obj["choices"][0]["delta"].get("content", "")
        if not delta:
            continue

        now = time.perf_counter()
        if stats.n_tokens == 0:
            stats.ttft_s = now - start  # first visible token
        else:
            stats.inter_token_s.append(now - last_token_t)  # type: ignore[arg-type]
        last_token_t = now
        stats.n_tokens += 1
        stats.text += delta

    stats.total_s = time.perf_counter() - start
    return stats


def main() -> None:
    answer = "Fireworks serves the fastest LLM inference in the industry."
    stats = consume_stream(simulated_sse_stream(answer))

    print("Reconstructed:", repr(stats.text))
    print(f"TTFT       : {stats.ttft_s*1000:6.1f} ms   (prefill latency the user feels first)")
    print(f"TPOT (avg) : {stats.tpot_s*1000:6.1f} ms   (decode latency per token)")
    print(f"tokens/s   : {stats.tokens_per_s:6.1f}")
    print(f"total      : {stats.total_s*1000:6.1f} ms")
    print(
        "\nInterview line: 'Total latency ~= TTFT + (out_tokens-1) x TPOT. "
        "We optimize TTFT with prefill tricks and TPOT with batching/quantization.'"
    )


if __name__ == "__main__":
    main()
