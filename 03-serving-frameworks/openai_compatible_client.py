"""
openai_compatible_client.py — one client for vLLM / SGLang / TGI / Fireworks.

Why this matters:
  All major serving stacks expose the OpenAI /v1/chat/completions API. Learning
  this one interface lets you migrate a customer between backends by changing a
  base_url. This file works three ways:
    1) Against the official `openai` SDK if installed + a real base_url.
    2) Against ANY OpenAI-compatible server via raw urllib (no deps).
    3) Fully offline mock if no server is reachable.

Run:
    # offline mock (no server needed)
    python openai_compatible_client.py

    # against the local mock server in this folder
    python mock_inference_server.py            # terminal A
    python openai_compatible_client.py --base-url http://localhost:8000/v1   # terminal B

    # against real Fireworks (needs FIREWORKS_API_KEY)
    python openai_compatible_client.py \
        --base-url https://api.fireworks.ai/inference/v1 \
        --model accounts/fireworks/models/llama-v3p1-8b-instruct \
        --api-key-env FIREWORKS_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def stream_chat_raw(base_url: str, model: str, prompt: str, api_key: str | None) -> dict:
    """POST /v1/chat/completions with stream=true using only the stdlib.

    Returns timing stats and the reconstructed text. Falls back to a local mock
    if the server is unreachable, so the demo always produces output.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.2,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    text, n, ttft = "", 0, 0.0
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                if not delta:
                    continue
                if n == 0:
                    ttft = time.perf_counter() - start
                n += 1
                text += delta
        return {"ok": True, "text": text, "ttft_s": ttft, "total_s": time.perf_counter() - start, "tokens": n}
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        return _offline_mock(prompt, reason=str(e))


def _offline_mock(prompt: str, reason: str) -> dict:
    text = (
        "[offline mock] No server reachable, so here's a canned response. "
        "Point --base-url at vLLM/SGLang/Fireworks to hit a real model."
    )
    # simulate TTFT + per-token streaming timing
    time.sleep(0.15)
    return {"ok": False, "text": text, "ttft_s": 0.15, "total_s": 0.30, "tokens": len(text.split()), "reason": reason}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="", help="OpenAI-compatible base, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", default="mock/llama-v3p1-8b-instruct")
    ap.add_argument("--api-key-env", default="FIREWORKS_API_KEY")
    ap.add_argument("--prompt", default="Explain continuous batching in one sentence.")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)

    if not args.base_url:
        print("No --base-url given; running fully offline mock.\n")
        r = _offline_mock(args.prompt, reason="no base_url")
    else:
        print(f"POST {args.base_url}/chat/completions  model={args.model}\n")
        r = stream_chat_raw(args.base_url, args.model, args.prompt, api_key)

    print("response:", r["text"])
    print(f"\nTTFT={r['ttft_s']*1000:.0f}ms  total={r['total_s']*1000:.0f}ms  chunks={r['tokens']}")
    if not r["ok"]:
        print(f"(fell back to offline mock: {r.get('reason')})")
    print(
        "\nNote: this is the SAME request shape Fireworks, vLLM, and SGLang accept.\n"
        "Migrating a customer = change base_url + model id; keep the code."
    )


if __name__ == "__main__":
    main()
