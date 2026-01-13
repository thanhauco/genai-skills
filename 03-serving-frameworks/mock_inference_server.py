"""
mock_inference_server.py — a tiny OpenAI-compatible endpoint, stdlib only.

Why this exists:
  Lets you practice the serving + client side WITHOUT a GPU, a model, or any
  dependency. It mimics POST /v1/chat/completions for both stream=false and
  stream=true (SSE), so the same client code you'd point at vLLM / SGLang /
  Fireworks works here.

Run:
    python mock_inference_server.py            # listens on :8000
    python mock_inference_server.py --port 9000

Then in another terminal:
    python openai_compatible_client.py --base-url http://localhost:8000/v1
    # or just: curl http://localhost:8000/v1/models
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "mock/llama-v3p1-8b-instruct"


def _fake_completion(prompt: str) -> str:
    return (
        "This is a mock completion. In production this endpoint would be vLLM, "
        "SGLang, or Fireworks serving a real model. Your prompt had "
        f"{len(prompt.split())} words."
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quieter logs
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        messages = req.get("messages", [])
        prompt = messages[-1].get("content", "") if messages else ""
        stream = bool(req.get("stream", False))
        text = _fake_completion(prompt)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if not stream:
            self._json(
                200,
                {
                    "id": cid,
                    "object": "chat.completion",
                    "model": req.get("model", MODEL_ID),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(text.split()),
                        "total_tokens": len(prompt.split()) + len(text.split()),
                    },
                },
            )
            return

        # Streaming: Server-Sent Events, chunk by ~3 chars to imitate tokens.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        time.sleep(0.15)  # simulate TTFT (prefill)
        for i in range(0, len(text), 3):
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": req.get("model", MODEL_ID),
                "choices": [{"index": 0, "delta": {"content": text[i : i + 3]}}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.02)  # simulate TPOT (decode)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Mock OpenAI-compatible server on http://{args.host}:{args.port}/v1  (Ctrl+C to stop)")
    print(f"  model: {MODEL_ID}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
