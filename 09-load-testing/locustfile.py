"""
locustfile.py — load-test an OpenAI-compatible LLM endpoint with Locust.

Locust is a popular open-source load tool with a live web UI. This scenario
streams chat completions and records a custom TTFT metric (Locust measures
total request time by default; for LLMs you want TTFT too).

Install + run:
    pip install locust
    locust -f locustfile.py --host http://localhost:8000
    # open http://localhost:8089, set users + spawn rate, watch p95 live
    # headless:
    locust -f locustfile.py --host http://localhost:8000 \
        --users 50 --spawn-rate 5 --run-time 2m --headless

Env:
    LLM_MODEL    model id (default mock/llama-v3p1-8b-instruct)
    LLM_API_KEY  bearer token if the endpoint requires it

If Locust isn't installed, this file just prints guidance (so it won't crash).
"""

from __future__ import annotations

import json
import os
import time

try:
    from locust import HttpUser, between, events, task

    _HAS_LOCUST = True
except Exception:  # noqa: BLE001
    _HAS_LOCUST = False


MODEL = os.environ.get("LLM_MODEL", "mock/llama-v3p1-8b-instruct")
API_KEY = os.environ.get("LLM_API_KEY")

PROMPTS = [
    "Explain continuous batching in one sentence.",
    "What causes a 429 and how should a client handle it?",
    "Summarize the trade-off between TTFT and throughput.",
    "Write a haiku about GPUs.",
]


if _HAS_LOCUST:

    class LLMUser(HttpUser):
        # think-time between requests; tune to model a realistic arrival pattern
        wait_time = between(0.1, 1.0)

        def _headers(self) -> dict:
            h = {"Content-Type": "application/json"}
            if API_KEY:
                h["Authorization"] = f"Bearer {API_KEY}"
            return h

        @task
        def chat_stream(self):
            import random

            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
                "max_tokens": 128,
                "stream": True,
            }
            start = time.perf_counter()
            ttft = None
            tokens = 0
            # catch_response lets us mark success/failure + log a custom TTFT metric
            with self.client.post(
                "/v1/chat/completions",
                data=json.dumps(payload),
                headers=self._headers(),
                stream=True,
                catch_response=True,
                name="/v1/chat/completions [stream]",
            ) as resp:
                try:
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode() if isinstance(raw, bytes) else raw
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta and ttft is None:
                            ttft = time.perf_counter() - start
                        if delta:
                            tokens += 1
                    resp.success()
                except Exception as e:  # noqa: BLE001
                    resp.failure(str(e))

            # emit a custom TTFT metric so it shows in stats
            if ttft is not None:
                events.request.fire(
                    request_type="TTFT",
                    name="time_to_first_token",
                    response_time=ttft * 1000,  # ms
                    response_length=tokens,
                    exception=None,
                    context={},
                )

else:

    def main() -> None:
        print(
            "Locust is not installed.\n"
            "  pip install locust\n"
            "  locust -f locustfile.py --host http://localhost:8000\n"
            "Then open http://localhost:8089 and set users + spawn rate.\n"
            "For an offline metric report with no install, use async_load_test.py."
        )

    if __name__ == "__main__":
        main()
