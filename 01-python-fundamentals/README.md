# 01 — Python Fundamentals (for Field Engineers)

The JD asks for **strong Python**: "comfortable reading, writing, and debugging production code." For an AI Field Engineer, the Python that matters most is:

- **Async & concurrency** — every LLM client is I/O-bound; you'll fan out requests.
- **Streaming** — SSE token streaming is the default UX for chat/agents.
- **Profiling & debugging** — you'll drop into a customer's codebase and find the bottleneck.
- **Clean data plumbing** — batching, retries, backpressure.

## Files

| File | What it teaches |
| --- | --- |
| [`async_concurrency.py`](async_concurrency.py) | `asyncio`, `gather`, semaphores, bounded concurrency, timeouts |
| [`streaming_client.py`](streaming_client.py) | Parsing SSE token streams; measuring TTFT/TPOT |
| [`retries_backoff.py`](retries_backoff.py) | Exponential backoff + jitter, idempotency, circuit-breaker sketch |
| [`profiling_debugging.py`](profiling_debugging.py) | `cProfile`, `timeit`, tracemalloc, structured logging |
| [`coding_drills.py`](coding_drills.py) | Common interview problems with tests (`python coding_drills.py`) |

## Run

```bash
python 01-python-fundamentals/async_concurrency.py
python 01-python-fundamentals/streaming_client.py
python 01-python-fundamentals/retries_backoff.py
python 01-python-fundamentals/profiling_debugging.py
python 01-python-fundamentals/coding_drills.py
```

All files run with the **standard library only** (no network needed) — they simulate an LLM endpoint.

## Interview Q&A (say these out loud)

1. **When do you reach for `asyncio` vs threads vs processes?**
   - `asyncio` for I/O-bound fan-out (HTTP to an inference endpoint). Threads when a library is blocking and not async-friendly. Processes (or a cluster) for CPU-bound work — Python's GIL means threads won't parallelize CPU.

2. **How do you bound concurrency so you don't hammer a customer's endpoint?**
   - `asyncio.Semaphore(N)` around each request, or a worker pool consuming a queue. This gives backpressure and a knob you can tie to the endpoint's rate limits.

3. **What's TTFT and how do you measure it from a stream?**
   - Time To First Token = wall-clock from request send to the first streamed chunk. You measure it by timestamping right before the request and on the first non-empty SSE delta. See `streaming_client.py`.

4. **A customer says "the script is slow." First three moves?**
   - Reproduce + measure (don't guess). `cProfile` for CPU hot spots, check whether it's actually I/O-bound (then it's a concurrency problem), and look at p95 not just the mean. See `profiling_debugging.py`.

5. **How do you make retries safe?**
   - Only retry idempotent / safe-to-repeat calls, use exponential backoff **with jitter** to avoid thundering herds, cap total attempts, and respect `Retry-After`. Add a circuit breaker so you stop hammering a down dependency. See `retries_backoff.py`.

## What "good" looks like in the codebase

- Type hints + small pure functions you can unit test.
- Errors handled at the boundary (network, parsing) — not swallowed everywhere.
- Concurrency is **bounded and observable** (you can see in-flight count, latencies).
- Logs are structured (key=value or JSON) so they're greppable in production.
