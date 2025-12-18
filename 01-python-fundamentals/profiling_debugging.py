"""
profiling_debugging.py — find the bottleneck instead of guessing.

Why this matters:
  You drop into a customer's codebase and "it's slow." A Field Engineer measures
  first. This file shows the four tools you'll reach for:
    - timeit / perf_counter   : micro-timing
    - cProfile + pstats       : where CPU time goes
    - tracemalloc             : where memory goes
    - structured logging      : greppable production signal

Run:
    python profiling_debugging.py
"""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
import time
import tracemalloc
from functools import wraps


# ---------- structured logging (greppable: key=value) ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s level=%(levelname)s %(message)s")
log = logging.getLogger("fde")


def timed(fn):
    """Decorator: log wall time as structured fields."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.info("event=timing func=%s duration_ms=%.2f", fn.__name__, (time.perf_counter() - t0) * 1000)

    return wrapper


# ---------- a deliberately inefficient function to profile ----------
def slow_build_string(n: int) -> str:
    """O(n^2) string concat — a classic hidden bottleneck."""
    s = ""
    for i in range(n):
        s += str(i % 10)
    return s


def fast_build_string(n: int) -> str:
    """O(n) with join — the fix."""
    return "".join(str(i % 10) for i in range(n))


@timed
def process(n: int) -> int:
    data = [slow_build_string(50) for _ in range(n)]
    return sum(len(x) for x in data)


def profile_cpu(n: int) -> None:
    pr = cProfile.Profile()
    pr.enable()
    process(n)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(6)
    print(buf.getvalue())


def profile_memory() -> None:
    tracemalloc.start()
    big = [fast_build_string(10_000) for _ in range(100)]  # noqa: F841
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"memory: current={current/1e6:.2f}MB peak={peak/1e6:.2f}MB")


def micro_compare() -> None:
    for name, fn in (("slow", slow_build_string), ("fast", fast_build_string)):
        t0 = time.perf_counter()
        for _ in range(200):
            fn(2_000)
        log.info("event=micro impl=%s duration_ms=%.2f", name, (time.perf_counter() - t0) * 1000)


def main() -> None:
    print("=== micro comparison (slow O(n^2) vs fast O(n) string build) ===")
    micro_compare()

    print("\n=== cProfile: where does the time go? ===")
    profile_cpu(500)

    print("=== tracemalloc: peak memory of a workload ===")
    profile_memory()

    print(
        "\nInterview narrative:\n"
        "  1) Reproduce + measure (p95, not mean).\n"
        "  2) cProfile to find the hot function.\n"
        "  3) Decide: CPU-bound (optimize/parallel-process) or I/O-bound (concurrency)?\n"
        "  4) Fix, re-measure, and add a regression guard."
    )


if __name__ == "__main__":
    main()
