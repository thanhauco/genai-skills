"""
retries_backoff.py — production-grade retry semantics for flaky upstreams.

Why this matters:
  Inference endpoints throttle (429), cold-start, or hit transient 5xx. Naive
  retries cause thundering herds and make outages worse. You need:
    - exponential backoff WITH jitter
    - a cap on attempts and total time
    - respect for Retry-After
    - a circuit breaker so you stop hammering a dead dependency
    - retries ONLY for idempotent / safe operations

Run:
    python retries_backoff.py
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


class TransientError(Exception):
    """Retryable (e.g., 429, 503, timeout)."""


class PermanentError(Exception):
    """Not retryable (e.g., 400 bad request, 401 auth)."""


def backoff_delays(base: float = 0.1, factor: float = 2.0, cap: float = 5.0, attempts: int = 6) -> list[float]:
    """Full-jitter exponential backoff (AWS architecture blog style)."""
    delays = []
    for i in range(attempts):
        exp = min(cap, base * (factor**i))
        delays.append(random.uniform(0, exp))  # full jitter
    return delays


def retry(call, *, attempts: int = 6, retry_after: float | None = None):
    """Run `call()` with bounded retries. `call` raises Transient/Permanent."""
    delays = backoff_delays(attempts=attempts)
    last: Exception | None = None
    for i in range(attempts):
        try:
            return call()
        except PermanentError:
            raise  # never retry these
        except TransientError as e:
            last = e
            if i == attempts - 1:
                break
            sleep_s = retry_after if retry_after is not None else delays[i]
            print(f"  attempt {i+1} failed ({e}); backing off {sleep_s:.2f}s")
            time.sleep(sleep_s)
    raise TransientError(f"exhausted retries: {last}")


@dataclass
class CircuitBreaker:
    """Open the circuit after `threshold` consecutive failures; cool down before retrying."""

    threshold: int = 5
    cooldown_s: float = 2.0
    _fails: int = 0
    _opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if time.perf_counter() - self._opened_at >= self.cooldown_s:
            self._opened_at = None  # half-open: allow a probe
            self._fails = 0
            return True
        return False

    def record(self, ok: bool) -> None:
        if ok:
            self._fails = 0
            self._opened_at = None
        else:
            self._fails += 1
            if self._fails >= self.threshold:
                self._opened_at = time.perf_counter()

    @property
    def state(self) -> str:
        return "open" if self._opened_at is not None else "closed"


def _flaky_endpoint(success_after: int):
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] < success_after:
            raise TransientError("503 from upstream")
        return "ok"

    return call


def main() -> None:
    random.seed(1)
    print("1) Retry with backoff (succeeds on attempt 3):")
    print("   result:", retry(_flaky_endpoint(success_after=3)))

    print("\n2) Backoff schedule (full jitter):")
    print("  ", [round(d, 3) for d in backoff_delays()])

    print("\n3) Circuit breaker trips then cools down:")
    cb = CircuitBreaker(threshold=3, cooldown_s=0.5)
    for i in range(6):
        if not cb.allow():
            print(f"   call {i}: SHORT-CIRCUITED (state={cb.state})")
            continue
        ok = i >= 4  # endpoint recovers at call 4
        cb.record(ok)
        print(f"   call {i}: {'ok' if ok else 'fail'} (state={cb.state})")


if __name__ == "__main__":
    main()
