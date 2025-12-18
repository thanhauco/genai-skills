"""
coding_drills.py — common Python interview problems, with self-checking tests.

These are the kinds of small problems that show up in a coding screen for a
hands-on field/forward-deployed role. Each has a clean solution + assertions.

Run:
    python coding_drills.py        # runs all tests, prints PASS/FAIL
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator


# 1) Sliding window: longest substring without repeating chars -----------------
def longest_unique_substring(s: str) -> int:
    seen: dict[str, int] = {}
    start = best = 0
    for i, ch in enumerate(s):
        if ch in seen and seen[ch] >= start:
            start = seen[ch] + 1
        seen[ch] = i
        best = max(best, i - start + 1)
    return best


# 2) Group anagrams ------------------------------------------------------------
def group_anagrams(words: Iterable[str]) -> list[list[str]]:
    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for w in words:
        key = tuple(sorted(Counter(w).items()))  # type: ignore[arg-type]
        groups[key].append(w)
    return list(groups.values())


# 3) Merge intervals (think: dedup/merge token spans, time windows) ------------
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


# 4) Batch an iterator into chunks (used everywhere in data pipelines) ---------
def batched(it: Iterable, n: int) -> Iterator[list]:
    if n < 1:
        raise ValueError("n must be >= 1")
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# 5) Token bucket rate limiter (a real FDE building block) ---------------------
class TokenBucket:
    def __init__(self, rate_per_s: float, capacity: float) -> None:
        self.rate = rate_per_s
        self.capacity = capacity
        self.tokens = capacity
        self._t = 0.0

    def allow(self, now: float, cost: float = 1.0) -> bool:
        self.tokens = min(self.capacity, self.tokens + (now - self._t) * self.rate)
        self._t = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


# 6) LRU cache from scratch (ordered dict / dll concept) -----------------------
class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.d: dict = {}
        self.order: deque = deque()

    def get(self, key):
        if key not in self.d:
            return None
        self.order.remove(key)
        self.order.append(key)
        return self.d[key]

    def put(self, key, value) -> None:
        if key in self.d:
            self.order.remove(key)
        elif len(self.d) >= self.cap:
            evict = self.order.popleft()
            del self.d[evict]
        self.d[key] = value
        self.order.append(key)


# 7) Flatten nested JSON-ish dict (config wrangling) ---------------------------
def flatten(d: dict, prefix: str = "", sep: str = ".") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


def _run_tests() -> None:
    checks: list[tuple[str, bool]] = []

    checks.append(("longest_unique_substring", longest_unique_substring("abcabcbb") == 3))
    checks.append(("longest_unique_empty", longest_unique_substring("") == 0))

    ga = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    checks.append(("group_anagrams", sorted(len(g) for g in ga) == [1, 2, 3]))

    checks.append(
        ("merge_intervals", merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)])
    )

    checks.append(("batched", list(batched(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]))

    tb = TokenBucket(rate_per_s=1.0, capacity=2.0)
    got = [tb.allow(0.0), tb.allow(0.0), tb.allow(0.0), tb.allow(2.0)]
    checks.append(("token_bucket", got == [True, True, False, True]))

    lru = LRUCache(2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.get("a")  # touch a -> b is now LRU
    lru.put("c", 3)  # evicts b
    checks.append(("lru_cache", lru.get("b") is None and lru.get("a") == 1 and lru.get("c") == 3))

    checks.append(
        ("flatten", flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3})
    )

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{passed}/{len(checks)} passed")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    _run_tests()
