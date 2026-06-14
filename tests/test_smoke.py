"""
Smoke + unit tests for the genai-skills repo.

Two layers:
  1) UNIT tests of pure functions across modules (correctness).
  2) SMOKE tests that every runnable script's main() executes without raising
     (except servers / interactive files, which are skipped).

Module directories start with digits and contain hyphens, so we load files by
PATH with importlib rather than importing them as packages.

Run:
    pip install pytest
    pytest -q
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relpath: str):
    """Load a module from a file path (handles digit/hyphen dir names)."""
    path = ROOT / relpath
    name = "mod_" + relpath.replace("/", "_").replace("-", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load {relpath}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve cls.__module__ via sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------- unit tests ---------------------------------------
def test_coding_drills():
    m = load("01-python-fundamentals/coding_drills.py")
    assert m.longest_unique_substring("abcabcbb") == 3
    assert m.merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]
    assert list(m.batched(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert m.flatten({"a": {"b": 1}}) == {"a.b": 1}
    lru = m.LRUCache(2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.get("a")
    lru.put("c", 3)  # evicts b
    assert lru.get("b") is None and lru.get("a") == 1


def test_backoff_bounds():
    m = load("01-python-fundamentals/retries_backoff.py")
    delays = m.backoff_delays(base=0.1, factor=2.0, cap=5.0, attempts=6)
    assert len(delays) == 6
    assert all(0.0 <= d <= 5.0 for d in delays)


def test_kv_cache_math():
    m = load("02-llm-inference/kv_cache_calculator.py")
    # 8B GQA on one H100 should fit and allow >0 concurrency
    r = m.concurrency_estimate(m.LLAMA_8B, m.H100, seq_len=2048)
    assert r["fits"] is True
    assert r["max_concurrent"] > 0
    # fp8 weights (1 byte) must free HBM vs fp16 (2 bytes) -> more concurrency
    r16 = m.concurrency_estimate(m.LLAMA_70B, m.H100, n_gpus=2, weight_dtype_bytes=2.0)
    r8 = m.concurrency_estimate(m.LLAMA_70B, m.H100, n_gpus=2, weight_dtype_bytes=1.0)
    assert r8["max_concurrent"] > r16["max_concurrent"]


def test_metrics():
    m = load("05-evaluation/metrics.py")
    assert m.accuracy([True, True, False, True]) == 0.75
    assert m.pass_at_k(10, 0, 3) == 0.0          # no correct -> 0
    assert m.pass_at_k(10, 10, 1) == 1.0         # all correct -> 1
    assert 0.0 < m.pass_at_k(10, 3, 5) < 1.0
    lo, hi = m.wilson_interval(7, 10)
    assert 0.0 <= lo < hi <= 1.0
    assert m.percentile([10, 20, 30, 40], 50) in (20, 30)
    assert m.cost_per_million_tokens(3.0, 0) == float("inf")


def test_vector_search():
    m = load("11-rag/vector_search.py")
    idx = m.VectorIndex(m.CORPUS)
    hits = idx.search("how do I get more KV cache?", k=2)
    assert len(hits) == 2
    # the fp8/HBM/KV-cache chunk (index 2) should be the top hit
    assert hits[0].index == 2
    # cosine of identical normalized vectors ~ 1
    v = idx.vectors[0]
    assert abs(m.cosine(v, v) - 1.0) < 1e-6


def test_retrieval_metrics():
    m = load("11-rag/retrieval_metrics.py")
    assert m.recall_at_k(["d1", "d2", "d3"], {"d1", "d2"}, 3) == 1.0
    assert m.precision_at_k(["d1", "x", "y"], {"d1"}, 3) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(["x", "d1"], {"d1"}) == 0.5
    assert m.reciprocal_rank(["x", "y"], {"d1"}) == 0.0
    assert m.hit_at_k(["x", "d1"], {"d1"}, 2) == 1.0


def test_json_schema_validation():
    m = load("13-structured-output/json_schema_demo.py")
    good = {"intent": "refund", "priority": 2, "order_id": "ORD-00042"}
    assert m.validate(good, m.SCHEMA) == []
    bad = {"intent": "refund", "priority": 9, "order_id": "ORD-12"}
    errs = m.validate(bad, m.SCHEMA)
    assert any("max" in e for e in errs)
    assert any("pattern" in e for e in errs)


def test_speculative_math():
    m = load("14-prompt-and-speculative/speculative_decoding.py")
    # closed form: 1 + p + p^2 + ... + p^k
    assert m.expected_tokens_per_step(4, 0.0) == pytest.approx(1.0)
    assert m.expected_tokens_per_step(2, 0.5) == pytest.approx(1 + 0.5 + 0.25)
    # higher acceptance -> more tokens per step
    assert m.expected_tokens_per_step(4, 0.9) > m.expected_tokens_per_step(4, 0.3)


# --------------------------- smoke tests --------------------------------------
# Every runnable demo's main() should execute without raising. Skip servers /
# interactive / dependency-gated files.
SMOKE_SCRIPTS = [
    "01-python-fundamentals/coding_drills.py",
    "01-python-fundamentals/streaming_client.py",
    "01-python-fundamentals/retries_backoff.py",
    "02-llm-inference/kv_cache_calculator.py",
    "02-llm-inference/batching_simulation.py",
    "02-llm-inference/roofline_latency.py",
    "02-llm-inference/tokenization_demo.py",
    "04-fine-tuning/data_prep.py",
    "04-fine-tuning/dpo_example.py",
    "05-evaluation/eval_harness.py",
    "05-evaluation/llm_as_judge.py",
    "05-evaluation/metrics.py",
    "06-agentic-systems/function_calling.py",
    "06-agentic-systems/react_agent.py",
    "06-agentic-systems/multi_tool_agent.py",
    "08-cloud-gpu-deployment/cost_estimator.py",
    "11-rag/chunking.py",
    "11-rag/vector_search.py",
    "11-rag/retrieval_metrics.py",
    "12-multimodal/multimodal_message.py",
    "13-structured-output/json_schema_demo.py",
    "14-prompt-and-speculative/speculative_decoding.py",
]


@pytest.mark.parametrize("relpath", SMOKE_SCRIPTS)
def test_main_runs(relpath):
    m = load(relpath)
    entry = getattr(m, "main", None) or getattr(m, "_run_tests", None)
    assert entry is not None, f"{relpath} has no main()/_run_tests()"
    with redirect_stdout(io.StringIO()):  # silence demo output
        entry()
