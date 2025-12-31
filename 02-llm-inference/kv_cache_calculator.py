"""
kv_cache_calculator.py — size a deployment from first principles.

This is THE calculation a Field Engineer does on a whiteboard: given a model and
a GPU, how many concurrent requests fit? What does quantization buy you?

KV cache bytes per token:
    2 (K and V) * n_layers * n_kv_heads * head_dim * dtype_bytes

Notes:
  - GQA/MQA reduce n_kv_heads (fewer KV heads than attention heads) -> smaller KV.
  - Weight quantization shrinks the weights term -> more HBM left for KV -> more
    concurrency -> higher throughput -> lower $/token.

Run:
    python kv_cache_calculator.py
Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1024**3


@dataclass
class ModelSpec:
    name: str
    n_params_b: float       # billions of parameters
    n_layers: int
    hidden: int             # model hidden size
    n_heads: int            # attention heads
    n_kv_heads: int         # KV heads (GQA/MQA -> < n_heads)

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        # 2 for K and V
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * dtype_bytes

    def weight_bytes(self, weight_dtype_bytes: float = 2.0) -> float:
        return self.n_params_b * 1e9 * weight_dtype_bytes


@dataclass
class GPUSpec:
    name: str
    hbm_gb: float
    mem_bandwidth_tbs: float  # TB/s, for decode roofline intuition


# A few representative specs (approximate; verify per real config).
LLAMA_8B = ModelSpec("Llama-3.1-8B", 8.0, 32, 4096, 32, 8)      # GQA: 8 KV heads
LLAMA_70B = ModelSpec("Llama-3.1-70B", 70.0, 80, 8192, 64, 8)   # GQA
MIXTRAL_8x7B = ModelSpec("Mixtral-8x7B (MoE)", 46.7, 32, 4096, 32, 8)

H100 = GPUSpec("H100-80GB", 80, 3.35)
A100 = GPUSpec("A100-80GB", 80, 2.04)
L40S = GPUSpec("L40S-48GB", 48, 0.86)


def concurrency_estimate(
    model: ModelSpec,
    gpu: GPUSpec,
    *,
    seq_len: int = 2048,
    weight_dtype_bytes: float = 2.0,   # fp16
    kv_dtype_bytes: int = 2,           # fp16 KV
    n_gpus: int = 1,
    overhead_frac: float = 0.10,       # activations, fragmentation, framework
) -> dict:
    total_hbm = gpu.hbm_gb * GB * n_gpus
    weights = model.weight_bytes(weight_dtype_bytes)
    usable = total_hbm * (1 - overhead_frac) - weights
    per_req = model.kv_bytes_per_token(kv_dtype_bytes) * seq_len
    max_conc = max(0, int(usable // per_req)) if per_req else 0
    return {
        "weights_gb": weights / GB,
        "usable_kv_gb": max(0.0, usable) / GB,
        "kv_per_token_kb": model.kv_bytes_per_token(kv_dtype_bytes) / 1024,
        "per_request_gb": per_req / GB,
        "max_concurrent": max_conc,
        "fits": weights < total_hbm,
    }


def show(model: ModelSpec, gpu: GPUSpec, n_gpus: int, **kw) -> None:
    r = concurrency_estimate(model, gpu, n_gpus=n_gpus, **kw)
    tag = "" if r["fits"] else "  <-- WEIGHTS DON'T FIT, need more GPUs / quantization"
    wq = kw.get("weight_dtype_bytes", 2.0)
    sl = kw.get("seq_len", 2048)
    print(
        f"{model.name:24s} on {n_gpus}x{gpu.name:11s} "
        f"w@{wq}B seq={sl:5d} | "
        f"weights={r['weights_gb']:6.1f}GB kv/tok={r['kv_per_token_kb']:6.1f}KB "
        f"/req={r['per_request_gb']:.3f}GB | max_conc={r['max_concurrent']:4d}{tag}"
    )


def main() -> None:
    print("=== How many concurrent requests fit? (seq_len = prompt + gen) ===\n")
    show(LLAMA_8B, H100, 1)
    show(LLAMA_8B, L40S, 1)
    show(LLAMA_70B, H100, 1)        # won't fit in fp16 on one GPU
    show(LLAMA_70B, H100, 2)        # TP=2
    show(LLAMA_70B, H100, 4)

    print("\n=== Quantization frees HBM -> more KV cache -> more concurrency ===\n")
    show(LLAMA_70B, H100, 2, weight_dtype_bytes=2.0)   # fp16 weights
    show(LLAMA_70B, H100, 2, weight_dtype_bytes=1.0)   # fp8 weights
    show(LLAMA_70B, H100, 2, weight_dtype_bytes=0.5)   # int4 weights

    print("\n=== Longer context costs linearly in KV ===\n")
    for sl in (2048, 8192, 32768):
        show(LLAMA_8B, H100, 1, seq_len=sl)

    print(
        "\nField-engineer translation:\n"
        "  'On 2xH100, moving 70B from fp16 to fp8 roughly doubles the KV budget,\n"
        "   which lets us run ~2x the concurrent requests at the same latency,\n"
        "   cutting $/1M tokens — pending an eval gate on your task.'"
    )


if __name__ == "__main__":
    main()
