# 07 — Kubernetes & GPU Infrastructure

The JD: "Familiarity with Kubernetes and infrastructure engineering" and "deploying models on GPU infrastructure." You'll deploy inference servers onto customers' clusters and tune them. You don't need to be a cluster admin — you need to **deploy, debug, and scale** GPU inference reliably.

## Files

| File | What it teaches |
| --- | --- |
| [`vllm-deployment.yaml`](vllm-deployment.yaml) | A real GPU vLLM Deployment + Service (requests/limits, node selector, probes, shm) |
| [`hpa.yaml`](hpa.yaml) | Horizontal Pod Autoscaler + notes on KEDA/custom metrics for LLM autoscaling |
| [`gpu-debugging.md`](gpu-debugging.md) | The "pod is Pending / CrashLoop / OOM" playbook for GPU workloads |
| [`kubectl_cheatsheet.md`](kubectl_cheatsheet.md) | The 25 commands you'll actually use in a customer's cluster |
| [`k8s_concepts.md`](k8s_concepts.md) | Pods/Deployments/Services/HPA, GPU scheduling, resource model |

## How to read these

The YAML is runnable on a GPU-enabled cluster (`kubectl apply -f vllm-deployment.yaml`). If you don't have a cluster, study the annotations — every field is commented with *why* it's there. Validate syntax locally with:

```bash
kubectl apply --dry-run=client -f 07-kubernetes-infra/vllm-deployment.yaml
# or, if you have it:
kubeval 07-kubernetes-infra/vllm-deployment.yaml
```

## The GPU-on-Kubernetes mental model

```
            ┌──────────────────────────── Cluster ─────────────────────────────┐
            │  Control plane (API server, scheduler)                            │
            │                                                                   │
            │   ┌─────────── GPU node pool ───────────┐   ┌──── CPU nodes ────┐ │
            │   │  node: nvidia.com/gpu: 1 (H100)     │   │  ingress, etc.    │ │
            │   │  ┌───────── Pod (vLLM) ──────────┐  │   └───────────────────┘ │
            │   │  │ container: vllm openai server │  │                         │
            │   │  │ resources.limits:             │  │   Service (ClusterIP)   │
            │   │  │   nvidia.com/gpu: 1           │◄─┼───  load-balances pods  │
            │   │  │ readiness/liveness probes     │  │                         │
            │   │  └───────────────────────────────┘  │   HPA scales replicas   │
            │   └──────────────────────────────────────┘  on a metric          │
            └───────────────────────────────────────────────────────────────────┘
```

Key facts:
- **GPUs are requested via `nvidia.com/gpu`** (exposed by the NVIDIA device plugin). They are **non-fractional** by default — one container gets whole GPU(s) (MIG/time-slicing can subdivide).
- **GPU = limit, not just request**: for `nvidia.com/gpu` the request and limit must be equal; you can't oversubscribe a GPU like CPU.
- **Pod placement** uses `nodeSelector`/`affinity`/`taints+tolerations` to land on GPU nodes.
- **Model weights are big**: pull from a registry/volume; large container images and long cold starts are real (set generous `readinessProbe` timeouts, consider pre-pulled images / PVC model cache).

## Interview Q&A

1. **A vLLM pod is stuck `Pending`. Diagnose.**
   - `kubectl describe pod` → look at Events. Usually: no node with a free `nvidia.com/gpu` (insufficient GPUs / all allocated), a `nodeSelector`/taint mismatch, or unschedulable due to resource requests. Fix: scale the GPU node pool, fix selectors/tolerations, or reduce the GPU request. See `gpu-debugging.md`.

2. **Pod `CrashLoopBackOff` right after start. First moves?**
   - `kubectl logs --previous` for the crash reason. Common: OOM (model too big for the GPU → use TP across GPUs or quantize), CUDA/driver mismatch, missing `--shm-size`/`/dev/shm` (NCCL/tensor-parallel needs shared memory), or bad model path/auth.

3. **How do you autoscale LLM serving? Why is CPU% a bad trigger?**
   - LLM pods are GPU- and queue-bound, not CPU-bound, so CPU% doesn't reflect load. Scale on a **custom metric**: queue depth / pending requests, GPU utilization, or tokens-in-flight via KEDA or Prometheus Adapter. Mind **slow scale-up** (model load = long cold start) → keep warm headroom / use a buffer.

4. **What resource fields matter for a GPU pod?**
   - `resources.limits["nvidia.com/gpu"]` (whole GPUs), CPU/memory requests+limits sized for the runtime, a large `/dev/shm` (emptyDir medium: Memory) for NCCL, and probes tuned for long model-load startup (`startupProbe` is your friend).

5. **Single big model across 4 GPUs — how does that look in K8s?**
   - One pod requesting `nvidia.com/gpu: 4` on a node with 4 GPUs (NVLink), vLLM `--tensor-parallel-size 4`. TP needs the GPUs co-located on one node with fast interconnect; across nodes you'd add pipeline/data parallel and a different topology.
