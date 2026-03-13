# Kubernetes concepts (the parts an AI Field Engineer needs)

You don't need to be a cluster admin. You need to deploy inference servers, debug them, and scale them. Here's the focused model.

## The objects you touch

```
 Deployment ──manages──► ReplicaSet ──manages──► Pods ──run──► Containers
     │                                              ▲
     │ desired replicas, rolling updates            │ scheduled onto Nodes
     ▼                                              │
 Service ──stable virtual IP + load-balances──────► Pods (by label selector)
     │
 Ingress / Gateway ──external HTTP ──► Service
 HPA ──scales replicas on a metric──► Deployment
```

- **Pod** — one or more containers sharing network + storage; the unit of scheduling. Ephemeral.
- **Deployment** — declarative desired state for a set of identical pods; handles rollouts + self-healing.
- **ReplicaSet** — keeps N pods alive (managed by the Deployment; you rarely touch it directly).
- **Service** — stable virtual IP + DNS name that load-balances across the pods matching a label selector (pods come and go; the Service endpoint is stable).
- **Ingress / Gateway API** — routes external HTTP(S) to Services.
- **HPA** — Horizontal Pod Autoscaler; changes replica count based on a metric (see `hpa.yaml`).
- **ConfigMap / Secret** — config + credentials injected as env/volumes (e.g., HF token).
- **Namespace** — soft tenancy boundary for grouping/quotas.
- **StatefulSet / DaemonSet / Job** — ordered stateful pods / one-per-node (device plugins) / run-to-completion (batch eval, fine-tune jobs).

## The resource model (why it matters for GPUs)

```
 requests = what the scheduler reserves (used for placement)
 limits   = the hard cap the kubelet enforces

 CPU:    compressible  — can request 4, limit 8 (burst); throttled, not killed
 Memory: incompressible — exceed the limit -> OOMKilled
 GPU (nvidia.com/gpu): whole devices, request MUST EQUAL limit, no oversubscription
```

- Set memory `requests`≈`limits` for predictable inference (avoid OOMKills).
- For GPUs, you request whole devices. Sharing requires **MIG** (hardware partitioning of A100/H100) or **time-slicing** (device-plugin config).

## How a GPU pod gets scheduled

1. You set `resources.limits["nvidia.com/gpu"]: N` + a `nodeSelector`/affinity for GPU nodes + tolerations for the GPU taint.
2. The **scheduler** finds a node with N free GPUs that matches selectors/taints.
3. The **NVIDIA device plugin** (a DaemonSet) advertises `nvidia.com/gpu` capacity on each node and wires the GPU into the container.
4. If no node fits → pod stays **Pending** (the cluster autoscaler may add a GPU node, slowly).

## Probes (critical for slow-loading models)

- **startupProbe** — guards the long boot (model download + load to GPU). Until it passes, liveness/readiness are paused. Set a high `failureThreshold`.
- **readinessProbe** — gates traffic; failing pods are removed from the Service endpoints.
- **livenessProbe** — restarts a wedged container. Don't set it so aggressive it kills a still-loading model (that's what startupProbe prevents).

## Rollouts & safety

- **RollingUpdate** with `maxUnavailable: 0` so you never dip below capacity while slow GPU pods come up.
- `kubectl rollout status` to watch, `kubectl rollout undo` to revert a bad image.
- Pin image versions in production (not `:latest`).

## Scaling levers

- **Pod autoscaling (HPA/KEDA)** — replicas, ideally on queue depth / GPU util, not CPU.
- **Node autoscaling (cluster autoscaler / Karpenter)** — adds GPU nodes when pods can't schedule; slow for GPUs, so keep headroom.
- **Vertical** — bigger GPUs or more GPUs per pod (TP) for larger models / lower latency.

## What usually goes wrong (links to the playbook)

- Pending → GPU scheduling (selector/taint/no free GPU).
- CrashLoop → CUDA OOM, `/dev/shm` for NCCL, driver mismatch, model auth.
- Not-Ready → slow model load mistaken for failure (use startupProbe).

See `gpu-debugging.md` for the full triage.
