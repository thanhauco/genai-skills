# GPU-on-Kubernetes debugging playbook

The failure modes you'll hit deploying inference servers on a customer's cluster, and how to triage them fast. This is the "debug a production issue with an ML engineer in the same afternoon" skill from the JD.

## First three commands, every time

```bash
kubectl get pods -o wide                      # status + which node
kubectl describe pod <pod>                     # Events at the bottom = the truth
kubectl logs <pod> [-c <container>] [--previous]  # --previous = the crashed instance
```

## Symptom → cause → fix

### Pod stuck `Pending`
- **`describe` Events say "Insufficient nvidia.com/gpu"** → no free GPU. Fix: scale the GPU node pool (cluster autoscaler), free a GPU, or reduce the GPU request.
- **"node(s) didn't match node selector / taint"** → your `nodeSelector`/`tolerations` don't match the GPU nodes. Fix the labels/tolerations.
- **"Insufficient cpu/memory"** → requests too high for the node. Lower requests or pick bigger nodes.
- **No NVIDIA device plugin** → `kubectl get pods -n kube-system | grep nvidia`; GPUs won't be advertised without it.

### `CrashLoopBackOff` / `Error`
- `kubectl logs --previous`:
  - **CUDA OOM / "out of memory"** → model too big for the GPU. Fix: tensor-parallel across more GPUs (`--tensor-parallel-size`), quantize (fp8/int4), or lower `--max-model-len` / `--gpu-memory-utilization`.
  - **NCCL / shared-memory errors with TP** → `/dev/shm` too small. Mount an `emptyDir{medium: Memory}` at `/dev/shm` (see vllm-deployment.yaml).
  - **CUDA driver/runtime mismatch** → container CUDA newer than node driver. Align image/driver versions.
  - **Auth/model not found** → gated model needs `HUGGING_FACE_HUB_TOKEN`; check the secret + model id.

### `ImagePullBackOff`
- Wrong image name/tag, private registry without `imagePullSecrets`, or rate-limited registry. `describe` shows the pull error.

### Pod `Running` but not `Ready` / 503s
- Readiness probe failing during the long model load. Use a `startupProbe` with a high `failureThreshold` so the slow boot isn't mistaken for a crash. Check `/health` actually returns 200 once loaded.

### `OOMKilled` (container memory, not GPU)
- `describe` shows `Reason: OOMKilled`. Raise memory limits, or the runtime is buffering too much (tune batch/queue). GPU OOM shows up in logs, not as OOMKilled.

### Node `NotReady` / GPU "fell off the bus"
- `kubectl describe node <node>`; check the NVIDIA device plugin + DCGM. Hardware/driver issue → cordon + drain the node, replace it.

## Useful deep-dive commands

```bash
kubectl get events --sort-by=.lastTimestamp -A | tail -30   # cluster-wide recent events
kubectl describe node <gpu-node> | grep -A5 Allocatable      # are GPUs advertised?
kubectl get pods -A -o wide | grep nvidia                    # device plugin / dcgm running?
kubectl exec -it <pod> -- nvidia-smi                         # GPU visible inside the pod?
kubectl top pod <pod>                                        # CPU/mem (needs metrics-server)
kubectl rollout status deploy/<name>                         # is the rollout healthy?
kubectl rollout undo deploy/<name>                           # roll back a bad deploy
```

## GPU resource gotchas (say these)

- `nvidia.com/gpu` **request must equal limit**; you can't oversubscribe a GPU like CPU.
- GPUs are **whole-device** by default; sharing needs **MIG** (partition an A100/H100) or **time-slicing** (config in the device plugin).
- **TP needs co-located GPUs** on one node with NVLink; don't try to tensor-parallel across nodes.
- **Model load dominates cold start** → cache weights on a PVC, pre-pull images, keep warm replicas.
- **Cluster autoscaler for GPU pools** is slow (provision a node → install drivers → schedule). Keep buffer capacity for spikes.

## The triage narrative (interview-ready)

> "I `get pods`, then `describe` to read the Events, then `logs --previous` for the crash. Pending almost always means GPU scheduling — selector/taint mismatch or no free GPU. CrashLoop is usually CUDA OOM (fix with TP or quantization) or `/dev/shm` for NCCL. Not-Ready is the slow model load — that's a `startupProbe`, not a real failure."
