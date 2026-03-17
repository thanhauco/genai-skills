# kubectl cheat sheet (the commands you'll actually use)

Scoped to what an AI Field Engineer needs when deploying + debugging inference on a customer's cluster.

## Context & namespaces

```bash
kubectl config get-contexts                 # which clusters you can reach
kubectl config use-context <ctx>            # switch cluster
kubectl config set-context --current --namespace=<ns>   # default a namespace
kubectl get ns                              # list namespaces
```

## Inspect

```bash
kubectl get pods -o wide                    # pods + node + IP
kubectl get pods -w                         # watch status live
kubectl get deploy,svc,hpa                  # the trio you deploy
kubectl describe pod <pod>                  # Events (scheduling/probe failures)
kubectl get events --sort-by=.lastTimestamp -A | tail -30
kubectl top pod / kubectl top node          # resource usage (needs metrics-server)
```

## Logs & exec

```bash
kubectl logs <pod>                          # current container logs
kubectl logs <pod> -c <container>           # multi-container pod
kubectl logs <pod> --previous               # the crashed instance (CrashLoop!)
kubectl logs -f deploy/<name>               # follow a deployment's logs
kubectl exec -it <pod> -- bash              # shell in
kubectl exec -it <pod> -- nvidia-smi        # is the GPU visible?
```

## Apply / edit / rollout

```bash
kubectl apply -f file.yaml                  # create/update from manifest
kubectl apply --dry-run=client -f f.yaml    # validate without applying
kubectl diff -f file.yaml                   # what would change
kubectl edit deploy/<name>                  # live edit (quick experiments)
kubectl set image deploy/<name> vllm=vllm/vllm-openai:v0.x.y
kubectl rollout status deploy/<name>        # watch a rollout
kubectl rollout undo deploy/<name>          # roll back
kubectl rollout restart deploy/<name>       # restart pods (e.g., re-pull/cache)
```

## Scale

```bash
kubectl scale deploy/<name> --replicas=4    # manual scale
kubectl get hpa                             # autoscaler status (current/target metric)
kubectl describe hpa <name>                 # why it scaled (or didn't)
```

## Networking / access

```bash
kubectl port-forward svc/<svc> 8000:80      # hit a ClusterIP service locally
kubectl get svc,ingress                     # endpoints
# quick in-cluster smoke test:
kubectl run curl --rm -it --image=curlimages/curl -- \
  curl -s http://<svc>/v1/models
```

## GPU / nodes

```bash
kubectl get nodes -L nvidia.com/gpu.present # which nodes have GPUs
kubectl describe node <node> | grep -A6 Allocatable    # GPUs advertised?
kubectl get pods -A -o wide | grep -i nvidia           # device plugin / dcgm
kubectl cordon <node> / kubectl drain <node> --ignore-daemonsets   # take a bad node out
```

## Secrets / config (for gated models, API keys)

```bash
kubectl create secret generic hf-token --from-literal=token=hf_xxx
kubectl create configmap vllm-args --from-file=args.txt
kubectl get secret hf-token -o jsonpath='{.data.token}' | base64 -d
```

## Cleanup

```bash
kubectl delete -f file.yaml
kubectl delete pod <pod> --grace-period=0 --force   # stuck pod (use sparingly)
```

## Muscle-memory triage chain

```bash
kubectl get pods -o wide \
  && kubectl describe pod <pod> \
  && kubectl logs <pod> --previous
```
Pending → describe Events (GPU scheduling). CrashLoop → logs --previous (OOM/NCCL/driver). Not-Ready → probes vs slow model load.
