# Kubernetes

```bash
kubectl create namespace agentzone
kubectl -n agentzone create secret generic agentzone --from-literal=network-secret=CHANGE-ME
kubectl -n agentzone create configmap agentzone-zone-yaml --from-file=zone.yaml=zone.yaml
kubectl -n agentzone apply -f deploy/k8s/agentzone.yaml
```

Edit `agentzone-config` (ZONE_NAME, ZONE_URL = the address peers will use, ZONE_SEED = any
existing zone). Push the image to your registry first, e.g.
`az acr build -r <acr> -t agentzones:latest -f deploy/Dockerfile .`, and set `image:`.
On AKS with GPU nodes, uncomment the `nvidia.com/gpu` limit on the Ollama deployment.
