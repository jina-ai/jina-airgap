Capacity planning for jina-on-prem deployments. Covers GPU vs CPU choice, VRAM by model, expected throughput, disk, and redundancy.

## The three knobs

```mermaid
flowchart LR
    A[Pick a model] --> B[Pick a runtime]
    B --> C[Size the host]
    C --> D[Estimate throughput]
    D -.- E[If under, add replicas
or move to bigger GPU]
```

## Runtime: GPU or CPU?

| | CPU image (`:cpu` tag) | GPU image (`:gpu` tag) |
|---|---|---|
| Base | `python:3.11-slim` | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` |
| Image size | 2-4 GB | 8-12 GB |
| Needs GPU on host | no | yes (CUDA 12.1+, driver 525+) |
| Latency, single request | 200-2000 ms | 10-200 ms |
| Throughput, batched | ~5-20 docs/s | 100-1000 docs/s |
| Cost on GCP (1 month, on-demand) | ~$50 (e2-standard-4) | ~$500 (g2-standard-4 + L4) |

**Rule of thumb**: < 10 QPS sustained -> CPU is fine and a lot cheaper. > 10 QPS or latency-sensitive interactive use -> GPU.

## Per-model VRAM and recommended GPU

| Model | Min VRAM | Recommended GPU | Notes |
|---|---|---|---|
| jina-embeddings-v5-text-nano | 2 GB | T4 / L4 | CPU also fine |
| jina-embeddings-v5-text-small | 3 GB | L4 / A10G | T4 OK with smaller batches |
| jina-embeddings-v5-omni-small | 8 GB | L4 / A10G / A100 | multimodal |
| jina-embeddings-v4 | 10 GB | A10G / A100 | multimodal, 32K |
| jina-reranker-v3.5 | 3 GB | L4 | 131K context |
| jina-reranker-v3 | 3 GB | L4 | 131K context |
| jina-clip-v2 | 4 GB | L4 | text + image |
| jina-code-embeddings-1.5b | 4 GB | L4 | |
| ReaderLM-v2 | 4 GB | L4 | 512K context, plan for bigger batches |

Full per-model VRAM in the [Model Catalog](Model-Catalog).

> **L4 is the workhorse**. 24GB, sane price, available in most regions, handles any model except v4-multimodal. Default to it unless the customer needs A100 latency or has L4 stockout (see [Troubleshooting -> L4 stockout](Troubleshooting#l4-stockout)).

## Reranker host memory

The listwise rerankers (`jina-reranker-v3`, `jina-reranker-v3.5`) score a block of
documents in one forward pass instead of one document at a time, so the memory a
single request needs grows with the **square** of the tokens in that request, not
linearly with the document count. Size the host from the largest request you
intend to send, not from the average.

On GPU this is not a practical constraint at L4 size: the full 16-document mixed
corpus below (~29K tokens, longest document 90,000 characters) completes in 8.4 s
using 8.7 GB of the L4's 23 GB.

On the `:cpu` image, host RAM is the binding constraint. Measured with
`jina-reranker-v3.5:cpu` on a `g2-standard-8` (8 vCPU, 31 GB RAM), uniform
2000-character English documents, one request at a time:

| Documents per request | Total characters | Result |
|---|---|---|
| 8 | 16,000 | 34 s |
| 12 | 24,000 | 67 s |
| 16 | 32,000 | 108 s |
| 24 | 48,000 | exceeds 31 GB; the host OOM-kills the container (exit 137) |

Two ways to stay inside a RAM budget: cap documents per request at the client and
page through the candidate list, or give the host more RAM. On the same 31 GB host,
`jina-reranker-v3:cpu` completed the mixed corpus that `jina-reranker-v3.5:cpu`
could not, so it is the safer pick where RAM is fixed and small.

These are the model's own batching defaults, not a jina-on-prem setting - there is
no knob in the container to change them.

## Disk planning

```mermaid
flowchart LR
    A[Connected builder
30-200 GB] --> B[.tar.gz
2-12 GB per model]
    B --> C[Air-gapped host
20-100 GB depending on
how many models]
```

- **Builder machine**: 100 GB minimum if bundling 1-2 models. 200 GB+ if bundling all 6 priority models. Each build holds: built image (2-12 GB) + tar.gz (1-7 GB) + Docker BuildKit cache (up to 20 GB).
- **Air-gapped host**: needs ~2x the image size at runtime (image + Docker layer cache). For a 4 GB image, plan 10 GB disk.
- **Reclaim**: `docker builder prune -af` between bundles. `docker system prune -f` after.

## Throughput math

Rough throughput on a single L4, batch size 32:

| Model | Tokens/s | Documents/s (avg 50 tokens) |
|---|---|---|
| v5-text-nano | ~30,000 | ~600 |
| v5-text-small | ~12,000 | ~240 |
| v5-omni-small (text only) | ~6,000 | ~120 |

(Measure on the customer hardware. These are starting estimates from `scripts/benchmark.py`.)

The listwise rerankers are not covered by `scripts/benchmark.py` - it drives
`SentenceTransformer.encode()`, and they are not SentenceTransformer models.
Measured instead through `POST /v1/rerank` on the `:gpu` images, one L4, batch of
32 documents per request, warmed, median of a 30 s steady-state run:

| Model | 50-token documents | 500-token documents |
|---|---|---|
| reranker-v3 | 229 doc-query pairs/s | 21 doc-query pairs/s |
| reranker-v3.5 | 193 doc-query pairs/s | 12 doc-query pairs/s |

Reranker throughput falls off sharply with document length because the whole
batch goes through one forward pass. Budget by total tokens per request, not by
document count.

For higher throughput:
1. **Batch at the client.** Send 32-256 inputs per request, not one at a time.
2. **Use matryoshka** to truncate output dims at query time if the index dim doesn't need to be full size.
3. **Multiple replicas.** A 4-vCPU + 1xL4 host can run one container. Two hosts -> 2x. Load balance with anything (nginx, HAProxy, ALB).
4. **Bigger GPU.** A10G is ~2x L4, A100 ~4x for these models.

## Deployment topologies in the customer environment

Where the jina-on-prem container sits in a typical customer architecture:

```
  outside  │   customer perimeter (firewall / VPN / ZTNA)
  ─────────┼────────────────────────────────────────────────────
           │
   user  ──┼──► ingress ──► app ──► jina-on-prem   ─►  ╳ internet
           │                   │     :8080              (blocked)
           │                   │
           │                   └──► Elasticsearch
           │                          └ (optional inference call)
           │                            back into jina-on-prem
```

Three common placements:

- **Sidecar to the app**: jina-on-prem and the calling app on the same VM/pod. Lowest latency (localhost). Best when the app needs many embedding calls per request.
- **Shared service**: one jina-on-prem host serving multiple apps via internal DNS. Easier to right-size; one place to patch.
- **Behind Elasticsearch**: ES inference service calls jina-on-prem at indexing and query time. Apps talk only to ES. Cleanest for search-only stacks.

## Redundancy

Three patterns:

```
  single host           active-active             k8s
  ──────────            ─────────────             ───
                                                  ┌──► pod A
  client ─► jina       client ─► LB ─► host A     │
                                  │     host B    LB ──► pod B
                                  └──► host B     │
                                                  └──► pod N
```

- **Single host** is fine for POC, dev, internal tools. No redundancy.
- **Active-active** with two hosts behind any L4 load balancer. Models are stateless so any request can go to any replica. Use this for production. Maintain spare image tarballs on disk so you can rebuild a host without rebundling.
- **Kubernetes** if the customer is already running it. Each pod is `docker run` with a `Service` and `Deployment`. Persistent volume not needed (model is in the image). NodeSelector for GPU nodes if you mix GPU and CPU.

A ready-to-apply manifest with Namespace + Deployment + Service + HPA + Ingress lives at [`k8s/jina-on-prem.yaml`](https://github.com/jina-ai/jina-on-prem/blob/main/k8s/jina-on-prem.yaml):

```bash
# Load the image on every node (no internal registry needed):
for n in node-1 node-2 node-3; do
  docker save jina/jina-embeddings-v5-text-small:gpu | ssh $n "docker load"
done

kubectl apply -f k8s/jina-on-prem.yaml
kubectl -n jina-on-prem rollout status deployment/jina-embed
kubectl -n jina-on-prem port-forward svc/jina-embed 8080:8080
curl http://localhost:8080/health
```

The manifest includes:

- `Deployment` with 2 replicas, rolling update strategy, GPU `nodeSelector`
- `Service` (ClusterIP) on port 8080
- `HorizontalPodAutoscaler` that scales 2-8 pods based on CPU
- `Ingress` example for nginx-ingress (comment out if not using)
- Readiness probe (30s initial, 60s grace) + liveness probe (90s initial)

To run two models (embed + rerank), copy the Deployment+Service blocks with different names and images.

## Customer-side prerequisites checklist

Before you go on-site or hand over a bundle:

- [ ] Host with Docker 24+ installed
- [ ] (GPU) NVIDIA driver >= 525, CUDA Container Toolkit installed (`nvidia-smi` works in container)
- [ ] Disk: 2-3x the bundle size free
- [ ] Port 8080 (or chosen port) free
- [ ] Network plan: how does the calling app reach this host? localhost / LAN IP / internal DNS?
- [ ] (Optional) Load balancer in front if multi-replica

If the customer has none of this set up, `scripts/bootstrap-gcp.sh` is a template - copy it and adapt for their cloud or on-prem.

## Next

- [Bundling Guide](Bundling-Guide) - how to actually build the .tar.gz
- [Quick Start](Quick-Start) - first-deploy walkthrough
- [Troubleshooting](Troubleshooting) - VRAM OOM, CUDA mismatch, etc
