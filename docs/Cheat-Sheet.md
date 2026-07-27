One-page reference. Screenshot this for customer calls.

## Pitch in one sentence

> Deploy Jina AI embedding, reranker, and reader models on customer infrastructure with zero outbound network calls.

## The two phases

```
   PHASE 1                       PHASE 2
   bundle on connected           deploy on offline
   ───────────────────           ─────────────────

   bundle  ─►  .tar.gz  ─►  docker load  ─►  docker run  ─►  port 8080
   (network)   (USB / SCP)                  (offline)        4 schemas
```

## Commands the customer runs

```bash
# Phase 1 (connected machine):
./scripts/pull-prebuilt.sh jina-embeddings-v5-text-nano cpu
# or to build from source:
python jina-on-prem.py bundle --model MODEL --cpu-only --yes

# Phase 2 (offline machine):
docker load < MODEL.tar.gz
docker run -d -p 8080:8080 jina/MODEL:cpu
curl http://localhost:8080/health
```

## Headline models

| Model | Use case | Params | VRAM | Prebuilt |
|---|---|---|---|---|
| `jina-embeddings-v5-text-nano` | first demo, OpenAI replacement | 239M | ~2GB | yes |
| `jina-embeddings-v5-text-small` | production multilingual | 677M | ~3GB | yes |
| `jina-embeddings-v5-omni-small` | multimodal RAG | 1.74B | ~8GB | yes |
| `jina-reranker-v3.5` | top-K reranking, 131K context, GPU only | 0.6B | ~3GB | yes |
| `jina-reranker-v3` | top-K reranking, CPU-capable | 597M | ~3GB | yes |
| `jina-clip-v2` | image + text search | 865M | ~4GB | yes |
| `jina-embeddings-v3` | stable, widely deployed | 570M | ~3GB | yes |

Full table: [Model Catalog](Model-Catalog).

## Four API schemas (all simultaneous, same container)

| Schema | Endpoint | One-line example |
|---|---|---|
| OpenAI | `POST /v1/embeddings` | `{"input":["hi"]}` |
| Cohere | `POST /v1/embed` | `{"texts":["hi"],"input_type":"search_query"}` |
| Gemini | `POST /v1/models/MODEL:embedContent` | `{"content":{"parts":[{"text":"hi"}]}}` |
| Reranker | `POST /v1/rerank` | `{"query":"...","documents":["...","..."],"top_n":2}` |
| Health | `GET /health` | (returns model + device + schemas) |

## Tasks (v5)

`retrieval` (default), `text-matching`, `classification`, `clustering`. Pass via `task` field. Different task = different vector.

## Matryoshka truncation

`{"input":["hi"], "dimensions": 128}` → returns 128-dim vector. Supported dims: 32, 64, 128, 256, 512, 768/1024/2048 depending on model.

## License

| License | Models | Commercial use |
|---|---|---|
| CC-BY-NC-4.0 | All v5, v4, v3, reranker-v3 / -v3.5, clip-v2, ColBERT, code-embeddings, ReaderLM, vlm | needs commercial license - contact [Elastic sales](https://www.elastic.co/contact) |
| Apache-2.0 | v2 (all variants), v1, clip-v1, reranker v1 | free for any use |

## Hardware quick rules

- < 10 QPS sustained → CPU is fine
- > 10 QPS or low-latency → L4 (24GB VRAM, handles any single model)
- > 100 QPS → multiple replicas behind an LB, or A10G/A100

## Customer pre-flight checklist

- [ ] Docker 24+ installed on target host
- [ ] (GPU) NVIDIA driver >= 525, NVIDIA Container Toolkit, `nvidia-smi` works
- [ ] 2-3x bundle size disk free
- [ ] Port 8080 (or chosen port) open
- [ ] Channel agreed for transferring the `.tar.gz`
- [ ] Decision: single host (POC), HA pair (prod), or k8s

## The three sentences that close most calls

1. "Your data and queries never leave your network - the env vars `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are baked into the image, no code path exists for an outbound call."
2. "It's OpenAI-compatible - any client that uses `openai.OpenAI(base_url=...)` works without code changes."
3. "You can verify the air-gap yourself: run on a host with no egress route, `docker logs` will be silent on outbound attempts. Source code on GitHub is Apache-2.0, auditable."

## Common first-mile errors

| Error | Fix |
|---|---|
| `permission denied connecting to docker API` | `sudo usermod -aG docker $USER`, reconnect |
| `Error response: unauthorized` (on `docker pull`) | `docker login ghcr.io -u USER`; needs `read:packages` PAT |
| `bind: address already in use` | Map a different port: `-p 9090:8080` |
| Container exits with CUDA error | Driver too old: `nvidia-smi` should show >=525 |
| `OOM` on GPU | Check [Model Catalog](Model-Catalog) VRAM; pick smaller model or bigger GPU |

Full list: [Troubleshooting](Troubleshooting).

## Links to keep handy

- Wiki home: https://github.com/jina-ai/jina-on-prem/wiki
- Repo: https://github.com/jina-ai/jina-on-prem
- Issues: https://github.com/jina-ai/jina-on-prem/issues
- Commercial license: https://www.elastic.co/contact
