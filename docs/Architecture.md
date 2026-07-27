How the pieces fit together — for engineers integrating or debugging jina-on-prem.

![mascot](images/pixel-mascot-shield.png)

## The two-phase model

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ PHASE 1: BUNDLE (connected machine)                             │
  │                                                                 │
  │  python jina-on-prem.py bundle                                   │
  │         │                                                       │
  │         ▼                                                       │
  │  read models/catalog.json  ─►  pinned deps + hf_repo            │
  │         │                                                       │
  │         ▼                                                       │
  │  DOCKER_BUILDKIT=1 docker build                                 │
  │  │   stage 1 (downloader): download weights from HF Hub,        │
  │  │                          patch model code,                   │
  │  │                          delete model repo requirements.txt  │
  │  │   stage 2 (runtime):    install pinned deps,                 │
  │  │                          copy weights from stage 1,          │
  │  │                          add server/app.py,                  │
  │  │                          HF_HUB_OFFLINE=1 + OFFLINE=1        │
  │  ▼                                                              │
  │  docker save | gzip > MODEL.tar.gz                              │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼  USB / SCP / approved channel
                                 │
  ┌──────────────────────────────┴──────────────────────────────────┐
  │ PHASE 2: DEPLOY (offline machine)                               │
  │                                                                 │
  │  docker load < MODEL.tar.gz                                     │
  │         │                                                       │
  │         ▼                                                       │
  │  docker run -d -p 8080:8080 jina/MODEL                          │
  │         │                                                       │
  │         ▼                                                       │
  │  FastAPI on :8080  ─►  customer app (ES / LlamaIndex / curl)    │
  │  multi-schema API                                               │
  └─────────────────────────────────────────────────────────────────┘
```

## What's in the Docker image

```
  jina/MODEL:cpu  (or :gpu)
  ┌─────────────────────────────────────────────────────────┐
  │  ENV         HF_HUB_OFFLINE=1   TRANSFORMERS_OFFLINE=1  │
  ├─────────────────────────────────────────────────────────┤
  │  SERVER      server/app.py + requirements.txt           │
  │              FastAPI multi-schema API                   │
  ├─────────────────────────────────────────────────────────┤
  │  CODE        custom model code, patched for offline     │
  ├─────────────────────────────────────────────────────────┤
  │  WEIGHTS     model weights + tokenizer   (500 MB - 10 GB)│
  ├─────────────────────────────────────────────────────────┤
  │  DEPS        pinned Python deps from catalog.json       │
  ├─────────────────────────────────────────────────────────┤
  │  BASE        python:3.11-slim    (CPU image)            │
  │              pytorch/pytorch:2.5.1-cuda12.1 (GPU image) │
  └─────────────────────────────────────────────────────────┘
              │
              │  docker run
              ▼
  HTTP server on :8080
```

Critical detail: **the model's own `requirements.txt`** (shipped in some HF repos) is deleted at build time by `docker/download_model.py`. This prevents the model from auto-upgrading transformers to a newer version at runtime, which would break it. The pinned deps win.

## The multi-schema server

```
   client                        endpoint
   ────────────────────────────────────────────────────────────────
   OpenAI SDK             ──►  POST /v1/embeddings           ┐
   Voyage SDK             ──►  POST /v1/embeddings (...)     │
   Cohere SDK             ──►  POST /v1/embed                ├──►  FastAPI server
   Google AI SDK          ──►  POST /v1/models/X:embedContent│         │
   reranker client        ──►  POST /v1/rerank               │         ▼
   ES inference           ──►  service: openai | cohere      ┘   model.encode()
                                                                       │
   GET /health  (status, schemas, multimodal flag)                     ▼
                                                              schema-specific
                                                              response shaper
                                                                       │
                                                                       ▼
                                                                   reply
```

One container, four protocols. The shape adapter is the only schema-aware code; the encode call is shared.

## How weights stay private

```mermaid
sequenceDiagram
    participant Build as Build (connected)
    participant HF as HuggingFace Hub
    participant Image as Docker image
    participant Deploy as Deploy (offline)
    participant App as Customer app

    Build->>HF: download weights, tokenizer, code
    Build->>Build: patch code for offline
    Build->>Build: delete model's requirements.txt
    Build->>Image: bake weights + deps + env
    Note over Image: HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

    Image->>Deploy: transfer .tar.gz (one-time)
    Deploy->>Deploy: docker load + docker run
    App->>Deploy: POST /v1/embeddings
    Deploy->>Deploy: encode locally
    Deploy->>App: vectors
    Note over Deploy,App: zero outbound calls
ever
```

At runtime, any code path that would try to download a missing file fails immediately because of the env vars. The image is self-sufficient.

## CLI structure

```
jina-on-prem.py (single file, stdlib only)
- cmd_list      - browse models/catalog.json
- cmd_bundle    - read catalog, write deps file, run docker build, save tar.gz
- cmd_deploy    - docker load + docker run + health check
- cmd_serve     - run server/app.py directly (skip Docker, if deps installed)
```

No third-party imports in the CLI. The model deps install inside the Docker image only.

## File layout

```
jina-on-prem/
- jina-on-prem.py             CLI (stdlib only)
- models/
  - catalog.json             29 models with pinned deps, vram, modality
- docker/
  - Dockerfile.cpu           two-stage: downloader + runtime
  - Dockerfile.gpu           two-stage with CUDA + cudnn
  - download_model.py        runs in stage 1, downloads + patches
- server/
  - app.py                   FastAPI with 4 schemas + /v1/rerank
  - requirements.txt         server framework deps
- scripts/
  - bootstrap-gcp.sh         provision a builder
  - pull-prebuilt.sh         pull GHCR image and tar.gz it
  - gen_catalog_md.py        regenerate the Model Catalog wiki page
  - sync-wiki.sh             push docs/ to the wiki
  - benchmark.py             throughput micro-benchmark
- docs/                      mirror of this wiki, edited via PRs
- tests/test_e2e.py          E2E API tests
- test_airgap.sh             quick smoke test for a built image
```

## What's intentionally NOT in scope

- **LLM chat / generation** (with a few exceptions like ReaderLM and jina-vlm). Use Ollama, vLLM, or LocalAI for chat models.
- **Distributed inference** across multiple GPUs. Each container is a single GPU. For more throughput, run more containers.
- **Fine-tuning**. Bundles serve as-is.
- **Streaming responses**. Embeddings and reranking are batch operations.
- **Auth / RBAC** at the API layer. Add this at your load balancer / API gateway.

## Next

- [API Reference](API-Reference) - exact request/response shapes per schema
- [Sizing & Hardware](Sizing-And-Hardware) - capacity planning
- [Troubleshooting](Troubleshooting) - debugging the pieces above
