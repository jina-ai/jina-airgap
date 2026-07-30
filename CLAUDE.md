# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An on-prem / air-gapped deployment toolkit for Jina AI models. It bundles embedding, reranker, reader, ColBERT, CLIP, and VLM models into self-contained Docker images that run fully offline. It is **not** a model-training or LLM-chat serving project — inference is embeddings + reranking, served via `sentence-transformers` / HuggingFace `transformers` on PyTorch (no vLLM, ONNX, or TensorRT).

## How this is publicly positioned

Launch post: [On-prem in under 5 minutes: Jina embedding models now available for on-prem deployment](https://www.elastic.co/search-labs/blog/on-prem-ai-jina-embedding-models), Scott Martens, Elastic Search Labs, 2026-07-23. Useful when writing docs or user-facing copy, so the repo and the public story stay consistent.

- The pitch is all 28 models as self-contained Docker containers with **no external connections, no license servers, and no telemetry**, deployable in under five minutes. The "five minutes" claim maps to the Phase-2 prebuilt path (`scripts/pull-prebuilt.sh` + `docker load`), not to `bundle`, which builds from scratch and takes much longer. The post notes users need a GitHub account and access token to pull from GHCR.
- Named use cases: air-gapped / firewalled environments (defense, intelligence), regulatory compliance (HIPAA, GDPR, data sovereignty), latency-critical edge (robotics, vehicles), cost predictability at high inference volume, and liability reduction (e.g. law firms with privileged material). Cloud APIs are positioned as still preferable for intermittent or batch usage.
- Headline models called out publicly: `jina-embeddings-v5-omni`, `jina-embeddings-v5-text-small`, `jina-embeddings-v5-text-nano`, `jina-reranker-v3`, `jina-clip-v2`.
- Positioned as a drop-in replacement at two levels: for the OpenAI / Cohere / Voyage AI / Gemini / Jina API schemas, and for models served by **EIS** (Elastic Inference Service) in air-gapped Elastic deployments.
- Licensing framing: commercial licensing via Elastic Sales; non-commercial use falls under CC BY-NC 4.0. "No license servers" is consistent with the local HMAC gate below, which never phones home.

## The two-phase mental model

Everything is organized around two phases, and confusing them causes most mistakes:

1. **Bundle (Phase 1, network-connected):** `jina-on-prem.py bundle` reads `models/catalog.json`, runs a two-stage `docker build` that downloads weights from HF Hub, patches model code for offline use, deletes the model repo's own `requirements.txt`, bakes in pinned deps, then `docker save | gzip` → `MODEL.tar.gz`.
2. **Deploy (Phase 2, offline):** `docker load < MODEL.tar.gz` then `docker run -p 8080:8080`. No repo, no Python, no network — just Docker. `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` are baked in so any code path that would hit the network fails instead of phoning home.

## Common commands

```bash
# Browse the model catalog
python jina-on-prem.py list [--type embedding|reranker|reader|colbert|vlm] [--modality text|multimodal|code] [-v] [--json]

# Bundle an image (Phase 1, needs network). Omit --cpu-only for the GPU variant.
python jina-on-prem.py bundle --model jina-embeddings-v5-text-nano --cpu-only --yes
python jina-on-prem.py bundle --model <id> --dry-run     # print the plan, build nothing

# Deploy a saved tarball (Phase 2, offline)
python jina-on-prem.py deploy --image MODEL.tar.gz --port 8080 [--gpu] [--detach]

# Serve directly without Docker (only if the model's deps are already installed locally)
python jina-on-prem.py serve --model jinaai/jina-embeddings-v5-text-nano --port 8080
python jina-on-prem.py serve --local-path /data/models/jina-v5-nano

# Mint an offline license key (see Licensing below)
python jina-on-prem.py keygen --sub acme-corp --days 90 [--model <id>] [--secret <s>] [--json]
```

The CLI (`jina-on-prem.py`) is intentionally **stdlib-only, zero third-party imports** — it must run on a bare builder machine before any deps are installed. Do not add imports of `torch`, `requests`, etc. to it. Model deps live inside the Docker image only. `bundle`/`deploy`/`serve`/`keygen` also have hidden single-word aliases.

## Tests

```bash
# License gate — pure unit tests, no server/Docker/network
python tests/test_license.py

# App import + default-task logic — imports server/app.py (pulls torch/transformers), no weights
python tests/test_default_task.py

# End-to-end API tests against a LIVE server (start a container first)
TEST_URL=http://localhost:8080 python tests/test_e2e.py

# Smoke-test a freshly built image (starts container, waits on /health, hits /v1/embeddings or /v1/rerank)
bash test_airgap.sh jina/jina-embeddings-v5-text-nano:cpu
```

There is no pytest harness or single-test selector — tests are plain scripts with a `check(name, cond)` helper that print PASS/FAIL and exit non-zero on failure. Run a whole file; to run one case, edit the file.

## Architecture and where things live

- **`models/catalog.json`** is the single source of truth: 28 models, each with `hf_repo`, `type`, `deps` (exact pinned versions), `vram_gb`, `matryoshka_dims`, `tasks`, `prebuilt`, etc. Bundle behavior is data-driven from here.
- **`docker/Dockerfile.cpu`** (`python:3.11-slim`) and **`docker/Dockerfile.gpu`** (`pytorch/pytorch` + CUDA, FP16) are both two-stage: stage 1 downloads+patches weights, stage 2 installs pinned deps and copies weights in. GPU image optionally compiles `flash-attn`.
- **`docker/download_model.py`** runs in build stage 1: downloads weights, applies offline-compatibility patches to model code (e.g. `custom_st.py`, `modeling_jina_embeddings_v5.py`), and **deletes every `requirements.txt` in the HF cache** so the model can't auto-upgrade `transformers` at runtime.
- **`server/app.py`** is one FastAPI server speaking **four schemas simultaneously** on `:8080`: OpenAI/Voyage (`/v1/embeddings`), Cohere (`/v1/embed`), Google Gemini (`:embedContent` / `:batchEmbedContents`), plus `/v1/rerank`, `/v1/multimodalembeddings`, and `/v1/chat/completions` (VLM/reader only). Only a schema-specific response shaper is protocol-aware; the underlying `model.encode()` call is shared. `_resolve_task()` maps API-level task names (e.g. Cohere `search_query`) to the model generation's own task names (v5/v4/v3 differ).
- **`server/license.py`** is the license gate (see below), wired in as a FastAPI middleware in `app.py`.
- **`scripts/`**: `bootstrap-gcp.sh` (provision a GCP builder), `pull-prebuilt.sh` (pull a GHCR image and tar.gz it for transport), `gen_catalog_md.py` (regenerate the Model Catalog wiki page from `catalog.json`), `sync-wiki.sh` (push `docs/` to the GitHub wiki).
- **`docs/`** mirrors the GitHub wiki and is edited via PRs, then synced. `docs/Architecture.md` and `docs/Licensing.md` are the deepest references.

## Non-obvious rules (from hard-won debugging — see CONTRIBUTING.md)

- **Never loosen the `transformers`/`sentence-transformers` pins in `catalog.json`.** They are exact for a reason (e.g. v5-text needs `Qwen3Config` from 4.51; v5-omni needs `Qwen3VLVisionConfig` from 4.57). HF model repos request `transformers>=5.x`, which breaks these models — that's why their `requirements.txt` is deleted at build time.
- **Rerankers load as `CrossEncoder`, not `SentenceTransformer`,** and Qwen3-based rerankers need `pad_token = eos_token` set after load or batch inference crashes. Test rerankers on `/v1/rerank`, not `/v1/embeddings`.
- **Do not test air-gap with `--network=none`** — it kills the host↔container network too. The real guarantee is the offline env vars + baked weights; test with normal `-p 8080:8080`.
- `app.py` monkey-patches `transformers.dynamic_module_utils.resolve_trust_remote_code` to always return `True`, using `*args, **kwargs` because the signature changed between transformers 4.51 and 5.x. Preserve that shape.
- **GHCR packages default to private** even from a public repo, and there is no API to flip them — it's a manual per-package web-UI step (`scripts/check-prebuilt-visibility.sh` audits state). Dockerfiles carry `LABEL org.opencontainers.image.source=...` so images link back to the repo.
- The `:gpu-opt` throughput images and their dynamic-batcher env vars (`JINA_BATCH_TOKENS`, `JINA_MERGE_TASK`, `JINA_LEAN`, `JINA_COMPILE`, …) are documented in the README, but that batcher serving stack is **not** in this tree — `server/app.py` is the single-request server. Don't assume those env vars exist in this code.

## Runtime env vars the server reads

`JINA_MODEL_ID` (which model is loaded), `JINA_DTYPE` / `JINA_CPU_AUTOCAST` (precision), `JINA_OFFLINE` / `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_HOME` (offline enforcement + cache), and the license vars below.

## Licensing (license gate)

`server/license.py` + the middleware in `app.py` implement an **offline, HMAC-signed, expiring entitlement signal — not DRM.** The signing secret ships in the image on purpose (防君子不防小人); a technical user can trivially mint or bypass a key. Key design invariant: **a paying, deployed customer must never be blocked.**

- Modes via `JINA_LICENSE_MODE`: **`warn`** (default, fail-open — always serves, only logs + reports in `/health`; ship sold customers here), **`enforce`** (returns 403 on missing/expired/invalid past grace — trials/POCs only), **`off`** (no checks). `JINA_LICENSE_GRACE_DAYS` (default 14) keeps an expired key working in enforce mode. Legacy `JINA_LICENSE_ENFORCE=0/1` still maps to off/enforce.
- Key format: `JINA-<base64url(payload)>.<base64url(hmac_sha256)>`; payload is `{sub, iat, exp, model, v}`. `decide()` is the single authority on whether to serve and is wrapped to fail open on any unexpected error. `/health`, `/`, `/docs`, `/redoc`, `/openapi.json`, `/favicon.ico` are always open; only inference POSTs are ever gated.

## Conventions

- Commit message prefixes in use: `fix:`, `feat:`, `docs:`, `refactor:`, `chore:`, `build:`, `experiment:`.
- Markdown in `docs/` mirrors the wiki — keep API examples to one curl per schema, no verbose request/response schemas (see CONTRIBUTING.md "README maintenance").
- When adding a model: add it to `catalog.json` with pinned `deps`, then regenerate the catalog wiki page via `scripts/gen_catalog_md.py`.
