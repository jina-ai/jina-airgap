Short definitions for terms used across the wiki.

## A

**Air-gap** — a network configuration where the host cannot reach the public internet (or even an internal network outside its security perimeter). Common in banks, government, defense, hospitals.

**API schema** — the shape of a request/response. jina-on-prem exposes four simultaneously: OpenAI, Cohere, Google Gemini, Voyage AI. Each is a recognized industry contract, so customer apps usually work without code change.

## B

**Bundle** — the act of building a self-contained Docker image with weights, deps, and server code baked in. Also the resulting `.tar.gz`. Phase 1 of jina-on-prem.

## C

**ColBERT** — a "late interaction" retrieval model. Instead of one vector per document, it produces many per-token vectors and scores at query time. Higher quality, more storage. jina-on-prem supports `jina-colbert-v1-en` and `jina-colbert-v2`.

**CrossEncoder** — a model that takes a (query, document) pair and outputs a relevance score. The architecture pattern behind rerankers.

**CUDA** — NVIDIA's GPU programming platform. Required for GPU images. Driver version >=525 needed for our CUDA 12.1 images.

## D

**Deploy** — Phase 2. `docker load` + `docker run` on the offline machine. Takes seconds. No network needed.

**Docker** — the container runtime jina-on-prem targets. Both bundle and deploy phases require Docker.

## E

**Embedding** — a fixed-size numerical vector that represents the meaning of a piece of text (or image, audio, video). Used for semantic search, clustering, classification.

**Elasticsearch inference service** — Elastic's plugin system for calling external models. jina-on-prem registers as `service: openai` (embeddings) or `service: cohere` (rerank).

**EOL (End of Life)** — for Jina SM, the point at which Jina/Elastic stops shipping fixes and standard support for a model generation. Unlike a hosted API, it does **not** stop your deployed model from running - the bundle keeps working indefinitely because you hold the weights. See [Product & Model Lifecycle](Product-And-Model-Lifecycle).

**Maintenance Term** — the window during which a model generation receives runtime security patches, dependency updates, bug fixes, and quantized builds. At least the longer of 36 months from GA or 24 months from its replacement's GA. Covers the runtime and packaging, never the model's trained quality. See [Product & Model Lifecycle](Product-And-Model-Lifecycle).

## F

**FastAPI** — the Python web framework powering the jina-on-prem server. Gives you `/docs` Swagger UI for free.

## G

**GHCR** — GitHub Container Registry (`ghcr.io`). Where prebuilt jina-on-prem images live. Requires `docker login` with a GitHub PAT (`read:packages` scope).

## H

**HF Hub** — Hugging Face Model Hub. Where weights are downloaded from during Phase 1. Never called at runtime (`HF_HUB_OFFLINE=1` baked in).

## L

**L4** — NVIDIA's mid-range datacenter GPU (24 GB VRAM). The workhorse for jina-on-prem deployments.

## M

**Matryoshka** — embeddings designed so any prefix is also a valid embedding. Pass `dimensions: 128` to a 1024-dim model to get a smaller vector with no quality cliff. Named after Russian nesting dolls.

**Multimodal** — handles multiple input types (text + image + audio + video). v5-omni, v4, clip-v2, vlm are multimodal. Other models are text-only.

## O

**Omni** — Jina's prefix for multimodal v5 models (`v5-omni-nano`, `v5-omni-small`). Accept any combination of text, image, audio, video.

**OpenAI-compatible** — accepts requests in OpenAI's `/v1/embeddings` shape. Drop-in for any client using `openai.OpenAI(base_url=...)`.

## P

**Phase 1 / Phase 2** — Phase 1 = bundle (requires network), Phase 2 = deploy (offline). The two-phase model is the whole product.

**Prebuilt** — a `.tar.gz` jina-on-prem already published to GHCR. Saves you from building yourself. See the "Prebuilt" column in the README.

## R

**Reranker** — a model that takes a query + a list of candidate documents and returns relevance scores. Used after retrieval to improve top-K precision. jina-on-prem supports `jina-reranker-v3.5` and earlier versions.

**Runtime** — `cpu` or `gpu`. Bundle once per runtime; deploy the appropriate one on the appropriate host.

## S

**SentenceTransformer** — the Python library used to load and run most embedding models. jina-on-prem uses it internally.

## T

**Task** — embedding optimization hint. v5 supports `retrieval` (default), `text-matching`, `classification`, `clustering`. Pass via the `task` field. Different tasks produce different embeddings for the same input.

**`.tar.gz`** — the gzipped Docker image. Output of `docker save | gzip`. The unit of transport for an air-gapped bundle.

## V

**v5 / v4 / v3 / v2 / v1** — model generations. v5 (2026) is the latest with multimodal omni variants. v3 (2024) is widely deployed and stable. v2/v1 are Apache-2.0 licensed (free for commercial use).

**Voyage AI** — a vendor whose API jina-on-prem is compatible with. Both `/v1/embeddings` (with `input_type` / `output_dimension`) and `/v1/multimodalembeddings` work.

**VLM** — Vision-Language Model. Like a chat LLM but accepts images. jina-on-prem has `jina-vlm` (2.4B params).

**VRAM** — GPU memory. Each model has a recommended minimum; see [Model Catalog](Model-Catalog).
