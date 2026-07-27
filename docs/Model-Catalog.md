All 29 models supported by jina-on-prem. Auto-generated from [`models/catalog.json`](https://github.com/jina-ai/jina-on-prem/blob/main/models/catalog.json) - re-run `python3 scripts/gen_catalog_md.py` to refresh.

**License note**: Models tagged `CC-BY-NC-4.0` need a commercial license for production use. Contact [Elastic sales](https://www.elastic.co/contact).

## Embeddings

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `jina-embeddings-v5-omni-small` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v5-omni-small) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v5-omni-small) | 1.74B | ~8GB | 32K | 1024 (matryoshka: 32-1024) | multimodal | CC-BY-NC-4.0 |
| `jina-embeddings-v5-omni-nano` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v5-omni-nano) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v5-omni-nano) | 1.04B | ~5GB | 8K | 768 (matryoshka: 32-768) | multimodal | CC-BY-NC-4.0 |
| `jina-embeddings-v5-text-small` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v5-text-small) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v5-text-small) | 677M | ~3GB | 32K | 1024 (matryoshka: 32-1024) | text | CC-BY-NC-4.0 |
| `jina-embeddings-v5-text-nano` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v5-text-nano) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v5-text-nano) | 239M | ~2GB | 8K | 768 (matryoshka: 32-768) | text | CC-BY-NC-4.0 |
| `jina-code-embeddings-1.5b` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-code-embeddings-1.5b) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-code-embeddings-1.5b) | 1.5B | ~4GB | 32K | 1536 (matryoshka: 128-1536) | code | CC-BY-NC-4.0 |
| `jina-code-embeddings-0.5b` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-code-embeddings-0.5b) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-code-embeddings-0.5b) | 494M | ~2GB | 32K | 896 (matryoshka: 64-896) | code | CC-BY-NC-4.0 |
| `jina-embeddings-v4` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v4) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v4) | 3.8B | ~10GB | 32K | 2048 (matryoshka: 128-2048) | multimodal | Qwen Research License |
| `jina-clip-v2` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-clip-v2) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-clip-v2) | 865M | ~4GB | 8K | 1024 (matryoshka: 64-1024) | multimodal | CC-BY-NC-4.0 |
| `jina-embeddings-v3` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v3) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v3) | 570M | ~3GB | 8K | 1024 (matryoshka: 32-1024) | text | CC-BY-NC-4.0 |
| `jina-clip-v1` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-clip-v1) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-clip-v1) | 223M | ~1GB | 8K | 768 | multimodal | Apache-2.0 |
| `jina-embeddings-v2-base-es` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-es) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-es) | 161M | ~1GB | 8K | 768 | text | Apache-2.0 |
| `jina-embeddings-v2-base-code` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-code) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-code) | 137M | ~1GB | 8K | 768 | code | Apache-2.0 |
| `jina-embeddings-v2-base-de` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-de) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-de) | 161M | ~1GB | 8K | 768 | text | Apache-2.0 |
| `jina-embeddings-v2-base-zh` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-zh) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-zh) | 161M | ~1GB | 8K | 768 | text | Apache-2.0 |
| `jina-embeddings-v2-base-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-en) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embeddings-v2-base-en) | 137M | ~1GB | 8K | 768 | text | Apache-2.0 |
| `jina-embedding-b-en-v1` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embedding-b-en-v1) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-embedding-b-en-v1) | 110M | ~1GB | 512 | 768 | text | Apache-2.0 |

## Rerankers

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `jina-reranker-v3.5` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v3.5) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v3.5) | 0.6B | ~3GB | 128K | - | text | CC-BY-NC-4.0 |
| `jina-reranker-v3` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v3) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v3) | 597M | ~3GB | 128K | - | text | CC-BY-NC-4.0 |
| `jina-reranker-m0` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-m0) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-m0) | 2.4B | ~6GB | 10K | - | multimodal | CC-BY-NC-4.0 |
| `jina-reranker-v2-base-multilingual` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v2-base-multilingual) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v2-base-multilingual) | 278M | ~1GB | 1K | - | text | CC-BY-NC-4.0 |
| `jina-reranker-v1-turbo-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v1-turbo-en) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v1-turbo-en) | 37.8M | ~1GB | 8K | - | text | Apache-2.0 |
| `jina-reranker-v1-tiny-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v1-tiny-en) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v1-tiny-en) | 33M | ~1GB | 8K | - | text | Apache-2.0 |
| `jina-reranker-v1-base-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v1-base-en) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-reranker-v1-base-en) | 137M | ~1GB | 8K | - | text | Apache-2.0 |

## ColBERT

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `jina-colbert-v2` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-colbert-v2) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-colbert-v2) | 560M | ~3GB | 8K | 128 (matryoshka: 64-128) | text | CC-BY-NC-4.0 |
| `jina-colbert-v1-en` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-colbert-v1-en) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-colbert-v1-en) | 137M | ~1GB | 8K | 128 | text | Apache-2.0 |

## Readers

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `ReaderLM-v2` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2FReaderLM-v2) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2FReaderLM-v2) | 1.54B | ~4GB | 512K | - | text | CC-BY-NC-4.0 |
| `reader-lm-1.5b` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Freader-lm-1.5b) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Freader-lm-1.5b) | 1.54B | ~4GB | 256K | - | text | CC-BY-NC-4.0 |
| `reader-lm-0.5b` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Freader-lm-0.5b) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Freader-lm-0.5b) | 494M | ~2GB | 256K | - | text | CC-BY-NC-4.0 |

## Vision-Language

| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |
|---|---|---|---|---|---|---|---|
| `jina-vlm` | [cpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-vlm) / [gpu](https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2Fjina-vlm) | 2.4B | ~6GB | 32K | - | multimodal | CC-BY-NC-4.0 |

## Picking a model

Quick rules of thumb:

- **First-time test / latency-critical**: `jina-embeddings-v5-text-nano` (239M, ~2GB, CPU-friendly).
- **Multilingual production embeddings**: `jina-embeddings-v5-text-small` or `jina-embeddings-v4`.
- **Multimodal (text + image)**: `jina-embeddings-v5-omni-small` or `jina-clip-v2`.
- **Code search**: `jina-code-embeddings-1.5b` (or 0.5b for smaller deploys).
- **Reranking after retrieval**: `jina-reranker-v3.5` (best quality) or `jina-reranker-v2-base-multilingual` (faster, CPU-friendly).
- **HTML/document cleanup**: `ReaderLM-v2` (largest context) or `reader-lm-0.5b` (lightweight).

See [API Reference](API-Reference) for the request shapes each model expects.
