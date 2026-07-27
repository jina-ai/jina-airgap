A deployed jina-on-prem server exposes four embedding API schemas (and a reranker endpoint) on the same port. Pick whichever your client already speaks - they all hit the same model.

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

| Schema | Endpoint | Drop-in for |
|---|---|---|
| OpenAI | `POST /v1/embeddings` | OpenAI SDK, Elasticsearch inference, LlamaIndex, LangChain |
| Voyage AI | `POST /v1/embeddings` (with `input_type` / `output_dimension`) | Voyage SDK |
| Cohere | `POST /v1/embed` | Cohere SDK |
| Google Gemini | `POST /v1/models/{model}:embedContent`, `:batchEmbedContents` | Google AI SDK |
| Reranker | `POST /v1/rerank` | Cohere reranker SDK |
| Utility | `GET /health` | health checks |

![api-schemas](images/04-schemas.gif)

## Health

```bash
curl http://localhost:8080/health
```
```json
{"status":"ok","model":"jinaai/jina-embeddings-v5-text-nano","device":"cpu","ready":true,"multimodal":false,"schemas":["openai","voyage","gemini","cohere"]}
```

## OpenAI / Voyage

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["Hello world"]}'
```

Optional fields:

| Field | Schema | Notes |
|---|---|---|
| `model` | OpenAI / Voyage | accepted; echoed in response. The actual model is fixed by the container |
| `task` | (extension) | `retrieval`, `text-matching`, `classification`, `clustering` for v5; `retrieval.query` / `.passage` for v3. Unknown values silently fall back to default |
| `input_type` | Voyage | `query` / `document` - mapped to `task` automatically |
| `dimensions` | OpenAI | matryoshka truncation - request any supported dim (32-768/1024/2048) |
| `output_dimension` | Voyage | alias for `dimensions` |

Python:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
resp = client.embeddings.create(model="jina-embeddings-v5-text-nano", input=["Hello world"])
```

Matryoshka:
```bash
curl -s http://localhost:8080/v1/embeddings \
  -d '{"input":["Hello"], "dimensions": 128}' \
  -H 'Content-Type: application/json' \
  | jq '.data[0].embedding | length'
# 128
```

## Cohere

```bash
curl -s http://localhost:8080/v1/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Hello world"], "input_type": "search_query"}'
```

Response shape (matches Cohere's `/v1/embed`):
```json
{
  "id": "...",
  "texts": ["Hello world"],
  "embeddings": {"float": [[ ... 768 floats ... ]]},
  "meta": {...},
  "response_type": "embedding"
}
```

`input_type` -> `task` mapping:
- `search_query` -> retrieval (query side)
- `search_document` -> retrieval (passage side)
- `classification` -> classification
- `clustering` -> clustering

## Gemini

```bash
# single
curl -s "http://localhost:8080/v1/models/jina-embeddings-v5-text-nano:embedContent" \
  -H 'Content-Type: application/json' \
  -d '{"content": {"parts": [{"text": "Hello"}]}, "taskType": "RETRIEVAL_QUERY"}'

# batch
curl -s "http://localhost:8080/v1/models/jina-embeddings-v5-text-nano:batchEmbedContents" \
  -H 'Content-Type: application/json' \
  -d '{"requests": [
    {"content": {"parts": [{"text": "first"}]}},
    {"content": {"parts": [{"text": "second"}]}}
  ]}'
```

Response uses Gemini's shape: `{"embedding": {"values": [...]}}` for single, `{"embeddings": [...]}` for batch.

`taskType` mapping: `RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`.

## Reranker

```bash
curl -s http://localhost:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "best programming language for AI",
    "documents": [
      "Python is the most popular language for ML",
      "Bananas are yellow",
      "PyTorch and TensorFlow use Python"
    ],
    "top_n": 2
  }'
```

Response (Cohere-compatible):
```json
{
  "model": "jinaai/jina-reranker-v3.5",
  "results": [
    {"index": 0, "relevance_score": 0.21, "document": {"text": "Python is the most popular..."}},
    {"index": 2, "relevance_score": -0.003, "document": {"text": "PyTorch and TensorFlow..."}}
  ],
  "meta": {"elapsed_ms": 1573.1}
}
```

> Reranker containers expose `/v1/rerank` only - calling `/v1/embeddings` on a reranker returns HTTP 500.

## Multimodal inputs (omni / clip / v4 / vlm)

Multimodal models accept images, audio, and video alongside text. Media must be base64-encoded inline (no URLs - air-gap forbids outbound fetches). Max 10 MB per input.

```bash
# Image-only embedding
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": [{"type": "image_base64",
                   "image_base64": {"base64": "<B64>", "mime_type": "image/png"}}]}'

# Fused text + image -> single embedding
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": [{"content": [
          {"type": "text", "text": "A red square"},
          {"type": "image", "format": "base64", "value": "<B64>"}
       ]}]}'
```

Text-only models return HTTP 400 with a helpful list of multimodal-capable model IDs if you send multimodal input.

## Embedding tasks

`task` is the high-leverage field. Embeddings optimized for retrieval differ from those for classification or clustering.

| Model family | Supported tasks |
|---|---|
| v5-text / v5-omni | `retrieval` (default), `text-matching`, `classification`, `clustering` |
| v4 | `retrieval` (default), `text-matching`, `code` |
| v3 | `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `separation` |
| v2 / v1 | no task field; passed values are ignored |

The same model called with `retrieval.query` vs `retrieval.passage` (v3) returns **different** vectors - this is the asymmetric retrieval pattern.

## Elasticsearch integration

Embeddings (`service: openai`):
```json
PUT _inference/text_embedding/jina-embed
{
  "service": "openai",
  "service_settings": {
    "url": "http://your-host:8080/v1/embeddings",
    "model_id": "jina-embeddings-v5-text-nano",
    "api_key": "not-needed"
  }
}
```

Reranker (`service: cohere`):
```json
PUT _inference/rerank/jina-rerank
{
  "service": "cohere",
  "service_settings": {
    "url": "http://your-host:8080/v1/rerank",
    "model_id": "jina-reranker-v3.5",
    "api_key": "not-needed"
  }
}
```

Then query as usual via the `inference_id`. The `api_key` is required by the ES schema but unused server-side.

## Auto-generated docs

Run the server and hit `GET /docs` for the FastAPI-generated Swagger UI - it's always in sync with the deployed code.

## Source of truth

Full request/response shapes: [`server/app.py`](https://github.com/jina-ai/jina-on-prem/blob/main/server/app.py).
