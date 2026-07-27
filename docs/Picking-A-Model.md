There are 29 models. Most customers need one of six. Use this decision tree.

```mermaid
flowchart TD
    A[What is the customer doing?] --> B{Just text?}
    B -- yes --> C{Multilingual?}
    B -- no --> D{Modalities?}
    C -- mostly English, latency matters --> M1[jina-embeddings-v5-text-nano
239M, ~2GB VRAM]
    C -- 89 languages, quality matters --> M2[jina-embeddings-v5-text-small
677M, ~3GB VRAM]
    C -- need 32K context --> M3[jina-embeddings-v5-text-small
or v4 for biggest]
    D -- text + image --> M4[jina-clip-v2
865M, ~4GB VRAM]
    D -- text + image + audio + video --> M5[jina-embeddings-v5-omni-small
1.74B, ~8GB VRAM]
    D -- code --> M6[jina-code-embeddings-1.5b
1.5B, ~4GB VRAM]

    A --> E{Reranking?}
    E -- top quality --> R1[jina-reranker-v3.5
0.6B, ~3GB VRAM, 131K context]
    E -- smaller host --> R3[jina-reranker-v3
597M, ~3GB VRAM]
    E -- fastest --> R2[jina-reranker-v1-turbo-en
37.8M, CPU OK]

    A --> F{HTML/document cleanup?}
    F -- yes --> RD1[ReaderLM-v2
1.54B, 512K context]
```

## Quick reference by use case

| Use case | Model | Why |
|---|---|---|
| First demo / proof of concept | `jina-embeddings-v5-text-nano` | smallest, fastest to bundle, CPU works |
| Production multilingual semantic search | `jina-embeddings-v5-text-small` | 89 languages, task-aware, 32K context |
| Drop-in for OpenAI `text-embedding-3-small` | `jina-embeddings-v5-text-nano` | similar dim (768), credible quality |
| Multimodal RAG (PDFs with images) | `jina-embeddings-v4` or `v5-omni-small` | image + text in same space |
| Image search | `jina-clip-v2` | optimized for text + image |
| Code search | `jina-code-embeddings-1.5b` or `0.5b` | trained on code |
| Reranking | `jina-reranker-v3.5` | best quality, long context |
| Reranking on a memory-tight host | `jina-reranker-v3` | lighter host-RAM profile on the `:cpu` image - see [Sizing](Sizing-And-Hardware#reranker-host-memory) |
| Cheap fast reranking | `jina-reranker-v1-turbo-en` | 38M params, CPU OK |
| HTML to clean markdown | `ReaderLM-v2` | 512K context, document focused |
| Legacy English embeddings | `jina-embeddings-v2-base-en` | Apache-2.0, no commercial license needed |

## Trade-offs

**Bigger isn't always better.** v5-text-small is a stronger model than v5-text-nano, but the nano is 3x smaller and runs on CPU. For a customer with tight latency SLA and English-only data, nano often wins.

**CC-BY-NC-4.0 license matters for sales.** All v5/v4/v3 models are CC-BY-NC-4.0 - the customer needs a commercial license from Elastic. v2 and v1 models are Apache-2.0 and free for any use. If the customer is allergic to vendor lock-in or wants to evaluate without buying, start with `jina-embeddings-v2-base-en` (or `-zh`, `-de`, `-es`).

**Multimodal models cost more memory.** A v5-omni-small needs ~8GB VRAM. A v5-text-nano needs 2GB. If the customer doesn't actually need image/audio/video, don't pay for it.

**Matryoshka truncation lets you downsize at query time.** v5/v4/v3 support requesting any dim from 32 to the model's max (768 / 1024 / 2048). A customer can index at 1024-dim and query at 128-dim if they want a smaller index. Pass `dimensions: 128` in the request.

## When to use multiple models

A real production stack often combines:

```mermaid
flowchart LR
    Q[User query] --> EMB[Embed query
v5-text-small]
    EMB --> VS[(Vector store
ES with v5-small index)]
    VS -->|top 100| RR[Rerank
reranker-v3.5]
    RR -->|top 10| APP[App]
```

- Embedding model for indexing and first-pass retrieval
- Reranker for high-precision top-K
- Optional ReaderLM if input is messy HTML/PDF

Both models can run side-by-side in two containers, OR you can deploy two replicas of each behind a load balancer if QPS demands it. See [Sizing & Hardware](Sizing-And-Hardware) for capacity planning.

## Next

- [Sizing & Hardware](Sizing-And-Hardware) - how much GPU/RAM/disk
- [Model Catalog](Model-Catalog) - full specs for all 28
- [Customer Scenarios](Customer-Scenarios) - applied examples
