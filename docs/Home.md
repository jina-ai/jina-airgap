**Deploy Jina AI's embedding, reranker, and reader models inside customer environments that cannot reach the internet.**

![bundle flow](images/pixel-bundle-flow.png)

For sales/SA/field engineers walking a customer through their first deployment, start with **[Why Air-Gap](Why-Airgap)**, then **[Quick Start](Quick-Start)**. For developers integrating the API into an application, start with **[API Reference](API-Reference)**.

## At a glance

```
    PHASE 1 (network)                              PHASE 2 (offline)
    ─────────────────                              ─────────────────

  ┌────────────────┐    USB / SCP / disk    ┌────────────────┐    port 8080
  │ connected host │   ──────────────────►  │ air-gapped host│   ──────►  app
  │                │       .tar.gz          │                │
  │ jina-on-prem.py │                        │  docker load   │  OpenAI / Cohere
  │   bundle       │                        │  docker run    │  Gemini / Voyage
  └────────────────┘                        └────────────────┘
        │                                          │
        ▼                                          ▼
  download weights                          serve embeddings,
  + deps from HF Hub                        reranking, readers
  docker build                              zero outbound calls
```

That's the whole product. The connected machine has internet to fetch model weights and dependencies. Everything is baked into a single Docker image and exported as a `.tar.gz`. The offline machine only needs Docker.

![demo](images/03-deploy.gif)

## What's supported

- **29 models**: Jina embeddings (v5, v4, v3, v2), rerankers, ColBERT, CLIP, ReaderLM, VLM. See [Model Catalog](Model-Catalog).
- **4 API schemas simultaneously**: OpenAI, Cohere, Google Gemini, Voyage AI - drop-in for any client.
- **Multimodal**: text + image + audio + video on omni/clip/v4 models.
- **GPU and CPU**: same model can be packaged either way.
- **Elasticsearch inference service**: works as a `service: openai` endpoint out of the box.

## Pick your starting point

| You are... | Start here |
|---|---|
| An SA/sales engineer evaluating jina-on-prem for a customer | [Why Air-Gap](Why-Airgap), then [Customer Scenarios](Customer-Scenarios) |
| Comparing this against Ollama / vLLM / ONNX / hosted API | [Comparison vs alternatives](Comparison) |
| A field engineer deploying at a customer site | [Quick Start](Quick-Start), then [Sizing & Hardware](Sizing-And-Hardware) |
| A developer integrating the API | [API Reference](API-Reference) |
| Building a new bundle from scratch | [Bundling Guide](Bundling-Guide) |
| Rolling out a new model version | [Versioning & Updates](Versioning-And-Updates) |
| Hitting an error | [Troubleshooting](Troubleshooting), [FAQ](FAQ) |

## License note

Most Jina v5/v4/v3 models are **CC-BY-NC-4.0**: commercial use needs a license. Contact [Elastic sales](https://www.elastic.co/contact). v2 and v1 models are Apache-2.0 and free for any use. Per-model license is in the [Model Catalog](Model-Catalog).
