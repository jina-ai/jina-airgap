Common questions with short answers.

## Business / licensing

**Do we need to buy a license?**
For most production use yes - all Jina v5/v4/v3 models are CC-BY-NC-4.0 and the "NC" forbids commercial use without a license. v2 and v1 models are Apache-2.0, free for any use. Contact [Elastic sales](https://www.elastic.co/contact). The CLI itself, the toolkit, and the Docker images are Apache-2.0.

**Can I evaluate without a commercial license?**
Yes - use any Apache-2.0 model (`jina-embeddings-v2-base-en`, `jina-clip-v1`, `jina-reranker-v1-*`). Quality is older but the deployment workflow is identical. Upgrade to a v5 model when you are ready for production.

**Where do the weights come from?**
HuggingFace Hub at bundle time. Repos under `jinaai/` org. Once bundled, the offline deploy never touches HuggingFace.

**Does this phone home or send telemetry?**
No. The toolkit has no analytics, no license check, no version-check ping. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are baked into the image.

## Technical

**Is internet required during deploy?**
No. Internet is only needed during *bundling* (downloading model weights). The bundle is a single `.tar.gz` that can be transferred via any approved channel.

**Is a GPU required?**
No, but throughput will be much lower on CPU. See [Sizing & Hardware](Sizing-And-Hardware).

**Can we run multiple models on the same host?**
Yes - each is a separate container on a separate port. With 1xL4 (24GB VRAM) you can hold two ~3GB-VRAM models comfortably.

**Does it support Kubernetes?**
Yes, no special integration - it's just a stateless container with one HTTP port. Sample manifest in [Sizing & Hardware -> Redundancy](Sizing-And-Hardware#redundancy).

**Can I run it on Apple Silicon?**
Yes, with a flag. The CPU image is `linux/amd64`, so `docker pull` fails with `no matching manifest for linux/arm64/v8` until you pass `--platform linux/amd64` to both `pull` and `run` (see [Troubleshooting](Troubleshooting#no-matching-manifest-on-apple-silicon)). It then works under Rosetta emulation but is slow. For Mac dev, run `python jina-on-prem.py serve --model X` directly via Python (the `serve` command bypasses Docker).

**Will it work with our load balancer / WAF?**
Yes - it's plain HTTP/JSON. Add TLS, auth headers, IP allow-lists at the LB layer.

**What if the model upstream gets pulled from HuggingFace?**
The bundle has the weights frozen. The deployed container will keep working forever, regardless of upstream changes.

## Integration

**Is it OpenAI-compatible?**
Yes. `POST /v1/embeddings` accepts and returns the OpenAI schema. Drop in via `openai.OpenAI(base_url=...)`.

**Does it work with Elasticsearch inference service?**
Yes - register it as `service: openai` for embeddings, `service: cohere` for reranking. Example in [API Reference -> Elasticsearch integration](API-Reference#elasticsearch-integration).

**Does it work with LlamaIndex / LangChain / Haystack?**
Yes - any framework that takes an OpenAI-compatible endpoint or Cohere-compatible endpoint works. Just point `base_url` at the deployed host.

**Can it serve embeddings AND reranking from the same container?**
No - one container hosts one model. Run two containers for two models.

**Does the Gemini schema work with the Google AI SDK?**
The endpoint shape matches `models/{model}:embedContent` and `:batchEmbedContents`. SDK clients that hit those endpoints by URL will work; SDK clients that go through Google's auth flow won't (this is an in-network endpoint, no auth).

## Operations

**How do I update to a new model version?**
Build a new bundle on the connected machine, transfer it, `docker load`, point the load balancer at the new container, drain the old one. No state to migrate.

**How do I monitor it?**
`GET /health` for liveness/readiness probes. Any HTTP-level metric (request rate, latency, error code) via your usual stack (Prometheus pull from in front of the LB, or sidecar exporter). The container itself doesn't expose metrics endpoints today.

**How do I know my embeddings are deterministic across deploys?**
Same model, same dtype (`gpu_dtype` in catalog), same transformers/torch versions, same input -> same output bit-exact. The bundle pins all of these. Confirm by hashing the response on two replicas.

**Can I rotate models without downtime?**
Yes - blue/green at the load balancer. Bring up a second container with the new model on a different port, switch the LB target, then tear down the old.

**What logs does the container emit?**
FastAPI access logs + model load info + warnings. `docker logs <container>` to see them. Quiet by default (no per-request payload logging).

## Comparison with alternatives

**"Why not Ollama / vLLM / LocalAI?"**
Those are great projects for hosted-yourself LLMs. jina-on-prem is specifically for *embeddings* and *rerankers* (the search workload), wraps Jina-specific models with their custom tokenizers and adapters, and standardizes on a multi-schema API. For LLM chat, use Ollama or vLLM separately.

**"Why not export the model to ONNX and run with onnxruntime?"**
You can, and for some models that's a fine path. jina-on-prem's value-add is: pinning the exact transformers/torch versions per model (custom code requires specific versions), wrapping the multi-schema API, and the bundle-and-transfer workflow with documented walkthroughs. ONNX export drops some model-specific code paths (LoRA adapters, custom rerankers) that the maintained Python implementation supports.

**"Can you guarantee it never sends data out?"**
Yes by design - HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 + no other outbound code. The image and source code on GitHub are auditable. Stronger guarantee: run it on a host with no egress route at all (most air-gapped environments already enforce this).

**"What's the long-term commitment?"**
The Apache-2.0 toolkit, Dockerfiles, and CLI are public. If Jina shuts down tomorrow, customers can keep rebundling existing models with their weights. The CC-BY-NC license on weights is what they're paying for, not the runtime. For the full support/maintenance horizon and what "end of life" means for an air-gapped model, see [Product & Model Lifecycle (EOL)](Product-And-Model-Lifecycle).

**"Can you patch the model to fix its accuracy on our task, like a software bug?"**
No. A model's quality is baked into its frozen, trained weights - there is no line of code to change. Better quality on a task only comes from a newer model generation (new weights), which you adopt as a migration on your own schedule. Within a generation we maintain the runtime, dependencies, and quantized builds of the same weights - not the model's learned behaviour. Full explanation: [Product & Model Lifecycle -> Models are not software](Product-And-Model-Lifecycle#models-are-not-software-what-maintenance-really-means).

## Next

- [Product & Model Lifecycle (EOL)](Product-And-Model-Lifecycle) - how long a deployed model is supported
- [Troubleshooting](Troubleshooting) - errors and fixes
- [Customer Scenarios](Customer-Scenarios) - applied use cases
- [API Reference](API-Reference) - exact request/response shapes
